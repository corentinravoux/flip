"""Differentiable forward model: initial conditions + LPT/PM evolution via JaxPM.

This module implements a JAX-differentiable pipeline:

    cosmology + P(k) + seed → Gaussian IC → LPT displacements → [PM] → fields

Key design choices (informed by copilot branch review):

* **DC-mode-safe IC generation**: :func:`_differentiable_linear_field` replaces
  ``jaxpm.pm.linear_field`` with a version that avoids NaN gradients at k=0
  by masking the DC mode multiplicatively (so its gradient is 0, not NaN).
* **Manual 1LPT**: :func:`_run_lpt` reimplements 1LPT without calling
  ``jaxpm.pm.lpt`` to avoid a known caching incompatibility in jaxpm when
  a JAX-traced cosmology is passed (breaks ``jax.grad``).
* **PM time-stepping** (optional): :class:`ForwardModel` also supports full
  leapfrog PM integration for beyond-ZA accuracy.

Typical usage::

    from flip.simulation.generator import ForwardModel, get_cosmology

    cosmo = get_cosmology(omega_m=0.3, sigma8=0.8)
    model = ForwardModel(mesh_shape=(64,64,64), box_size=(512.,512.,512.),
                         a_final=1.0, lpt_only=True)
    density, velocity = model.get_fields(cosmo, seed=42)
"""

import math

try:
    import jax
    import jax.numpy as jnp
    import jax_cosmo as jc
    from jax_cosmo import background, power
    from jaxpm.distributed import fft3d, ifft3d, normal_field
    from jaxpm.growth import growth_factor, growth_rate
    from jaxpm.kernels import fftk
    from jaxpm.painting import cic_paint_dx
    from jaxpm.pm import pm_forces
except ImportError:
    import numpy as jnp

from flip.utils import create_log

log = create_log()

#: H₀ in km/s/(Mpc/h). The h factors cancel: H₀ = 100h km/s/Mpc, 1 Mpc/h = (1/h) Mpc.
_H0_UNIT = 100.0


def get_cosmology(
    omega_m,
    sigma8,
    h=0.6774,
    omega_b=0.0486,
    n_s=0.9667,
    w0=-1.0,
    wa=0.0,
    omega_k=0.0,
):
    """Build a ``jax_cosmo.Cosmology`` from standard parameters.

    Args:
        omega_m (float): Total matter density Ω_m.
        sigma8 (float): RMS matter fluctuation amplitude σ₈.
        h (float): Dimensionless Hubble parameter. Default 0.6774.
        omega_b (float): Baryon density Ω_b. Default 0.0486.
        n_s (float): Scalar spectral index. Default 0.9667.
        w0 (float): Dark energy EoS constant. Default -1.0.
        wa (float): Dark energy EoS evolution. Default 0.0.
        omega_k (float): Curvature density. Default 0.0.

    Returns:
        jax_cosmo.Cosmology: Ready for JaxPM simulations.
    """
    return jc.Cosmology(
        h=h,
        Omega_b=omega_b,
        Omega_c=omega_m - omega_b,
        w0=w0,
        wa=wa,
        n_s=n_s,
        sigma8=sigma8,
        Omega_k=omega_k,
    )


def _differentiable_linear_field(mesh_shape, box_size, pk_fn, seed):
    """Generate a Gaussian linear density field with NaN-safe gradients at k=0.

    ``jaxpm.pm.linear_field`` evaluates P(k) at all modes including k=0
    (the DC mode).  For typical jax_cosmo power spectra, P(0) = 0 but
    ∂P/∂σ₈ at k=0 is NaN, which corrupts ``jax.grad``.  This function
    avoids the issue by:

    1. Replacing k=0 with a dummy value (k=1) before calling ``pk_fn``.
    2. Setting the DC amplitude to zero with a **multiplicative** mask
       (not ``jnp.where``), so the gradient of the mask itself is 0, not NaN.

    Args:
        mesh_shape (tuple[int, int, int]): Grid dimensions.
        box_size (jnp.ndarray): Box size [Lx, Ly, Lz] in Mpc/h.
        pk_fn (callable): P(k) mapping wavenumbers [h/Mpc] → [(Mpc/h)³].
        seed (jax.random.PRNGKey): Random seed.

    Returns:
        jnp.ndarray: Real-space linear density field, shape ``mesh_shape``.
    """
    field = normal_field(seed=seed, shape=mesh_shape)
    field = fft3d(field)

    kvec = fftk(field)
    kmesh = sum(
        (kk / box_size[i] * mesh_shape[i]) ** 2 for i, kk in enumerate(kvec)
    ) ** 0.5

    # Replace k=0 with 1.0 to avoid NaN from pk_fn at DC mode
    kmesh_safe = jnp.where(kmesh > 0, kmesh, jnp.ones_like(kmesh))

    volume = jnp.prod(jnp.array(mesh_shape)) / jnp.prod(box_size)
    pkmesh = pk_fn(kmesh_safe) * volume

    # Multiplicative mask: gradient is 0 at DC mode (not NaN as with jnp.where)
    dc_mask = (kmesh > 0).astype(jnp.float32)

    field = field * jnp.sqrt(pkmesh) * dc_mask
    return ifft3d(field)


def _run_lpt(cosmo, initial_conditions, a):
    """Manual 1LPT (Zel'dovich approximation): avoids jaxpm caching bug.

    ``jaxpm.pm.lpt`` calls ``dGfa`` internally, which has a known caching
    incompatibility when the cosmology is a JAX-traced value (as during
    ``jax.grad``).  This function reimplements 1LPT directly:

        dx = D₁(a) × Ψ    (Zel'dovich displacement)
        p  = a² f₁ H(a) dx  (canonical momentum)

    where Ψ is the Zel'dovich displacement field from the gravitational force.

    Args:
        cosmo (jax_cosmo.Cosmology): Cosmological parameters.
        initial_conditions (jnp.ndarray): Linear density field on the mesh,
            shape ``mesh_shape``.
        a (float): Scale factor.

    Returns:
        tuple[jnp.ndarray, jnp.ndarray]: ``(dx, p)`` — displacement and
        canonical momentum, both shape ``(*mesh_shape, 3)``.
    """
    mesh_shape = initial_conditions.shape
    a_arr = jnp.atleast_1d(a)

    E = jnp.sqrt(background.Esqr(cosmo, a_arr))[0]
    D1 = growth_factor(cosmo, a_arr)[0]
    f1 = growth_rate(cosmo, a_arr)[0]

    # Gravitational force from the linear density field (Ψ in ZA)
    particles = jnp.zeros((*mesh_shape, 3))
    delta_k = fft3d(initial_conditions)
    psi = pm_forces(particles, delta=delta_k, paint_absolute_pos=False)

    dx = D1 * psi
    p = a_arr[0] ** 2 * f1 * E * dx
    return dx, p


class ForwardModel:
    """JAX-differentiable forward model: cosmology → density & velocity fields.

    Wraps JaxPM's LPT (and optionally PM time-stepping) to produce density
    contrast and peculiar velocity fields on a 3D Cartesian mesh. The full
    pipeline is differentiable w.r.t. cosmological parameters.

    Uses a DC-mode-safe IC generator and a manual 1LPT implementation
    to ensure ``jax.grad`` works correctly through the full pipeline.

    Args:
        mesh_shape (tuple[int, int, int]): Grid resolution (Nx, Ny, Nz).
            Fixed at construction; JAX traces static shapes.
        box_size (tuple[float, float, float]): Box dimensions in Mpc/h.
        a_final (float): Target scale factor (``1/(1+z_survey)``). Default: 1.0.
        a_initial (float): Scale factor for LPT initial conditions when
            ``lpt_only=False``. Default: 0.1.
        n_steps (int): PM leapfrog time steps. Ignored when ``lpt_only=True``.
        lpt_only (bool): If True, skip PM and use 1LPT positions directly.
            Faster and differentiable; less accurate at small scales.
        pk_fn (callable | None): P(k) callable ``k [h/Mpc] → P(k) [(Mpc/h)³]``.
            If None, uses ``jax_cosmo.power.linear_matter_power`` evaluated at
            ``a_final``. Pass a custom callable to use CLASS/CAMB spectra.
    """

    def __init__(
        self,
        mesh_shape,
        box_size,
        a_final=1.0,
        a_initial=0.1,
        n_steps=10,
        lpt_only=True,
        pk_fn=None,
    ):
        self.mesh_shape = tuple(int(n) for n in mesh_shape)
        self.box_size = jnp.array(box_size, dtype=jnp.float32)
        self.a_final = float(a_final)
        self.a_initial = float(a_initial)
        self.n_steps = int(n_steps)
        self.lpt_only = bool(lpt_only)
        self._pk_fn_override = pk_fn

        log.info(
            "ForwardModel: mesh=%s, box=%s Mpc/h, a_final=%.4f, lpt_only=%s",
            self.mesh_shape,
            tuple(float(x) for x in self.box_size),
            self.a_final,
            self.lpt_only,
        )

    # ------------------------------------------------------------------
    # Core pipeline steps
    # ------------------------------------------------------------------

    def _get_pk_fn(self, cosmo):
        """Return the P(k) callable for initial condition generation."""
        if self._pk_fn_override is not None:
            return self._pk_fn_override
        a = self.a_final

        def pk(k):
            return power.linear_matter_power(cosmo, k, a=a)

        return pk

    def generate_initial_conditions(self, cosmo, seed):
        """Draw a Gaussian random linear density field.

        Args:
            cosmo (jax_cosmo.Cosmology): Cosmological parameters.
            seed (int | jax.random.PRNGKey): Random seed.

        Returns:
            jnp.ndarray: Linear density field of shape ``mesh_shape``.
        """
        if isinstance(seed, int):
            seed = jax.random.PRNGKey(seed)
        pk_fn = self._get_pk_fn(cosmo)
        return _differentiable_linear_field(self.mesh_shape, self.box_size, pk_fn, seed)

    def run_lpt(self, cosmo, initial_conditions, a=None):
        """Apply 1LPT to produce displacements and momenta.

        Args:
            cosmo (jax_cosmo.Cosmology): Cosmological parameters.
            initial_conditions (jnp.ndarray): Linear density field.
            a (float | None): Scale factor. Defaults to ``a_final``.

        Returns:
            tuple[jnp.ndarray, jnp.ndarray]: ``(dx, p)`` — displacement and
            canonical momentum, both shape ``(*mesh_shape, 3)``.
        """
        if a is None:
            a = self.a_final
        return _run_lpt(cosmo, initial_conditions, a)

    def run_pm(self, cosmo, dx_init, p_init, a_out=None):
        """Integrate PM equations of motion from ``a_initial`` to ``a_final``.

        Args:
            cosmo (jax_cosmo.Cosmology): Cosmological parameters.
            dx_init (jnp.ndarray): LPT displacements at ``a_initial``.
            p_init (jnp.ndarray): LPT momenta at ``a_initial``.
            a_out (jnp.ndarray | None): Output scale factors. Last entry is
                the output epoch. Defaults to linspace.

        Returns:
            tuple[jnp.ndarray, jnp.ndarray]: ``(dx, p)`` at last ``a_out``.
        """
        from jax.experimental.ode import odeint
        from jaxpm.pm import make_ode_fn

        if a_out is None:
            a_out = jnp.linspace(self.a_initial, self.a_final, self.n_steps)

        ode_fn = make_ode_fn(self.mesh_shape)
        traj = odeint(
            ode_fn,
            [dx_init, p_init],
            a_out,
            cosmo,
            self.mesh_shape,
            self.box_size,
        )
        return traj[-1, 0], traj[-1, 1]

    # ------------------------------------------------------------------
    # High-level entry points
    # ------------------------------------------------------------------

    def __call__(self, cosmo, seed, a_out=None):
        """Run forward model: cosmology + seed → (displacement, momentum).

        Args:
            cosmo (jax_cosmo.Cosmology): Cosmological parameters.
            seed (int | jax.random.PRNGKey): IC random seed.
            a_out (jnp.ndarray | None): PM output scale factors (PM mode only).

        Returns:
            tuple[jnp.ndarray, jnp.ndarray]: ``(dx, p)`` at ``a_final``.
        """
        ic = self.generate_initial_conditions(cosmo, seed)
        dx, p = self.run_lpt(cosmo, ic)
        if self.lpt_only:
            return dx, p
        return self.run_pm(cosmo, dx, p, a_out=a_out)

    def get_fields(self, cosmo, seed, a_out=None):
        """Run simulation and return density contrast + velocity fields.

        Args:
            cosmo (jax_cosmo.Cosmology): Cosmological parameters.
            seed (int | jax.random.PRNGKey): IC random seed.
            a_out (jnp.ndarray | None): PM output scale factors (PM mode).

        Returns:
            tuple[jnp.ndarray, jnp.ndarray]:
              * ``density_field``: δ(x), shape ``mesh_shape``.
              * ``velocity_field``: v(x) in km/s, shape ``(*mesh_shape, 3)``.
        """
        dx, p = self(cosmo, seed, a_out=a_out)

        density_field = cic_paint_dx(dx)

        a_arr = jnp.atleast_1d(self.a_final)
        E = jnp.sqrt(background.Esqr(cosmo, a_arr))[0]
        cell_size = self.box_size / jnp.array(self.mesh_shape, dtype=jnp.float32)
        velocity_field = p / (a_arr[0] ** 2 * E) * cell_size * _H0_UNIT

        return density_field, velocity_field

    # ------------------------------------------------------------------
    # Convenience constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_survey_geometry(
        cls,
        rcom_max,
        cell_size_mpc,
        z_survey=0.0,
        lpt_only=True,
        n_steps=10,
        padding_factor=1.2,
    ):
        """Build a ForwardModel sized to enclose a given survey volume.

        Box side is ``2 × rcom_max × padding_factor``, snapped to a power of
        two for FFT efficiency.

        Args:
            rcom_max (float): Maximum survey comoving distance [Mpc/h].
            cell_size_mpc (float): Target cell size [Mpc/h].
            z_survey (float): Survey redshift (sets ``a_final``).
            lpt_only (bool): Use LPT only (default True).
            n_steps (int): PM steps when ``lpt_only=False``.
            padding_factor (float): Box padding. Default 1.2.

        Returns:
            ForwardModel: Configured instance.
        """
        box_side_raw = 2.0 * rcom_max * padding_factor
        n_mesh = int(2 ** round(math.log2(box_side_raw / cell_size_mpc)))
        box_side = n_mesh * cell_size_mpc
        a_final = 1.0 / (1.0 + float(z_survey))

        log.info(
            "ForwardModel.from_survey_geometry: rcom_max=%.1f, cell=%.1f → "
            "mesh=(%d,%d,%d), box=%.1f Mpc/h, a_final=%.4f",
            rcom_max, cell_size_mpc, n_mesh, n_mesh, n_mesh, box_side, a_final,
        )
        return cls(
            mesh_shape=(n_mesh, n_mesh, n_mesh),
            box_size=(box_side, box_side, box_side),
            a_final=a_final,
            lpt_only=lpt_only,
            n_steps=n_steps,
        )

    def __repr__(self):
        return (
            f"ForwardModel(mesh={self.mesh_shape}, "
            f"box={tuple(float(x) for x in self.box_size)}, "
            f"a_final={self.a_final:.4f}, lpt_only={self.lpt_only})"
        )
