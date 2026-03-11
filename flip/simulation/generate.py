"""Differentiable simulation of density and velocity fields using JaxPM.

This module provides JAX-differentiable functions to generate large-scale
structure simulations using Lagrangian Perturbation Theory (LPT) via the
JaxPM package.  All operations are implemented in JAX to support automatic
differentiation with respect to cosmological parameters.

Notes:
    The ``jaxpm`` and ``jax_cosmo`` packages must be installed to use this
    module::

        pip install jaxpm jax_cosmo

Examples:
    >>> import jax
    >>> import jax.numpy as jnp
    >>> import jax_cosmo as jc
    >>> from flip.simulation import generate
    >>> cosmo = generate.get_cosmology(omega_m=0.3, sigma8=0.8)
    >>> seed = jax.random.PRNGKey(0)
    >>> density, velocity = generate.generate_density_and_velocity(
    ...     cosmo, mesh_shape=(32, 32, 32), box_size=[256., 256., 256.],
    ...     seed=seed
    ... )
"""

import jax
import jax.numpy as jnp
import jax_cosmo as jc
from jaxpm.distributed import fft3d, ifft3d, normal_field
from jaxpm.growth import growth_factor, growth_rate
from jaxpm.kernels import fftk
from jaxpm.painting import cic_paint_dx
from jaxpm.pm import pm_forces

from flip.utils import create_log

log = create_log()

#: Conversion factor: 1 Mpc/h * H_0 = 100 km/s.
#: The h factors cancel because H_0 = 100h km/s/Mpc and 1 Mpc/h = (1/h) Mpc.
_H0_UNIT = 100.0  # km/s / (Mpc/h)


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
    """Create a ``jax_cosmo.Cosmology`` object from standard parameters.

    Args:
        omega_m (float): Total matter density parameter Omega_m.
        sigma8 (float): RMS matter density fluctuation on 8 Mpc/h scales.
        h (float): Dimensionless Hubble parameter H_0 / (100 km/s/Mpc).
            Default 0.6774.
        omega_b (float): Baryon density parameter Omega_b. Default 0.0486.
        n_s (float): Spectral index of the primordial power spectrum.
            Default 0.9667.
        w0 (float): Dark energy equation of state at a=1. Default -1.0.
        wa (float): Dark energy equation of state evolution parameter.
            Default 0.0.
        omega_k (float): Curvature density parameter. Default 0.0.

    Returns:
        jax_cosmo.Cosmology: Cosmology instance compatible with JaxPM.
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
    """Generate a linear density field with a JAX-differentiable power spectrum.

    This replaces ``jaxpm.pm.linear_field`` with a numerically safe version
    that avoids a NaN gradient at the DC mode (k=0).  The DC mode of the
    power spectrum is identically zero in any reasonable cosmology, but the
    automatic derivative of ``jax_cosmo`` power spectra at k=0 is NaN.
    This function handles this by replacing k=0 with a dummy value before
    calling ``pk_fn``, and then zeroing the DC mode with a multiplicative
    mask (which has zero gradient).

    Args:
        mesh_shape (tuple[int, int, int]): Grid dimensions.
        box_size (jnp.ndarray): Box size in Mpc/h, shape ``(3,)``.
        pk_fn (callable): Power spectrum function P(k).
        seed (jax.random.PRNGKey): Random seed.

    Returns:
        jnp.ndarray: Real-space linear density field, shape ``mesh_shape``.
    """
    # Draw Gaussian random Fourier coefficients
    field = normal_field(seed=seed, shape=mesh_shape)
    field = fft3d(field)

    # Wavenumber magnitude in units of 1/Mpc/h
    kvec = fftk(field)
    kmesh = sum(
        (kk / box_size[i] * mesh_shape[i]) ** 2 for i, kk in enumerate(kvec)
    ) ** 0.5

    # Replace k=0 (DC mode) with 1.0 to avoid NaN in pk_fn at k=0.
    # The DC contribution will be zeroed by dc_mask below.
    kmesh_safe = jnp.where(kmesh > 0, kmesh, jnp.ones_like(kmesh))

    # Dimensionless power spectrum amplitude on the mesh
    volume = jnp.prod(jnp.array(mesh_shape)) / jnp.prod(box_size)
    pkmesh = pk_fn(kmesh_safe) * volume

    # Multiplicative DC mask: 1 for k>0, 0 for k=0.
    # Using multiplication instead of jnp.where ensures the gradient at k=0
    # is 0 rather than NaN (both branches of jnp.where are evaluated in JAX).
    dc_mask = (kmesh > 0).astype(jnp.float64)

    field = field * jnp.sqrt(pkmesh) * dc_mask
    return ifft3d(field)


def _run_lpt(cosmo, initial_conditions, a):
    """Run first-order Lagrangian Perturbation Theory (1LPT).

    This is a manual 1LPT implementation that returns particle displacements
    and momenta without computing the force derivative ``dGfa``, which has a
    known caching incompatibility in jaxpm when used with JAX-traced
    cosmology objects.

    Args:
        cosmo (jax_cosmo.Cosmology): Cosmological parameters.
        initial_conditions (jnp.ndarray): Linear density field on the mesh,
            shape ``mesh_shape``.
        a (float): Scale factor at which to evaluate LPT.

    Returns:
        tuple:
            - dx (jnp.ndarray): Particle displacement from lattice positions
              in mesh cell units, shape ``(*mesh_shape, 3)``.
            - p (jnp.ndarray): Particle momentum in internal units
              ``[cells * H_0]``, shape ``(*mesh_shape, 3)``.
    """
    mesh_shape = initial_conditions.shape
    a_arr = jnp.atleast_1d(a)
    a_scalar = a_arr[0]

    E = jnp.sqrt(jc.background.Esqr(cosmo, a_arr))[0]

    # Start particles at lattice positions (zero displacement)
    particles = jnp.zeros((*mesh_shape, 3))

    # Compute gravitational force from the linear density field
    delta_k = fft3d(initial_conditions)
    initial_force = pm_forces(particles, delta=delta_k, paint_absolute_pos=False)

    # 1LPT displacement: dx = D1(a) * Psi (Zel'dovich approximation)
    D1 = growth_factor(cosmo, a_arr)[0]
    f1 = growth_rate(cosmo, a_arr)[0]

    dx = D1 * initial_force
    # Momentum: p = a^2 * H(a) * f1 * dx  (in internal units)
    p = a_scalar**2 * f1 * E * dx

    return dx, p


def generate_density_and_velocity(
    cosmo,
    mesh_shape,
    box_size,
    seed,
    a=1.0,
):
    """Generate differentiable density and velocity fields using 1LPT.

    Runs a first-order Lagrangian Perturbation Theory simulation (Zel'dovich
    approximation) using the JaxPM package and returns the density contrast
    and peculiar velocity fields on a regular 3D Cartesian mesh.  All
    operations are JAX-differentiable with respect to ``cosmo``.

    Args:
        cosmo (jax_cosmo.Cosmology): Cosmological parameters.  Create with
            :func:`get_cosmology`.
        mesh_shape (tuple[int, int, int]): Number of mesh cells along each
            axis, e.g. ``(64, 64, 64)``.
        box_size (array-like): Box dimensions in Mpc/h along each axis,
            e.g. ``[256., 256., 256.]``.
        seed (jax.random.PRNGKey): Random seed for the Gaussian initial
            conditions.
        a (float): Scale factor at which to evaluate the fields.
            Default 1.0 (z=0).

    Returns:
        tuple:
            - density_field (jnp.ndarray): Density contrast delta(x) on the
              mesh, shape ``mesh_shape``.  The mean value is approximately
              zero.
            - velocity_field (jnp.ndarray): Peculiar velocity field in km/s
              on the mesh, shape ``(*mesh_shape, 3)`` with components
              ``(vx, vy, vz)`` in Cartesian coordinates.
    """
    box_size = jnp.array(box_size)

    def linear_pk_fn(k):
        return jc.power.linear_matter_power(cosmo, k, a=a)

    # Generate Gaussian random initial conditions from the linear power spectrum
    initial_conditions = _differentiable_linear_field(mesh_shape, box_size, linear_pk_fn, seed)

    # Run 1LPT to get particle displacements and momenta
    dx, p = _run_lpt(cosmo, initial_conditions, a)

    # Paint displaced particles to obtain density contrast field delta(x)
    density_field = cic_paint_dx(dx)

    # Convert momentum to velocity field in km/s
    # p = a^2 * f * E * dx  =>  v_dimensionless = p / (a^2 * E) = f * dx [cells]
    # v_km_s = v_dimensionless * cell_size [Mpc/h] * H_0 [km/s / (Mpc/h)]
    a_arr = jnp.atleast_1d(a)
    E = jnp.sqrt(jc.background.Esqr(cosmo, a_arr))[0]
    cell_size = box_size / jnp.array(mesh_shape, dtype=jnp.float64)
    velocity_field = p / (a_arr[0] ** 2 * E) * cell_size * _H0_UNIT

    return density_field, velocity_field


def interpolate_velocity_to_positions(velocity_field, positions, box_size, mesh_shape):
    """Interpolate velocity field at arbitrary Cartesian positions using CIC.

    Performs trilinear (CIC) interpolation of a 3D velocity field at the
    provided positions.  Periodic boundary conditions are applied.

    Args:
        velocity_field (jnp.ndarray): Velocity field in km/s, shape
            ``(*mesh_shape, 3)`` as returned by
            :func:`generate_density_and_velocity`.
        positions (jnp.ndarray): Galaxy Cartesian positions in Mpc/h,
            shape ``(N, 3)``.  Coordinates should be within ``[0, box_size)``
            along each axis.
        box_size (array-like): Box dimensions in Mpc/h, shape ``(3,)``.
        mesh_shape (array-like): Number of mesh cells per axis, shape ``(3,)``.

    Returns:
        jnp.ndarray: Velocity vector in km/s at each position, shape
        ``(N, 3)``.
    """
    box_size = jnp.array(box_size)
    mesh_shape_arr = jnp.array(mesh_shape, dtype=jnp.float64)

    # Convert Cartesian Mpc/h positions to mesh cell units [0, Ni)
    pos_mesh = positions / box_size * mesh_shape_arr

    # Read each velocity component using CIC interpolation
    velocities = jnp.stack(
        [_cic_read(velocity_field[..., i], pos_mesh) for i in range(3)],
        axis=-1,
    )
    return velocities


def _cic_read(grid_mesh, positions):
    """Read a 3D scalar field at arbitrary positions using CIC interpolation.

    This is a JAX-differentiable trilinear interpolation compatible with
    arbitrary batch sizes (unlike jaxpm's ``cic_read`` which requires
    positions to match the grid shape).

    Args:
        grid_mesh (jnp.ndarray): 3D scalar field, shape ``(Nx, Ny, Nz)``.
        positions (jnp.ndarray): Positions in mesh cell units ``[0, Ni)``,
            shape ``(N, 3)``.

    Returns:
        jnp.ndarray: Interpolated field values at each position, shape
        ``(N,)``.
    """
    # Add neighbour-offset dimension: positions (N, 1, 3)
    pos = jnp.expand_dims(positions, -2)

    # 8 CIC neighbour offsets, shape (1, 8, 3)
    offsets = jnp.array(
        [
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
            [1, 1, 0],
            [1, 0, 1],
            [0, 1, 1],
            [1, 1, 1],
        ],
        dtype=jnp.float64,
    )
    offsets = offsets[jnp.newaxis, ...]  # (1, 8, 3)

    floor_pos = jnp.floor(pos)  # (N, 1, 3)
    neighbours = floor_pos + offsets  # (N, 8, 3)

    # CIC kernel weights
    kernel = 1.0 - jnp.abs(pos - neighbours)  # (N, 8, 3)
    kernel = kernel[..., 0] * kernel[..., 1] * kernel[..., 2]  # (N, 8)

    # Periodic wrap of neighbour indices
    grid_shape = jnp.array(grid_mesh.shape)
    idx = jnp.mod(neighbours.astype(jnp.int32), grid_shape)  # (N, 8, 3)

    # Gather and weight
    values = grid_mesh[idx[..., 0], idx[..., 1], idx[..., 2]]  # (N, 8)
    return (values * kernel).sum(axis=-1)  # (N,)


def _safe_normalize(positions):
    """Compute unit vectors along the line-of-sight, handling zero-radius positions.

    Returns the unit vector for each position.  When the Euclidean norm of a
    position is zero (observer at the origin coincides with the galaxy), the
    unit vector is set to zero to avoid division by zero.

    Args:
        positions (jnp.ndarray): Cartesian positions in Mpc/h, shape ``(N, 3)``.

    Returns:
        jnp.ndarray: Unit vectors along line-of-sight, shape ``(N, 3)``.
    """
    r = jnp.linalg.norm(positions, axis=-1, keepdims=True)
    return positions / jnp.where(r > 0, r, jnp.ones_like(r))


def compute_los_velocity(velocities, positions):
    """Project 3D peculiar velocities onto the line-of-sight direction.

    Computes the radial (line-of-sight) component of the peculiar velocity
    for each galaxy.  The observer is assumed to be at the Cartesian origin.

    Args:
        velocities (jnp.ndarray): 3D peculiar velocities in km/s, shape
            ``(N, 3)``.
        positions (jnp.ndarray): Galaxy Cartesian positions in Mpc/h, shape
            ``(N, 3)``.

    Returns:
        jnp.ndarray: Line-of-sight peculiar velocity in km/s, shape ``(N,)``.
    """
    los_unit = _safe_normalize(positions)
    return jnp.sum(velocities * los_unit, axis=-1)


def radec_to_cartesian(ra, dec, r_com):
    """Convert spherical sky coordinates to Cartesian positions.

    Args:
        ra (array-like): Right ascension in degrees.
        dec (array-like): Declination in degrees.
        r_com (array-like): Comoving distance in Mpc/h.

    Returns:
        jnp.ndarray: Cartesian positions in Mpc/h, shape ``(N, 3)``.
    """
    ra_rad = jnp.deg2rad(jnp.asarray(ra))
    dec_rad = jnp.deg2rad(jnp.asarray(dec))
    r = jnp.asarray(r_com)
    x = r * jnp.cos(dec_rad) * jnp.cos(ra_rad)
    y = r * jnp.cos(dec_rad) * jnp.sin(ra_rad)
    z = r * jnp.sin(dec_rad)
    return jnp.stack([x, y, z], axis=-1)
