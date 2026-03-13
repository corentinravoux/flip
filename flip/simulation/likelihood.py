"""Gaussian likelihood for forward-model velocity field comparison.

This module provides a differentiable Gaussian likelihood that compares a
simulated peculiar velocity field (generated with :mod:`flip.simulation.generate`)
to observed peculiar velocity measurements from a :class:`flip.data_vector.DataVector`.
All operations are implemented in JAX.

Examples:
    >>> import jax
    >>> import jax.numpy as jnp
    >>> import jax_cosmo as jc
    >>> from flip.simulation import generate, likelihood
    >>> # Build likelihood from a DataVector and galaxy positions
    >>> lik = likelihood.VelocityFieldLikelihood(
    ...     data_vector=my_velocity_data_vector,
    ...     positions_cartesian=galaxy_xyz,
    ...     mesh_shape=(64, 64, 64),
    ...     box_size=[512., 512., 512.],
    ...     seed=jax.random.PRNGKey(0),
    ... )
    >>> neg_log_lik = lik({"omega_m": 0.3, "sigma8": 0.8})
"""

from flip.simulation import generate
from flip.utils import create_log

log = create_log()

try:
    import jax.numpy as jnp
    import jax.scipy as jsc

    jax_installed = True

except ImportError:
    jax_installed = False
    log.add(
        "Install jax to use the simulation likelihood module",
        level="warning",
    )


def log_likelihood_gaussian(simulated_velocity, observed_velocity, observed_variance):
    """Compute the Gaussian log-likelihood between simulated and observed velocities.

    Evaluates the log-likelihood under independent Gaussian measurement errors
    (diagonal noise covariance).  When ``observed_variance`` is a 2-D matrix,
    the full covariance is used via Cholesky factorisation.

    The diagonal form evaluates:

    .. math::

        \\log\\mathcal{L} = -\\frac{1}{2}\\sum_i
        \\left[\\frac{(v^{\\rm obs}_i - v^{\\rm sim}_i)^2}{\\sigma_i^2}
        + \\log(2\\pi\\sigma_i^2)\\right]

    Args:
        simulated_velocity (jnp.ndarray): Simulated line-of-sight velocities
            in km/s, shape ``(N,)``.
        observed_velocity (jnp.ndarray): Observed peculiar velocities in km/s,
            shape ``(N,)``.
        observed_variance (jnp.ndarray): Measurement (co)variance.  Either a
            1-D array of shape ``(N,)`` for independent errors, or a 2-D
            positive-definite matrix of shape ``(N, N)`` for correlated errors.

    Returns:
        float: Log-likelihood value.

    Raises:
        ImportError: If JAX is not installed.
    """
    if not jax_installed:
        raise ImportError(
            "'log_likelihood_gaussian' requires jax. "
            "Install it with: pip install jax"
        )
    residual = observed_velocity - simulated_velocity
    if observed_variance.ndim == 2:
        # Full covariance: use Cholesky factorisation for numerical stability
        chol = jsc.linalg.cho_factor(observed_variance)
        logdet = 2.0 * jnp.sum(jnp.log(jnp.diag(chol[0])))
        chi2 = jnp.dot(residual, jsc.linalg.cho_solve(chol, residual))
        n = residual.size
        return -0.5 * (chi2 + logdet + n * jnp.log(2.0 * jnp.pi))
    else:
        # Diagonal covariance: independent Gaussian errors
        logdet = jnp.sum(jnp.log(2.0 * jnp.pi * observed_variance))
        chi2 = jnp.sum(residual**2 / observed_variance)
        return -0.5 * (chi2 + logdet)


class VelocityFieldLikelihood:
    """Gaussian likelihood comparing a JaxPM simulation to observed velocities.

    Given cosmological parameters, this callable runs a forward simulation
    (N-body ODE by default, or 1LPT when ``method='lpt'``), interpolates the
    resulting velocity field at observed galaxy positions, projects onto the
    line of sight, and returns the Gaussian log-likelihood of the observations.

    The likelihood is fully JAX-differentiable with respect to the input
    cosmological parameters.

    Args:
        data_vector (flip.data_vector.DataVector): Velocity data vector that
            provides observed velocities and their measurement variances via
            its ``give_data_and_variance()`` method.
        positions_cartesian (array-like): Galaxy Cartesian positions in Mpc/h,
            shape ``(N, 3)``.  These should be defined in the same frame as
            the simulation box (origin at box corner, box extends to
            ``box_size``).
        mesh_shape (tuple[int, int, int]): Number of simulation mesh cells per
            axis, e.g. ``(64, 64, 64)``.
        box_size (array-like): Simulation box dimensions in Mpc/h.
        seed (jax.random.PRNGKey): Random seed for the initial conditions.
        a (float): Scale factor at which to evaluate the fields. Default 1.0.
        method (str): Simulation method forwarded to
            :func:`~flip.simulation.generate.generate_density_and_velocity`.
            Either ``"nbody"`` (default, full N-body ODE integration) or
            ``"lpt"`` (faster Zel'dovich approximation, for testing).
        fixed_cosmo_params (dict | None): Cosmological parameters that are
            held fixed during optimization.  These are merged with the
            ``cosmo_params`` dict at each likelihood call, with
            ``cosmo_params`` taking precedence.  Useful for fixing ``omega_m``
            while fitting only ``sigma8``, for example.  Default ``None``.
        parameter_values_dict (dict | None): Additional parameters consumed
            by the data vector (e.g. ``{"M_0": -19.3}`` for
            ``VelFromHDres``).  If ``None``, an empty dict is used.
        **simulation_kwargs: Extra keyword arguments forwarded to the
            simulation function (e.g. ``ode_rtol``, ``ode_atol`` for
            ``method='nbody'``).

    Examples:
        >>> lik = VelocityFieldLikelihood(
        ...     data_vector=vel_vec,
        ...     positions_cartesian=xyz,
        ...     mesh_shape=(32, 32, 32),
        ...     box_size=[256., 256., 256.],
        ...     seed=jax.random.PRNGKey(1),
        ... )
        >>> neg_log_lik = lik({"omega_m": 0.3, "sigma8": 0.8})
    """

    def __init__(
        self,
        data_vector,
        positions_cartesian,
        mesh_shape,
        box_size,
        seed,
        a=1.0,
        method="nbody",
        fixed_cosmo_params=None,
        parameter_values_dict=None,
        **simulation_kwargs,
    ):
        self.data_vector = data_vector
        if not jax_installed:
            raise ImportError(
                "'VelocityFieldLikelihood' requires jax. "
                "Install it with: pip install jax"
            )
        self.positions_cartesian = jnp.array(positions_cartesian)
        self.mesh_shape = mesh_shape
        self.box_size = jnp.array(box_size)
        self.seed = seed
        self.a = a
        self.method = method
        self.fixed_cosmo_params = fixed_cosmo_params or {}
        self.simulation_kwargs = simulation_kwargs
        self.parameter_values_dict = parameter_values_dict or {}

        # Pre-compute observed velocities and their measurement (co)variances
        # from the data vector.  This is done once at construction time.
        observed_velocity, observed_variance = self.data_vector.give_data_and_variance(
            self.parameter_values_dict
        )
        self.observed_velocity = jnp.array(observed_velocity)
        self.observed_variance = jnp.array(observed_variance)

        log.add(
            f"VelocityFieldLikelihood: {len(self.observed_velocity)} "
            f"velocity observations, mesh {mesh_shape}, "
            f"box {list(box_size)} Mpc/h, method='{method}'."
        )

    def __call__(self, cosmo_params):
        """Evaluate the negative log-likelihood for a set of cosmological parameters.

        Runs the full JAX-differentiable forward model:

        1. Build cosmology from ``cosmo_params`` merged with
           ``fixed_cosmo_params``.
        2. Generate density and velocity fields via the configured simulation
           method (N-body ODE or LPT).
        3. Interpolate velocity field at observed galaxy positions.
        4. Project onto line of sight.
        5. Compute Gaussian log-likelihood and return its negation.

        Args:
            cosmo_params (dict): Cosmological parameters to optimize, accepted
                by :func:`~flip.simulation.generate.get_cosmology`.  These are
                merged with ``fixed_cosmo_params`` (``cosmo_params`` takes
                precedence), so only the free parameters need to be provided.

        Returns:
            float: Negative log-likelihood (suitable for minimization).
        """
        # Merge fixed parameters with the free ones (free params take precedence)
        full_params = {**self.fixed_cosmo_params, **cosmo_params}
        cosmo = generate.get_cosmology(**full_params)

        _, velocity_field = generate.generate_density_and_velocity(
            cosmo,
            self.mesh_shape,
            self.box_size,
            self.seed,
            a=self.a,
            method=self.method,
            **self.simulation_kwargs,
        )

        velocities_3d = generate.interpolate_velocity_to_positions(
            velocity_field,
            self.positions_cartesian,
            self.box_size,
            self.mesh_shape,
        )

        simulated_los = generate.compute_los_velocity(
            velocities_3d, self.positions_cartesian
        )

        return -log_likelihood_gaussian(
            simulated_los,
            self.observed_velocity,
            self.observed_variance,
        )
