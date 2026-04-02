"""Gaussian field-level likelihood for forward-model velocity inference.

Provides:

* :func:`log_likelihood_gaussian` — scalar log-likelihood under diagonal
  or full Gaussian noise, differentiable via JAX.
* :class:`VelocityFieldLikelihood` — full likelihood callable that wraps a
  :class:`~flip.simulation.generator.ForwardModel` and compares its output to
  a flip :class:`~flip.data_vector.basic.DataVector`.

The likelihood is end-to-end differentiable w.r.t. cosmological parameters,
enabling gradient-based maximization (:class:`~flip.simulation.fitter.SimulationFitter`)
or HMC sampling.

Example::

    import jax
    from flip.simulation.generator import ForwardModel, get_cosmology
    from flip.simulation.likelihood import VelocityFieldLikelihood

    model = ForwardModel(mesh_shape=(32,32,32), box_size=(256.,256.,256.))
    lik = VelocityFieldLikelihood(
        forward_model=model,
        data_vector=my_vel_dv,
        positions_cartesian=galaxy_xyz,
        seed=jax.random.PRNGKey(0),
    )
    neg_log_lik = lik({"omega_m": 0.3, "sigma8": 0.8})
"""

import jax.numpy as jnp
import jax.scipy as jsc

from flip.simulation import generator as gen
from flip.simulation import painter
from flip.utils import create_log

log = create_log()


def log_likelihood_gaussian(simulated_velocity, observed_velocity, observed_variance):
    """Gaussian log-likelihood between simulated and observed LOS velocities.

    Supports both **diagonal** (independent Gaussian errors) and **full**
    covariance matrices.

    Diagonal form:

    .. math::

        \\log\\mathcal{L} = -\\tfrac{1}{2}\\sum_i
        \\left[\\frac{(v^\\text{obs}_i - v^\\text{sim}_i)^2}{\\sigma_i^2}
        + \\log(2\\pi\\sigma_i^2)\\right]

    Args:
        simulated_velocity (jnp.ndarray): Simulated LOS velocities [km/s],
            shape ``(N,)``.
        observed_velocity (jnp.ndarray): Observed velocities [km/s],
            shape ``(N,)``.
        observed_variance (jnp.ndarray): Measurement (co)variance.  1-D array
            of shape ``(N,)`` for independent errors; 2-D array of shape
            ``(N, N)`` for full covariance.

    Returns:
        float: Log-likelihood value.
    """
    residual = observed_velocity - simulated_velocity
    if observed_variance.ndim == 2:
        chol = jsc.linalg.cho_factor(observed_variance)
        logdet = 2.0 * jnp.sum(jnp.log(jnp.diag(chol[0])))
        chi2 = jnp.dot(residual, jsc.linalg.cho_solve(chol, residual))
        n = residual.size
        return -0.5 * (chi2 + logdet + n * jnp.log(2.0 * jnp.pi))
    else:
        logdet = jnp.sum(jnp.log(2.0 * jnp.pi * observed_variance))
        chi2 = jnp.sum(residual**2 / observed_variance)
        return -0.5 * (chi2 + logdet)


class VelocityFieldLikelihood:
    """Gaussian likelihood comparing a forward simulation to observed velocities.

    For a given cosmology, this callable:

    1. Runs the :class:`~flip.simulation.generator.ForwardModel` to get δ and v fields.
    2. Interpolates the velocity field at observed galaxy positions (Cartesian,
       box frame).
    3. Projects onto the line of sight.
    4. Evaluates the Gaussian log-likelihood and returns its negation
       (for minimization).

    The entire chain is JAX-differentiable w.r.t. cosmological parameters,
    so gradients for optimization or HMC are available via ``jax.grad``.

    Args:
        forward_model (ForwardModel): Configured simulation instance.
        data_vector (flip DataVector): Velocity data vector with a
            ``give_data_and_variance(**kwargs)`` method.
        positions_cartesian (array-like): Galaxy Cartesian positions [Mpc/h]
            in the **box frame** (origin at box corner), shape ``(N, 3)``.
            Use :func:`~flip.simulation.painter.sky_to_cartesian` +
            :func:`~flip.simulation.painter.cartesian_to_box_frame` to convert.
        seed (int | jax.random.PRNGKey): IC random seed. Fixed across calls.
        parameter_values_dict (dict | None): Extra parameters for the data
            vector's ``give_data_and_variance`` method (e.g. ``{"M_0": -19.3}``
            for ``VelFromHDres``). Default: empty dict.
    """

    def __init__(
        self,
        forward_model,
        data_vector,
        positions_cartesian,
        seed,
        parameter_values_dict=None,
    ):
        self.forward_model = forward_model
        self.data_vector = data_vector
        self.positions_cartesian = jnp.array(positions_cartesian, dtype=jnp.float32)
        self.seed = seed
        self.parameter_values_dict = parameter_values_dict or {}

        # Pre-compute observed velocities and variances (fixed across calls)
        obs_vel, obs_var = self.data_vector.give_data_and_variance(
            self.parameter_values_dict
        )
        self.observed_velocity = jnp.array(obs_vel)
        self.observed_variance = jnp.array(obs_var)

        log.info(
            "VelocityFieldLikelihood: %d observations, mesh %s, box %s Mpc/h",
            len(self.observed_velocity),
            self.forward_model.mesh_shape,
            tuple(float(x) for x in self.forward_model.box_size),
        )

    def __call__(self, cosmo_params):
        """Evaluate the negative log-likelihood.

        Args:
            cosmo_params (dict): Cosmological parameters accepted by
                :func:`~flip.simulation.generator.get_cosmology`.  At minimum
                ``omega_m`` and ``sigma8`` must be provided.

        Returns:
            float: Negative log-likelihood (suitable for minimization).
        """
        cosmo = gen.get_cosmology(**cosmo_params)

        _, velocity_field = self.forward_model.get_fields(cosmo, self.seed)

        velocities_3d = painter.interpolate_velocity_to_positions(
            velocity_field,
            self.positions_cartesian,
            self.forward_model.box_size,
            self.forward_model.mesh_shape,
        )

        simulated_los = painter.compute_los_velocity(
            velocities_3d, self.positions_cartesian
        )

        return -log_likelihood_gaussian(
            simulated_los,
            self.observed_velocity,
            self.observed_variance,
        )
