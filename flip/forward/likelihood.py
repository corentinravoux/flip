import abc
from functools import partial

import jax
import jax.numpy as jnp
import jax_cosmo as jcosmo

from flip.forward.field_utils import density_from_grid, radial_velocity_from_grid

# CR - replace this function
# We need something more general for the redshift to distance conversion.

_REDSHIFT_LOOKUP = jnp.arange(0, 0.5, 0.001)
_REDSHIFT_LOOKUP_A = jcosmo.utils.z2a(_REDSHIFT_LOOKUP)
_REDSHIFT_LOOKUP_DIST = jcosmo.background.radial_comoving_distance(
    cosmo=jcosmo.Planck15(), a=_REDSHIFT_LOOKUP_A
)


@jax.jit
def redshift_from_dist_mpch(distance):
    return jnp.interp(distance, _REDSHIFT_LOOKUP_DIST, _REDSHIFT_LOOKUP)


@partial(jax.jit, static_argnames=("box_size", "number_bins"))
def log_likelihood_velocity_density_link(
    redshift,
    redshift_error,
    velocity,
    line_of_sight,
    box_size,
    number_bins,
    comoving_distance_targets,
    sigma_v,
):

    redshift_cosmo = redshift_from_dist_mpch(comoving_distance_targets)

    radial_velocity_estimator = (
        jcosmo.constants.c * (redshift - redshift_cosmo) / (1 + redshift_cosmo)
    )

    radial_velocity_simulator = radial_velocity_from_grid(
        velocity,
        comoving_distance_targets,
        line_of_sight,
        box_size,
        number_bins,
    )
    error = jnp.sqrt(
        (jcosmo.constants.c * redshift_error / (1 + redshift_cosmo)) ** 2 + sigma_v**2
    )

    log_likelihood_velocity_density = (
        -0.5 * jnp.log(2 * jnp.pi * error**2)
        - 0.5 * ((radial_velocity_estimator - radial_velocity_simulator) / error) ** 2
    )
    return jnp.sum(log_likelihood_velocity_density)


@jax.jit
def log_likelihood_magnitude_distance_link(
    observed_distance_modulus,
    observed_distance_modulus_err,
    comoving_distance_targets,
    h,
):

    redshift_cosmo = redshift_from_dist_mpch(comoving_distance_targets)
    theoretical_distance_modulus = (
        5 * jnp.log10((1 + redshift_cosmo) * comoving_distance_targets / h) + 25
    )

    log_likelihood_magnitude_distance = (
        -0.5 * jnp.log(2 * jnp.pi * observed_distance_modulus_err**2)
        - 0.5
        * (
            (observed_distance_modulus - theoretical_distance_modulus)
            / observed_distance_modulus_err
        )
        ** 2
    )
    return jnp.sum(log_likelihood_magnitude_distance)


@partial(jax.jit, static_argnames=("box_size", "number_bins"))
def log_prior_density_function(
    delta_fourier,
    power_spectrum_grid,
    box_size,
    number_bins,
):
    # No 0.5 factor in front of the sum, as we are not considering the complex conjugate modes.
    # No sigma8 because the modes are already scaled by sigma8 in the simulator.
    variance = power_spectrum_grid * (number_bins / box_size) ** 3 * number_bins**3
    prior_density = -jnp.abs(delta_fourier) ** 2 / variance - jnp.log(jnp.pi * variance)
    return jnp.sum(prior_density)


@partial(jax.jit, static_argnames=("box_size", "number_bins"))
def log_prior_position_function(
    density,
    line_of_sight,
    box_size,
    number_bins,
    comoving_distance_targets,
    cutoff=200.0,
):

    density_values = density_from_grid(
        density,
        comoving_distance_targets,
        line_of_sight,
        box_size,
        number_bins,
    )

    # CR - rethink this whole part

    log_density_values = jnp.where(
        (comoving_distance_targets > cutoff) + (comoving_distance_targets < 0),
        -((comoving_distance_targets - cutoff) ** 2),
        jnp.log(density_values),
    )

    return jnp.sum(log_density_values)


# CR - all self.simulator. functions should be general to all simulators,
# This will require a lot of refactoring, but will make the code more modular and easier to maintain.


class BaseLikelihood(abc.ABC):

    def __init__(
        self,
        parameter_names=None,
        likelihood_properties=None,
    ):
        self.parameter_names = parameter_names
        self.likelihood_properties = likelihood_properties
        self.likelihood_evaluation, self.likelihood_grad = self._init_likelihood()

    @abc.abstractmethod
    def _init_likelihood(self):
        pass


class CandleGridGaussianLikelihood(BaseLikelihood):

    def __init__(
        self,
        simulator=None,
        velocity_data_vector=None,
        coordinates_velocity=None,
        parameter_names=None,
        likelihood_properties=None,
    ):

        self.simulator = simulator
        self.velocity_data_vector = velocity_data_vector
        self.coordinates_velocity = coordinates_velocity

        super(CandleGridGaussianLikelihood, self).__init__(
            parameter_names=parameter_names,
            likelihood_properties=likelihood_properties,
        )

    def _init_likelihood(self):

        ra = self.coordinates_velocity[0]
        dec = self.coordinates_velocity[1]

        simulation_positions_at_target_positions = (
            self.simulator.get_voxels_in_direction(
                ra,
                dec,
                dist_range=[0, 200],
                physical_unit=True,
                unique=False,
            )
        )
        self.line_of_sight = self.simulator.get_line_of_sight(ra, dec)

        self.simulation_distances_at_target_positions = self.simulator._dist_mpch[
            simulation_positions_at_target_positions[:, 0],
            simulation_positions_at_target_positions[:, 1],
            simulation_positions_at_target_positions[:, 2],
        ]

        self.simulation_positions_at_target_positions = (
            simulation_positions_at_target_positions
        )

        def likelihood_evaluation(
            parameter_values,
        ):
            parameter_values_dict = dict(zip(self.parameter_names, parameter_values))
            delta_fourier, density, velocity = self.get_fields_from_delta_modes(
                parameter_values_dict
            )
            log_likelihood_delta_fourier_val = self.get_log_prior_fields(delta_fourier)
            log_likelihood_targets_val = self.get_log_likelihood_targets(
                parameter_values_dict, density, velocity
            )

            log_likelihood_total = (
                log_likelihood_delta_fourier_val + log_likelihood_targets_val
            )
            return log_likelihood_total

        likelihood_grad = jax.jit(jax.grad(likelihood_evaluation))
        return likelihood_evaluation, likelihood_grad

    def get_fields_from_delta_modes(self, parameter_values_dict):
        delta_fourier, density, velocity = (
            self.simulator.sample_density_velocity_fields_from_modes(
                parameter_values_dict,
            )
        )
        return delta_fourier, density, velocity

    def get_log_prior_fields(self, delta_fourier):

        return log_prior_density_function(
            delta_fourier,
            self.simulator.power_spectrum_grid,
            self.simulator.box_size,
            self.simulator.number_bins,
        )

    def get_log_likelihood_targets(self, parameter_values_dict, density, velocity):
        comoving_distance_targets = parameter_values_dict["comoving_distance_targets"]
        sigma_v = parameter_values_dict["sigma_v"]
        h = parameter_values_dict["h"]

        observed_distance_modulus = (
            self.velocity_data_vector.compute_observed_distance_modulus(
                parameter_values_dict
            )
        )
        observed_distance_modulus_err = jnp.sqrt(
            self.velocity_data_vector.compute_observed_distance_modulus_variance(
                parameter_values_dict
            )
        )
        redshift = self.velocity_data_vector._data["zobs"]
        if "zobs_error" in self.velocity_data_vector._data.keys():
            redshift_error = self.velocity_data_vector._data["zobs_error"]
        else:
            redshift_error = jnp.zeros_like(redshift)

        log_likelihood_velocity_density = log_likelihood_velocity_density_link(
            redshift,
            redshift_error,
            velocity,
            self.line_of_sight,
            self.simulator.box_size,
            self.simulator.number_bins,
            comoving_distance_targets,
            sigma_v,
        )

        log_likelihood_magnitude_distance = log_likelihood_magnitude_distance_link(
            observed_distance_modulus,
            observed_distance_modulus_err,
            comoving_distance_targets,
            h,
        )

        log_prior_position = log_prior_position_function(
            density,
            self.line_of_sight,
            self.simulator.box_size,
            self.simulator.number_bins,
            comoving_distance_targets,
            cutoff=200.0,
        )

        return (
            log_likelihood_velocity_density
            + log_likelihood_magnitude_distance
            + log_prior_position
        )

    def __call__(self, parameter_values):
        """Evaluate likelihood at parameter values.

        Args:
            parameter_values (array-like): Parameter vector aligned with `parameter_names`.

        Returns:
            float: Likelihood value, sign controlled by `negative_log_likelihood`.
        """
        return self.likelihood_evaluation(parameter_values)
