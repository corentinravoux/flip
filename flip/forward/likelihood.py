import abc
from functools import partial

import jax
import jax.numpy as jnp
import jax_cosmo as jcosmo
from jax.scipy.special import gammaln

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


@jax.jit
def log_prior_position_constant_gaussian_wall(
    comoving_distance_targets,
    maximal_distance,
):

    log_density_values = jnp.where(
        (comoving_distance_targets > maximal_distance)
        + (comoving_distance_targets < 0),
        -((comoving_distance_targets - maximal_distance) ** 2),
        1.0,
    )
    return jnp.sum(log_density_values)


@partial(jax.jit, static_argnames=("box_size", "number_bins"))
def log_prior_position_density_gaussian_wall(
    density,
    line_of_sight,
    box_size,
    number_bins,
    comoving_distance_targets,
    maximal_distance,
):

    density_values = density_from_grid(
        density,
        comoving_distance_targets,
        line_of_sight,
        box_size,
        number_bins,
    )

    log_density_values = jnp.where(
        (comoving_distance_targets > maximal_distance)
        + (comoving_distance_targets < 0),
        -((comoving_distance_targets - maximal_distance) ** 2),
        jnp.log(density_values),
    )

    return jnp.sum(log_density_values)


@jax.jit
def log_prior_position_constant_sigmoid(
    comoving_distance_targets,
    maximal_distance,
):
    log_window = jax.nn.log_sigmoid(
        (maximal_distance - comoving_distance_targets)
    ) + jax.nn.log_sigmoid(comoving_distance_targets)
    return jnp.sum(log_window)


@jax.jit
def log_prior_position_homogeneous_sigmoid(
    comoving_distance_targets,
    maximal_distance,
):
    log_num = 2.0 * jnp.log(comoving_distance_targets)
    log_window = jax.nn.log_sigmoid(
        (maximal_distance - comoving_distance_targets)
    ) + jax.nn.log_sigmoid(comoving_distance_targets)
    return jnp.sum(log_num + log_window)


@partial(jax.jit, static_argnames=("box_size", "number_bins"))
def log_prior_position_lognormal(
    density,
    line_of_sight,
    box_size,
    number_bins,
    comoving_distance_targets,
    maximal_distance,
):
    # --- numerator: log(chi^2 * n) at sampled chi_i ---
    n_i = density_from_grid(
        density, comoving_distance_targets, line_of_sight, box_size, number_bins
    )
    log_num = 2.0 * jnp.log(comoving_distance_targets) + jnp.log(jnp.clip(n_i, 1e-30))

    # --- normalization Z_i via logsumexp(trapezoid) along each LOS ---
    n_chi_grid = 300
    chi_grid = jnp.linspace(0.0, maximal_distance, n_chi_grid)  # (G,)
    n_g = density_from_grid(
        density,
        chi_grid[None, :],  # (1,G)
        line_of_sight[:, None, :],  # (Nt,1,3)
        box_size,
        number_bins,
    )  # -> (Nt,G)
    dchi = chi_grid[1] - chi_grid[0]
    w = jnp.full_like(chi_grid, dchi).at[0].set(0.5 * dchi).at[-1].set(0.5 * dchi)
    log_w = jnp.log(jnp.clip(chi_grid**2, 1e-30)) + jnp.log(w)
    log_Z = jax.scipy.special.logsumexp(
        log_w[None, :] + jnp.log(jnp.clip(n_g, 1e-30)), axis=-1
    )

    return jnp.sum(log_num - log_Z)


@jax.jit
def log_prior_position_exponential_cutoff(
    comoving_distance_targets,
    a,
    b,
    c,
):
    log_num = (
        a * jnp.log(comoving_distance_targets) - (comoving_distance_targets / b) ** c
    )
    # log N(a,b,c) = (a+1) log b - log c + lgamma((a+1)/c); constant if a,b,c fixed
    log_norm = (a + 1.0) * jnp.log(b) - jnp.log(c) + gammaln((a + 1.0) / c)
    return jnp.sum(log_num - log_norm)


@jax.jit
def log_prior_position_piecewise_gaussian(
    comoving_distance_targets,
    a,
    b,
    c,
):
    sigma = jnp.where(comoving_distance_targets <= a, b, c)
    log_num = -0.5 * ((comoving_distance_targets - a) / sigma) ** 2
    log_norm = jnp.log(jnp.sqrt(2.0 * jnp.pi) * (b + c))
    return jnp.sum(log_num - log_norm)


@jax.jit
def log_prior_position_histogram(
    comoving_distance_targets,
    hist_bin_centers,
    hist_log_density,
):
    log_density = jnp.interp(
        comoving_distance_targets, hist_bin_centers, hist_log_density
    )
    return jnp.sum(log_density)


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

    _default_likelihood_properties = {
        "distance_prior_name": "constant_sigmoid",
    }

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
        if likelihood_properties is None:
            likelihood_properties = {}

        likelihood_properties = {
            **self._default_likelihood_properties,
            **likelihood_properties,
        }

        super(CandleGridGaussianLikelihood, self).__init__(
            parameter_names=parameter_names,
            likelihood_properties=likelihood_properties,
        )

    def _init_likelihood(self):

        ra = self.coordinates_velocity[0]
        dec = self.coordinates_velocity[1]
        self.line_of_sight = self.simulator.get_line_of_sight(ra, dec)

        def likelihood_evaluation(
            parameter_values,
        ):
            parameter_values_dict = dict(zip(self.parameter_names, parameter_values))
            delta_fourier, density, velocity = self.get_fields_from_delta_modes(
                parameter_values_dict
            )
            log_likelihood_targets_val = self.get_log_likelihood_targets(
                parameter_values_dict, velocity
            )
            log_prior_fields_val = self.get_log_prior_fields(delta_fourier)

            log_prior_distance_val = self.get_log_prior_targets(
                parameter_values_dict, density
            )

            log_likelihood_val = (
                log_likelihood_targets_val
                + log_prior_fields_val
                + log_prior_distance_val
            )
            return log_likelihood_val

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

    def get_log_prior_targets(
        self,
        parameter_values_dict,
        density,
    ):

        prior_name = self.likelihood_properties["distance_prior_name"]

        # CR - Maximal distance considered chose to avoid coding
        # periodic boundary conditions in the box.

        maximal_distance = self.likelihood_properties.get(
            "maximal_distance",
            self.simulator.box_size / 2.0,
        )
        comoving_distance_targets = parameter_values_dict["comoving_distance_targets"]

        if prior_name == "no_prior":
            log_prior_distance = 0.0

        elif prior_name == "constant_gaussian_wall":
            log_prior_distance = log_prior_position_constant_gaussian_wall(
                comoving_distance_targets,
                maximal_distance,
            )

        elif prior_name == "density_gaussian_wall":
            log_prior_distance = log_prior_position_density_gaussian_wall(
                density,
                self.line_of_sight,
                self.simulator.box_size,
                self.simulator.number_bins,
                comoving_distance_targets,
                maximal_distance,
            )
        elif prior_name == "constant_sigmoid":
            log_prior_distance = log_prior_position_constant_sigmoid(
                comoving_distance_targets,
                maximal_distance,
            )
        elif prior_name == "homogeneous_sigmoid":
            log_prior_distance = log_prior_position_homogeneous_sigmoid(
                comoving_distance_targets,
                maximal_distance,
            )
        elif prior_name == "piecewise_gaussian":
            a = parameter_values_dict["piecewise_gaussian_a"]
            b = parameter_values_dict["piecewise_gaussian_b"]
            c = parameter_values_dict["piecewise_gaussian_c"]
            log_prior_distance = log_prior_position_piecewise_gaussian(
                comoving_distance_targets,
                a,
                b,
                c,
            )

        elif prior_name == "exponential_cutoff":
            a = parameter_values_dict["exponential_cutoff_a"]
            b = parameter_values_dict["exponential_cutoff_b"]
            c = parameter_values_dict["exponential_cutoff_c"]
            log_prior_distance = log_prior_position_exponential_cutoff(
                comoving_distance_targets,
                a,
                b,
                c,
            )
        elif prior_name == "histogram":
            redshift_cosmo = self.velocity_data_vector._data["zobs"]

            a_obs = jcosmo.utils.z2a(redshift_cosmo)
            d_z = jnp.asarray(
                jcosmo.background.radial_comoving_distance(jcosmo.Planck15(), a_obs)
            )
            density, edges = jnp.histogram(d_z, bins=50, density=True)
            hist_bin_centers = 0.5 * (edges[:-1] + edges[1:])
            hist_log_density = jnp.log(jnp.clip(density, 1e-30, None))

            log_prior_distance = log_prior_position_histogram(
                comoving_distance_targets,
                hist_bin_centers,
                hist_log_density,
            )

        elif prior_name == "lognormal":
            log_prior_distance = log_prior_position_lognormal(
                density,
                self.line_of_sight,
                self.simulator.box_size,
                self.simulator.number_bins,
                comoving_distance_targets,
                maximal_distance,
            )

        return log_prior_distance

    def get_log_likelihood_targets(
        self,
        parameter_values_dict,
        velocity,
    ):
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

        return log_likelihood_velocity_density + log_likelihood_magnitude_distance

    def __call__(self, parameter_values):
        """Evaluate likelihood at parameter values.

        Args:
            parameter_values (array-like): Parameter vector aligned with `parameter_names`.

        Returns:
            float: Likelihood value, sign controlled by `negative_log_likelihood`.
        """
        return self.likelihood_evaluation(parameter_values)
