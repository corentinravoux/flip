import abc
from functools import partial

import jax
import jax.numpy as jnp
import jax_cosmo as jcosmo
import tensorflow_probability.substrates.jax as tfp


@jax.jit
def log_likelihood_delta_fourier(
    delta_fourier,
    power_spectrum_grid,
    number_bins,
    sigma8,
):
    variance = power_spectrum_grid * number_bins ** (3 / 2) * sigma8**2
    return jnp.sum(-0.5 * jnp.abs(delta_fourier) ** 2 / (variance))


@partial(jax.jit, static_argnames=("box_size", "number_bins"))
def velocity_from_grid(v, dist_mpch, box_size, number_bins):
    r1d = jnp.linspace(0, box_size, number_bins) - box_size / 2
    v_interp = jax.scipy.interpolate.RegularGridInterpolator((r1d, r1d, r1d), v)
    v_model = v_interp(dist_mpch)
    return v_model


@jax.jit
def radial_velocity_from_velocity(velocity, dist_mpch_vec):
    radial_velocity = jnp.sum(velocity * dist_mpch_vec, axis=-1)
    return radial_velocity


# CR - replace this function
@jax.jit
def redshift_from_dist_mpch(distance, cosmo=jcosmo.Planck15(), zbins="0:0.5:0.001"):
    redshifts = eval(f"jnp.r_[{zbins}]")
    a = jcosmo.utils.z2a(redshifts)
    cosmo_dist = jcosmo.background.radial_comoving_distance(cosmo=cosmo, a=a)
    return jnp.interp(distance, cosmo_dist, redshifts)


@partial(jax.jit, static_argnames=("box_size", "number_bins"))
def log_likelihood_targets(
    observed_distance_modulus,
    observed_distance_modulus_err,
    redshift,
    redshift_error,
    density,
    velocity,
    comoving_distance_vectors,
    simulation_distances_at_target_positions,
    simulation_positions_at_target_positions,
    box_size,
    number_bins,
    comoving_distance_targets,
    sigma_v,
    h,
):

    # distance modulus log-likelihood
    redshift_cosmo = redshift_from_dist_mpch(comoving_distance_targets)
    theoretical_distance_modulus = (
        5 * jnp.log10((1 + redshift_cosmo) * comoving_distance_targets / h) + 25
    )

    log_likelihood_distance_modulus = tfp.distributions.Normal(
        loc=theoretical_distance_modulus, scale=observed_distance_modulus_err
    ).log_prob(observed_distance_modulus)

    # redshift log-likelihood
    vi = velocity_from_grid(
        velocity,
        comoving_distance_targets[:, None] * comoving_distance_vectors,
        box_size,
        number_bins,
    )
    vr = radial_velocity_from_velocity(vi, comoving_distance_vectors)
    std = jnp.sqrt(
        (jcosmo.constants.c * redshift_error / (1 + redshift_cosmo)) ** 2
        + (sigma_v) ** 2
    )
    log_likelihood_redshift = tfp.distributions.Normal(loc=vr, scale=std).log_prob(
        jcosmo.constants.c * (redshift - redshift_cosmo) / (1 + redshift_cosmo)
    )

    # density log-likelihood
    target_density_los = density[
        simulation_positions_at_target_positions[:, 0],
        simulation_positions_at_target_positions[:, 1],
        simulation_positions_at_target_positions[:, 2],
    ]

    index_ = jnp.argmin(
        (simulation_distances_at_target_positions - comoving_distance_targets[:, None])
        ** 2,
        axis=-1,
    )
    log_likelihood_density = target_density_los.at[
        jnp.arange(len(target_density_los)), index_
    ].get()

    log_likelihood_targets = (
        log_likelihood_distance_modulus
        + log_likelihood_redshift
        + log_likelihood_density
    )
    return jnp.sum(log_likelihood_targets, axis=0)


# CR - all self.simulator. functions should be general to all simulators,
# This will require a lot of refactoring, but will make the code more modular and easier to maintain.


class BaseLikelihood(object):

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
        self.comoving_distance_vectors = self.simulator.get_unity_3dcoords(ra, dec)

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
            log_likelihood_delta_fourier_val = self.get_log_likelihood_delta_fourier(
                delta_fourier, parameter_values_dict
            )
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

    def get_log_likelihood_delta_fourier(self, delta_fourier, parameter_values_dict):
        sigma8 = parameter_values_dict["s8"]

        return log_likelihood_delta_fourier(
            delta_fourier,
            self.simulator.power_spectrum_grid,
            self.simulator.number_bins,
            sigma8,
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

        return log_likelihood_targets(
            observed_distance_modulus,
            observed_distance_modulus_err,
            redshift,
            redshift_error,
            density,
            velocity,
            self.comoving_distance_vectors,
            self.simulation_distances_at_target_positions,
            self.simulation_positions_at_target_positions,
            self.simulator.box_size,
            self.simulator.number_bins,
            comoving_distance_targets,
            sigma_v,
            h,
        )

    def __call__(self, parameter_values):
        """Evaluate likelihood at parameter values.

        Args:
            parameter_values (array-like): Parameter vector aligned with `parameter_names`.

        Returns:
            float: Likelihood value, sign controlled by `negative_log_likelihood`.
        """
        return self.likelihood_evaluation(parameter_values)


# Use this for distance interpolation:

# def cic_read(grid_mesh, positions):
#     """CIC-interpolate a 3D scalar field at arbitrary positions.

#     Custom JAX-differentiable trilinear interpolation compatible with
#     arbitrary batch sizes (unlike some jaxpm versions whose ``cic_read``
#     requires positions to match the grid shape).  Periodic BCs applied.

#     Args:
#         grid_mesh (jnp.ndarray): Scalar field of shape ``(Nx, Ny, Nz)``.
#         positions (jnp.ndarray): Positions in mesh-cell units ``[0, Ni)``,
#             shape ``(N, 3)``.

#     Returns:
#         jnp.ndarray: Interpolated values at each position, shape ``(N,)``.
#     """
#     pos = jnp.expand_dims(positions, -2)  # (N, 1, 3)

#     offsets = jnp.array(
#         [
#             [0, 0, 0],
#             [1, 0, 0],
#             [0, 1, 0],
#             [0, 0, 1],
#             [1, 1, 0],
#             [1, 0, 1],
#             [0, 1, 1],
#             [1, 1, 1],
#         ],
#         dtype=jnp.float32,
#     )[
#         jnp.newaxis
#     ]  # (1, 8, 3)

#     floor_pos = jnp.floor(pos)  # (N, 1, 3)
#     neighbours = floor_pos + offsets  # (N, 8, 3)
#     kernel = 1.0 - jnp.abs(pos - neighbours)  # (N, 8, 3)
#     kernel = kernel[..., 0] * kernel[..., 1] * kernel[..., 2]  # (N, 8)

#     grid_shape = jnp.array(grid_mesh.shape)
#     idx = jnp.mod(neighbours.astype(jnp.int32), grid_shape)  # (N, 8, 3)
#     values = grid_mesh[idx[..., 0], idx[..., 1], idx[..., 2]]  # (N, 8)
#     return (values * kernel).sum(axis=-1)  # (N,)
