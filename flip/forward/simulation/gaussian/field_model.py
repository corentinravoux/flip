import functools

import jax
import jax.numpy as jnp

from .box import FourierBox


def get_seed_from_parameter_values_dict(parameter_values_dict):
    if "seed" not in parameter_values_dict:
        raise ValueError("seed must be provided in parameter_values_dict")
    seed = parameter_values_dict["seed"]
    seed = jax.random.PRNGKey(seed) if isinstance(seed, int) else seed
    return seed


@functools.partial(jax.jit, static_argnums=(1, 2))
def delta_fourier_from_power_spectrum(
    power_spectrum_grid,
    number_bins,
    box_size,
    seed,
):
    w = jax.random.normal(seed, shape=(number_bins, number_bins, number_bins))
    modes = jnp.fft.rfftn(w)
    modes *= jnp.sqrt(power_spectrum_grid * (number_bins / box_size) ** 3)
    return modes


@jax.jit
def velocity_from_delta_fourier(
    wavenumber_ratio,
    delta_fourier,
    f,
    sigma8,
):

    # Main model assumptions
    # CR - this part will be improved to implement the full model
    H0 = 100.0  # Mph/h coordinates
    redshift = 0.0  # Expressed at redshift 0.0
    D_growth = 1.0  # Expressed at redshift 0.0

    # Model parameter, GR not assumed
    # D_growth added to account for delta_fourier redshift dependence.
    vmodes = (
        1j
        * wavenumber_ratio
        * (1 + redshift) ** (-1)
        * f
        * sigma8
        * H0
        * D_growth
        * jnp.expand_dims(delta_fourier, -1)
    )
    velocity = jnp.fft.irfftn(vmodes, axes=(0, 1, 2))
    return velocity


@jax.jit
def density_from_delta_fourier(
    delta_fourier,
    b,
    sigma8,
):
    density = 1 + b * sigma8 * jnp.fft.irfftn(delta_fourier)
    return density


# CR -  log normal model
# delta_real = irfftn(delta_fourier)
# density = exp(b * sigma8 * delta_real - 0.5 * (b * sigma8)**2 * sigma_delta**2)


class GaussianRandomFieldBox(FourierBox):

    def __init__(
        self,
        number_bins=None,
        box_size=None,
        wavenumber=None,
        power_spectrum=None,
        kmax=0.1,
        **kwargs,
    ):
        super().__init__(number_bins, box_size)
        self.wavenumber = wavenumber
        self.init_delta_fourier()
        self.power_spectrum = power_spectrum
        self.compute_power_spectrum_grid(kmax=kmax, **kwargs)
        self.kmaxindex = jnp.where(self.wavenumber_norm_squared <= kmax**2)

    def init_delta_fourier(self):
        self.delta_fourier_base = jnp.zeros(
            shape=(self.number_bins, self.number_bins, self.number_bins // 2 + 1),
            dtype="complex64",
        )

    def compute_power_spectrum_grid(self, kmax=0.1, fillvalue=1e-10):
        power_spectrum = self.power_spectrum.copy()
        if kmax is not None:
            power_spectrum[self.wavenumber > kmax] = fillvalue
        self.power_spectrum_grid = jnp.interp(
            jnp.sqrt(self.wavenumber_norm_squared), self.wavenumber, power_spectrum
        )

    def sample_delta_fourier(self, parameter_values_dict):
        seed = get_seed_from_parameter_values_dict(parameter_values_dict)
        delta_fourier = delta_fourier_from_power_spectrum(
            self.power_spectrum_grid,
            self.number_bins,
            self.box_size,
            seed,
        )
        return delta_fourier

    def paint_modes_on_delta_fourier(self, parameter_values_dict):
        delta_mode_real = parameter_values_dict["delta_modes_real"]
        delta_mode_imag = parameter_values_dict["delta_modes_imag"]
        delta_fourier = self.delta_fourier_base.at[self.kmaxindex].set(
            delta_mode_real + delta_mode_imag * 1j
        )
        return delta_fourier

    def get_density_from_delta_fourier(
        self,
        delta_fourier,
        parameter_values_dict,
    ):
        return density_from_delta_fourier(
            delta_fourier,
            parameter_values_dict["b"],
            parameter_values_dict["s8"],
        )

    def get_velocity_from_delta_fourier(
        self,
        delta_fourier,
        parameter_values_dict,
    ):
        return velocity_from_delta_fourier(
            self.wavenumber_ratio,
            delta_fourier,
            parameter_values_dict["f"],
            parameter_values_dict["s8"],
        )

    def sample_density_velocity_fields(self, parameter_values_dict):
        delta_fourier = self.sample_delta_fourier(parameter_values_dict)
        density_field = self.get_density_from_delta_fourier(
            delta_fourier,
            parameter_values_dict,
        )
        velocity_field = self.get_velocity_from_delta_fourier(
            delta_fourier,
            parameter_values_dict,
        )
        return delta_fourier, density_field, velocity_field

    def sample_density_velocity_fields_from_modes(
        self,
        parameter_values_dict,
    ):
        delta_fourier = self.paint_modes_on_delta_fourier(parameter_values_dict)
        density_field = self.get_density_from_delta_fourier(
            delta_fourier,
            parameter_values_dict,
        )
        velocity_field = self.get_velocity_from_delta_fourier(
            delta_fourier,
            parameter_values_dict,
        )
        return delta_fourier, density_field, velocity_field

    def sample_density_velocity_fields_from_modes_lists(
        self,
        parameter_values_dict,
    ):
        key = list(parameter_values_dict.keys())[0]
        average_delta_fourier, average_density_field, average_velocity_field = (
            [],
            [],
            [],
        )
        for i in range(len(parameter_values_dict[key])):
            single_parameter_values_dict = {
                key: parameter_values_dict[key][i] for key in parameter_values_dict
            }

            delta_fourier = self.paint_modes_on_delta_fourier(
                single_parameter_values_dict
            )
            density_field = self.get_density_from_delta_fourier(
                delta_fourier,
                single_parameter_values_dict,
            )
            velocity_field = self.get_velocity_from_delta_fourier(
                delta_fourier,
                single_parameter_values_dict,
            )
            average_density_field.append(density_field)
            average_velocity_field.append(velocity_field)
            average_delta_fourier.append(delta_fourier)

        average_density_field = jnp.mean(jnp.stack(average_density_field), axis=0)
        average_velocity_field = jnp.mean(jnp.stack(average_velocity_field), axis=0)
        average_delta_fourier = jnp.mean(jnp.stack(average_delta_fourier), axis=0)

        return average_delta_fourier, average_density_field, average_velocity_field

    def draw_targets(
        self,
        size_sample,
        parameter_values_dict,
        observator=None,
    ):
        _, density, velocity = self.sample_density_velocity_fields(
            parameter_values_dict
        )

        if observator is None:
            centroid = self.get_centroid(physical_unit=False)

        seed = get_seed_from_parameter_values_dict(parameter_values_dict)
        sampled_voxels = self.draw_voxelid(size_sample, seed, density=density)
        i, j, k = self.get_voxel_coordinates(
            sampled_voxels,
            physical_unit=False,
            as_spherical=False,
        )
        ijk = jnp.stack([i, j, k])
        xyz = ijk * self.bins_to_physical

        centroid_physical_unit = centroid * self.bins_to_physical
        ra, dec, distance = self.xyz_to_radecdist(xyz, centroid=centroid_physical_unit)
        target_velocities = velocity[i.astype(int), j.astype(int), k.astype(int)].T
        target_densities = density[i.astype(int), j.astype(int), k.astype(int)].T
        di, dj, dk = jnp.asarray(ijk) - jnp.asarray(centroid)[:, None]
        normed = jnp.sqrt(jnp.sum(jnp.stack([di**2, dj**2, dk**2]), axis=0))
        direction_unit = jnp.stack([di, dj, dk]) / normed
        radial_velocities = jnp.sum(target_velocities * direction_unit, axis=0)
        return ra, dec, distance, radial_velocities, target_velocities, target_densities
