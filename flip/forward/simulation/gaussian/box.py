import jax
import jax.numpy as jnp
from flip.forward.simulation.simulator import BaseSimulator
from flip.utils import cartesian_to_spherical, spherical_to_cartesian


def compute_wavenumber_grid(number_bins, box_size):
    wavenumber_grid = jnp.moveaxis(
        jnp.stack(
            jnp.meshgrid(
                jnp.fft.fftfreq(number_bins, box_size / number_bins),
                jnp.fft.fftfreq(number_bins, box_size / number_bins),
                jnp.fft.rfftfreq(number_bins, box_size / number_bins),
                indexing="ij",
            )
        )
        * 2
        * jnp.pi,
        0,
        -1,
    )
    wavenumber_norm_squared = jnp.sum(wavenumber_grid**2, axis=-1)[..., None]
    wavenumber_norm_squared = wavenumber_norm_squared.at[0, 0, 0].set(1)
    return (
        wavenumber_grid,
        wavenumber_grid / wavenumber_norm_squared,
        wavenumber_norm_squared[..., 0],
    )


class FourierBox(BaseSimulator):

    def __init__(self, number_bins, box_size):
        self._number_bins = number_bins
        self._box_size = box_size
        self._load_wavenumber_grid()

        self._xcube, self._ycube, self._zcube = jnp.meshgrid(
            jnp.arange(number_bins), jnp.arange(number_bins), jnp.arange(number_bins)
        )
        xyz = (
            jnp.vstack(
                [self._xcube.flatten(), self._ycube.flatten(), self._zcube.flatten()]
            )
            - self.get_centroid()[:, None]
        )
        self._ra, self._dec, self._dist_mpch = jnp.stack(cartesian_to_spherical(*xyz))
        self._dist_mpch *= self.bins_to_physical  # in distance

        self._dist_mpch = self._dist_mpch.reshape(self.shape)
        self._ra = self._ra.reshape(self.shape)
        self._dec = self._dec.reshape(self.shape)

    def _load_wavenumber_grid(self):
        self._wavenumber_grid, self._wavenumber_ratio, self._wavenumber_norm_squared = (
            compute_wavenumber_grid(self.number_bins, self.box_size)
        )

    def get_centroid(self, physical_unit=False):
        centroid = jnp.asarray(self.centroid)
        if physical_unit:
            centroid = centroid * self.bins_to_physical

        return centroid

    def get_voxel_coordinates(
        self, ids=None, centroid=None, physical_unit=False, as_spherical=False
    ):
        # convert into coordinates
        X, Y, Z = jnp.meshgrid(
            jnp.linspace(0, self.number_bins - 1, self.number_bins),
            jnp.linspace(0, self.number_bins - 1, self.number_bins),
            jnp.linspace(0, self.number_bins - 1, self.number_bins),
        )

        x_flat, y_flat, z_flat = (
            X.reshape(self.shape_flat),
            Y.reshape(self.shape_flat),
            Z.reshape(self.shape_flat),
        )
        # selected coordinates
        if ids is not None:
            xyz = jnp.stack([x_flat[ids], y_flat[ids], z_flat[ids]])
        else:
            xyz = jnp.stack([x_flat, y_flat, z_flat])

        if physical_unit:
            xyz *= self.bins_to_physical

        if as_spherical:
            xyz = self.xyz_to_radecdist(
                xyz, centroid=centroid, physical_unit=physical_unit
            )

        return xyz

    def xyz_to_radecdist(self, xyz, centroid=None, physical_unit=False):
        if centroid is None:
            centroid = self.get_centroid(physical_unit=physical_unit)

        dx, dy, dz = (xyz.T - centroid).T
        xyz = jnp.stack(cartesian_to_spherical(dx, dy, dz))
        return xyz

    def draw_voxelid(self, size, seed, density=None):
        if density is not None:
            density_flat = density.reshape(self.shape_flat)
            density_flat_normed = density_flat / density_flat.sum()
        else:
            density_flat_normed = None

        # sampled
        if isinstance(size, int):
            size = (size,)  # jax format

        sampled_voxels = jax.random.choice(
            seed, self.voxel_id, shape=size, p=density_flat_normed
        )
        return sampled_voxels

    def get_line_of_sight(self, ra, dec):
        return jnp.stack(
            spherical_to_cartesian(
                ra,
                dec,
                jnp.ones_like(ra),
            )
        ).T

    @property
    def number_bins(self):
        """size of the box"""
        return self._number_bins

    @property
    def box_size(self):
        """physical size of the box"""
        return self._box_size

    @property
    def wavenumber_grid(self):
        """indice of the wavenumber"""
        if not hasattr(self, "_wavenumber_grid") or self._wavenumber_grid is None:
            self._load_wavenumber_grid()

        return self._wavenumber_grid

    @property
    def wavenumber_ratio(self):
        """delta to velocity"""
        if not hasattr(self, "_wavenumber_ratio") or self._wavenumber_ratio is None:
            self._load_wavenumber_grid()

        return self._wavenumber_ratio

    @property
    def wavenumber_norm_squared(self):
        """norm of k"""
        if (
            not hasattr(self, "_wavenumber_norm_squared")
            or self._wavenumber_norm_squared is None
        ):
            self._load_wavenumber_grid()

        return self._wavenumber_norm_squared

    @property
    def shape(self):
        """shape of the box (number_bins, number_bins, number_bins)"""
        return (self.number_bins, self.number_bins, self.number_bins)

    @property
    def shape_flat(self):
        """shape of the box (number_bins, number_bins, number_bins)"""
        return jnp.prod(jnp.asarray(self.shape))

    @property
    def centroid(self):
        """ """
        return (self.number_bins / 2, self.number_bins / 2, self.number_bins / 2)

    @property
    def bins_to_physical(self):
        """physical size of a bins (box_size/number_bins)"""
        return self.box_size / self.number_bins

    @property
    def voxel_id(self):
        """linear 1d array of the voxel"""

        return jnp.arange(self.shape_flat)
