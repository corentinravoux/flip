try:
    import jax_cosmo as jcosmo
except ImportError:
    raise ImportError(
        "jax_cosmo is not installed. Please install it to use the GaussianRandomFieldBox class."
    )

import jax
import jax.numpy as jnp
import tensorflow_probability.substrates.jax as tfp
from flip.forward.flox.fourrier_box import FourierBox


def v_from_grid(v, dist_mpch, r1d):
    v_interp = jax.scipy.interpolate.RegularGridInterpolator((r1d, r1d, r1d), v)
    v_model = v_interp(dist_mpch)
    return v_model


def vr_from_v(v, dist_mpch_vec):
    vr = jnp.sum(v * dist_mpch_vec, axis=-1)
    return vr


def deltak_from_power_spectrum(
    power_spectrum_grid,
    number_bins,
    size_box,
    seed,
):
    mean = jnp.zeros((number_bins, number_bins, number_bins))
    std = jnp.ones((number_bins, number_bins, number_bins))
    w = tfp.distributions.Normal(loc=mean, scale=std).sample(seed=seed)
    modes = jnp.fft.rfftn(w)
    modes *= jnp.sqrt(power_spectrum_grid * (number_bins / size_box) ** 3)
    return modes


def n_from_deltak(
    deltak,
    b_bias=0.2,
):
    density = 1 + b_bias * jnp.fft.irfftn(deltak)
    return density


def v_from_deltak(
    deltak,
    d2v,
    f,
    D,
    H0,
):
    # jcosmo.constant.H0 is the 100 by definition.
    vmodes = 1j * d2v * f * D * H0 * jnp.expand_dims(deltak, -1)
    v = jnp.fft.irfftn(vmodes, axes=(0, 1, 2))
    return v


def vr_from_v_and_position(velocities, positions, centroid=(0, 0, 0)):
    # create the norm unit
    di, dj, dk = jnp.asarray(positions) - jnp.asarray(centroid)[:, None]
    normed = jnp.sqrt(jnp.sum(jnp.stack([di**2, dj**2, dk**2]), axis=0))

    # direction units
    direction_unit = jnp.stack([di, dj, dk]) / normed

    vradial = jnp.sum(velocities * direction_unit, axis=0)
    return vradial


def redshift_from_dist_mpch(distance, cosmo=jcosmo.Planck15(), zbins="0:0.5:0.001"):
    redshifts = eval(f"jnp.r_[{zbins}]")
    a = jcosmo.utils.z2a(redshifts)
    cosmo_dist = jcosmo.background.radial_comoving_distance(cosmo=cosmo, a=a)

    return jnp.interp(distance, cosmo_dist, redshifts)


class GaussianRandomFieldBox(FourierBox):

    def __init__(
        self, nbins, lsize, cosmo, wavenumber, power_spectrum, kmax=0.1, **kwargs
    ):
        super().__init__(nbins, lsize)
        self.wavenumber = wavenumber
        self.power_spectrum = power_spectrum
        self.compute_power_spectrum_grid(kmax=kmax, **kwargs)
        self.cosmo = cosmo

    def compute_power_spectrum_grid(self, kmax=0.1, fillvalue=1e-10):
        """loads the power spectrum P(k)

        Parameters
        ----------
        kmax: float or None
            the power-spectrum will be set to `fillvalue` at k>kmax

        **kwargs goes to pk_from_cosmo ; fillvalue etc

        Returns
        -------
        None
        """

        power_spectrum = self.power_spectrum.copy()
        if kmax is not None:
            power_spectrum[self.wavenumber > kmax] = fillvalue
        self._power_spectrum_grid = jnp.interp(
            jnp.sqrt(self.k2), self.wavenumber, power_spectrum
        )

    # Sampling
    def sample_deltak(self, seed):
        modes = deltak_from_power_spectrum(
            power_spectrum_grid=self._power_spectrum_grid,
            number_bins=self.nbins,
            size_box=self.lsize,
            seed=seed,
        )
        return modes

    def sample_n_from_deltak(self, deltak, b_bias=0.2):
        n = n_from_deltak(deltak, b_bias=b_bias)
        return n

    def sample_v_from_deltak(self, deltak, a=1.0):
        a = jnp.atleast_1d(a)  # jax
        growth_factor = jcosmo.background.growth_factor(self.cosmo, a=a)  # D
        growth_rate = jcosmo.background.growth_rate(self.cosmo, a=a)  # f
        H0 = jcosmo.constants.H0
        v = v_from_deltak(deltak, d2v=self.d2v, f=growth_rate, D=growth_factor, H0=H0)

        return v

    def draw_targets(self, size, seed, deltak=None, b_bias=None, observator=None, a=1):
        """draw ra, dec, z_cosmo, v_radial given in the input cosmology

        This draws targets following the baryonic matter density.

        Parameters
        ----------
        size: int
            number of particle to draw

        seed: PRNGKey
            a PRNG key used as the random key.
            if unsure, do: seed = jax.random.PRNGKey(0)

        deltak:
           mode of the density field as returned by self.sample_deltak

        b_bias:
            density bias

        Returns
        -------
        tuple
            1. RA
            2. Dec
            3. cosmological redshift
            4. distance (Mpc/h)
            5. radial peculiar velocity (km/s)
            6. density at the target location

        """

        if observator is None:
            centroid = self.get_centroid(physical_unit=False)  # (3) center of the box

        if deltak is None:
            deltak = self.sample_deltak(seed)

        if b_bias is not None:
            prop_density = dict(b_bias=b_bias)
        else:
            prop_density = {}

        # convert then into
        # -> density fields
        density = self.sample_n_from_deltak(
            deltak, **prop_density
        )  # (nbins, nbins, nbins)
        velocities = self.sample_v_from_deltak(deltak, a=a)  # (nbins, nbins, nbins, 3)

        # -> Draw targets in the box
        sampled_voxels = self.draw_voxelid(size, seed, density=density)  # (ntargets)

        # get their 3d-positions
        i, j, k = self.get_voxel_coordinates(
            sampled_voxels,  # (ntargets, ntargets, ntargets)
            physical_unit=False,
            as_spherical=False,
        )
        ijk = jnp.stack([i, j, k])
        xyz = ijk * self.bins_to_physical  # (ntargets, ntargets, ntargets)

        # -> xyz is needed for vr_from_v
        centroid_physical_unit = centroid * self.bins_to_physical  # (3)
        distance, ra, dec = self.xyz_to_distradec(
            xyz, centroid=centroid_physical_unit
        )  # (ntargets, ntargets, ntargets)

        # convert distance to redshift
        zcosmo = redshift_from_dist_mpch(distance)  # (ntargets) | assume Planck15

        # vpec
        target_velocities = velocities[
            i.astype(int), j.astype(int), k.astype(int)
        ].T  # (ntargets, ntargets, ntargets)
        target_densities = density[
            i.astype(int), j.astype(int), k.astype(int)
        ].T  # (ntargets, ntargets, ntargets)
        vradials = vr_from_v_and_position(
            target_velocities, ijk, centroid=centroid
        )  # (ntargets)

        return ra, dec, zcosmo, distance, vradials, target_densities

    @property
    def power_spectrum_grid(self):
        """power spectrum grid"""
        return self._power_spectrum_grid
