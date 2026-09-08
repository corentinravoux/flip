import jax
import jax.numpy as jnp
import jax_cosmo as jcosmo
import tensorflow_probability.substrates.jax as tfp


def n_from_deltak(deltak, b_bias=0.2):
    """get a particle density given the k-mode

    Assumed model: density = 1 + bias * (delta)

    Parameters
    ----------
    deltak: `jax.numpy.Array`
        fourier modes of the density fields

    b_bias: float
        matter bias.

    Returns
    -------
    `jax.numpy.Array`
        baryonic particle density field (nbins, nbins, nbins)
    """
    density = 1 + b_bias * jnp.fft.irfftn(deltak)
    return density


def v_from_deltak(deltak, d2v, f, D):
    """get the (linear) velocity field

    Parameters
    ----------
    deltak: `jax.numpy.Array`
        fourier modes of the density fields

    d2v:

    f: float

    D: float

    Returns
    -------
    `jax.numpy.Array`
        linear velocity (nbins, nbins, nbins, 3)
    """
    # jcosmo.constant.H0 is the 100 by definition.
    vmodes = 1j * d2v * f * D * jcosmo.constants.H0 * jnp.expand_dims(deltak, -1)
    v = jnp.fft.irfftn(vmodes, axes=(0, 1, 2))
    return v


def v_from_grid(v, dist_mpch, r1d):

    v_interp = jax.scipy.interpolate.RegularGridInterpolator((r1d, r1d, r1d), v)
    v_model = v_interp(dist_mpch)

    return v_model


def vr_from_v(v, dist_mpch_vec):

    vr = jnp.sum(v * dist_mpch_vec, axis=-1)

    return vr


def redshift_from_dist_mpch(distance, cosmo=jcosmo.Planck15(), zbins="0:0.5:0.001"):
    """cosmological redshift for given distance and cosmology

    Parameters
    ----------
    distance: float or array
        distance in Mpc/h

    cosmo: `jax_cosmo.core.Cosmology`
        Cosmology used to draw the power spectrum

    zbins: string
        structure accepted by jnp / np r_
        e.g. `zarray = np.r_[0:0.5:0.001]`


    Returns
    -------
    float or array
        same format as input distance.
    """
    redshifts = eval(f"jnp.r_[{zbins}]")
    a = jcosmo.utils.z2a(redshifts)
    cosmo_dist = jcosmo.background.radial_comoving_distance(cosmo=cosmo, a=a)

    return jnp.interp(distance, cosmo_dist, redshifts)


# ============== #
#  Top level     #
# ============== #
@jax.jit
def get_logprob(
    data_candles,
    magabs,
    sigma8,
    h,
    dist_mpch,
    sigma_nl,
    modes_real,
    modes_imag,
    box_struct,
    use_density=True,
):
    """
    Get the global log likelihood. Combine the log likelihood on redshift, magnitude, distance and cosmo.

    Parameters:
    -----------
    mi: array
      magnitudes of supernvae

    zi: array
      supernovae redshifts

    sigma_mi: float
      Error on supernovae magnitude

    M: float
      Absolute magnitude of supernovae


    deltak: DeviceArray
      Fourier modes of density field

    d2v: DeviceArray
      Coefficients k/k2

    dist_mpch: array
      Norm of comoving distances of supernovae (Mpc/h)

    dist_mpch_vec: array
      Comoving distances of supernovae

    sigma_zi: float
      Error on redshift

    sigma_nl: int
      Non-linearity parameter

    sigma8: float
      growth rate

    pk0: DeviceArray
      Initial linear matter power spectrum

    nbins: int
      grid size

    f: float
      growth rate

    D: float
      growth factor

    H: float
      Hubble cste

    r1d: 1d vector of the box

    Returns:
    --------
    logprob_tot: DeviceArray
      SN global loglikelihood
    """
    # Cosmology
    cosmo = jcosmo.Planck15(sigma8=sigma8, h=h)

    # Cosmology => fields
    density_fields, velocity_fields, logprob_deltak = get_logprob_fields(
        cosmo,
        modes_real,
        modes_imag,
        # structure
        deltak_sampling=box_struct["deltak_sampling"],
        d2v=box_struct["d2v"],
        kmaxindex=box_struct["kmaxindex"],
        pk0=box_struct["pk0"],
        nbins=box_struct["nbins"],
    )

    # fields => target => to data
    logprob_sn = get_logprob_candles(
        data_candles,
        # nuisance
        dist_mpch=dist_mpch,
        sigma_nl=sigma_nl,
        magabs=magabs,
        h=cosmo.h,
        # fields:
        velocity_fields=velocity_fields,
        density_fields=density_fields,
        # structure
        dist_mpch_vec=box_struct["dist_mpch_vec"],
        r1d=box_struct["r1d"],
        targets_voxel_dir=box_struct["targets_voxel_dir"],
        dist_mpch_los=box_struct["dist_mpch_los"],
        use_density=use_density,
    )

    logprob_tot = logprob_sn + logprob_deltak

    return logprob_tot


# ============== #
#   Fields       #
# ============== #
@jax.jit
def get_logprob_fields(
    cosmo, modes_real, modes_imag, deltak_sampling, kmaxindex, pk0, nbins, d2v
):
    """ """
    # Cosmology and fields

    f = jcosmo.background.growth_rate(
        cosmo=cosmo, a=jnp.asarray([1.0], dtype="float32")
    )
    D = jcosmo.background.growth_factor(
        cosmo=cosmo, a=jnp.asarray([1.0], dtype="float32")
    )
    # Deltak
    deltak = deltak_sampling.at[kmaxindex].set(modes_real + modes_imag * 1j)
    # velocity
    fs8 = f * cosmo.sigma8
    velocity_fields = v_from_deltak(deltak, d2v, fs8, D)
    density_fields = n_from_deltak(deltak)  # (nbins, nbins, nbins)

    logprob_deltak = get_logprob_deltak(
        deltak=deltak[kmaxindex], pk0=pk0[kmaxindex], nbins=nbins
    )
    return density_fields, velocity_fields, logprob_deltak


# ============== #
#   Target       #
# ============== #
# i.e. redshifts & magnitudes
@jax.jit
def get_logprob_candles(
    data_candles,
    # nuisance:
    dist_mpch,
    sigma_nl,
    magabs,
    h,
    # fields:
    velocity_fields,
    density_fields,
    # structure:
    dist_mpch_vec,
    r1d,
    targets_voxel_dir,
    dist_mpch_los,
    use_density=True,
):
    """ """

    zcosmo = redshift_from_dist_mpch(dist_mpch)  # cosmo done before

    #
    # Magnitudes (incl. h0)
    #
    dl_mpc = (1 + zcosmo) * dist_mpch * h  # dist_mpch is in Mpc/h.
    mag = 5 * jnp.log10(dl_mpc) + 25 + magabs
    # vs. data
    logprob_magnitudes = tfp.distributions.Normal(
        loc=mag, scale=data_candles["mag_err"]
    ).log_prob(data_candles["mag"])

    #
    # Redshifts
    #
    vi = v_from_grid(velocity_fields, dist_mpch[:, None] * dist_mpch_vec, r1d)
    vr = vr_from_v(vi, dist_mpch_vec)
    # vs. data
    std = jnp.sqrt(
        (jcosmo.constants.c * data_candles["redshift_err"] / (1 + zcosmo)) ** 2
        + (sigma_nl) ** 2
    )
    logprob_redshifts = tfp.distributions.Normal(loc=vr, scale=std).log_prob(
        jcosmo.constants.c * (data_candles["redshift"] - zcosmo) / (1 + zcosmo)
    )

    #
    # distance logprob
    #
    logprob_dist_mpch = get_logprob_dist_mpch(
        dist_mpch=dist_mpch,
        density=density_fields,
        targets_voxel_dir=targets_voxel_dir,
        dist_mpch_los=dist_mpch_los,
    )

    #    logprob_dist_mpch = get_logprob_dist_mpch_nodensity(dist_mpch=dist_mpch)

    # output
    logprob_targets = logprob_redshifts + logprob_magnitudes + logprob_dist_mpch
    return jnp.sum(logprob_targets, axis=0)


# ============== #
#   Magnitude    #
# ============== #


# ============== #
#  Densities     #
# ============== #
@jax.jit
def get_logprob_deltak(deltak, pk0, nbins):
    """
    Get the log likelihood of cosmology.

    Parameters
    ----------
    deltak: DeviceArray
      Overdensity modes indexed on the grid in the box

    pk0: DeviceArray
      Initial linear matter power spectrum

    nbins: int
      grid size

    Returns:
    --------
    logprob_deltak: Device Array
        Cosmo log likelihood
    """
    var = pk0 * nbins ** (3 / 2)
    return jnp.sum(-0.5 * jnp.abs(deltak) ** 2 / (var))


# ============== #
#  Distances     #
# ============== #
@jax.jit
def get_logprob_dist_mpch_nodensity(dist_mpch, cutoff=200, **kwargs):
    """
    Get the log likelihood of SN distance.

    Parameters
    ----------
    dist_mpch: array
      Norm of comoving distances of supernovae

    Returns:
    --------
    logprob_dist_mpch: DeviceArray
      Distance log likelihood
    """
    #### don't forget s_8 | not included yet.
    return jnp.where(
        (dist_mpch > cutoff) + (dist_mpch < 0),
        jnp.exp(-((dist_mpch - cutoff) ** 2)),
        1.0,
    )


@jax.jit
def get_logprob_dist_mpch(
    dist_mpch, density, targets_voxel_dir, dist_mpch_los, cutoff=200
):
    """
    Get the log likelihood of SN distance.

    Parameters
    ----------
    dist_mpch: array
      Norm of comoving distances of supernovae

    Returns:
    --------
    logprob_dist_mpch: DeviceArray
      Distance log likelihood
    """
    logprob_los = density_to_logprob_los(
        density,
        dist_mpch,
        targets_voxel_dir=targets_voxel_dir,
        dist_mpch_los=dist_mpch_los,
    )

    #### don't forget s_8 | not included yet.
    return jnp.where(
        (dist_mpch > cutoff) + (dist_mpch < 0),
        jnp.exp(-((dist_mpch - cutoff) ** 2)),
        logprob_los,
    )


# jnp.zeros_like(dist_mpch)


@jax.jit
def density_to_logprob_los(
    density,
    dist_mpch,
    # structure
    targets_voxel_dir,
    dist_mpch_los,
):
    """ """
    # Density along line of sight
    target_density_los = density[
        targets_voxel_dir[:, 0], targets_voxel_dir[:, 1], targets_voxel_dir[:, 2]
    ]

    index_ = jnp.argmin((dist_mpch_los - dist_mpch[:, None]) ** 2, axis=-1)
    logprob_los = target_density_los.at[
        jnp.arange(len(target_density_los)), index_
    ].get()
    return logprob_los
