import jax.numpy as jnp
from jax import jit

from . import probabilities


@jit
def get_log_prob_tot_without_zcosmo(
    mi,
    zi,
    sigma_mi,
    M,
    d2v,
    xi,
    xi_vec,
    sigma_zi,
    sigma_nl,
    sigma8,
    pk0,
    nbins,
    f,
    D,
    H,
    cosmo,
    r1d,
    modes_real,
    modes_imag,
    kmaxindex,
    deltak_sampling=None,
    **kwargs,
):
    deltak = deltak_sampling.at[kmaxindex].set(modes_real + modes_imag * 1j)

    log_prob_sn = get_log_prob_sn_without_zcosmo(
        mi=mi,
        zi=zi,
        sigma_mi=sigma_mi,
        M=M,
        deltak=deltak,
        d2v=d2v,
        xi=xi,
        xi_vec=xi_vec,
        sigma_zi=sigma_zi,
        sigma_nl=sigma_nl,
        f=f,
        sigma8=sigma8,
        D=D,
        H=H,
        cosmo=cosmo,
        r1d=r1d,
    )

    log_prob_deltaq = probabilities.get_log_prob_deltaq(
        deltaq=deltak[kmaxindex], sigma8=None, pk0=pk0[kmaxindex], nbins=nbins
    )
    log_prob_sigma8 = probabilities.get_log_prob_sigma8(sigma8)
    log_prob_tot = log_prob_sn + log_prob_deltaq + log_prob_sigma8

    return log_prob_tot


def get_log_prob_sn_without_zcosmo(
    mi,
    zi,
    sigma_mi,
    M,
    deltak,
    d2v,
    xi,
    xi_vec,
    sigma_zi,
    sigma_nl,
    f,
    D,
    H,
    cosmo,
    r1d,
    sigma8=None,
    **kwargs,
):

    log_prob_zi = jnp.sum(
        probabilities.get_log_prob_zi_without_zcosmo(
            zi=zi,
            deltak=deltak,
            d2v=d2v,
            xi=xi,
            xi_vec=xi_vec,
            sigma_zi=sigma_zi,
            sigma_nl=sigma_nl,
            f=f,
            sigma8=sigma8,
            D=D,
            H=H,
            cosmo=cosmo,
            r1d=r1d,
        ),
        axis=0,
    )
    log_prob_mi = jnp.sum(
        probabilities.get_log_prob_mi_without_zcosmo(
            xi=xi, mi=mi, sigma_mi=sigma_mi, M=M, cosmo=cosmo
        ),
        axis=0,
    )
    log_prob_xi = jnp.sum(
        probabilities.get_log_prob_xi(xi, sigma8=None, **kwargs), axis=0
    )

    return log_prob_zi + log_prob_mi + log_prob_xi


def get_log_prob_sn(
    mi, zi, sigma_mi, M, distance_norm, zcosmo, sigma_zi, sigma_nl, vr, **kwargs
):

    log_prob_zi = jnp.sum(
        probabilities.get_log_prob_zi(
            zi=zi, zcosmo=zcosmo, vr=vr, sigma_zi=sigma_zi, sigma_nl=sigma_nl
        ),
        axis=0,
    )
    log_prob_mi = jnp.sum(
        probabilities.get_log_prob_mi(
            mi=mi, zcosmo=zcosmo, distance_norm=distance_norm, sigma_mi=sigma_mi, M=M
        ),
        axis=0,
    )
    log_prob_xi = jnp.sum(
        probabilities.get_log_prob_xi(distance_norm, **kwargs), axis=0
    )

    return log_prob_zi + log_prob_mi + log_prob_xi


def get_log_prob_tot(
    mi,
    zi,
    sigma_mi,
    M,
    distance_norm,
    zcosmo,
    sigma_zi,
    sigma_nl,
    vr,
    deltak,
    sigma8,
    pk0,
    nbins,
    **kwargs,
):
    deltaq = deltak
    log_prob_sn = get_log_prob_sn(
        mi=mi,
        zi=zi,
        sigma_mi=sigma_mi,
        M=M,
        distance_norm=distance_norm,
        zcosmo=zcosmo,
        sigma_zi=sigma_zi,
        sigma_nl=sigma_nl,
        vr=vr,
        **kwargs,
    )
    log_prob_deltaq = probabilities.get_log_prob_deltaq(
        deltaq=deltaq, sigma8=sigma8, pk0=pk0, nbins=nbins
    )
    log_prob_tot = log_prob_sn + log_prob_deltaq

    return log_prob_tot
