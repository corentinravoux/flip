import multiprocessing as mp
from functools import partial

import numpy as np
from scipy.special import spherical_jn

from flip.covariance import cov_utils


def angle_between(ra_0, ra_1, dec_0, dec_1):
    """Compute cos of the angle between r0 and r1."""
    cos_alpha = np.cos(ra_1 - ra_0) * np.cos(dec_0) * np.cos(dec_1) + np.sin(
        dec_0
    ) * np.sin(dec_1)
    return cos_alpha


def separation(r_0, r_1, cos_alpha):
    """Compute separation between r_0 and r_1."""
    return np.sqrt(r_0**2 + r_1**2 - 2 * r_0 * r_1 * cos_alpha)


def window(r_0, r_1, cos_alpha, sep, j0kr, j2kr):
    """Note: here, the bisector angle definition is used in wide-angle"""
    win = 1 / 3 * (j0kr - 2 * j2kr) * cos_alpha
    win += j2kr * r_0 * r_1 / sep**2 * (1 - cos_alpha**2)
    return win


def intp(win, k, pk):
    pint = win.T * pk
    return np.trapz(pint, x=k)


def covariance_vv(
    ra_in,
    dec_in,
    rcomov_in,
    k_in,
    pk_in,
    grid_window_in=None,
    size_batch=100_000,
    number_worker=8,
):
    N = len(ra_in)

    if grid_window_in is not None:
        pk = pk_in * grid_window_in**2
    else:
        pk = pk_in

    n_task = int((N * (N + 1)) / 2) - N

    batches = []
    for n in range(0, n_task, size_batch):
        brange = np.arange(n, np.min((n + size_batch, n_task)))
        i_list, j_list = cov_utils.compute_i_j(N, brange)
        r_comovi, rai, deci = rcomov_in[i_list], ra_in[i_list], dec_in[i_list]
        r_comovj, raj, decj = rcomov_in[j_list], ra_in[j_list], dec_in[j_list]
        batches.append([rai, raj, deci, decj, r_comovi, r_comovj])

    with mp.Pool(number_worker) as pool:
        func = partial(compute_coef, k_in, pk)
        pool_results = pool.map(func, batches)
    values = np.concatenate(pool_results)

    var_val = np.trapz(pk / 3, x=k_in)
    cov = np.insert(values, 0, var_val)
    cov = 100**2 / (2 * np.pi**2) * cov
    return cov


def compute_coef(k, pk, coord):
    cos = angle_between(coord[0], coord[1], coord[2], coord[3])
    sep = separation(coord[4], coord[5], cos)
    ksep = np.outer(k, sep)
    j0 = spherical_jn(0, ksep)
    j2 = spherical_jn(2, ksep)
    res = window(coord[4], coord[5], cos, sep, j0, j2)
    res = intp(res, k, pk)
    return res


def generate_covariance(
    model_type,
    power_spectrum_dict,
    coordinates_density=False,
    coordinates_velocity=None,
    **kwargs,
):
    """
    The generate_covariance function generates a covariance matrix for the velocity field.

    Args:
        model_type: Specify the type of model to generate
        power_spectrum_dict: Pass the power spectrum to the function
        coordinates_density: Specify the coordinates of the density field
        coordinates_velocity: Generate the covariance matrix
        **kwargs: Pass additional parameters to the function
        : Generate the covariance matrix for the velocity field
        The wide angle used is the bisector.
    Returns:
        A dictionary with a single key &quot;vv&quot;

    """
    assert model_type == "velocity"
    cov_utils.check_generator_need(
        model_type,
        coordinates_density,
        coordinates_velocity,
    )
    number_densities = None
    number_velocities = len(coordinates_velocity[0])
    cov_vv = covariance_vv(
        coordinates_velocity[0],
        coordinates_velocity[1],
        coordinates_velocity[2],
        power_spectrum_dict["vv"][0][0],
        power_spectrum_dict["vv"][0][1],
        **kwargs,
    )

    los_definition = "bisector"
    redshift_dict = None

    return (
        {"vv": np.array([cov_vv])},
        number_densities,
        number_velocities,
        los_definition,
        redshift_dict,
    )
