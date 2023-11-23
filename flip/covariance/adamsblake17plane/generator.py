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


def window_vv(r_0, r_1, cos_alpha, sep, j0kr, j2kr):
    """Note: here, the bisector angle definition is used to compute"""
    win = 1 / 3 * (j0kr + j2kr)
    alpha = np.arccos(np.clip(cos_alpha, -1.0, 1.0))
    phi = cov_utils.compute_phi_bisector(sep, alpha, r_0, r_1)
    win += -j2kr * np.cos(phi) ** 2
    return win


def window_vg(r_0, r_1, cos_alpha, sep, j1kr):
    """Note: here, the bisector angle definition is used to compute"""
    alpha = np.arccos(np.clip(cos_alpha, -1.0, 1.0))
    phi = cov_utils.compute_phi_bisector(sep, alpha, r_0, r_1)
    win = j1kr * np.cos(phi)
    return win


def intp(win, k, pk):
    pint = win.T * pk
    return np.trapz(pint, x=k)


def compute_coef_gg(k, pk, coord):
    cos = angle_between(coord[0], coord[1], coord[2], coord[3])
    sep = separation(coord[4], coord[5], cos)
    ksep = np.outer(k, sep)
    j0 = spherical_jn(0, ksep)
    res = intp(j0, k, pk * k**2)
    return res


def compute_coef_gv(k, pk, coord):
    cos = angle_between(coord[0], coord[1], coord[2], coord[3])
    sep = separation(coord[4], coord[5], cos)
    ksep = np.outer(k, sep)
    j1 = spherical_jn(1, ksep)
    res = window_vg(coord[4], coord[5], cos, sep, j1)
    res = intp(res, k, pk)
    return res


def compute_coef_vv(k, pk, coord):
    cos = angle_between(coord[0], coord[1], coord[2], coord[3])
    sep = separation(coord[4], coord[5], cos)
    ksep = np.outer(k, sep)
    j0 = spherical_jn(0, ksep)
    j2 = spherical_jn(2, ksep)
    res = window_vv(coord[4], coord[5], cos, sep, j0, j2)
    res = intp(res, k, pk)
    return res


def covariance_vv(
    ra_v,
    dec_v,
    rcomov_v,
    wavenumber,
    power_spectrum,
    size_batch=100_000,
    number_worker=8,
):
    N = len(ra_v)
    n_task = int((N * (N + 1)) / 2) - N
    batches = []
    for n in range(0, n_task, size_batch):
        brange = np.arange(n, np.min((n + size_batch, n_task)))
        i_list, j_list = cov_utils.compute_i_j(N, brange)
        r_comovi, rai, deci = rcomov_v[i_list], ra_v[i_list], dec_v[i_list]
        r_comovj, raj, decj = rcomov_v[j_list], ra_v[j_list], dec_v[j_list]
        batches.append([rai, raj, deci, decj, r_comovi, r_comovj])

    with mp.Pool(number_worker) as pool:
        func = partial(compute_coef_vv, wavenumber, power_spectrum)
        pool_results = pool.map(func, batches)
    values = np.concatenate(pool_results)

    var_val = np.trapz(power_spectrum / 3, x=wavenumber)
    cov = np.insert(values, 0, var_val)
    cov = 100**2 / (2 * np.pi**2) * cov
    return cov


def covariance_gv(
    ra_g,
    dec_g,
    rcomov_g,
    ra_v,
    dec_v,
    rcomov_v,
    wavenumber,
    power_spectrum,
    size_batch=100_000,
    number_worker=8,
):
    number_objects_g = len(ra_g)
    number_objects_v = len(ra_v)

    n_task = int(number_objects_g * number_objects_v)
    batches = []
    for n in range(0, n_task, size_batch):
        brange = np.arange(n, np.min((n + size_batch, n_task)))
        i_list, j_list = cov_utils.compute_i_j_cross_matrix(number_objects_v, brange)
        r_comovi, rai, deci = rcomov_g[i_list], ra_g[i_list], dec_g[i_list]
        r_comovj, raj, decj = rcomov_v[j_list], ra_v[j_list], dec_v[j_list]
        batches.append([rai, raj, deci, decj, r_comovi, r_comovj])

    with mp.Pool(number_worker) as pool:
        func = partial(compute_coef_gv, wavenumber, power_spectrum)
        pool_results = pool.map(func, batches)
    values = np.concatenate(pool_results)
    var_val = compute_coef_gv(
        wavenumber, power_spectrum, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    )
    cov = np.insert(values, 0, var_val)
    cov = 100 / (2 * np.pi**2) * cov
    return cov


def covariance_gg(
    ra_g,
    dec_g,
    rcomov_g,
    wavenumber,
    power_spectrum,
    size_batch=100_000,
    number_worker=8,
):
    N = len(ra_g)
    n_task = int((N * (N + 1)) / 2) - N
    batches = []
    for n in range(0, n_task, size_batch):
        brange = np.arange(n, np.min((n + size_batch, n_task)))
        i_list, j_list = cov_utils.compute_i_j(N, brange)
        r_comovi, rai, deci = rcomov_g[i_list], ra_g[i_list], dec_g[i_list]
        r_comovj, raj, decj = rcomov_g[j_list], ra_g[j_list], dec_g[j_list]
        batches.append([rai, raj, deci, decj, r_comovi, r_comovj])

    with mp.Pool(number_worker) as pool:
        func = partial(compute_coef_gg, wavenumber, power_spectrum)
        pool_results = pool.map(func, batches)
    values = np.concatenate(pool_results)
    var_val = compute_coef_gg(
        wavenumber, power_spectrum, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    )
    cov = np.insert(values, 0, var_val)
    cov = 1 / (2 * np.pi**2) * cov
    return cov
