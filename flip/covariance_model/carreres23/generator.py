"""Function to compute the covariance."""
import numpy as np
from scipy.special import spherical_jn
import multiprocessing as mp
from functools import partial
from flip.covariance_model import cov_utils


def window(r_0, r_1, cos_alpha, sep, j0kr, j2kr):
    """From Johnson et al. 2014"""
    win = 1 / 3 * (j0kr - 2 * j2kr) * cos_alpha
    win += j2kr * r_0 * r_1 / sep**2 * (1 - cos_alpha**2)
    return win


def intp(win, k, pk):
    pint = win.T * pk
    return np.trapz(pint, x=k)


def finalize_cov(N, val, k, pk, pk_nogrid=None, nobj=None):
    var_val = np.trapz(pk / 3, x=k)

    # If there is a grid account for sampling problem
    cov_val = np.zeros((N, N))
    vi, vj = np.triu_indices(N, k=1)
    cov_val[vi, vj] = val
    cov_val[vj, vi] = val

    if nobj is not None:
        var_nogrid = np.trapz(pk_nogrid / 3, x=k)
        var_val = var_val + (var_nogrid - var_val) / nobj

    var_val = var_val * np.eye(N)
    cov = (cov_val + var_val) * 100**2 / (2 * np.pi**2)
    return cov


def covariance_vv(
    ra_in,
    dec_in,
    rcomov_in,
    k_in,
    pk_in,
    grid_window_in=None,
    nobj_in=None,
    n_per_batch=100_000,
    fullcov=True,
    number_worker=8,
):
    """
    Compute the covariance matrix for a set of angular positions.

    Parameters
    ----------
    ra_in : array_like
        Right ascension of angular positions in radians.
    dec_in : array_like
        Declination of angular positions in radians.
    rcomov_in : array_like
        Comoving distance to the angular positions in Mpc/h.
    k_in : array_like
        Wavenumber array for the power spectrum in h/Mpc.
    pk_in : array_like
        Power spectrum in (Mpc/h)^3.
    grid_window_in : array_like, optional
        Grid window function for the power spectrum in h/Mpc corresponding to bin size. (default is None)
    nobj_in : array_like, optional
        Number of objects in each angular bin. (default is None)
    n_per_batch : int, optional
        Number of elements to compute in each batch. (default is 100_000)
    fullcov : bool, optional
        Flag to compute the full covariance matrix. If False, give flatten matrix. (default is True)

    Returns
    -------
    cov : array_like
        Covariance matrix.

    Notes
    -----
    This docstring has been generated using ChatGPT ^^
    """

    N = len(ra_in)

    if grid_window_in is not None:
        pk = pk_in * grid_window_in**2
        pk_nogrid = pk_in
    else:
        pk = pk_in
        pk_nogrid = None
        nobj = None

    if nobj_in is not None:
        nobj = nobj_in
    else:
        nobj = None

    n_task = int((N * (N + 1)) / 2) - N

    batches = []
    for n in range(0, n_task, n_per_batch):
        brange = np.arange(n, np.min((n + n_per_batch, n_task)))
        i_list, j_list = cov_utils.compute_i_j(N, brange)
        r_comovi, rai, deci = rcomov_in[i_list], ra_in[i_list], dec_in[i_list]
        r_comovj, raj, decj = rcomov_in[j_list], ra_in[j_list], dec_in[j_list]
        batches.append([rai, raj, deci, decj, r_comovi, r_comovj, k_in, pk])

    with mp.Pool(number_worker) as pool:
        func = partial(compute_coef, k_in, pk)
        pool_results = pool.map(func, batches)
    values = np.concatenate(pool_results)

    if fullcov:
        cov = finalize_cov(N, values, k_in, pk, pk_nogrid=pk_nogrid, nobj=nobj)
    else:
        var_val = np.trapz(pk / 3, x=k_in)
        cov = np.insert(values, 0, var_val)
        cov = 100**2 / (2 * np.pi**2) * cov
    return cov


def compute_coef(k, pk, coord):
    cos = cov_utils.angle_between(coord[0], coord[1], coord[2], coord[3])
    sep = cov_utils.separation(coord[4], coord[5], cos)
    ksep = np.outer(k, sep)
    j0 = spherical_jn(0, ksep)
    j2 = spherical_jn(2, ksep)
    res = window(coord[4], coord[5], cos, sep, j0, j2)
    res = intp(res, k, pk)
    return res
