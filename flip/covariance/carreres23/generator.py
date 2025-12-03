import multiprocessing as mp
from functools import partial

import numpy as np
from scipy.special import spherical_jn

from flip.covariance import cov_utils


def angle_between(ra_0, ra_1, dec_0, dec_1):
    """Compute cos of the angle between two directions.

    Args:
        ra_0: Right ascension of first object (radians).
        ra_1: Right ascension of second object (radians).
        dec_0: Declination of first object (radians).
        dec_1: Declination of second object (radians).

    Returns:
        Cosine of the angle between the two directions.
    """
    cos_alpha = np.cos(ra_1 - ra_0) * np.cos(dec_0) * np.cos(dec_1) + np.sin(
        dec_0
    ) * np.sin(dec_1)
    return cos_alpha


def separation(r_0, r_1, cos_alpha):
    """Compute comoving separation given distances and angular cosine.

    Args:
        r_0: Comoving distance of first object.
        r_1: Comoving distance of second object.
        cos_alpha: Cosine of the angular separation.

    Returns:
        Scalar separation ``|r_0 - r_1|`` generalized to non-colinear case.
    """
    return np.sqrt(r_0**2 + r_1**2 - 2 * r_0 * r_1 * cos_alpha)


def window(r_0, r_1, cos_alpha, sep, j0kr, j2kr):
    """Wide-angle window function using bisector definition.

    Args:
        r_0: Comoving distance of first object.
        r_1: Comoving distance of second object.
        cos_alpha: Cosine of angle between directions.
        sep: Comoving separation.
        j0kr: Spherical Bessel ``j_0(k r)`` evaluated on grid.
        j2kr: Spherical Bessel ``j_2(k r)`` evaluated on grid.

    Returns:
        Window values per k contributing to the vv covariance integral.
    """
    win = 1 / 3 * (j0kr - 2 * j2kr) * cos_alpha
    win += j2kr * r_0 * r_1 / sep**2 * (1 - cos_alpha**2)
    return win


def intp(win, k, pk):
    """Integrate window times power spectrum over k.

    Args:
        win: Window array per k (shape compatible with k).
        k: Wavenumber grid.
        pk: Power spectrum values at k.

    Returns:
        Scalar integral value via trapezoidal rule.
    """
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
    """Compute Carreres23 vv covariance using bisector wide-angle.

    Applies optional grid window to the velocity power spectrum and integrates
    per pair in batches using multiprocessing.

    Args:
        ra_in: Right ascensions (radians).
        dec_in: Declinations (radians).
        rcomov_in: Comoving distances.
        k_in: Wavenumber grid.
        pk_in: Velocity power spectrum values at ``k_in``.
        grid_window_in: Optional window to apply to ``pk_in``.
        size_batch: Number of pairs per batch for processing.
        number_worker: Number of parallel workers.

    Returns:
        Flattened covariance vector: variance at index 0, then upper triangle.
    """
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
    """Compute covariance coefficient for a single pair.

    Args:
        k: Wavenumber grid.
        pk: Power spectrum values at k.
        coord: Tuple/list ``(ra_i, ra_j, dec_i, dec_j, r_i, r_j)``.

    Returns:
        Scalar integral value contributing to the covariance.
    """
    cos = angle_between(coord[0], coord[1], coord[2], coord[3])
    sep = separation(coord[4], coord[5], cos)
    ksep = np.outer(k, sep)
    j0 = spherical_jn(0, ksep)
    j2 = spherical_jn(2, ksep)
    res = window(coord[4], coord[5], cos, sep, j0, j2)
    res = intp(res, k, pk)
    return res


def generate_covariance(
    model_kind,
    power_spectrum_dict,
    coordinates_density=False,
    coordinates_velocity=None,
    **kwargs,
):
    """Generate Carreres23 covariance for velocity-only model.

    Wide-angle definition uses the bisector. Only ``vv`` block is produced.

    Args:
        model_kind: Must be ``"velocity"``.
        power_spectrum_dict: Dict with ``{"vv": [(k,), (pk,)]}`` entries.
        coordinates_density: Ignored; kept for interface compatibility.
        coordinates_velocity: Tuple ``(ra, dec, rcom)`` for velocity tracers.
        **kwargs: Forwarded to ``covariance_vv`` (e.g., batching/workers/window).

    Returns:
        Tuple ``(covariance_dict, number_densities, number_velocities, los_definition)``.
    """
    assert model_kind == "velocity"
    cov_utils.check_generator_need(
        model_kind,
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

    return (
        {"vv": np.array([cov_vv])},
        number_densities,
        number_velocities,
        los_definition,
    )
