import multiprocessing as mp
from functools import partial

import numpy as np
from scipy.special import spherical_jn

from flip.covariance import cov_utils


def angle_between(ra_0, ra_1, dec_0, dec_1):
    """Compute cos of the angle between two sky directions.

    Args:
        ra_0: Right ascension of first object (radians).
        ra_1: Right ascension of second object (radians).
        dec_0: Declination of first object (radians).
        dec_1: Declination of second object (radians).

    Returns:
        Cosine of the angular separation.
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
        cos_alpha: Cosine of angular separation.

    Returns:
        Scalar separation between points.
    """
    return np.sqrt(r_0**2 + r_1**2 - 2 * r_0 * r_1 * cos_alpha)


def window_vv(r_0, r_1, cos_alpha, sep, j0kr, j2kr):
    """Wide-angle window for vv using bisector definition.

    Args:
        r_0, r_1: Comoving distances of the two objects.
        cos_alpha: Cosine of angle between directions.
        sep: Comoving separation.
        j0kr, j2kr: Spherical Bessel terms evaluated on ``k*sep``.

    Returns:
        Window values per k contributing to vv covariance integral.
    """
    win = 1 / 3 * (j0kr + j2kr)
    alpha = np.arccos(np.clip(cos_alpha, -1.0, 1.0))
    phi = cov_utils.compute_phi_bisector_theorem(sep, alpha, r_0, r_1)
    win += -j2kr * np.cos(phi) ** 2
    return win


def window_vg(r_0, r_1, cos_alpha, sep, j1kr):
    """Wide-angle window for gv using bisector definition.

    Args:
        r_0, r_1: Comoving distances of the two objects.
        cos_alpha: Cosine of angle between directions.
        sep: Comoving separation.
        j1kr: Spherical Bessel term ``j_1(k*sep)``.

    Returns:
        Window values per k contributing to gv covariance integral.
    """
    alpha = np.arccos(np.clip(cos_alpha, -1.0, 1.0))
    phi = cov_utils.compute_phi_bisector_theorem(sep, alpha, r_0, r_1)
    win = j1kr * np.cos(phi)
    return win


def intp(win, k, pk):
    """Integrate window times spectrum over k using trapezoidal rule.

    Args:
        win: Window array per k.
        k: Wavenumber grid.
        pk: Spectrum values at k.

    Returns:
        Scalar integral value.
    """
    pint = win.T * pk
    return np.trapz(pint, x=k)


def compute_coef_gg(k, pk, coord):
    """Coefficient contributing to gg covariance for a single pair.

    Args:
        k: Wavenumber grid.
        pk: Matter spectrum values at k.
        coord: Tuple/list ``(ra_i, ra_j, dec_i, dec_j, r_i, r_j)``.

    Returns:
        Scalar integral value for gg.
    """
    cos = angle_between(coord[0], coord[1], coord[2], coord[3])
    sep = separation(coord[4], coord[5], cos)
    ksep = np.outer(k, sep)
    j0 = spherical_jn(0, ksep)
    res = intp(j0, k, pk * k**2)
    return res


def compute_coef_gv(k, pk, coord):
    """Coefficient contributing to gv covariance for a single pair.

    Args mirror those of ``compute_coef_gg`` but using cross-spectrum and window.

    Returns:
        Scalar integral value for gv.
    """
    cos = angle_between(coord[0], coord[1], coord[2], coord[3])
    sep = separation(coord[4], coord[5], cos)
    ksep = np.outer(k, sep)
    j1 = spherical_jn(1, ksep)
    res = window_vg(coord[4], coord[5], cos, sep, j1)
    res = intp(res, k, pk * k)
    return res


def compute_coef_vv(k, pk, coord):
    """Coefficient contributing to vv covariance for a single pair.

    Args mirror those of ``compute_coef_gg`` but using velocity spectrum and window.

    Returns:
        Scalar integral value for vv.
    """
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
    """Compute velocity-velocity covariance (plane-parallel wide-angle bisector).

    Args:
        ra_v, dec_v, rcomov_v: Velocity tracer coordinates and distances.
        wavenumber: Wavenumber grid for velocity spectrum.
        power_spectrum: Spectrum values at wavenumber.
        size_batch: Number of pairs per batch.
        number_worker: Number of parallel workers.

    Returns:
        Flattened covariance vector with variance at index 0.
    """
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
    """Compute density-velocity covariance (plane-parallel wide-angle bisector).

    Returns flattened covariance vector with variance at index 0.
    """
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
    """Compute density-density covariance (plane-parallel)."""
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


def generate_covariance(
    model_kind,
    power_spectrum_dict,
    coordinates_velocity=None,
    coordinates_density=None,
    **kwargs,
):
    """Generate covariance blocks using Adams & Blake (2017) plane approximation.

    Wide-angle definition uses the bisector. Supports ``gg``, ``vv``, and ``gv``.

    Args:
        model_kind: One of ``"density"``, ``"velocity"``, ``"full"``, ``"density_velocity"``.
        power_spectrum_dict: Dict providing required spectra grids and values.
        coordinates_velocity: Tuple ``(ra, dec, rcom)`` for velocity tracers.
        coordinates_density: Tuple ``(ra, dec, rcom)`` for density tracers.
        **kwargs: Forwarded to low-level covariance functions (batch/workers).

    Returns:
        Tuple ``(covariance_dict, number_densities, number_velocities, los_definition)``.
    """
    cov_utils.check_generator_need(
        model_kind,
        coordinates_density,
        coordinates_velocity,
    )
    covariance_dict = {}

    if model_kind in ["density", "full", "density_velocity"]:
        covariance_dict["gg"] = np.array(
            [
                covariance_gg(
                    coordinates_density[0],
                    coordinates_density[1],
                    coordinates_density[2],
                    power_spectrum_dict["gg"][0][0],
                    power_spectrum_dict["gg"][0][1],
                    **kwargs,
                )
            ]
        )
        number_densities = len(coordinates_density[0])
    else:
        number_densities = None

    if model_kind in ["velocity", "full", "density_velocity"]:
        covariance_dict["vv"] = np.array(
            [
                covariance_vv(
                    coordinates_velocity[0],
                    coordinates_velocity[1],
                    coordinates_velocity[2],
                    power_spectrum_dict["vv"][0][0],
                    power_spectrum_dict["vv"][0][1],
                    **kwargs,
                )
            ]
        )
        number_velocities = len(coordinates_velocity[0])
    else:
        number_velocities = None

    if model_kind == "full":
        covariance_dict["gv"] = np.array(
            [
                covariance_gv(
                    coordinates_density[0],
                    coordinates_density[1],
                    coordinates_density[2],
                    coordinates_velocity[0],
                    coordinates_velocity[1],
                    coordinates_velocity[2],
                    power_spectrum_dict["gv"][0][0],
                    power_spectrum_dict["gv"][0][1],
                    **kwargs,
                )
            ]
        )

    los_definition = "bisector"
    return (
        covariance_dict,
        number_densities,
        number_velocities,
        los_definition,
    )
