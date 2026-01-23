import itertools
import multiprocessing as mp
from functools import partial

import cosmoprimo
import numpy as np
from scipy import integrate
from scipy.interpolate import interp1d
from scipy.special import factorial, spherical_jn

from flip.covariance import cov_utils


def compute_correlation_coefficient_simple_integration(p, q, ell, r, k, pk):
    """Compute correlation coefficient via direct integration.

    Implements Lai et al. formalism with the ``(2π^2)`` factor inside the
    integrand. The small-scale velocity dispersion (``sigma_u``) should be
    applied to ``pk`` upstream when required.

    Args:
        p: Power of k from derivative order on point 1.
        q: Power of k from derivative order on point 2.
        ell: Spherical Bessel order.
        r: Separation values at which to evaluate.
        k: Wavenumbers.
        pk: Power spectrum values at ``k``.

    Returns:
        Array of correlation coefficient values at separations ``r``.
    """
    kr = np.outer(k, r)
    integrand = (
        spherical_jn(ell, kr).T * k**2 * k ** (2 * (p + q)) * pk / (2 * np.pi**2)
    )
    return integrate.simpson(integrand, x=k)


def compute_correlation_coefficient_hankel(
    p, q, ell, r, k, pk, hankel_overhead_coefficient=2
):
    """Compute correlation coefficient using FFTLog Hankel transform.

    Uses cosmoprimo's ``PowerToCorrelation`` to accelerate the computation.
    Falls back to direct integration for small ``r`` values where the Hankel
    result may be unreliable, controlled by ``hankel_overhead_coefficient``.

    Args:
        p: Power of k from derivative order on point 1.
        q: Power of k from derivative order on point 2.
        ell: Spherical Bessel order.
        r: Separation values.
        k: Wavenumbers.
        pk: Power spectrum values.
        hankel_overhead_coefficient: Safety margin for the small-``r`` mask.

    Returns:
        Array of correlation coefficient values at separations ``r``.
    """
    integrand = k ** (2 * (p + q)) * pk
    Hankel = cosmoprimo.fftlog.PowerToCorrelation(k, ell=ell, q=0, complex=False)
    Hankel.set_fft_engine("numpy")
    r_hankel, xi_hankel = Hankel(integrand)
    mask = r < np.min(r_hankel) * hankel_overhead_coefficient
    output = np.empty_like(r)
    output[mask] = compute_correlation_coefficient_simple_integration(
        p, q, ell, r[mask], k, pk
    )
    output[~mask] = (-1) ** (ell // 2) * interp1d(r_hankel, xi_hankel)(r[~mask])
    return output


def compute_cov_vv(
    ra,
    dec,
    comoving_distance,
    wavenumber_tt,
    power_spectrum_tt,
    grid_window_v_tt=None,
    size_batch=10_000,
    number_worker=8,
    hankel=True,
    los_definition="bisector",
):
    """Compute velocity-velocity covariance in Lai22 wide-angle formalism.

    Args:
        ra: Right ascensions of objects (radians).
        dec: Declinations of objects (radians).
        comoving_distance: Comoving distances of objects.
        wavenumber_tt: Wavenumbers for velocity-velocity spectrum.
        power_spectrum_tt: P_tt values at ``wavenumber_tt``.
        grid_window_v_tt: Optional window function applied to ``P_tt``.
        size_batch: Batch size for pair processing.
        number_worker: Number of parallel workers (1 for serial).
        hankel: If True, use Hankel transform acceleration where applicable.
        los_definition: Line-of-sight definition ("bisector").

    Returns:
        Flattened covariance vector with the diagonal term at index 0 followed by upper-triangle terms.
    """
    if grid_window_v_tt is not None:
        power_spectrum_tt = power_spectrum_tt * grid_window_v_tt**2

    number_objects = len(ra)
    n_task = int((number_objects * (number_objects + 1)) / 2) - number_objects

    parameters = []
    for n in range(0, n_task, size_batch):
        batches = np.arange(n, np.min((n + size_batch, n_task)))
        i_list, j_list = cov_utils.compute_i_j(number_objects, batches)
        r_i, ra_i, dec_i = comoving_distance[i_list], ra[i_list], dec[i_list]
        r_j, ra_j, dec_j = comoving_distance[j_list], ra[j_list], dec[j_list]
        r, theta, phi = cov_utils.angle_separation(
            ra_i, ra_j, dec_i, dec_j, r_i, r_j, los_definition=los_definition
        )
        parameters.append([r, theta, phi])

    func = partial(coefficient_vv, wavenumber_tt, power_spectrum_tt, hankel=hankel)
    if number_worker == 1:
        cov_vv = np.concatenate([func(param) for param in parameters])
    else:
        with mp.Pool(number_worker) as pool:
            cov_vv = np.concatenate(pool.map(func, parameters))

    variance_val = coefficient_vv(
        wavenumber_tt,
        power_spectrum_tt,
        [0, 0, 0],
        hankel=False,
    )

    cov_vv = np.insert(cov_vv, 0, variance_val)
    # Multiplication by H_0 ** 2
    cov_vv = 100**2 * cov_vv

    return cov_vv


def coefficient_vv(wavenumber, power_spectrum_tt, coord, hankel=True):
    """Coefficient contributing to vv covariance for a given separation.

    Args:
        wavenumber: Wavenumbers array.
        power_spectrum_tt: Velocity-velocity power spectrum at ``wavenumber``.
        coord: Tuple/list ``(r, theta, phi)`` separation coordinates.
        hankel: If True, use Hankel transform; else use direct integration.

    Returns:
        Scalar covariance contribution at the provided separation.
    """
    result = 0
    for ell in [0, 2]:
        if hankel:
            correlation = compute_correlation_coefficient_hankel(
                -0.5, -0.5, ell, coord[0], wavenumber, power_spectrum_tt
            )
        else:
            correlation = compute_correlation_coefficient_simple_integration(
                -0.5, -0.5, ell, coord[0], wavenumber, power_spectrum_tt
            )
        h_function = eval(f"h_terms.H_vv_l{ell}")(coord[1], coord[2])
        result = result + np.real(1j ** (ell)) * correlation * h_function
    return result


def compute_cov_gg(
    pmax,
    qmax,
    ra,
    dec,
    comoving_distance,
    wavenumber_mm,
    wavenumber_mt,
    wavenumber_tt,
    power_spectrum_mm,
    power_spectrum_mt,
    power_spectrum_tt,
    grid_window_m_mm=None,
    grid_window_m_mt=None,
    grid_window_v_mt=None,
    grid_window_v_tt=None,
    size_batch=10_000,
    number_worker=8,
    sig_damp_mm_gg_m=None,
    hankel=True,
    los_definition="bisector",
):
    """Compute density-density covariance terms for Lai22 wide-angle.

    Builds per-``m`` contributions for b², f² and bf components using sums over
    ``p,q,ℓ`` multipoles with optional damping on the matter-matter spectrum.

    Args:
        pmax: Maximum p index.
        qmax: Maximum q index.
        ra: Right ascensions (radians).
        dec: Declinations (radians).
        comoving_distance: Comoving distances.
        wavenumber_mm: Wavenumbers for matter-matter spectrum.
        wavenumber_mt: Wavenumbers for matter-velocity spectrum.
        wavenumber_tt: Wavenumbers for velocity-velocity spectrum.
        power_spectrum_mm: P_mm at ``wavenumber_mm``.
        power_spectrum_mt: P_mv at ``wavenumber_mt``.
        power_spectrum_tt: P_vv at ``wavenumber_tt``.
        grid_window_m_mm: Optional window for P_mm.
        grid_window_m_mt: Optional window for P_mv.
        grid_window_v_mt: Optional velocity window multiplying P_mv.
        grid_window_v_tt: Optional window for P_vv.
        size_batch: Batch size for pair processing.
        number_worker: Number of workers (1 for serial).
        sig_damp_mm_gg_m: Damping scale for P_mm high-k suppression in gg b² terms.
        hankel: Use Hankel transform acceleration when True.
        los_definition: Line-of-sight definition ("bisector").

    Returns:
        Tuple ``(m_index, cov_gg_b2, cov_gg_f2, cov_gg_bf)`` with lists of flattened per-m covariance vectors (variance inserted at index 0).
    """
    if grid_window_m_mm is not None:
        power_spectrum_mm = power_spectrum_mm * grid_window_m_mm**2

    if grid_window_m_mt is not None:
        power_spectrum_mt = power_spectrum_mt * grid_window_m_mt

    if grid_window_v_mt is not None:
        power_spectrum_mt = power_spectrum_mt * grid_window_v_mt

    if grid_window_v_tt is not None:
        power_spectrum_tt = power_spectrum_tt * grid_window_v_tt**2

    number_objects = len(ra)
    n_task = int((number_objects * (number_objects + 1)) / 2) - number_objects
    parameters = []
    for n in range(0, n_task, size_batch):
        batches = np.arange(n, np.min((n + size_batch, n_task)))
        i_list, j_list = cov_utils.compute_i_j(number_objects, batches)
        r_i, ra_i, dec_i = comoving_distance[i_list], ra[i_list], dec[i_list]
        r_j, ra_j, dec_j = comoving_distance[j_list], ra[j_list], dec[j_list]
        r, theta, phi = cov_utils.angle_separation(
            ra_i, ra_j, dec_i, dec_j, r_i, r_j, los_definition=los_definition
        )
        parameters.append([r, theta, phi])

    p_index = np.arange(pmax + 1)
    q_index = np.arange(qmax + 1)
    m_index = np.arange(0, 2 * (qmax + pmax) + 1, 2)
    iter_pq = np.array(list(itertools.product(p_index, q_index)))
    sum_iter_pq = 2 * np.sum(iter_pq, axis=1)
    function_covariance_dict = {}
    for m in m_index:
        function_covariance_dict[f"func_gg_b2_{m}"] = partial(
            coefficient_gg_b2_m,
            wavenumber_mm,
            power_spectrum_mm,
            iter_pq,
            sum_iter_pq,
            m,
            sig_damp_mm_gg_m,
            hankel=hankel,
        )
        function_covariance_dict[f"func_gg_f2_{m}"] = partial(
            coefficient_gg_f2_m,
            wavenumber_tt,
            power_spectrum_tt,
            iter_pq,
            sum_iter_pq,
            m,
            hankel=hankel,
        )
        function_covariance_dict[f"func_gg_bf_{m}"] = partial(
            coefficient_gg_bf_m,
            wavenumber_mt,
            power_spectrum_mt,
            iter_pq,
            sum_iter_pq,
            m,
            hankel=hankel,
        )

    if number_worker == 1:
        for m in m_index:
            function_covariance_dict[f"cov_gg_b2_{m}"] = np.concatenate(
                [
                    function_covariance_dict[f"func_gg_b2_{m}"](param)
                    for param in parameters
                ]
            )
            function_covariance_dict[f"cov_gg_f2_{m}"] = np.concatenate(
                [
                    function_covariance_dict[f"func_gg_f2_{m}"](param)
                    for param in parameters
                ]
            )
            function_covariance_dict[f"cov_gg_bf_{m}"] = np.concatenate(
                [
                    function_covariance_dict[f"func_gg_bf_{m}"](param)
                    for param in parameters
                ]
            )
    else:
        with mp.Pool(number_worker) as pool:
            for m in m_index:
                function_covariance_dict[f"map_async_gg_b2_{m}"] = pool.map_async(
                    function_covariance_dict[f"func_gg_b2_{m}"], parameters
                )
                function_covariance_dict[f"map_async_gg_f2_{m}"] = pool.map_async(
                    function_covariance_dict[f"func_gg_f2_{m}"], parameters
                )
                function_covariance_dict[f"map_async_gg_bf_{m}"] = pool.map_async(
                    function_covariance_dict[f"func_gg_bf_{m}"], parameters
                )

            for m in m_index:
                function_covariance_dict[f"cov_gg_b2_{m}"] = np.concatenate(
                    function_covariance_dict[f"map_async_gg_b2_{m}"].get()
                )
                function_covariance_dict[f"cov_gg_f2_{m}"] = np.concatenate(
                    function_covariance_dict[f"map_async_gg_f2_{m}"].get()
                )
                function_covariance_dict[f"cov_gg_bf_{m}"] = np.concatenate(
                    function_covariance_dict[f"map_async_gg_bf_{m}"].get()
                )

    for m in m_index:
        variance_val_gg_b2_m = coefficient_gg_b2_m(
            wavenumber_mm,
            power_spectrum_mm,
            iter_pq,
            sum_iter_pq,
            m,
            sig_damp_mm_gg_m,
            [0, 0, 0],
            hankel=False,
        )

        variance_val_gg_f2_m = coefficient_gg_f2_m(
            wavenumber_tt,
            power_spectrum_tt,
            iter_pq,
            sum_iter_pq,
            m,
            [0, 0, 0],
            hankel=False,
        )

        variance_val_gg_bf_m = coefficient_gg_bf_m(
            wavenumber_mt,
            power_spectrum_mt,
            iter_pq,
            sum_iter_pq,
            m,
            [0, 0, 0],
            hankel=False,
        )  # does not work with hankel

        function_covariance_dict[f"cov_gg_b2_{m}"] = np.insert(
            function_covariance_dict[f"cov_gg_b2_{m}"], 0, variance_val_gg_b2_m
        )
        function_covariance_dict[f"cov_gg_f2_{m}"] = np.insert(
            function_covariance_dict[f"cov_gg_f2_{m}"], 0, variance_val_gg_f2_m
        )
        function_covariance_dict[f"cov_gg_bf_{m}"] = np.insert(
            function_covariance_dict[f"cov_gg_bf_{m}"], 0, variance_val_gg_bf_m
        )

    cov_gg_b2 = [function_covariance_dict[f"cov_gg_b2_{m}"] for m in m_index]
    cov_gg_f2 = [function_covariance_dict[f"cov_gg_f2_{m}"] for m in m_index]
    cov_gg_bf = [function_covariance_dict[f"cov_gg_bf_{m}"] for m in m_index]

    return m_index, cov_gg_b2, cov_gg_f2, cov_gg_bf


def compute_cov_gg_add(
    pmax,
    qmax,
    ra,
    dec,
    comoving_distance,
    wavenumber_mm,
    power_spectrum_mm,
    grid_window_m_mm=None,
    size_batch=10_000,
    number_worker=8,
    sig_damp_mm_gg_m=None,
    hankel=True,
    los_definition="bisector",
):
    """Compute additional gg b² terms only (e.g. for additive components).

    Args:
        pmax: Maximum p index.
        qmax: Maximum q index.
        ra: Right ascensions (radians).
        dec: Declinations (radians).
        comoving_distance: Comoving distances.
        wavenumber_mm: Wavenumbers for matter-matter spectrum.
        power_spectrum_mm: P_mm values.
        grid_window_m_mm: Optional window for P_mm.
        size_batch: Batch size for pair processing.
        number_worker: Number of workers.
        sig_damp_mm_gg_m: Damping scale for P_mm high-k suppression.
        hankel: Use Hankel transform acceleration when True.
        los_definition: Line-of-sight definition.

    Returns:
        Tuple ``(m_index, cov_gg_b2_add)`` with flattened covariance vectors including variance at index 0.
    """
    if grid_window_m_mm is not None:
        power_spectrum_mm = power_spectrum_mm * grid_window_m_mm**2

    number_objects = len(ra)
    n_task = int((number_objects * (number_objects + 1)) / 2) - number_objects
    parameters = []
    for n in range(0, n_task, size_batch):
        batches = np.arange(n, np.min((n + size_batch, n_task)))
        i_list, j_list = cov_utils.compute_i_j(number_objects, batches)
        r_i, ra_i, dec_i = comoving_distance[i_list], ra[i_list], dec[i_list]
        r_j, ra_j, dec_j = comoving_distance[j_list], ra[j_list], dec[j_list]
        r, theta, phi = cov_utils.angle_separation(
            ra_i, ra_j, dec_i, dec_j, r_i, r_j, los_definition=los_definition
        )
        parameters.append([r, theta, phi])

    p_index = np.arange(pmax + 1)
    q_index = np.arange(qmax + 1)
    m_index = np.arange(0, 2 * (qmax + pmax) + 1, 2)
    iter_pq = np.array(list(itertools.product(p_index, q_index)))
    sum_iter_pq = 2 * np.sum(iter_pq, axis=1)
    function_covariance_dict = {}
    for m in m_index:
        function_covariance_dict[f"func_gg_b2_{m}"] = partial(
            coefficient_gg_b2_m,
            wavenumber_mm,
            power_spectrum_mm,
            iter_pq,
            sum_iter_pq,
            m,
            sig_damp_mm_gg_m,
            hankel=hankel,
        )

    if number_worker == 1:
        for m in m_index:
            function_covariance_dict[f"cov_gg_b2_{m}"] = np.concatenate(
                [
                    function_covariance_dict[f"func_gg_b2_{m}"](param)
                    for param in parameters
                ]
            )
    else:
        with mp.Pool(number_worker) as pool:
            for m in m_index:
                function_covariance_dict[f"map_async_gg_b2_{m}"] = pool.map_async(
                    function_covariance_dict[f"func_gg_b2_{m}"], parameters
                )

            for m in m_index:
                function_covariance_dict[f"cov_gg_b2_{m}"] = np.concatenate(
                    function_covariance_dict[f"map_async_gg_b2_{m}"].get()
                )

    for m in m_index:
        variance_val_gg_b2_m = coefficient_gg_b2_m(
            wavenumber_mm,
            power_spectrum_mm,
            iter_pq,
            sum_iter_pq,
            m,
            sig_damp_mm_gg_m,
            [0, 0, 0],
            hankel=False,
        )

        function_covariance_dict[f"cov_gg_b2_{m}"] = np.insert(
            function_covariance_dict[f"cov_gg_b2_{m}"], 0, variance_val_gg_b2_m
        )

    cov_gg_b2_add = [function_covariance_dict[f"cov_gg_b2_{m}"] for m in m_index]

    return m_index, cov_gg_b2_add


def coefficient_gg_b2_m(
    wavenumber_mm,
    power_spectrum_mm,
    iter_pq,
    sum_iter_pq,
    m_value,
    sig_damp_mm_gg_m,
    coord,
    hankel=True,
):
    """b² coefficient for gg block at fixed m.

    Args:
        wavenumber_mm: Wavenumbers for P_mm.
        power_spectrum_mm: P_mm values (with optional damping applied inside).
        iter_pq: Array of all (p, q) combinations.
        sum_iter_pq: Array of ``2*(p+q)`` per (p, q) used for m filtering.
        m_value: Even m index for basis term.
        sig_damp_mm_gg_m: Damping scale for P_mm when ``m_value != 0``.
        coord: Separation coordinates ``(r, theta, phi)``.
        hankel: Use Hankel transform acceleration when True.

    Returns:
        Scalar contribution for the given separation.
    """
    if (sig_damp_mm_gg_m is not None) and (m_value != 0):
        power_spectrum_mm = power_spectrum_mm * np.exp(
            -((wavenumber_mm * sig_damp_mm_gg_m) ** 4) / 2
        )
    result = 0
    pq_index = iter_pq[sum_iter_pq == m_value]
    for pq in pq_index:
        p, q = pq[0], pq[1]
        lmax = 2 * (p + q + 1)
        for ell in range(0, lmax + 1, 2):
            if hankel:
                correlation = compute_correlation_coefficient_hankel(
                    p, q, ell, coord[0], wavenumber_mm, power_spectrum_mm
                )
            else:
                correlation = compute_correlation_coefficient_simple_integration(
                    p, q, ell, coord[0], wavenumber_mm, power_spectrum_mm
                )
            h_function = eval(f"h_terms.H_gg_l{ell}_p{p}_q{q}")(coord[1], coord[2])
            coeff = (
                np.real(1j ** (ell))
                * ((-1) ** (p + q))
                / (2 ** (p + q) * factorial(p) * factorial(q))
            )
            result = result + coeff * correlation * h_function
    return result


def coefficient_gg_f2_m(
    wavenumber_tt,
    power_spectrum_tt,
    iter_pq,
    sum_iter_pq,
    m_value,
    coord,
    hankel=True,
):
    """f² coefficient for gg block at fixed m.

    Args mirror those of ``coefficient_gg_b2_m`` but using P_tt and h-terms shifted by +1.

    Returns:
        Scalar contribution for the given separation.
    """
    result = 0
    pq_index = iter_pq[sum_iter_pq == m_value]
    for pq in pq_index:
        p, q = pq[0], pq[1]
        lmax = 2 * (p + q + 1)
        for ell in range(0, lmax + 1, 2):
            if hankel:
                correlation = compute_correlation_coefficient_hankel(
                    p, q, ell, coord[0], wavenumber_tt, power_spectrum_tt
                )
            else:
                correlation = compute_correlation_coefficient_simple_integration(
                    p, q, ell, coord[0], wavenumber_tt, power_spectrum_tt
                )
            h_function = eval(f"h_terms.H_gg_l{ell}_p{p+1}_q{q+1}")(coord[1], coord[2])
            coeff = (
                np.real(1j ** (ell))
                * ((-1) ** (p + q))
                / (2 ** (p + q) * factorial(p) * factorial(q))
            )
            result = result + coeff * correlation * h_function
    return result


def coefficient_gg_bf_m(
    wavenumber_mt,
    power_spectrum_mt,
    iter_pq,
    sum_iter_pq,
    m_value,
    coord,
    hankel=True,
):
    """bf cross coefficient for gg block at fixed m.

    Combines the two h-terms with indices (p+1,q) and (p,q+1).

    Args mirror those of ``coefficient_gg_b2_m`` but using P_mt.

    Returns:
        Scalar contribution for the given separation.
    """
    result = 0
    pq_index = iter_pq[sum_iter_pq == m_value]
    for pq in pq_index:
        p, q = pq[0], pq[1]
        lmax = 2 * (p + q + 1)
        for ell in range(0, lmax + 1, 2):
            if hankel:
                correlation = compute_correlation_coefficient_hankel(
                    p, q, ell, coord[0], wavenumber_mt, power_spectrum_mt
                )
            else:
                correlation = compute_correlation_coefficient_simple_integration(
                    p, q, ell, coord[0], wavenumber_mt, power_spectrum_mt
                )
            h_function = eval(f"h_terms.H_gg_l{ell}_p{p+1}_q{q}")(
                coord[1], coord[2]
            ) + eval(f"h_terms.H_gg_l{ell}_p{p}_q{q+1}")(coord[1], coord[2])
            coeff = (
                np.real(1j ** (ell))
                * ((-1) ** (p + q))
                / (2 ** (p + q) * factorial(p) * factorial(q))
            )
            result = result + coeff * correlation * h_function
    return result


def compute_cov_gv(
    pmax,
    ra_g,
    dec_g,
    comoving_distance_g,
    ra_v,
    dec_v,
    comoving_distance_v,
    wavenumber_mt,
    wavenumber_tt,
    power_spectrum_mt,
    power_spectrum_tt,
    grid_window_m_mt=None,
    grid_window_v_mt=None,
    grid_window_v_tt=None,
    size_batch=10_000,
    number_worker=8,
    hankel=True,
    los_definition="bisector",
):
    """Compute density-velocity covariance terms for Lai22 wide-angle.

    Args:
        pmax: Maximum p index.
        ra_g, dec_g, comoving_distance_g: Galaxy coordinates and distances.
        ra_v, dec_v, comoving_distance_v: Velocity coordinates and distances.
        wavenumber_mt, wavenumber_tt: Wavenumbers for cross and velocity spectra.
        power_spectrum_mt, power_spectrum_tt: Corresponding spectra values.
        grid_window_m_mt, grid_window_v_mt, grid_window_v_tt: Optional windows.
        size_batch: Batch size.
        number_worker: Number of workers.
        hankel: Use Hankel acceleration when True.
        los_definition: Line-of-sight definition.

    Returns:
        Tuple ``(m_index, cov_gv_f2, cov_gv_bf)`` with lists of flattened covariance vectors.
    """
    if grid_window_m_mt is not None:
        power_spectrum_mt = power_spectrum_mt * grid_window_m_mt

    if grid_window_v_mt is not None:
        power_spectrum_mt = power_spectrum_mt * grid_window_v_mt

    if grid_window_v_tt is not None:
        power_spectrum_tt = power_spectrum_tt * grid_window_v_tt**2

    number_objects_v = len(ra_v)
    number_objects_g = len(ra_g)

    n_task = int(number_objects_g * number_objects_v)
    parameters = []
    for n in range(0, n_task, size_batch):
        batches = np.arange(n, np.min((n + size_batch, n_task)))
        i_list, j_list = cov_utils.compute_i_j_cross_matrix(number_objects_v, batches)
        r_i, ra_i, dec_i = comoving_distance_g[i_list], ra_g[i_list], dec_g[i_list]
        r_j, ra_j, dec_j = comoving_distance_v[j_list], ra_v[j_list], dec_v[j_list]
        r, theta, phi = cov_utils.angle_separation(
            ra_i, ra_j, dec_i, dec_j, r_i, r_j, los_definition=los_definition
        )
        parameters.append([r, theta, phi])

    cov_gv_f2 = []
    cov_gv_bf = []
    p_index = np.arange(pmax + 1)
    m_index = 2 * p_index
    function_covariance_dict = {}
    for p in p_index:
        function_covariance_dict[f"func_gv_f2_{p}"] = partial(
            coefficient_gv_f2_p,
            wavenumber_tt,
            power_spectrum_tt,
            p,
            hankel=hankel,
        )
        function_covariance_dict[f"func_gv_bf_{p}"] = partial(
            coefficient_gv_bf_p,
            wavenumber_mt,
            power_spectrum_mt,
            p,
            hankel=hankel,
        )
    if number_worker == 1:
        for p in p_index:
            function_covariance_dict[f"cov_gv_f2_{p}"] = np.concatenate(
                [
                    function_covariance_dict[f"func_gv_f2_{p}"](param)
                    for param in parameters
                ]
            )
            function_covariance_dict[f"cov_gv_bf_{p}"] = np.concatenate(
                [
                    function_covariance_dict[f"func_gv_bf_{p}"](param)
                    for param in parameters
                ]
            )
    else:
        with mp.Pool(number_worker) as pool:
            for p in p_index:
                function_covariance_dict[f"map_async_gv_f2_{p}"] = pool.map_async(
                    function_covariance_dict[f"func_gv_f2_{p}"], parameters
                )
                function_covariance_dict[f"map_async_gv_bf_{p}"] = pool.map_async(
                    function_covariance_dict[f"func_gv_bf_{p}"], parameters
                )

            for p in p_index:
                function_covariance_dict[f"cov_gv_f2_{p}"] = np.concatenate(
                    function_covariance_dict[f"map_async_gv_f2_{p}"].get()
                )
                function_covariance_dict[f"cov_gv_bf_{p}"] = np.concatenate(
                    function_covariance_dict[f"map_async_gv_bf_{p}"].get()
                )

    for p in p_index:
        # Multiplication by H_0
        function_covariance_dict[f"cov_gv_f2_{p}"] = (
            100 * function_covariance_dict[f"cov_gv_f2_{p}"]
        )
        function_covariance_dict[f"cov_gv_bf_{p}"] = (
            100 * function_covariance_dict[f"cov_gv_bf_{p}"]
        )

    cov_gv_f2 = [function_covariance_dict[f"cov_gv_f2_{p}"] for p in p_index]
    cov_gv_bf = [function_covariance_dict[f"cov_gv_bf_{p}"] for p in p_index]

    return m_index, cov_gv_f2, cov_gv_bf


def coefficient_gv_f2_p(
    wavenumber_tt,
    power_spectrum_tt,
    p,
    coord,
    hankel=True,
):
    """f² coefficient for gv block at fixed p.

    Args:
        wavenumber_tt: Wavenumbers for P_tt.
        power_spectrum_tt: P_tt values.
        p: p index.
        coord: Separation coordinates ``(r, theta, phi)``.
        hankel: Use Hankel acceleration when True.

    Returns:
        Scalar contribution for the given separation.
    """
    result = 0
    lmax = 2 * (p + 1)
    for ell in range(1, lmax + 1, 2):
        if hankel:
            correlation = compute_correlation_coefficient_hankel(
                p, -0.5, ell, coord[0], wavenumber_tt, power_spectrum_tt
            )
        else:
            correlation = compute_correlation_coefficient_simple_integration(
                p, -0.5, ell, coord[0], wavenumber_tt, power_spectrum_tt
            )
        h_function = eval(f"h_terms.H_gv_l{ell}_p{p+1}")(coord[1], coord[2])
        coeff = np.real(1j ** (ell + 1)) * ((-1) ** p) / (2**p * factorial(p))
        result = result + coeff * correlation * h_function
    return result


def coefficient_gv_bf_p(
    wavenumber_mt,
    power_spectrum_mt,
    p,
    coord,
    hankel=True,
):
    """bf cross coefficient for gv block at fixed p.

    Args mirror those of ``coefficient_gv_f2_p`` but using P_mt and different h-term.

    Returns:
        Scalar contribution for the given separation.
    """
    result = 0
    lmax = 2 * (p + 1)
    for ell in range(1, lmax + 1, 2):
        if hankel:
            correlation = compute_correlation_coefficient_hankel(
                p, -0.5, ell, coord[0], wavenumber_mt, power_spectrum_mt
            )
        else:
            correlation = compute_correlation_coefficient_simple_integration(
                p, -0.5, ell, coord[0], wavenumber_mt, power_spectrum_mt
            )
        h_function = eval(f"h_terms.H_gv_l{ell}_p{p}")(coord[1], coord[2])
        coeff = np.real(1j ** (ell + 1)) * ((-1) ** p) / (2**p * factorial(p))
        result = result + coeff * correlation * h_function
    return result


def return_matrix_covariance(cov):
    """Convert flattened covariance vector (variance + upper triangle) to matrix.

    Args:
        cov: 1D array with diagonal at index 0 and upper-triangle entries following.

    Returns:
        Square covariance matrix reconstructed from the flattened representation.
    """
    variance_val = cov[0]

    non_diagonal_cov = np.delete(cov, 0)
    number_objects = int((1 + np.sqrt(1 + 8 * non_diagonal_cov.size)) / 2)

    variance_val = variance_val * np.eye(number_objects)

    full_cov = np.zeros((number_objects, number_objects))
    vi, vj = np.triu_indices(number_objects, k=1)
    full_cov[vi, vj] = non_diagonal_cov
    full_cov[vj, vi] = non_diagonal_cov

    full_cov = full_cov + variance_val
    return full_cov


def return_correlation_matrix(cov):
    """Compute correlation matrix from covariance matrix.

    Args:
        cov: Square covariance matrix.

    Returns:
        Correlation matrix with unit diagonal.
    """
    sigma = np.sqrt(np.diag(cov))
    corr_matrix = cov / np.outer(sigma, sigma)
    return corr_matrix


def compute_all_matrices(
    ra_density,
    dec_density,
    rcom_density,
    ra_vel,
    dec_vel,
    rcom_vel,
    wavenumber_mm,
    wavenumber_mt,
    wavenumber_tt,
    power_spectrum_gg_mm,
    power_spectrum_gg_mt,
    power_spectrum_gg_tt,
    power_spectrum_gv_mt,
    power_spectrum_gv_tt,
    power_spectrum_vv_tt,
    grid_window_m_mm=None,
    grid_window_m_mt=None,
    grid_window_v_mt=None,
    grid_window_v_tt=None,
    pmax=3,
    qmax=3,
    size_batch=10_000,
    number_worker=1,
    hankel=True,
):
    """Compute all Lai22 covariance blocks and assemble matrices.

    Args:
        ra_density, dec_density, rcom_density: Galaxy coordinates and distances.
        ra_vel, dec_vel, rcom_vel: Velocity coordinates and distances.
        wavenumber_mm, wavenumber_mt, wavenumber_tt: Wavenumber grids.
        power_spectrum_gg_mm, power_spectrum_gg_mt, power_spectrum_gg_tt: Spectra for gg terms.
        power_spectrum_gv_mt, power_spectrum_gv_tt: Spectra for gv terms.
        power_spectrum_vv_tt: Spectrum for vv term.
        grid_window_*: Optional window functions.
        pmax, qmax: Max indices for expansions.
        size_batch: Batch size.
        number_worker: Parallel workers.
        hankel: Use Hankel acceleration when True.

    Returns:
        Tuple of matrices and indices: ``(cov_gg_b2_m, cov_gg_bf_m, cov_gg_f2_m, m_index_gg, cov_gv_bf_m, cov_gv_f2_m, m_index_gv, cov_vv)``.
    """
    m_index_gg, cov_gg_b2, cov_gg_f2, cov_gg_bf = compute_cov_gg(
        pmax,
        qmax,
        ra_density,
        dec_density,
        rcom_density,
        wavenumber_mm,
        wavenumber_mt,
        wavenumber_tt,
        power_spectrum_gg_mm,
        power_spectrum_gg_mt,
        power_spectrum_gg_tt,
        grid_window_m_mm=grid_window_m_mm,
        grid_window_m_mt=grid_window_m_mt,
        grid_window_v_mt=grid_window_v_mt,
        grid_window_v_tt=grid_window_v_tt,
        size_batch=size_batch,
        number_worker=number_worker,
        hankel=hankel,
    )
    cov_gg_b2_m = [
        return_matrix_covariance(cov_gg_b2[i]) for i, m in enumerate(m_index_gg)
    ]
    cov_gg_bf_m = [
        return_matrix_covariance(cov_gg_bf[i]) for i, m in enumerate(m_index_gg)
    ]
    cov_gg_f2_m = [
        return_matrix_covariance(cov_gg_f2[i]) for i, m in enumerate(m_index_gg)
    ]

    m_index_gv, cov_gv_f2, cov_gv_bf = compute_cov_gv(
        pmax,
        ra_density,
        dec_density,
        rcom_density,
        ra_vel,
        dec_vel,
        rcom_vel,
        wavenumber_mt,
        wavenumber_tt,
        power_spectrum_gv_mt,
        power_spectrum_gv_tt,
        grid_window_m_mt=grid_window_m_mt,
        grid_window_v_mt=grid_window_v_mt,
        grid_window_v_tt=grid_window_v_tt,
        size_batch=size_batch,
        number_worker=number_worker,
        hankel=hankel,
    )
    cov_gv_bf_m = [
        cov_gv_bf[i].reshape((ra_density.size, ra_vel.size))
        for i, _ in enumerate(m_index_gv)
    ]
    cov_gv_f2_m = [
        cov_gv_f2[i].reshape((ra_density.size, ra_vel.size))
        for i, _ in enumerate(m_index_gv)
    ]

    cov_vv = return_matrix_covariance(
        compute_cov_vv(
            ra_vel,
            dec_vel,
            rcom_vel,
            wavenumber_tt,
            power_spectrum_vv_tt,
            grid_window_v_tt=grid_window_v_tt,
            size_batch=size_batch,
            number_worker=number_worker,
            hankel=hankel,
        )
    )

    return (
        cov_gg_b2_m,
        cov_gg_bf_m,
        cov_gg_f2_m,
        m_index_gg,
        cov_gv_bf_m,
        cov_gv_f2_m,
        m_index_gv,
        cov_vv,
    )


def generate_covariance(
    model_kind,
    power_spectrum_dict,
    coordinates_velocity=None,
    coordinates_density=None,
    pmax=3,
    qmax=3,
    **kwargs,
):
    """Generate Lai22 covariance blocks for requested model kind.

    Wide-angle definition follows the bisector as in Lai et al. (2022).

    Args:
        model_kind: One of ``"density"``, ``"velocity"``, ``"full"``, ``"density_velocity"`` indicating which blocks to compute.
        power_spectrum_dict: Dict containing required spectra for gg/gv/vv blocks.
        coordinates_velocity: Tuple ``(ra, dec, rcom)`` for velocity tracers.
        coordinates_density: Tuple ``(ra, dec, rcom)`` for density tracers.
        pmax: Maximum p index.
        qmax: Maximum q index.
        **kwargs: Extra options forwarded to lower-level functions (e.g., windowing, batching, hankel).

    Returns:
        Tuple ``(covariance_dict, number_densities, number_velocities, los_definition)``.
    """

    los_definition = "bisector"

    cov_utils.check_generator_need(
        model_kind,
        coordinates_density,
        coordinates_velocity,
    )
    covariance_dict = {}

    if model_kind in ["density", "full", "density_velocity"]:
        covariance_dict["gg"] = compute_cov_gg(
            pmax,
            qmax,
            coordinates_density[0],
            coordinates_density[1],
            coordinates_density[2],
            power_spectrum_dict["gg"][0][0],
            power_spectrum_dict["gg"][1][0],
            power_spectrum_dict["gg"][2][0],
            power_spectrum_dict["gg"][0][1],
            power_spectrum_dict["gg"][1][1],
            power_spectrum_dict["gg"][2][1],
            los_definition=los_definition,
            **kwargs,
        )
        number_densities = len(coordinates_density[0])
    else:
        number_densities = None

    if model_kind in ["velocity", "full", "density_velocity"]:
        covariance_dict["vv"] = compute_cov_vv(
            coordinates_velocity[0],
            coordinates_velocity[1],
            coordinates_velocity[2],
            power_spectrum_dict["vv"][0][0],
            power_spectrum_dict["vv"][1][0],
            los_definition=los_definition,
            **kwargs,
        )
        number_velocities = len(coordinates_velocity[0])
    else:
        number_velocities = None

    if model_kind == "full":
        covariance_dict["gv"] = compute_cov_gv(
            pmax,
            coordinates_density[0],
            coordinates_density[1],
            coordinates_density[2],
            coordinates_velocity[0],
            coordinates_velocity[1],
            coordinates_velocity[2],
            power_spectrum_dict["gv"][0][0],
            power_spectrum_dict["gv"][1][0],
            power_spectrum_dict["gv"][0][1],
            power_spectrum_dict["gv"][1][1],
            los_definition=los_definition,
            **kwargs,
        )

    return (
        covariance_dict,
        number_densities,
        number_velocities,
        los_definition,
    )
