import importlib
import multiprocessing as mp
from functools import partial

import cosmoprimo
import mpmath
import numpy as np
from scipy import integrate
from scipy.signal import savgol_filter
from scipy.special import spherical_jn

from flip.covariance import cov_utils
from flip.utils import create_log

log = create_log()
_avail_models = [
    "adamsblake17plane",
    "adamsblake17",
    "adamsblake20",
    "lai22",
    "carreres23",
    "ravouxcarreres",
    "ravouxnoanchor25",
    "rcrk24",
]
_avail_regularization_option = [
    None,
    "mpmath",
    "savgol",
    "lowk_asymptote",
]


def correlation_integration(ell, r, k, integrand):
    """
    The correlation_integration function is used to calculate the correlation function for a given multipole.
    It does this by integrating over k, which is the magnitude of the wavevector. The integration is performed using
    the Simpson's rule (integrate.simps). This function takes in four arguments: l, r, k and integrand. l represents
    the multipole that we are calculating for; it can be 0 (monopole), 2 (quadrupole) or 4 (hexadecapole). r represents
    the radial distance at which we want to evaluate our correlation function at; it can be any value between 0 and 200 M

    Args:
        l: Determine the order of the spherical bessel function
        r: Calculate the spherical bessel function
        k: Calculate the spherical bessel function
        integrand: Calculate the integrand of the correlation function

    Returns:
        ndarray: Correlation function $xi_\ell(r)$ via Simpson integration.

    """
    kr = np.outer(k, r)
    integrand = (
        (-1) ** (ell // 2)
        * (k**2 / (2 * np.pi**2))
        * integrand
        * spherical_jn(ell, kr).T
    )
    return (-1) ** (ell % 2) * integrate.simpson(integrand, x=k)


def correlation_hankel(ell, r, k, integrand, hankel_overhead_coefficient=2, kmin=None):
    """
    The correlation_hankel function is a wrapper for the cosmoprimo.fftlog.PowerToCorrelation function,
    which computes the correlation function from power spectrum using FFTLog (Hamilton 2000).
    The PowerToCorrelation class takes in an array of k values and an array of P(k) values, and returns
    an array of r values and an array of xi(r) values. The PowerToCorrelation class has two methods: set_fft_engine()
    and __call__(). The set_fft_engine() method sets which fft engine to use; it can be either &quot;n

    Args:
        l: Determine the parity of the integrand, and therefore whether to add a 1j term
        r: Set the radius of the sphere
        k: Set the kmin and kmax values in the powertocorrelation function
        integrand: Pass the integrand to the correlation_hankel function
        hankel_overhead_coefficient: Determine the minimum r value for which to use the hankel transform

    Returns:
        ndarray: Correlation function $xi_\ell(r)$ combining Hankel and direct integration.

    Note:
        If l is odd, count a 1j term in the integrand, without the need for adding it
    """
    Hankel = cosmoprimo.fftlog.PowerToCorrelation(k, ell=ell, q=0, complex=False)
    Hankel.set_fft_engine("numpy")
    r_hankel, xi_hankel = Hankel(integrand)
    mask = r < np.min(r_hankel) * hankel_overhead_coefficient
    if np.any(r > np.max(r_hankel)):
        raise ValueError(
            "Min pw spectrum k is too high, please take a lower one. Use kmin parameter to lower bound integration."
        )
    output = np.empty_like(r)
    output[mask] = correlation_integration(ell, r[mask], k, integrand)
    output[~mask] = (-1) ** (ell % 2) * np.interp(r[~mask], r_hankel, xi_hankel)

    # Regularization
    if kmin is not None:
        kreg = np.geomspace(np.min(k), kmin, int(len(k) / 20))
        output -= correlation_integration(ell, r, kreg, np.interp(kreg, k, integrand))
    return output


def coefficient_hankel(
    model_name,
    covariance_type,
    term_index,
    lmax,
    wavenumber,
    power_spectrum,
    coord,
    additional_parameters_values=None,
    **kwargs,
):
    """
    The coefficient_hankel function computes the covariance between two terms of a given model.
    The function takes as input:
        - The name of the model (e.g., adamsblake20, lai22, carreres23 or ravouxcarreres)
        - The type of term (i.e., either 'delta' or 'v')
        - The index i for the first term in question (i = 1 corresponds to delta_0 and v_0; i = 2 corresponds to delta_2 and v_2; etc.)
        - lmax: maximum multipole moment

    Args:
        model_name: Call the dictionary_subterms function from the flip_terms file
        type: Determine which subterms to use
        term_index: Select the term in the dictionary_subterms
        lmax: Determine the number of terms in the sum
        wavenumber: Calculate the hankel transform
        power_spectrum: Calculate the correlation function
        coord: Pass the coordinates of the point where we want to evaluate
        additional_parameters_values: Pass the values of additional parameters to the functions that
        : Define the model name

    Returns:
        ndarray: Covariance contribution for term `term_index`.

    """
    flip_terms = importlib.import_module(
        f"flip.covariance.analytical.{model_name}.flip_terms", package=__package__
    )

    cov_ab_i = 0
    flip_terms.set_backend("numpy")
    dictionary_subterms = flip_terms.dictionary_subterms
    regularize_M_terms = flip_terms.regularize_M_terms

    for ell in range(lmax + 1):
        number_terms = dictionary_subterms[f"{covariance_type}_{term_index}_{ell}"]
        for j in range(number_terms):
            M_ab_i_l_j = getattr(
                flip_terms, f"M_{covariance_type}_{term_index}_{ell}_{j}"
            )
            M_ab_i_l_j_evaluated = regularize_M(
                M_ab_i_l_j,
                wavenumber,
                regularize_M_terms,
                covariance_type,
                flip_terms,
                additional_parameters_values,
            )

            N_ab_i_l_j = getattr(
                flip_terms, f"N_{covariance_type}_{term_index}_{ell}_{j}"
            )(coord[1], coord[2])
            hankel_ab_i_l_j = correlation_hankel(
                ell,
                coord[0],
                wavenumber,
                M_ab_i_l_j_evaluated * power_spectrum,
                **kwargs,
            )
            cov_ab_i += N_ab_i_l_j * hankel_ab_i_l_j
    return cov_ab_i


def coefficient_trapz(
    model_name,
    covariance_type,
    term_index,
    lmax,
    wavenumber,
    power_spectrum,
    coord,
    additional_parameters_values=None,
    **kwargs,
):
    """
    The coefficient_trapz function computes the covariance matrix element for a given term.

    Args:
        model_name: Call the dictionary_subterms function from the flip_terms file
        type: Distinguish between the different terms in the covariance matrix
        term_index: Specify which term of the model is being used
        lmax: Specify the maximum order of spherical harmonics
        wavenumber: Calculate the integrand
        power_spectrum: Multiply the integrand
        coord: Pass in the coordinates of the point at which we want to evaluate
        additional_parameters_values: Pass the values of the additional parameters
        : Determine the model to be used

    Returns:
        ndarray: Covariance contribution for term `term_index`.

    """
    cov_ab_i = 0
    flip_terms = importlib.import_module(
        f"flip.covariance.analytical.{model_name}.flip_terms", package=__package__
    )
    flip_terms.set_backend("numpy")
    dictionary_subterms = flip_terms.dictionary_subterms
    regularize_M_terms = flip_terms.regularize_M_terms

    for ell in range(lmax + 1):
        number_terms = dictionary_subterms[f"{covariance_type}_{term_index}_{ell}"]
        for j in range(number_terms):
            M_ab_i_l_j = getattr(
                flip_terms, f"M_{covariance_type}_{term_index}_{ell}_{j}"
            )
            M_ab_i_l_j_evaluated = regularize_M(
                M_ab_i_l_j,
                wavenumber,
                regularize_M_terms,
                covariance_type,
                flip_terms,
                additional_parameters_values,
            )
            N_ab_i_l_j = getattr(
                flip_terms, f"N_{covariance_type}_{term_index}_{ell}_{j}"
            )(coord[1], coord[2])
            kr = np.outer(wavenumber, coord[0])
            integrand = (
                (-1) ** (ell // 2)
                * (wavenumber**2 / (2 * np.pi**2))
                * M_ab_i_l_j_evaluated
                * power_spectrum
                * spherical_jn(ell, kr).T
            )
            hankel_ab_i_l_j = (-1) ** (ell % 2) * np.trapezoid(integrand, x=wavenumber)
            cov_ab_i += N_ab_i_l_j * hankel_ab_i_l_j
    return cov_ab_i


def regularize_M(
    M_function,
    wavenumber,
    regularize_M_terms,
    covariance_type,
    flip_terms,
    additional_parameters_values,
    savgol_window=50,
    savgol_polynomial=3,
    lowk_unstable_threshold=0.1,
    lowk_unstable_mean_filtering=10,
    mpmmath_decimal_precision=50,
):
    """Evaluate and optionally regularize M(k) term.

    Applies one of the supported regularizations to stabilize numerical
    behavior (mpmath high-precision, Savitzky–Golay smoothing, or low-k
    asymptote detection).

    Args:
        M_function (callable): Function returning M(k) given additional parameters.
        wavenumber (ndarray): k samples.
        regularize_M_terms (dict|None): Per type regularization option.
        covariance_type (str): `"gg"`, `"gv"`, or `"vv"`.
        flip_terms (module): Terms module to switch backend when needed.
        additional_parameters_values (dict|tuple): Extra parameters for M.
        savgol_window (int): Window size for Savitzky–Golay.
        savgol_polynomial (int): Polynomial order for Savitzky–Golay.
        lowk_unstable_threshold (float): Threshold for low-k asymptote detection.
        lowk_unstable_mean_filtering (int): Window for mean filtering indices.
        mpmmath_decimal_precision (int): Decimal precision for mpmath.

    Returns:
        ndarray: Evaluated M(k) after regularization.
    """
    if regularize_M_terms is None:
        return M_function(*additional_parameters_values)(wavenumber)
    else:
        regularization_option = regularize_M_terms[covariance_type]

        if regularization_option is None:
            return M_function(*additional_parameters_values)(wavenumber)

        elif regularization_option == "mpmath":
            flip_terms.set_backend("mpmath")
            mpmath.mp.dps = mpmmath_decimal_precision
            wavenumber_mpmath = wavenumber * mpmath.mpf(1)
            additional_parameters_values_mpf = tuple(
                [mpmath.mpf(par) for par in additional_parameters_values]
            )
            M_function_evaluated = np.array(
                np.frompyfunc(M_function(*additional_parameters_values_mpf), 1, 1)(
                    wavenumber_mpmath
                ).tolist(),
                dtype=float,
            )
            flip_terms.set_backend("numpy")

        elif regularization_option == "savgol":
            M_function_evaluated = M_function(*additional_parameters_values)(wavenumber)
            M_function_evaluated = savgol_filter(
                M_function_evaluated,
                savgol_window,
                savgol_polynomial,
            )

        elif regularization_option == "lowk_asymptote":
            # The low k region presents numerical instabilities for density models.
            # All the M density function should present and asymptotic behaviour at low k.
            # This method detect low k asymptote and force it for all M functions.
            M_function_evaluated = M_function(*additional_parameters_values)(wavenumber)
            diff = np.diff(M_function_evaluated, append=[M_function_evaluated[-1]])
            mask_asymptote = np.abs(diff) < lowk_unstable_threshold * np.mean(
                np.abs(diff[wavenumber > wavenumber[len(wavenumber) // 2]])
            )
            mask_asymptote &= wavenumber < wavenumber[3 * len(wavenumber) // 4]

            if len(mask_asymptote[mask_asymptote]) > 0:
                index_mask_asymptote = np.argwhere(mask_asymptote)[:, 0]
                diff_index = np.diff(
                    index_mask_asymptote, prepend=[index_mask_asymptote[0]]
                )
                mean_diff_index = np.convolve(
                    diff_index,
                    np.ones(lowk_unstable_mean_filtering)
                    / lowk_unstable_mean_filtering,
                    mode="same",
                )
                mask_best_value_asymptote = np.abs(mean_diff_index - 1.0) < 10**-5
                if len(mask_best_value_asymptote[mask_best_value_asymptote]) == 0:
                    index_asymptote = index_mask_asymptote[-1]
                else:
                    index_asymptote = index_mask_asymptote[mask_best_value_asymptote][0]
                mask_unstable_region = wavenumber < wavenumber[index_asymptote]
                M_function_evaluated[mask_unstable_region] = M_function_evaluated[
                    index_asymptote
                ]
        else:
            raise ValueError(
                f"regularization option {regularization_option} is not available"
                f"Please choose in: {_avail_regularization_option}"
            )

    return M_function_evaluated


def compute_coordinates(
    covariance_type,
    coordinates_density=None,
    coordinates_velocity=None,
    size_batch=10_000,
    los_definition="bisector",
):
    """
    The compute_coordinates function computes the spherical coordinates of all pairs of objects in a given catalog.

    Args:
        covariance_type: Determine whether the covariance is a cross-covariance or an auto-covariance
        coordinates_density: Store the coordinates of the density field
        coordinates_velocity: Pass the coordinates of the velocity field
        size_batch: Control the size of the batches
        : Compute the angle separation between two objects

    Returns:
        list: Per-batch arrays `[r, theta, phi]`.
    """
    if covariance_type == "gg":
        ra = coordinates_density[0]
        dec = coordinates_density[1]
        comoving_distance = coordinates_density[2]
        number_objects = len(ra)
    elif covariance_type == "vv":
        ra = coordinates_velocity[0]
        dec = coordinates_velocity[1]
        comoving_distance = coordinates_velocity[2]
        number_objects = len(ra)
    elif covariance_type == "gv":
        ra_g = coordinates_density[0]
        dec_g = coordinates_density[1]
        comoving_distance_g = coordinates_density[2]
        ra_v = coordinates_velocity[0]
        dec_v = coordinates_velocity[1]
        comoving_distance_v = coordinates_velocity[2]
        number_objects_g = len(ra_g)
        number_objects_v = len(ra_v)

    if (covariance_type == "gg") | (covariance_type == "vv"):
        cross = False
        n_task = int((number_objects * (number_objects + 1)) / 2) - number_objects
    else:
        cross = True
        n_task = int(number_objects_g * number_objects_v)

    parameters = []
    for n in range(0, n_task, size_batch):
        batches = np.arange(n, np.min((n + size_batch, n_task)))
        if cross:
            i_list, j_list = cov_utils.compute_i_j_cross_matrix(
                number_objects_v, batches
            )
            ra_i, dec_i, r_i = (
                ra_g[i_list],
                dec_g[i_list],
                comoving_distance_g[i_list],
            )
            ra_j, dec_j, r_j = (
                ra_v[j_list],
                dec_v[j_list],
                comoving_distance_v[j_list],
            )
        else:
            i_list, j_list = cov_utils.compute_i_j(number_objects, batches)
            ra_i, dec_i, r_i = (
                ra[i_list],
                dec[i_list],
                comoving_distance[i_list],
            )
            ra_j, dec_j, r_j = (
                ra[j_list],
                dec[j_list],
                comoving_distance[j_list],
            )
        r, theta, phi = cov_utils.angle_separation(
            ra_i, ra_j, dec_i, dec_j, r_i, r_j, los_definition=los_definition
        )
        parameters.append([r, theta, phi])
    return parameters


def compute_coeficient(
    parameters,
    model_name,
    covariance_type,
    power_spectrum_list,
    additional_parameters_values=None,
    number_worker=8,
    hankel=True,
    kmin=None,
):
    """
    The compute_coeficient function computes the covariance matrix for a given model.

    Args:
        parameters: Compute the covariance matrix for each parameter
        model_name: Select the model to be used
        covariance_type: Select the type of covariance we want to compute
        power_spectrum_list: Compute the power spectrum of each term in the covariance matrix
        additional_parameters_values: Pass the values of additional parameters, such as the redshift
        number_worker: Specify the number of cores to use for multiprocessing
        hankel: Choose between the trapezoidal rule and the hankel transform
        : Define the number of threads used to compute the covariance matrix

    Returns:
        ndarray: Stacked covariance arrays per term index.
    """
    if additional_parameters_values is None:
        additional_parameters_values = {}
    if hankel:
        coefficient = coefficient_hankel
    else:
        coefficient = coefficient_trapz

    flip_terms = importlib.import_module(
        f"flip.covariance.analytical.{model_name}.flip_terms", package=__package__
    )
    term_index_list = getattr(flip_terms, "dictionary_terms")[covariance_type]
    lmax_list = getattr(flip_terms, "dictionary_lmax")[covariance_type]
    multi_index_model = getattr(flip_terms, "multi_index_model")

    function_covariance_dict = {}
    for i, index in enumerate(term_index_list):
        if multi_index_model:
            index_power_spectrum = int(index[0])
        else:
            index_power_spectrum = i

        function_covariance_dict[f"func_{index}"] = partial(
            coefficient,
            model_name,
            covariance_type,
            index,
            lmax_list[i],
            power_spectrum_list[index_power_spectrum][0],
            power_spectrum_list[index_power_spectrum][1],
            additional_parameters_values=additional_parameters_values,
            kmin=kmin,
        )
    if number_worker == 1:
        for i, index in enumerate(term_index_list):
            function_covariance_dict[f"cov_{index}"] = np.concatenate(
                [
                    function_covariance_dict[f"func_{index}"](param)
                    for param in parameters
                ]
            )
    else:
        with mp.Pool(number_worker) as pool:
            for i, index in enumerate(term_index_list):
                function_covariance_dict[f"map_async_{index}"] = pool.map_async(
                    function_covariance_dict[f"func_{index}"],
                    parameters,
                )
            for i, index in enumerate(term_index_list):
                function_covariance_dict[f"cov_{index}"] = np.concatenate(
                    function_covariance_dict[f"map_async_{index}"].get()
                )

    # In the case of autocorrelation, add the theoretical variance.
    if covariance_type[0] == covariance_type[1]:
        for i, index in enumerate(term_index_list):
            if multi_index_model:
                index_power_spectrum = int(index[0])
            else:
                index_power_spectrum = i
            variance_t = coefficient(
                model_name,
                covariance_type,
                index,
                lmax_list[i],
                power_spectrum_list[index_power_spectrum][0],
                power_spectrum_list[index_power_spectrum][1],
                np.zeros((5, 1)),
                additional_parameters_values=additional_parameters_values,
            )[0]

            function_covariance_dict[f"cov_{index}"] = np.insert(
                function_covariance_dict[f"cov_{index}"],
                0,
                variance_t,
            )

    return np.array(
        [
            function_covariance_dict[f"cov_{index}"]
            for _, index in enumerate(term_index_list)
        ]
    )


def compute_cov(
    model_name,
    covariance_type,
    power_spectrum_list,
    coordinates_density=None,
    coordinates_velocity=None,
    additional_parameters_values=None,
    size_batch=10_000,
    number_worker=8,
    hankel=True,
    los_definition="bisector",
    kmin=None,
):
    """
    The compute_cov function computes the covariance matrix for a given model.

    Args:
        model_name: Choose the model to be used
        covariance_type: Determine the type of covariance to be computed
        power_spectrum_list: Compute the power spectrum of the density field
        coordinates_density: Set the range of values for the density coordinates
        coordinates_velocity: Compute the velocity covariance
        additional_parameters_values: Pass the values of the additional parameters to be used in the model
        size_batch: Split the computation of the covariance matrix in smaller matrices
        number_worker: Specify the number of cores to be used
        hankel: Choose between the hankel transform or the fourier transform
        : Compute the covariance matrix for a given model

    Returns:
        ndarray: Covariance arrays per term index.

    """
    if model_name not in _avail_models:
        log.add(
            f"Model {model_name} not available."
            f"Please choose between: {_avail_models}"
        )

    parameters = compute_coordinates(
        covariance_type,
        coordinates_density=coordinates_density,
        coordinates_velocity=coordinates_velocity,
        size_batch=size_batch,
        los_definition=los_definition,
    )
    covariance = compute_coeficient(
        parameters,
        model_name,
        covariance_type,
        power_spectrum_list,
        additional_parameters_values=additional_parameters_values,
        number_worker=number_worker,
        hankel=hankel,
        kmin=kmin,
    )

    return covariance


def generate_covariance(
    model_name,
    model_kind,
    power_spectrum_dict,
    coordinates_velocity=None,
    coordinates_density=None,
    additional_parameters_values=None,
    size_batch=10_000,
    number_worker=8,
    hankel=True,
    los_definition="bisector",
    kmin=None,
):
    """
    The generate_flip function computes the covariance matrix for a given model.

    Args:
        model_name: Select the model to use
        model_kind: Determine the type of model to generate
        power_spectrum_dict: Store the power spectra of the different fields
        coordinates_velocity: Specify the coordinates of the velocity field
        coordinates_density: Specify the coordinates of the density field
        additional_parameters_values: Pass the values of the additional parameters to be used in the computation of covariance matrices
        size_batch: Split the computation of the covariance matrix into smaller batches
        number_worker: Specify the number of cores to use for computing the covariance matrix
        hankel: Decide whether to use the hankel transform or not
        : Define the number of workers to use for the computation

    Returns:
        tuple: `(covariance_dict, number_densities, number_velocities)`.

    """
    cov_utils.check_generator_need(
        model_kind,
        coordinates_density,
        coordinates_velocity,
    )
    covariance_dict = {}

    if model_kind in ["density", "full", "density_velocity"]:
        covariance_dict["gg"] = compute_cov(
            model_name,
            "gg",
            power_spectrum_dict["gg"],
            coordinates_density=coordinates_density,
            coordinates_velocity=coordinates_velocity,
            additional_parameters_values=additional_parameters_values,
            size_batch=size_batch,
            number_worker=number_worker,
            hankel=hankel,
            los_definition=los_definition,
            kmin=kmin,
        )
        number_densities = len(coordinates_density[0])
    else:
        number_densities = None

    if model_kind in ["velocity", "full", "density_velocity"]:
        covariance_dict["vv"] = compute_cov(
            model_name,
            "vv",
            power_spectrum_dict["vv"],
            coordinates_density=coordinates_density,
            coordinates_velocity=coordinates_velocity,
            additional_parameters_values=additional_parameters_values,
            size_batch=size_batch,
            number_worker=number_worker,
            hankel=hankel,
            los_definition=los_definition,
            kmin=kmin,
        )
        number_velocities = len(coordinates_velocity[0])
    else:
        number_velocities = None

    if model_kind == "full":
        covariance_dict["gv"] = compute_cov(
            model_name,
            "gv",
            power_spectrum_dict["gv"],
            coordinates_density=coordinates_density,
            coordinates_velocity=coordinates_velocity,
            additional_parameters_values=additional_parameters_values,
            size_batch=size_batch,
            number_worker=number_worker,
            hankel=hankel,
            los_definition=los_definition,
            kmin=kmin,
        )

    return covariance_dict, number_densities, number_velocities
