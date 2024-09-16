import multiprocessing as mp
from functools import partial

import cosmoprimo
import numpy as np
from scipy import integrate
from scipy.ndimage import uniform_filter
from scipy.signal import savgol_filter
from scipy.special import spherical_jn

from flip.covariance import cov_utils
from flip.covariance.adamsblake17plane import flip_terms as flip_terms_adamsblake17plane
from flip.covariance.adamsblake20 import flip_terms as flip_terms_adamsblake20
from flip.covariance.carreres23 import flip_terms as flip_terms_carreres23
from flip.covariance.lai22 import flip_terms as flip_terms_lai22
from flip.covariance.ravouxcarreres import flip_terms as flip_terms_ravouxcarreres
from flip.covariance.rcrk24 import flip_terms as flip_terms_rcrk24
from flip.utils import create_log

log = create_log()
_avail_models = [
    "adamsblake17plane",
    "adamsblake20",
    "lai22",
    "carreres23",
    "ravouxcarreres",
    "rcrk24",
]


def correlation_integration(l, r, k, integrand):
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
        The integral of the integrand

    """
    kr = np.outer(k, r)
    integrand = (
        (-1) ** (l // 2) * (k**2 / (2 * np.pi**2)) * integrand * spherical_jn(l, kr).T
    )
    return (-1) ** (l % 2) * integrate.simpson(integrand, x=k)


def correlation_hankel(l, r, k, integrand, hankel_overhead_coefficient=2, kmin=None):
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
        The correlation function

    Note:
        If l is odd, count a 1j term in the integrand, without the need for adding it
    """
    Hankel = cosmoprimo.fftlog.PowerToCorrelation(k, ell=l, q=0, complex=False)
    Hankel.set_fft_engine("numpy")
    r_hankel, xi_hankel = Hankel(integrand)
    mask = r < np.min(r_hankel) * hankel_overhead_coefficient
    if np.any(r > np.max(r_hankel)):
        raise ValueError(
            "Min pw spectrum k is too high, please take a lower one. Use kmin parameter to lower bound integration."
        )
    output = np.empty_like(r)
    output[mask] = correlation_integration(l, r[mask], k, integrand)
    output[~mask] = (-1) ** (l % 2) * np.interp(r[~mask], r_hankel, xi_hankel)

    # Regularization
    if kmin is not None:
        kreg = np.geomspace(np.min(k), kmin, int(len(k) / 20))
        output -= correlation_integration(l, r, kreg, np.interp(kreg, k, integrand))
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
        covariance_type: Determine which subterms to use
        term_index: Select the term in the dictionary_subterms
        lmax: Determine the number of terms in the sum
        wavenumber: Calculate the hankel transform
        power_spectrum: Calculate the correlation function
        coord: Pass the coordinates of the point where we want to evaluate
        additional_parameters_values: Pass the values of additional parameters to the functions that
        : Define the model name

    Returns:
        The covariance of the a-th and b-th terms

    """
    cov_ab_i = 0
    dictionary_subterms = eval(f"flip_terms_{model_name}.dictionary_subterms")
    regularize_M_terms = eval(f"flip_terms_{model_name}.regularize_M_terms")
    for l in range(lmax + 1):
        number_terms = dictionary_subterms[f"{covariance_type}_{term_index}_{l}"]
        for j in range(number_terms):
            M_ab_i_l_j = eval(
                f"flip_terms_{model_name}.M_{covariance_type}_{term_index}_{l}_{j}"
            )(*additional_parameters_values)
            N_ab_i_l_j = eval(
                f"flip_terms_{model_name}.N_{covariance_type}_{term_index}_{l}_{j}"
            )(coord[1], coord[2])
            if regularize_M_terms is not None:
                M_ab_i_l_j_evaluated = regularize_M(
                    M_ab_i_l_j,
                    wavenumber,
                    regularize_M_terms[covariance_type],
                )
            else:
                M_ab_i_l_j_evaluated = M_ab_i_l_j(wavenumber)
            hankel_ab_i_l_j = correlation_hankel(
                l,
                coord[0],
                wavenumber,
                M_ab_i_l_j_evaluated * power_spectrum,
                **kwargs,
            )
            cov_ab_i = cov_ab_i + N_ab_i_l_j * hankel_ab_i_l_j
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
        covariance_type: Distinguish between the different terms in the covariance matrix
        term_index: Specify which term of the model is being used
        lmax: Specify the maximum order of spherical harmonics
        wavenumber: Calculate the integrand
        power_spectrum: Multiply the integrand
        coord: Pass in the coordinates of the point at which we want to evaluate
        additional_parameters_values: Pass the values of the additional parameters
        : Determine the model to be used

    Returns:
        A matrix

    """
    cov_ab_i = 0
    dictionary_subterms = eval(f"flip_terms_{model_name}.dictionary_subterms")
    regularize_M_terms = eval(f"flip_terms_{model_name}.regularize_M_terms")
    for l in range(lmax + 1):
        number_terms = dictionary_subterms[f"{covariance_type}_{term_index}_{l}"]
        for j in range(number_terms):
            M_ab_i_l_j = eval(
                f"flip_terms_{model_name}.M_{covariance_type}_{term_index}_{l}_{j}"
            )(*additional_parameters_values)
            N_ab_i_l_j = eval(
                f"flip_terms_{model_name}.N_{covariance_type}_{term_index}_{l}_{j}"
            )(coord[1], coord[2])
            if regularize_M_terms is not None:
                M_ab_i_l_j_evaluated = regularize_M(
                    M_ab_i_l_j,
                    wavenumber,
                    regularize_M_terms,
                    covariance_type,
                )
            else:
                M_ab_i_l_j_evaluated = M_ab_i_l_j(wavenumber)
            kr = np.outer(wavenumber, coord[0])
            integrand = (
                (-1) ** (l // 2)
                * (wavenumber**2 / (2 * np.pi**2))
                * M_ab_i_l_j_evaluated
                * power_spectrum
                * spherical_jn(l, kr).T
            )
            hankel_ab_i_l_j = (-1) ** (l % 2) * np.trapz(integrand, x=wavenumber)
            cov_ab_i = cov_ab_i + N_ab_i_l_j * hankel_ab_i_l_j
    return cov_ab_i


def regularize_M(
    M_function,
    wavenumber,
    regularization_option,
    savgol_window=50,
    savgol_polynomial=3,
    running_std_window=10,
    unstable_level_detection=10,
    unstable_overhead=2,
):
    M_function_evaluated = M_function(wavenumber)

    if regularization_option == "savgol":
        M_function_evaluated = savgol_filter(
            M_function_evaluated,
            savgol_window,
            savgol_polynomial,
        )

    elif regularization_option == "lowk_asymptote":
        M_function_evaluated = savgol_filter(
            M_function_evaluated,
            savgol_window,
            savgol_polynomial,
        )

        simple = uniform_filter(
            M_function_evaluated,
            running_std_window * 2,
            mode="constant",
            origin=-running_std_window,
        )
        square = uniform_filter(
            M_function_evaluated * M_function_evaluated,
            running_std_window * 2,
            mode="constant",
            origin=-running_std_window,
        )
        running_std = np.sqrt((square - simple * simple))
        mask_unstable = (
            running_std > unstable_level_detection * running_std[len(running_std) // 2]
        ) & (wavenumber < wavenumber[len(wavenumber) // 2])
        if len(mask_unstable[mask_unstable]) == 0:
            return M_function_evaluated

        k_unstable = wavenumber[mask_unstable][-1]
        mask = wavenumber < unstable_overhead * k_unstable
        M_function_evaluated[mask] = M_function_evaluated[~mask][0]

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
        A list of parameters
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
            ra_i, dec_i, r_i = ra_g[i_list], dec_g[i_list], comoving_distance_g[i_list]
            ra_j, dec_j, r_j = ra_v[j_list], dec_v[j_list], comoving_distance_v[j_list]
        else:
            i_list, j_list = cov_utils.compute_i_j(number_objects, batches)
            ra_i, dec_i, r_i = ra[i_list], dec[i_list], comoving_distance[i_list]
            ra_j, dec_j, r_j = ra[j_list], dec[j_list], comoving_distance[j_list]
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
        A list of arrays
    """
    if additional_parameters_values is None:
        additional_parameters_values = ()
    if hankel:
        coefficient = coefficient_hankel
    else:
        coefficient = coefficient_trapz

    term_index_list = eval(f"flip_terms_{model_name}.dictionary_terms")[covariance_type]
    lmax_list = eval(f"flip_terms_{model_name}.dictionary_lmax")[covariance_type]
    multi_index_model = eval(f"flip_terms_{model_name}.multi_index_model")

    for i, index in enumerate(term_index_list):
        if multi_index_model:
            index_power_spectrum = int(index[0])
        else:
            index_power_spectrum = i

        locals()[f"func_{index}"] = partial(
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
            loc = locals()
            locals()[f"cov_{index}"] = np.concatenate(
                [eval(f"func_{index}", loc)(param) for param in parameters]
            )
    else:
        with mp.Pool(number_worker) as pool:
            for i, index in enumerate(term_index_list):
                locals()[f"map_async_{index}"] = pool.map_async(
                    eval(f"func_{index}"), parameters
                )
            for i, index in enumerate(term_index_list):
                locals()[f"cov_{index}"] = np.concatenate(
                    eval(f"map_async_{index}").get()
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
                np.zeros((3, 1)),
                additional_parameters_values=additional_parameters_values,
            )[0]

            locals()[f"cov_{index}"] = np.insert(eval(f"cov_{index}"), 0, variance_t)

    loc = locals()
    return np.array(
        [eval(f"cov_{index}", loc) for _, index in enumerate(term_index_list)]
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
        The covariance matrix for a given model and set of parameters

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


def generate_redshift_dict(
    model_name,
    model_type,
    redshift_velocity=None,
    redshift_density=None,
    coordinates_velocity=None,
    coordinates_density=None,
):
    redshift_dependent_model = eval(f"flip_terms_{model_name}.redshift_dependent_model")
    if redshift_dependent_model:
        redshift_dict = {}
    else:
        return None

    if model_type in ["density", "full", "density_velocity"]:
        if redshift_dependent_model:
            if redshift_density is not None:
                redshift_dict["g"] = redshift_density
            else:
                if len(coordinates_density) < 4:
                    raise ValueError(
                        "You are using a model which is redshift dependent."
                        "Please provide redshifts as the fourth field"
                        "of the coordinates_density value"
                    )
                else:
                    redshift_dict["g"] = coordinates_density[3]
    if model_type in ["velocity", "full", "density_velocity"]:
        if redshift_dependent_model:
            if redshift_velocity is not None:
                redshift_dict["v"] = redshift_velocity
            else:
                if len(coordinates_velocity) < 4:
                    raise ValueError(
                        "You are using a model which is redshift dependent."
                        "Please provide redshifts as the fourth field"
                        "of the coordinates_velocity value"
                    )
                else:
                    redshift_dict["v"] = coordinates_velocity[3]
    return redshift_dict


def generate_covariance(
    model_name,
    model_type,
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
        model_type: Determine the type of model to generate
        power_spectrum_dict: Store the power spectra of the different fields
        coordinates_velocity: Specify the coordinates of the velocity field
        coordinates_density: Specify the coordinates of the density field
        additional_parameters_values: Pass the values of the additional parameters to be used in the computation of covariance matrices
        size_batch: Split the computation of the covariance matrix into smaller batches
        number_worker: Specify the number of cores to use for computing the covariance matrix
        hankel: Decide whether to use the hankel transform or not
        : Define the number of workers to use for the computation

    Returns:
        A dictionary with the covariance matrices and their dimensions

    """
    cov_utils.check_generator_need(
        model_type,
        coordinates_density,
        coordinates_velocity,
    )
    covariance_dict = {}

    if model_type in ["density", "full", "density_velocity"]:
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

    if model_type in ["velocity", "full", "density_velocity"]:
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

    if model_type == "full":
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

    redshift_dict = generate_redshift_dict(
        model_name,
        model_type,
        coordinates_velocity=coordinates_velocity,
        coordinates_density=coordinates_density,
    )
    return covariance_dict, number_densities, number_velocities, redshift_dict
