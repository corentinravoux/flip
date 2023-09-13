import numpy as np
from flip.covariance import cov_utils
import multiprocessing as mp
from functools import partial
from scipy.special import spherical_jn
from scipy import integrate
from scipy.interpolate import interp1d
import cosmoprimo

from flip.covariance.adamsblake20 import flip_terms as flip_terms_adamsblake20
from flip.covariance.lai22 import flip_terms as flip_terms_lai22
from flip.covariance.carreres23 import flip_terms as flip_terms_carreres23
from flip.covariance.ravouxcarreres import flip_terms as flip_terms_ravouxcarreres

from flip.utils import create_log

log = create_log()
_avail_models = ["adamsblake20", "lai22", "carreres23", "ravouxcarreres"]


def correlation_integration(l, r, k, integrand):
    kr = np.outer(k, r)
    integrand = (
        (-1) ** (l // 2)
        * (k**2 / (2 * np.pi**2))
        * integrand
        * spherical_jn(l, kr).T
    )
    return (-1) ** (l % 2) * integrate.simps(integrand, x=k)


def correlation_hankel(l, r, k, integrand, hankel_overhead_coefficient=2):
    """If l is odd, count a 1j term in the integrand, without the need for adding it"""
    Hankel = cosmoprimo.fftlog.PowerToCorrelation(k, ell=l, q=0, complex=False)
    Hankel.set_fft_engine("numpy")
    r_hankel, xi_hankel = Hankel(integrand)
    mask = r < np.min(r_hankel) * hankel_overhead_coefficient
    output = np.empty_like(r)
    output[mask] = correlation_integration(l, r[mask], k, integrand)
    output[~mask] = (-1) ** (l % 2) * interp1d(r_hankel, xi_hankel)(r[~mask])
    return output


def coefficient_hankel(
    model_name,
    type,
    term_index,
    lmax,
    wavenumber,
    power_spectrum,
    coord,
    additional_parameters_values=None,
):
    cov_ab_i = 0
    dictionary_subterms = eval(f"flip_terms_{model_name}.dictionary_subterms")
    for l in range(lmax + 1):
        number_terms = dictionary_subterms[f"{type}_{term_index}_{l}"]
        for j in range(number_terms):
            M_ab_i_l_j = eval(f"flip_terms_{model_name}.M_{type}_{term_index}_{l}_{j}")(
                *additional_parameters_values
            )
            N_ab_i_l_j = eval(f"flip_terms_{model_name}.N_{type}_{term_index}_{l}_{j}")(
                coord[1], coord[2]
            )
            hankel_ab_i_l_j = correlation_hankel(
                l, coord[0], wavenumber, M_ab_i_l_j(wavenumber) * power_spectrum
            )
            cov_ab_i = cov_ab_i + N_ab_i_l_j * hankel_ab_i_l_j
    return cov_ab_i


def coefficient_trapz(
    model_name,
    type,
    term_index,
    lmax,
    wavenumber,
    power_spectrum,
    coord,
    additional_parameters_values=None,
):
    cov_ab_i = 0
    dictionary_subterms = eval(f"flip_terms_{model_name}.dictionary_subterms")
    for l in range(lmax + 1):
        number_terms = dictionary_subterms[f"{type}_{term_index}_{l}"]
        for j in range(number_terms):
            M_ab_i_l_j = eval(f"flip_terms_{model_name}.M_{type}_{term_index}_{l}_{j}")(
                *additional_parameters_values
            )
            N_ab_i_l_j = eval(f"flip_terms_{model_name}.N_{type}_{term_index}_{l}_{j}")(
                coord[1], coord[2]
            )

            kr = np.outer(wavenumber, coord[0])
            integrand = (
                (-1) ** (l // 2)
                * (wavenumber**2 / (2 * np.pi**2))
                * M_ab_i_l_j(wavenumber)
                * power_spectrum
                * spherical_jn(l, kr).T
            )
            hankel_ab_i_l_j = (-1) ** (l % 2) * np.trapz(integrand, x=wavenumber)
            cov_ab_i = cov_ab_i + N_ab_i_l_j * hankel_ab_i_l_j
    return cov_ab_i


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
):
    """Compute the covariance matrix.

    Parameters
    ----------
    model_name : str
        Name of the model of covariance matrix to use.
    covariance_type : str
        The covariance type to compute 'gg', 'vv' or 'gv'.
    power_spectrum_list : list(list)
        [k, Pk] # BC : Not sure of the format
    coordinates_density : list(list, list, list) optional
        RA, Dec and RCOM [Mpc/h] for density data, by default None
    coordinates_velocity : list(list, list, list), optional
        RA, Dec and RCOM [Mpc/h] for velocity data, by default None
    additional_parameters_values : _type_, optional
        _description_, by default None
    size_batch : int, optional
        Number of coefficient computed by batch, by default 10_000
    number_worker : int, optional
        Number of processes to use, by default 8

    Returns
    -------
    list
        Matrix coefficient in ordered list.
    """    
    if model_name not in _avail_models:
        log.add(
            f"Model {model_name} not available."
            "Please choose between: {_avail_models}"
        )

    if additional_parameters_values is None:
        additional_parameters_values = ()
    if hankel:
        coefficient = coefficient_hankel
    else:
        coefficient = coefficient_trapz

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
        r, theta, phi = cov_utils.angle_separation(ra_i, ra_j, dec_i, dec_j, r_i, r_j)
        parameters.append([r, theta, phi])

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
    covariance = [eval(f"cov_{index}", loc) for i, index in enumerate(term_index_list)]
    return covariance
