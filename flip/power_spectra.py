import os
import numpy as np
from flip.utils import create_log
log = create_log()

try:
    from classy import Class
except:
    log.add("Install CLASS https://github.com/lesgourg/class_public to use power_spectra module",
            level="warning")
    

def get_power_spectrum_suffix(
    z,
    kmin,
    kmax,
    number_points,
    logspace,
):
    return f"z{z}_kmin{kmin:.4f}_kmax{kmax:.4f}_N{number_points}{'_log' if logspace else '_lin'}"


def compute_power_spectra(
    class_settings,
    z,
    kmin,
    kmax,
    number_points,
    logspace,
    path,
):
    """
    The compute_power_spectra function computes the linear and non-linear power spectra from CLASS.

    Args:
        class_settings: Set the cosmological parameters for class
        z: Set the redshift of the power spectrum
        kmin: Set the minimum value of k
        kmax: Define the maximum value of k in the power spectrum
        number_points: Set the number of points in the k array
        path: Save the power spectra in a folder
        logspace: Determine whether the k values are logarithmically or linearly spaced
        : Set the path where the power spectra are saved

    Returns:
        The power spectra for the linear and non-linear case
    """
    # Computing power spectra from class (linear and non linear HF)
    if logspace:
        k_class = np.logspace(np.log10(kmin), np.log10(kmax), number_points)
    else:
        k_class = np.linspace(kmin, kmax, number_points)
    model = Class()
    model.set(class_settings)
    model.compute()
    pk_class_lin, pk_class_hf = [], []
    for k in k_class:
        pk_class_lin.append(model.pk_lin(k * model.h(), z) * model.h() ** 3)
        pk_class_hf.append(model.pk(k * model.h(), z) * model.h() ** 3)
    pk_class_lin = np.array(pk_class_lin)
    pk_class_hf = np.array(pk_class_hf)

    # Normalization power spectra by fiducial sigma8 value

    sigma_8 = model.sigma(8 / model.h(), z)
    pk_class_lin_normalized = pk_class_lin / sigma_8**2
    pk_class_hf_normalized = pk_class_hf / sigma_8**2

    fsigma_8 = model.scale_independent_f_sigma8(z)
    suffix = get_power_spectrum_suffix(z, kmin, kmax, number_points, logspace)
    header = f"fiducial fsigma_8 = {fsigma_8} & fiducial sigma_8 = {sigma_8}"

    # Computation of non linear power spectra
    a1, a2, a3, invkdelta, b = bel_coefficients(sigma_8)
    p_tt = pk_class_lin_normalized * np.exp(
        -k_class * (a1 + a2 * k_class + a3 * k_class**2)
    )
    p_mt_hf = np.sqrt(pk_class_lin_normalized * pk_class_hf_normalized) * np.exp(
        -invkdelta * k_class - b * k_class**6
    )
    p_mm_hf = pk_class_hf_normalized

    p_mt_nohf = pk_class_lin_normalized * np.exp(
        -invkdelta * k_class - b * k_class**6
    )
    p_mm_nohf = pk_class_lin_normalized

    # Saving

    save_power_spectrum(
        k_class, pk_class_lin_normalized, "linearclass", suffix, header, path
    )
    save_power_spectrum(k_class, p_tt, "linearbel_tt", suffix, header, path)
    save_power_spectrum(k_class, p_mt_nohf, "linearbel_mt", suffix, header, path)
    save_power_spectrum(k_class, p_mm_nohf, "linearbel_mm", suffix, header, path)
    save_power_spectrum(k_class, p_tt, "hfbel_tt", suffix, header, path)
    save_power_spectrum(k_class, p_mt_hf, "hfbel_mt", suffix, header, path)
    save_power_spectrum(k_class, p_mm_hf, "hfbel_mm", suffix, header, path)


def save_power_spectrum(
    wavenumber,
    power_spectrum,
    name_model,
    suffix,
    header,
    path,
):
    np.savetxt(
        os.path.join(
            path,
            f"power_spectrum_{name_model}_{suffix}.txt",
        ),
        [wavenumber, power_spectrum],
        header=header,
    )


def bel_coefficients(sigma_8):
    """
    The bel_coefficients function takes in a value for sigma_8 and returns the values of the coefficients
    a_i, i = 1, 2, 3 and invkdelta as well as b. These are used to calculate the non-linear power spectrum using
    the fitting formula from Bel et al. (2007). The function is called by Class when calculating P(k)

    Args:
        sigma_8: Calculate the coefficients of the bessel function

    Returns:
        The coefficients a_i, i = 1,

    """
    a1 = -0.817 + 3.198 * sigma_8
    a2 = 0.877 - 4.191 * sigma_8
    a3 = -1.199 + 4.629 * sigma_8
    invkdelta = -0.017 + 1.496 * sigma_8**2
    b = 0.091 + 0.702 * sigma_8**2
    return a1, a2, a3, invkdelta, b


def get_fiducial_fs8(class_settings, z):
    """
    The get_fiducial_fs8 function takes a CLASS settings dictionary and redshift as input,
    and returns the fiducial value of f*sigma8 at that redshift. The function uses the CLASS
    Python wrapper to compute this quantity.

    Args:
        class_settings: Set the parameters of the class model
        z: Set the redshift, and the class_settings parameter is used to set the cosmological parameters

    Returns:
        The fiducial fs8 value for the given cosmology and redshift

    """
    model = Class()
    model.set(class_settings)
    model.compute()
    return model.scale_independent_f_sigma8(z)


def get_fiducial_s8(class_settings, z):
    """
    The get_fiducial_s8 function takes a CLASS settings dictionary and a redshift,
    and returns the fiducial sigma_8 value for that cosmology at that redshift.


    Args:
        class_settings: Set the parameters of the class model
        z: Set the redshift of the fiducial model

    Returns:
        The value of sigma_8 at a given redshift

    """
    model = Class()
    model.set(class_settings)
    model.compute()
    return model.sigma(8 / model.h(), z)
