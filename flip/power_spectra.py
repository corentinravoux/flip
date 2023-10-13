import os
import numpy as np
from classy import Class


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


def compute_power_spectra(
    class_settings,
    z,
    kmin,
    kmax,
    number_points,
    path,
    logspace=False,
    halofit=False,
):
    """
    The compute_power_spectra function computes the linear and non-linear power spectra
        from CLASS. The linear power spectrum is computed using the pk_lin function of class,
        while the non-linear one is computed using a combination of pk_lin and pk (halofit).

    Args:
        class_settings: Set the cosmological parameters of the model
        z: Set the redshift of the power spectra
        kmin: Set the minimum value of k
        kmax: Set the maximum value of k
        number_points: Specify the number of points in the k-space that we want to compute
        path: Specify the path where the power spectra are saved
        logspace: Define the type of k-space sampling (log or linear)
        halofit: Choose whether to use the halofit power spectrum or not
        : Define the path where the power spectra will be saved

    Returns:
        The linear and nonlinear power spectra for a given redshift

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

    # Saving of linear power spectra
    np.savetxt(
        os.path.join(
            path,
            f"./power_spectrum_{kmin:.4f}_{kmax:.4f}_{number_points}_{'log' if logspace else 'lin'}_z{z}_class.txt",
        ),
        [k_class, pk_class_lin_normalized],
    )

    # Computation of non linear power spectra
    a1, a2, a3, invkdelta, b = bel_coefficients(sigma_8)
    p_tt = pk_class_lin_normalized * np.exp(
        -k_class * (a1 + a2 * k_class + a3 * k_class**2)
    )
    if halofit:
        p_mt = np.sqrt(pk_class_lin_normalized * pk_class_hf_normalized) * np.exp(
            -invkdelta * k_class - b * k_class**6
        )
        p_mm = pk_class_hf_normalized
    else:
        p_mt = pk_class_lin_normalized * np.exp(-invkdelta * k_class - b * k_class**6)
        p_mm = pk_class_lin_normalized

    # Saving of non linear power spectra
    np.savetxt(
        os.path.join(
            path,
            f"power_spectrum_tt_{kmin:.4f}_{kmax:.4f}_{number_points}_{'log' if logspace else 'lin'}{'' if halofit else '_nohf'}_z{z}_bel.txt",
        ),
        [k_class, p_tt],
    )
    np.savetxt(
        os.path.join(
            path,
            f"power_spectrum_mt_{kmin:.4f}_{kmax:.4f}_{number_points}_{'log' if logspace else 'lin'}{'' if halofit else '_nohf'}_z{z}_bel.txt",
        ),
        [k_class, p_mt],
    )
    np.savetxt(
        os.path.join(
            path,
            f"power_spectrum_mm_{kmin:.4f}_{kmax:.4f}_{number_points}_{'log' if logspace else 'lin'}{'' if halofit else '_nohf'}_z{z}_bel.txt",
        ),
        [k_class, p_mm],
    )


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
