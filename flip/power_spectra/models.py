import numpy as np


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


def get_nonlinearbel_model(
    wavenumber,
    power_spectrum_linear,
    power_spectrum_non_linear,
    sigma_8=None,
):
    if sigma_8 is None:
        raise ValueError("Fiducial sigma_8 value is needed for nonlinearbel model")

    if power_spectrum_non_linear is None:
        raise ValueError("Non linear power spectrum is needed for nonlinearbel model")

    a1, a2, a3, invkdelta, b = bel_coefficients(sigma_8)
    power_spectrum_tt = power_spectrum_linear * np.exp(
        -wavenumber * (a1 + a2 * wavenumber + a3 * wavenumber**2)
    )
    power_spectrum_mt = np.sqrt(
        power_spectrum_linear * power_spectrum_non_linear
    ) * np.exp(-invkdelta * wavenumber - b * wavenumber**6)
    power_spectrum_mm = power_spectrum_non_linear

    return power_spectrum_mm, power_spectrum_mt, power_spectrum_tt


def get_linearbel_model(
    wavenumber,
    power_spectrum_linear,
    power_spectrum_non_linear=None,
    sigma_8=None,
):
    if sigma_8 is None:
        raise ValueError("Fiducial sigma_8 value is needed for linearbel model")

    a1, a2, a3, invkdelta, b = bel_coefficients(sigma_8)
    power_spectrum_tt = power_spectrum_linear * np.exp(
        -wavenumber * (a1 + a2 * wavenumber + a3 * wavenumber**2)
    )
    power_spectrum_mt = power_spectrum_linear * np.exp(
        -invkdelta * wavenumber - b * wavenumber**6
    )
    power_spectrum_mm = power_spectrum_linear

    return power_spectrum_mm, power_spectrum_mt, power_spectrum_tt
