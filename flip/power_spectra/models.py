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


def get_bel_model(
    wavenumber,
    power_spectrum_linear,
    **kwargs,
):
    """
    The get_bel_model function takes in the linear power spectrum, wavenumber, and sigma_8 value.
    It then calculates the nonlinear matter-matter power spectrum using a fitting function from Bel et al. (2014).
    The function also returns the nonlinear matter-matter cross correlation coefficient.

    Args:
        wavenumber: Calculate the power spectrum
        power_spectrum_linear: Calculate the nonlinear power spectrum
        **kwargs: Pass a variable number of keyword arguments to a function
        : Calculate the nonlinear power spectrum

    Returns:
        The nonlinear matter power spectrum and the nonlinear galaxy clustering power spectrum
    """
    if "sigma_8" not in kwargs.keys():
        raise ValueError("Fiducial sigma_8 value is needed for nonlinearbel model")

    a1, a2, a3, invkdelta, b = bel_coefficients(kwargs["sigma_8"])

    power_spectrum_tt = power_spectrum_linear * np.exp(
        -wavenumber * (a1 + a2 * wavenumber + a3 * wavenumber**2)
    )

    power_spectrum_mt = power_spectrum_linear * np.exp(
        -invkdelta * wavenumber - b * wavenumber**6
    )

    return power_spectrum_mt, power_spectrum_tt


def get_nonlinearbel_model(
    wavenumber,
    power_spectrum_linear,
    **kwargs,
):
    """
    The get_nonlinearbel_model function returns the nonlinear power spectra for a given linear power spectrum.

    Args:
        wavenumber: Get the wavenumber of the power spectrum
        power_spectrum_linear: Calculate the power_spectrum_mt and power_spectrum_tt
        **kwargs: Pass a variable number of keyword arguments to a function
        : Get the nonlinear power spectrum

    Returns:
        The nonlinear power spectrum of matter, the cross power spectrum of matter and tracers and the auto-power spectrum of tracers

    """
    if "power_spectrum_non_linear" not in kwargs.keys():
        raise ValueError("Non linear power spectrum is needed for nonlinearbel model")

    power_spectrum_mt, power_spectrum_tt = get_bel_model(
        wavenumber, power_spectrum_linear, **kwargs
    )

    power_spectrum_mt *= np.sqrt(
        kwargs["power_spectrum_non_linear"] / power_spectrum_linear
    )
    power_spectrum_mm = kwargs["power_spectrum_non_linear"]

    return power_spectrum_mm, power_spectrum_mt, power_spectrum_tt


def get_linearbel_model(
    wavenumber,
    power_spectrum_linear,
    **kwargs,
):
    """
    The get_linearbel_model function returns the linear BEL model for a given power spectrum.

    Args:
        wavenumber: Calculate the wavenumber in h/mpc
        power_spectrum_linear: Calculate the power spectrum of the matter field
        **kwargs: Pass a variable number of keyword arguments to the function
        : Set the value of the power spectrum

    Returns:
        The linear power spectrum and the
    """
    power_spectrum_mt, power_spectrum_tt = get_bel_model(
        wavenumber, power_spectrum_linear, **kwargs
    )
    power_spectrum_mm = power_spectrum_linear

    return power_spectrum_mm, power_spectrum_mt, power_spectrum_tt


def get_linear_model(
    wavenumber,
    power_spectrum_linear,
    **kwargs,
):

    return power_spectrum_linear, power_spectrum_linear, power_spectrum_linear
