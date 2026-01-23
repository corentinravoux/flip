import numpy as np


def bel_coefficients(sigma_8):
    """Compute BEL model coefficients as a function of $\sigma_8$.

    Uses the parameterization from Bel et al. to construct coefficients
    for the non-linear damping terms.

    Args:
        sigma_8 (float): Amplitude of matter fluctuations in 8 Mpc/h spheres.

    Returns:
        tuple: `(a1, a2, a3, invkdelta, b)` coefficients.
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
    """Apply BEL damping to linear spectrum to get TT and MT forms.

    Args:
        wavenumber (ndarray): $k$ values in h/Mpc.
        power_spectrum_linear (ndarray): Linear matter power spectrum $P_{mm}^{lin}(k)$.
        **kwargs: Must include `sigma_8`.

    Returns:
        tuple: `(P_mt(k), P_tt(k))` BEL-damped spectra.
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
    """Return BEL-damped spectra using an external non-linear $P_{mm}$.

    Args:
        wavenumber (ndarray): $k$ values in h/Mpc.
        power_spectrum_linear (ndarray): Linear $P_{mm}^{lin}(k)$.
        **kwargs: Must include `power_spectrum_non_linear` and `sigma_8`.

    Returns:
        tuple: `(P_mm(k), P_mt(k), P_tt(k))` with $P_mm$ taken from engine.
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
    """Return linear spectra with BEL damping applied to MT and TT.

    Args:
        wavenumber (ndarray): $k$ values in h/Mpc.
        power_spectrum_linear (ndarray): Linear $P_{mm}^{lin}(k)$.
        **kwargs: Must include `sigma_8`.

    Returns:
        tuple: `(P_mm, P_mt, P_tt)` with `P_mm = P_lin`.
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
    """Return purely linear spectra for MM, MT, and TT.

    Args:
        wavenumber (ndarray): $k$ values (unused).
        power_spectrum_linear (ndarray): Linear $P_{mm}^{lin}(k)$.

    Returns:
        tuple: `(P_mm, P_mt, P_tt)` all equal to `P_lin`.
    """

    return power_spectrum_linear, power_spectrum_linear, power_spectrum_linear
