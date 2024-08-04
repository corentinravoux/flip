import numpy as np
from astropy.cosmology import FlatLambdaCDM
from flip.covariance.rcrk24.flip_terms import power_spectrum_amplitude_function, dpsafdO0, dpsafdgamma


# The flip convention is to split the power spectrum into several terms
# where linearity assumptions are made
# P_ab = A * B * P0_xy
#
# A is the coefficients where
# P_ab = A P_xy
# B is the cofficient where
# P_xy = B P0_xy
# and
# P0_xy is the power spectrum for a fiducial cosmology at z=0

# The derivative cofficients we need are
# dA/dp B + A dB/dp

# for vv
# A = (aHfs8)_1 * (aHfs8_1)
# B = psaf_1 * psaf_2


def get_partial_derivative_coefficients(
    model_type,
    parameter_values_dict,
    variant=None,
    redshift_dict=None,
    power_spectrum_amplitude_function=None,
):


    redshift_velocities = redshift_dict["v"]
    a = 1 / (1 + redshift_velocities)

    cosmo = FlatLambdaCDM(H0=100, Om0=parameter_values_dict["Om0"])

    # The Om0-gamma model f=Omega(Om0)^gamma

    # pre-calculate a few things that are used repeatedly
    cosmoOm = np.array(cosmo.Om(redshift_velocities))
    f = cosmoOm ** parameter_values_dict["gamma"]
    f0 = parameter_values_dict["Om0"] ** parameter_values_dict["gamma"]
    s80 = power_spectrum_amplitude_function(0, parameter_values_dict)

    # Calculation of s8 and its derivatives requires an integral.  It is useful to
    # expand Omega in terms of (1-a), which allows analytic solutions

    # # approximation
    def lnD(a):
        return np.log(a) * (
            f0
            + f0
            * 3
            * parameter_values_dict["gamma"]
            * (1 - parameter_values_dict["Om0"])
        ) + (1 - a) * f0 * 3 * parameter_values_dict["gamma"] * (
            1 - parameter_values_dict["Om0"]
        )

    def dlnDdOm0(a):
        return (
            parameter_values_dict["gamma"]
            * parameter_values_dict["Om0"] ** (parameter_values_dict["gamma"] - 1)
            * (
                3
                * (a - 1)
                * (
                    parameter_values_dict["gamma"] * (parameter_values_dict["Om0"] - 1)
                    + parameter_values_dict["Om0"]
                )
                + np.log(a)
                * (
                    -3
                    * parameter_values_dict["gamma"]
                    * (parameter_values_dict["Om0"] - 1)
                    - 3 * parameter_values_dict["Om0"]
                    + 1
                )
            )
        )

    def dlnDdgamma(a):
        return (
            3 * (1 - a) * (1 - parameter_values_dict["Om0"]) * f0
            + 3
            * (1 - a)
            * parameter_values_dict["gamma"]
            * (1 - parameter_values_dict["Om0"])
            * f0
            * np.log(parameter_values_dict["Om0"])
            + np.log(a)
            * (
                3 * (1 - parameter_values_dict["Om0"]) * f0
                + 3
                * parameter_values_dict["gamma"]
                * (1 - parameter_values_dict["Om0"])
                * f0
                * np.log(parameter_values_dict["Om0"])
                + f0 * np.log(parameter_values_dict["Om0"])
            )
        )

    def s8(a):
        return s80 * np.exp(lnD(a))

    def aHfs8(a):
        return (
            f
            * s8(a)
            / (1 + redshift_velocities)
            * cosmo.H(redshift_velocities)
            / cosmo.H0
        )

    # now for the partials

    def dOmdOm0(a):
        numerator = parameter_values_dict["Om0"] * a ** (-3)
        denominator = numerator + 1 - parameter_values_dict["Om0"]
        return a ** (-3) / denominator - numerator / denominator**2 * (a ** (-3) - 1)

    def dfdOm0(a):
        return parameter_values_dict["gamma"] * f / cosmoOm * dOmdOm0(a)

    def dfdgamma(a):
        return np.log(cosmoOm) * f

    def ds8dOm0(a):
        return s8(a) * dlnDdOm0(a)

    def ds8dgamma(a):
        return s8(a) * dlnDdgamma(a)

    def dAdOm0(a):
        return a * cosmo.H(a) / cosmo.H0 * (dfdOm0(a) * s8(a) + f * ds8dOm0(a))

    def dAdgamma(a):
        return a * cosmo.H(a) / cosmo.H0 * (dfdgamma(a) * s8(a) + f * ds8dgamma(a))

    power_spectrum_amplitude_values = power_spectrum_amplitude_function(redshift_velocities, parameter_values_dict)
    aHfs8_values = aHfs8(a)
    aHfs8power_spectrum_amplitude = aHfs8_values * power_spectrum_amplitude_values
    Omega_m_partial_derivative_coefficients = ( dAdOm0(a) * power_spectrum_amplitude_values + 
        aHfs8_values * dpsafdO0(redshift_velocities, parameter_values_dict, power_spectrum_amplitude_values=power_spectrum_amplitude_values)
        )

    gamma_partial_derivative_coefficients = ( dAdgamma(a) * power_spectrum_amplitude_values + 
        aHfs8_values * dpsafdgamma(redshift_velocities, parameter_values_dict, power_spectrum_amplitude_values=power_spectrum_amplitude_values)
        )

    # in the fs8 case
    def s8_fs8(a):
        return s80 + parameter_values_dict["fs8"] * np.log(a)

    def ds8dfs8(a):
        return np.log(a)

    fs8_partial_derivative_coefficients = (
        a
        * cosmo.H(redshift_velocities)
        / cosmo.H0
        * (s8_fs8(a) + parameter_values_dict["fs8"] * ds8dfs8(a))
    )

    aHfs8s8_fs8 = (
        a
        * cosmo.H(redshift_velocities)
        / cosmo.H0
        * parameter_values_dict["fs8"]
        * s8_fs8(a)
    )
    partial_coefficients_dict = {
        "Omegam": {
            "vv": [
                np.outer(
                    Omega_m_partial_derivative_coefficients,
                    aHfs8power_spectrum_amplitude,
                )
                + np.outer(
                    aHfs8power_spectrum_amplitude,
                    Omega_m_partial_derivative_coefficients,
                ),
            ],
        },
        "gamma": {
            "vv": [
                np.outer(
                    gamma_partial_derivative_coefficients,
                    aHfs8power_spectrum_amplitude,
                )
                + np.outer(
                    aHfs8power_spectrum_amplitude,
                    gamma_partial_derivative_coefficients,
                ),
            ],
        },
        "fs8": {
            "vv": [
                np.outer(
                    fs8_partial_derivative_coefficients,
                    aHfs8s8_fs8,
                )
                + np.outer(
                    aHfs8s8_fs8,
                    fs8_partial_derivative_coefficients,
                ),
            ],
        },
    }

    return partial_coefficients_dict
