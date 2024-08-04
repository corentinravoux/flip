import numpy as np
from astropy.cosmology import FlatLambdaCDM
from flip.covariance.rcrk24.flip_terms import * #power_spectrum_amplitude_function, dpsafdO0, dpsafdgamma, lnD, dlnDdOm0, dlnDdgamma, dOmdOm0


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

    f0 = parameter_values_dict["Om0"] ** parameter_values_dict["gamma"]
    s80 = power_spectrum_amplitude_function(0, parameter_values_dict)

    cosmoOm = np.array(cosmo.Om(redshift_velocities))
    f = cosmoOm ** parameter_values_dict["gamma"]
    s8 = s80 * np.exp(lnD(a, parameter_values_dict))
    H = cosmo.H(redshift_velocities)/cosmo.H0
    aHfs8 = a * H * f * s8 # aka A
    power_spectrum_amplitude_values = power_spectrum_amplitude_function(redshift_velocities, parameter_values_dict)
    aHfs8power_spectrum_amplitude = aHfs8  * power_spectrum_amplitude_values

    # now for the partials
    dfdOm0 = parameter_values_dict["gamma"] * f / cosmoOm * dOmdOm0(a, parameter_values_dict)
    dfdgamma = np.log(cosmoOm) * f
    ds8dOm0 = s8 * dlnDdOm0(a, parameter_values_dict)
    ds8dgamma = s8 * dlnDdgamma(a, parameter_values_dict)

    # A = aHfs8
    dAdOm0 = a * H * (dfdOm0 * s8 + f * ds8dOm0)
    dAdgamma = a * H * (dfdgamma * s8 + f * ds8dgamma)

    Omega_m_partial_derivative_coefficients = ( dAdOm0 * power_spectrum_amplitude_values + 
        aHfs8 * dpsafdO0(redshift_velocities, parameter_values_dict, power_spectrum_amplitude_values=power_spectrum_amplitude_values)
        )

    gamma_partial_derivative_coefficients = ( dAdgamma * power_spectrum_amplitude_values + 
        aHfs8 * dpsafdgamma(redshift_velocities, parameter_values_dict, power_spectrum_amplitude_values=power_spectrum_amplitude_values)
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
