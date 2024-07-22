import numpy as np
from astropy.cosmology import FlatLambdaCDM


def get_partial_derivative_coefficients(
    model_type,
    parameter_values_dict,
    variant=None,
    redshift_dict=None,
    power_spectrum_amplitude_function=None,
):

    cosmo = FlatLambdaCDM(H0=100, Om0=parameter_values_dict["Om0"])
    redshift_velocities = redshift_dict["v"]

    cosmoOm=np.array(cosmo.Om(redshift_velocities))

    Omega_m_partial_derivative_coefficients = (
        2
        * np.array(cosmo.Om(redshift_velocities)) ** (2 * parameter_values_dict["gamma"])
        * parameter_values_dict["gamma"]
        * power_spectrum_amplitude_function(redshift_velocities) ** 2
        / np.array(cosmo.Om(redshift_velocities))
    )

    gamma_partial_derivative_coefficients = (
        2
        * np.array(cosmo.Om(redshift_velocities)) ** (2 * parameter_values_dict["gamma"])
        * power_spectrum_amplitude_function(redshift_velocities) ** 2
        * np.log(cosmo.Om(redshift_velocities))
    )

    Omega_m_partial_derivative_coefficients = (
        cosmoOm ** (parameter_values_dict["gamma"]-1)
        * parameter_values_dict["gamma"]
        * power_spectrum_amplitude_function(redshift_velocities)
    )

    gamma_partial_derivative_coefficients = (
        cosmoOm ** (parameter_values_dict["gamma"])
        * power_spectrum_amplitude_function(redshift_velocities)
        * np.log(cosmoOm)
    )


    s8_partial_derivative_coefficients = (
        2
        * cosmoOm ** (2 * parameter_values_dict["gamma"])
        * power_spectrum_amplitude_function(redshift_velocities)
    )

    partial_coefficients_dict = {
        "Omegam": {
            "vv": [
                np.outer(
                    Omega_m_partial_derivative_coefficients,
                    cosmoOm ** (parameter_values_dict["gamma"]),
                ) +
                np.outer(
                    cosmoOm ** (parameter_values_dict["gamma"]),
                    Omega_m_partial_derivative_coefficients,
                ),
            ],
        },
        "gamma": {
            "vv": [
                np.outer(
                    gamma_partial_derivative_coefficients,
                    cosmoOm ** (parameter_values_dict["gamma"]),
                ) +
                np.outer(
                    cosmoOm ** (parameter_values_dict["gamma"]),
                    gamma_partial_derivative_coefficients,
                ),
            ],
        },
        "s8": {
            "vv": [
                np.outer(
                    s8_partial_derivative_coefficients,
                    s8_partial_derivative_coefficients,
                ),
            ],
        },
    }

    return partial_coefficients_dict
