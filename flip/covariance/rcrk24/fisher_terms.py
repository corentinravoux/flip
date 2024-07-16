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

    partial_coefficients_dict = {
        "Omegam": {
            "vv": [
                np.outer(
                    2
                    * cosmo.Om(redshift_velocities).value
                    ** (2 * parameter_values_dict["gamma"])
                    * parameter_values_dict["gamma"]
                    * power_spectrum_amplitude_function(redshift_velocities) ** 2
                    / cosmo.Om(redshift_velocities).value
                ),
            ],
        },
        "gamma": {
            "vv": [
                np.outer(
                    2
                    * cosmo.Om(redshift_velocities).value
                    ** (2 * parameter_values_dict["gamma"])
                    * power_spectrum_amplitude_function(redshift_velocities) ** 2
                    * np.log(cosmo.Om(redshift_velocities).value)
                ),
            ],
        },
        "s8": {
            "vv": [
                np.outer(
                    2
                    * cosmo.Om(redshift_velocities).value
                    ** (2 * parameter_values_dict["gamma"])
                    * power_spectrum_amplitude_function(redshift_velocities)
                ),
            ],
        },
    }

    return partial_coefficients_dict
