import numpy as np
from astropy.cosmology import FlatLambdaCDM
from astropy.cosmology import Planck18 as cosmo_background


def power_spectrum_amplitude_function(redshift, parameter_values_dict):
    return 1


def get_coefficients(
    model_type,
    parameter_values_dict,
    variant=None,
    redshift_dict=None,
):
    coefficients_dict = {}
    if variant == "growth_index":
        redshift_velocities = redshift_dict["v"]
        cosmo = FlatLambdaCDM(H0=100, Om0=parameter_values_dict["Om0"])
        coefficient_vector = (
            np.array(cosmo.Om(redshift_velocities)) ** parameter_values_dict["gamma"]
            * cosmo_background.H(redshift_velocities)
            / cosmo_background.H0
            * power_spectrum_amplitude_function(
                redshift_velocities, parameter_values_dict
            )
            / (1 + redshift_velocities)
        )

        coefficients_dict["vv"] = [np.outer(coefficient_vector, coefficient_vector)]
    elif variant == "growth_rate":
        redshift_velocities = redshift_dict["v"]

        coefficient_vector = (
            parameter_values_dict["fs8"]
            * cosmo_background.H(redshift_velocities)
            / cosmo_background.H0
            * power_spectrum_amplitude_function(
                redshift_velocities, parameter_values_dict
            )
            / (1 + redshift_velocities)
        )

        coefficients_dict["vv"] = [np.outer(coefficient_vector, coefficient_vector)]
    else:
        raise ValueError(
            "For the rcrk24 model, "
            "you need to chose variant between growth_index and growth_rate "
            "when you initialize the covariance matrix "
        )
    return coefficients_dict


def get_diagonal_coefficients(model_type, parameter_values_dict):
    coefficients_dict = {}
    coefficients_dict["vv"] = parameter_values_dict["sigv"] ** 2
    return coefficients_dict
