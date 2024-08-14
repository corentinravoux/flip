import numpy as np
from astropy.cosmology import FlatLambdaCDM
from flip.covariance.rcrk24.flip_terms import cosmo_background

def get_coefficients(
    model_type,
    parameter_values_dict,
    variant=None,
    redshift_dict=None,
    power_spectrum_amplitude_function=None,
):
    coefficients_dict = {}
    if variant == "growth_index":
        redshift_velocities = redshift_dict["v"]

        coefficient_vector = (
            np.array(cosmo.Om(redshift_velocities)) ** parameter_values_dict["gamma"]
            * cosmo_background.H(redshift_velocities)
            / cosmo_background.H0
            * power_spectrum_amplitude_function(redshift_velocities, parameter_values_dict)
            / (1 + redshift_velocities)
        )

        coefficients_dict["vv"] = [np.outer(coefficient_vector, coefficient_vector)]
    elif variant == "growth_rate":
        redshift_velocities = redshift_dict["v"]

        coefficient_vector = (
            parameter_values_dict["f"]
            * cosmo_background.H(redshift_velocities)
            / cosmo_background.H0
            * power_spectrum_amplitude_function(redshift_velocities, parameter_values_dict)
            / (1 + redshift_velocities)
        )

        coefficients_dict["vv"] = [np.outer(coefficient_vector, coefficient_vector)]
    return coefficients_dict


def get_diagonal_coefficients(model_type, parameter_values_dict):
    coefficients_dict = {}
    coefficients_dict["vv"] = parameter_values_dict["sigv"] ** 2
    return coefficients_dict
