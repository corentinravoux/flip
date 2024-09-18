import numpy as np
from astropy.cosmology import FlatLambdaCDM
from flip.covariance.rcrk24.flip_terms import s8
from flip.covariance.rcrk24.fisher_terms import cosmo_background

# vv model is (aHf)(aHf) P = (aHf)(aHf) (s^2 * P_fid/s^2_fid)

def get_coefficients(
    model_type,
    parameter_values_dict,
    variant=None,
    redshift_dict=None,
    fiducial_dict=None,
):
    if fiducial_dict is None:
        raise Exception("rcrk24 model requires fiducial_dict "
                        "to be sure the user understands the "
                        "the fiducial power spectrum")

    coefficients_dict = {}
    if variant == "growth_index":
        # Omega - gamma parameterization 

        # vv
        #      P=(a H O**g s8)(a H O**g s8) (P_fid/s8^2_fid)

        redshift_velocities = redshift_dict["v"]
        cosmo = FlatLambdaCDM(H0=100, Om0=parameter_values_dict["Om0"])
        coefficient_vector = (
            np.array(cosmo.Om(redshift_velocities)) ** parameter_values_dict["gamma"]
            * cosmo_background.H(redshift_velocities)
            / cosmo_background.H0
            * s8(redshift_velocities, parameter_values_dict)
            / (1 + redshift_velocities)
        )

        coefficients_dict["vv"] = [np.outer(coefficient_vector, coefficient_vector)]
    elif variant == "growth_rate":
        # fs8 parameterization 

        # vv
        #      P = (aHfs8)(aHfs8) (P_fid/s8^2_fid)

        redshift_velocities = redshift_dict["v"]

        coefficient_vector = (
            parameter_values_dict["fs8"]
            * cosmo_background.H(redshift_velocities)
            / cosmo_background.H0
            / (1 + redshift_velocities)
        )

        coefficients_dict["vv"] = [np.outer(coefficient_vector, coefficient_vector)]
    else:
        raise ValueError("For the rcrk24 model, "
                         "you need to chose variant between growth_index and growth_rate "
                         "when you initialize the covariance matrix ")
    return coefficients_dict


def get_diagonal_coefficients(model_type, parameter_values_dict):
    coefficients_dict = {}
    coefficients_dict["vv"] = parameter_values_dict["sigv"] ** 2
    return coefficients_dict

