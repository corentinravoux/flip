import numpy as np
import astropy.cosmology
from astropy import units as u
from astropy import constants as const

class FisherTerms(object):

    def __init__(self, coordinates_velocity, cosmo=astropy.cosmology.Planck18):
        self._coordinates_velocity = coordinates_velocity
        self._cosmo = cosmo

        z = astropy.cosmology.z_at_value(self._cosmo.comoving_distance, self._coordinates_velocity[2]*u.Mpc)
        self._cov_factors = {"v": self._cosmo.H(z)/(1+z) / self._cosmo.H0, "g": None} # need to implement "g""

    @property
    def cov_factors(self):
        return self._cov_factors

    def get_partial_derivative_coefficients(self,
        model_type,
        parameter_values_dict,
        variant=None,
    ):
        if variant == "growth_index":
            partial_coefficients_dict = {
                "Omegam": {
                    "vv": [
                        2
                        * parameter_values_dict["Omegam"]
                        ** (2 * parameter_values_dict["gamma"])
                        * parameter_values_dict["gamma"]
                        * parameter_values_dict["s8"] ** 2
                        / parameter_values_dict["Omegam"],
                    ],
                },
                "gamma": {
                    "vv": [
                        2
                        * parameter_values_dict["Omegam"]
                        ** (2 * parameter_values_dict["gamma"])
                        * parameter_values_dict["s8"] ** 2
                        * np.log(parameter_values_dict["Omegam"]),
                    ],
                },
                "s8": {
                    "vv": [
                        2
                        * parameter_values_dict["Omegam"]
                        ** (2 * parameter_values_dict["gamma"])
                        * parameter_values_dict["s8"],
                    ],
                },
            }
        else:
            partial_coefficients_dict = {
                "fs8": {
                    "vv": [
                        2 * parameter_values_dict["fs8"],
                    ],
                },
            }
        return partial_coefficients_dict
