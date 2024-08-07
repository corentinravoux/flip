import numpy as np
from astropy.cosmology import FlatLambdaCDM
# from flip.covariance.rcrk24.flip_terms import * #power_spectrum_amplitude_function, dpsafdO0, dpsafdgamma, lnD, dlnDdOm0, dlnDdgamma, dOmdOm0
import matplotlib.pyplot as plt

# The flip convention is to split the power spectrum into several terms
# where linearity assumptions are made
# P_ab = AA * BB * P0_xy
#
# A is the coefficients where
# P_ab = AA P_xy
# B is the cofficient where
# P_xy = BB P0_xy
# and
# P0_xy is the power spectrum for a fiducial cosmology at z=0

# The derivative cofficients we need are
# dA/dp B + A dB/dp

# for vv
# A = (aHfs8)_1
# B = psaf_1 = s8_1
#
# B is "power_spectrum_amplitude_values" and it and its derivatives are calculated in flip_terms.py
# as it is needed by coefficients.py
# Note however that the derivatives of s8_1 in A and B are different!  A is normalized at z=0. B
# is normalized a z=z_cmb

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
    H = cosmo.H(redshift_velocities)/cosmo.H0
    s80 = 0.832

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
