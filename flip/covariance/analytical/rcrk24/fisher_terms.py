import numpy as np
from flip.covariance.analytical.rcrk24.coefficients import (
    aH,
    dfdgamma,
    dfdOm0,
    ds8dgamma_approx,
    ds8dgamma_exact,
    ds8dO0_approx,
    ds8dO0_exact,
    f,
    s8_approx,
    s8_exact,
)

exact = False
if exact:
    s8 = s8_exact
    ds8dgamma = ds8dgamma_exact
    ds8dO0 = ds8dO0_exact
else:
    s8 = s8_approx
    ds8dgamma = ds8dgamma_approx
    ds8dO0 = ds8dO0_approx


def get_partial_derivative_coefficients(
    model_kind,
    parameter_values_dict,
    variant=None,
    covariance_prefactor_dict=None,
):
    partial_coefficients_dict = None
    redshift_velocity = covariance_prefactor_dict["redshift_velocity"]
    a = 1 / (1 + redshift_velocity)

    if variant == "growth_index":
        # vv
        # for a parameterization Omega_gamma:
        #      P=(a H O**g s)(a H O**g s) (P_fid/s^2_fid)

        Om0 = parameter_values_dict["Om0"]
        gamma = parameter_values_dict["gamma"]

        # The Om0-gamma model f=Omega(Om0)^gamma
        aH_values = aH(a)
        f_values = f(a, Om0, gamma)  # cosmoOm ** parameter_values_dict["gamma"]
        s8_values = s8(redshift_velocity, Om0, gamma)
        aHfs8 = aH_values * f_values * s8_values

        dfdOm0_values = dfdOm0(a, Om0, gamma)
        dfdgamma_values = dfdgamma(a, Om0, gamma)

        ds8dO0_values = ds8dO0(
            redshift_velocity,
            Om0,
            gamma,
            s8_values=s8_values,
        )
        ds8dgamma_values = ds8dgamma(
            redshift_velocity,
            Om0,
            gamma,
            s8_values=s8_values,
        )

        Omega_m_partial_derivative_coefficients = (
            aH_values * dfdOm0_values * s8_values + aH_values * f_values * ds8dO0_values
        )

        gamma_partial_derivative_coefficients = (
            aH_values * dfdgamma_values * s8_values
            + aH_values * f_values * ds8dgamma_values
        )

        partial_coefficients_dict = {
            "Omegam": {
                "vv": [
                    np.outer(
                        Omega_m_partial_derivative_coefficients,
                        aHfs8,
                    )
                    + np.outer(
                        aHfs8,
                        Omega_m_partial_derivative_coefficients,
                    ),
                ],
            },
            "gamma": {
                "vv": [
                    np.outer(
                        gamma_partial_derivative_coefficients,
                        aHfs8,
                    )
                    + np.outer(
                        aHfs8,
                        gamma_partial_derivative_coefficients,
                    ),
                ],
            },
        }
    elif variant == "growth_rate":
        fs8 = parameter_values_dict["fs8"]
        aH_values = aH(a)

        fs8_partial_derivative_coefficients = aH_values

        aHfs8 = aH_values * fs8

        partial_coefficients_dict = {
            "fs8": {
                "vv": [
                    np.outer(
                        fs8_partial_derivative_coefficients,
                        aHfs8,
                    )
                    + np.outer(
                        aHfs8,
                        fs8_partial_derivative_coefficients,
                    ),
                ],
            },
        }

    return partial_coefficients_dict
