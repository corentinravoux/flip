import numpy as np
from astropy.cosmology import FlatLambdaCDM

def get_partial_derivative_coefficients(
    model_type,
    parameter_values_dict,
    variant=None,
    redshift_dict=None,
    power_spectrum_amplitude_function=None,
):

    # The convention is to split the power spectrum into several terms
    # P_ab = A * B * P0_xy
    # To first order and under certain conventions
    # A is the coefficient that
    # P_ab = A P_xy
    # B is the cofficient that
    # P_xy = P0_xy

    # P0_xy is the power spectrum for a fiducial cosmology so has no parameter dependence
    # The derivative cofficients we need are
    # dA/dp B + A dB/dp

    # for vv
    # A = (aHfs8)_1 * (aHfs8_1)
    # B = s8_1 * s8_2

    redshift_velocities = redshift_dict["v"]
    cosmo = FlatLambdaCDM(H0=100, Om0=parameter_values_dict["Om0"])
    # calculate one a few things that are used repeatedly
    cosmoOm=np.array(cosmo.Om(redshift_velocities))
    f = cosmoOm**parameter_values_dict["gamma"]
    f0 = parameter_values_dict["Om0"]**parameter_values_dict["gamma"]

    s80 = 0.832  # from Planck for the fiducial model

   # D normalized so D(z=0)=1
    def s8(a):
        return s80* a**(f0 + f0*3*parameter_values_dict["gamma"]*(1-parameter_values_dict["Om0"])) \
            * np.exp((1-a)*f0*3* parameter_values_dict["gamma"]*(1-parameter_values_dict["Om0"]))

    def aHfs8(a):
        return f * s8(a) / (1+redshift_velocities) * cosmo.H(redshift_velocities)/cosmo.H0 

    # now for the partials

    def dOmdOm0(a):
        numerator = parameter_values_dict["Om0"] * a**(-3) 
        denominator = (numerator + 1 - parameter_values_dict["Om0"])
        return a**(-3)/denominator  - numerator/denominator**2*(a**(-3)-1)

    def dfdOm0(a):
        return parameter_values_dict["gamma"] * f / cosmoOm * dOmdOm0(a)

    def dfdgamma(a):
        return np.log(cosmoOm)  * f

    # for the following it is useful and fast to expand Omega in terms of (1-a)
    def ds8dOm0(a):
        domegapgamma =  parameter_values_dict["gamma"] *parameter_values_dict["Om0"]**(parameter_values_dict["gamma"]-1)
        domegapgammap1 =  (parameter_values_dict["gamma"]+1) *parameter_values_dict["Om0"]**parameter_values_dict["gamma"]
        return s80* np.log(a)*(domegapgamma + 3*parameter_values_dict["gamma"]*(domegapgamma - domegapgammap1)) \
            + (1-a)*3*parameter_values_dict["gamma"]*(domegapgamma-domegapgammap1)

    def ds8dgamma(a):
        omegapgamma = parameter_values_dict["Om0"]**parameter_values_dict["gamma"]
        lnOm0 = np.log(parameter_values_dict["Om0"])
        return s80*np.log(a) * (lnOm0 * omegapgamma * (1 + 3*parameter_values_dict["gamma"]*(1-parameter_values_dict["Om0"])) + omegapgamma*3) \
            + 3*(1-a)*(1-parameter_values_dict["Om0"])*(lnOm0 * omegapgamma *parameter_values_dict["gamma"] + omegapgamma)

    def dAdOm0(a):
        return a*cosmo.H(a)/cosmo.H0*(dfdOm0(a)*s8(a) + f*ds8dOm0(a))

    def dAdgamma(a):
        return a*cosmo.H(a)/cosmo.H0*(dfdgamma(a)*s8(a) + f*ds8dgamma(a))

    a = 1/(1+redshift_velocities)

    aHfs8s8 = aHfs8(a)*s8(a)

    Omega_m_partial_derivative_coefficients = (
        dAdOm0(a) * s8(a) + aHfs8(a) * ds8dOm0(a)
    )

    gamma_partial_derivative_coefficients = (
        dAdgamma(a) * s8(a) + aHfs8(a) * ds8dgamma(a)
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
                    aHfs8s8,
                ) +
                np.outer(
                    aHfs8s8,
                    Omega_m_partial_derivative_coefficients,
                ),
            ],
        },
        "gamma": {
            "vv": [
                np.outer(
                    gamma_partial_derivative_coefficients,
                    aHfs8s8,
                ) +
                np.outer(
                    aHfs8s8,
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
