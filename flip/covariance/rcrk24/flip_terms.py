import numpy as np
import scipy
from astropy.cosmology import FlatLambdaCDM
import scipy.integrate as integrate

def M_vv_0_0_0():
    def func(k):
        return (10000 / 9) / k**2

    return func


def N_vv_0_0_0(theta, phi):
    return 3 * np.cos(theta)


def M_vv_0_2_0():
    def func(k):
        return (10000 / 9) / k**2

    return func


def N_vv_0_2_0(theta, phi):
    return (9 / 2) * np.cos(2 * phi) + (3 / 2) * np.cos(theta)


def lnD(a, parameter_values_dict):
    f0 = parameter_values_dict["Om0"] ** parameter_values_dict["gamma"]
    return np.log(a) * (
        f0
        + f0
        * 3
        * parameter_values_dict["gamma"]
        * (1 - parameter_values_dict["Om0"])
    ) + (1 - a) * f0 * 3 * parameter_values_dict["gamma"] * (
        1 - parameter_values_dict["Om0"]
    )


# Normalization anchored to CMB
s8_cmb= 0.811
a_cmb = 1089.92
lna_cmb = np.log(a_cmb)

def psaf_objective(lna, parameter_values_dict):
    cosmo = FlatLambdaCDM(H0=100, Om0=parameter_values_dict["Om0"])
    z=1/np.exp(lna)-1
    return cosmo.Om(z) ** parameter_values_dict["gamma"]

def power_spectrum_amplitude_function(r, parameter_values_dict):
    r = np.asarray(r)
    scalar_input = False
    if r.ndim == 0:
        r = r[None]  # Makes x 1D
        scalar_input = True

    ret=[]
    a = 1/(1+r)
    for _a in a:
        ret.append(integrate.quad(psaf_objective, np.log(_a), lna_cmb, args=parameter_values_dict)[0])

    if scalar_input:
        return np.squeeze(s8_cmb* np.exp(ret))
    return s8_cmb * np.exp(ret)

def dOmdOm0(a, parameter_values_dict):
    numerator = parameter_values_dict["Om0"] * a ** (-3)
    denominator = numerator + 1 - parameter_values_dict["Om0"]
    return a ** (-3) / denominator - numerator / denominator**2 * (a ** (-3) - 1)

def psaf_O0_objective(lna, parameter_values_dict):
    cosmo = FlatLambdaCDM(H0=100, Om0=parameter_values_dict["Om0"])
    a = np.exp(lna)
    z=1/a-1
    Om = cosmo.Om(z)
    return parameter_values_dict["gamma"] * Om ** (parameter_values_dict["gamma"]-1) * dOmdOm0(a,parameter_values_dict)

# Partials are 
def dpsafdO0(r, parameter_values_dict, power_spectrum_amplitude_values=None):
    r = np.asarray(r)
    scalar_input = False
    if r.ndim == 0:
        r = r[None]  # Makes x 1D
        scalar_input = True

    ret=[]
    a = 1/(1+r)
    for _a in a:
        ret.append(integrate.quad(psaf_O0_objective, np.log(_a), lna_cmb, args=parameter_values_dict)[0])

    if power_spectrum_amplitude_values is None:
        power_spectrum_amplitude_values = power_spectrum_amplitude_function(r, parameter_values_dict)

    if scalar_input:
        return power_spectrum_amplitude_values * np.squeeze(ret)
    return power_spectrum_amplitude_values * ret

def psaf_gamma_objective(lna, parameter_values_dict):
    cosmo = FlatLambdaCDM(H0=100, Om0=parameter_values_dict["Om0"])
    a = np.exp(lna)
    z=1/a-1
    Om = cosmo.Om(z)
    return np.log(Om) * Om ** parameter_values_dict["gamma"]

# Partials are 
def dpsafdgamma(r, parameter_values_dict, power_spectrum_amplitude_values=None):
    r = np.asarray(r)
    scalar_input = False
    if r.ndim == 0:
        r = r[None]  # Makes x 1D
        scalar_input = True

    ret=[]
    a = 1/(1+r)
    for _a in a:
        ret.append(integrate.quad(psaf_gamma_objective, np.log(_a), lna_cmb, args=parameter_values_dict)[0])

    if power_spectrum_amplitude_values is None:
        power_spectrum_amplitude_values = power_spectrum_amplitude_function(r, parameter_values_dict)

    if scalar_input:
        return power_spectrum_amplitude_values * np.squeeze(ret)
    return power_spectrum_amplitude_values * ret


dictionary_terms = {"vv": ["0"]}
dictionary_lmax = {"vv": [2]}
dictionary_subterms = {"vv_0_0": 1, "vv_0_1": 0, "vv_0_2": 1}
multi_index_model = False
redshift_dependent_model = True
