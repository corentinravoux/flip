import numpy as np
import scipy


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
    f0=parameter_values_dict["Om0"] ** parameter_values_dict["gamma"]
    return np.log(a) * (
        f0
        + f0
        * 3
        * parameter_values_dict["gamma"]
        * (1 - parameter_values_dict["Om0"])
    ) + (1 - a) * f0 * 3 * parameter_values_dict["gamma"] * (
        1 - parameter_values_dict["Om0"]
    )


def s8(a, parameter_values_dict):
    s80 = 0.832
    return s80 * np.exp(lnD(a, parameter_values_dict))

def power_spectrum_amplitude_function(r, parameter_values_dict):
    return s8(1./(1+r), parameter_values_dict)

dictionary_terms = {"vv": ["0"]}
dictionary_lmax = {"vv": [2]}
dictionary_subterms = {"vv_0_0": 1, "vv_0_1": 0, "vv_0_2": 1}
multi_index_model = False
redshift_dependent_model = True
