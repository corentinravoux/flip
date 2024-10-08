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


dictionary_terms = {"vv": ["0"]}
dictionary_lmax = {"vv": [2]}
dictionary_subterms = {"vv_0_0": 1, "vv_0_1": 0, "vv_0_2": 1}
multi_index_model = False
redshift_dependent_model = False
