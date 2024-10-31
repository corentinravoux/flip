import numpy as np
import scipy


def M_gg_0_0_0():
    def func(k):
        return 1

    return func


def N_gg_0_0_0(theta, phi):
    return 1


def M_gv_0_1_0():
    def func(k):
        return (100 / 3) / k

    return func


def N_gv_0_1_0(theta, phi):
    return 3 * np.cos(phi)


def M_vv_0_0_0():
    def func(k):
        return (10000 / 3) / k**2

    return func


def N_vv_0_0_0(theta, phi):
    return 1


def M_vv_0_2_0():
    def func(k):
        return (4000 / 3) / k**2

    return func


def N_vv_0_2_0(theta, phi):
    return (15 / 4) * np.cos(2 * phi) + 5 / 4


dictionary_terms = {"gg": ["0"], "gv": ["0"], "vv": ["0"]}
dictionary_lmax = {"gg": [0], "gv": [1], "vv": [2]}
dictionary_subterms = {
    "gg_0_0": 1,
    "gv_0_0": 0,
    "gv_0_1": 1,
    "vv_0_0": 1,
    "vv_0_1": 0,
    "vv_0_2": 1,
}
multi_index_model = False
redshift_dependent_model = False
