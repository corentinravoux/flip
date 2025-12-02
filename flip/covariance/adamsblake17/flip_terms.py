import mpmath
import numpy
import scipy


def set_backend(module):
    global np, erf
    if module == "numpy":
        np = numpy
        erf = scipy.special.erf
    elif module == "mpmath":
        np = mpmath.mp
        erf = mpmath.erf


set_backend("numpy")


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
    return -3 * np.cos(phi + (1 / 2) * theta)


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


dictionary_terms = {"gg": ["0"], "gv": ["0"], "vv": ["0"]}
dictionary_lmax = {"gg": [2], "gv": [2], "vv": [2]}
dictionary_subterms = {
    "gg_0_0": 1,
    "gg_0_1": 0,
    "gg_0_2": 0,
    "gv_0_0": 0,
    "gv_0_1": 1,
    "gv_0_2": 0,
    "vv_0_0": 1,
    "vv_0_1": 0,
    "vv_0_2": 1,
}
multi_index_model = False
regularize_M_terms = None
