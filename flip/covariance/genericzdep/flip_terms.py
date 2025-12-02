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


def M_vv_0_0_0(kNL):
    def func(k):
        return (10000 / 9) / k**2

    return func


def N_vv_0_0_0(theta, phi):
    return 3 * np.cos(theta)


def M_vv_0_2_0(kNL):
    def func(k):
        return (10000 / 9) / k**2

    return func


def N_vv_0_2_0(theta, phi):
    return (9 / 2) * np.cos(2 * phi) + (3 / 2) * np.cos(theta)


def M_vv_1_0_0(kNL):
    def func(k):
        return (10000 / 9) / kNL**2

    return func


def N_vv_1_0_0(theta, phi):
    return 3 * np.cos(theta)


def M_vv_1_2_0(kNL):
    def func(k):
        return (10000 / 9) / kNL**2

    return func


def N_vv_1_2_0(theta, phi):
    return (9 / 2) * np.cos(2 * phi) + (3 / 2) * np.cos(theta)


def M_vv_2_0_0(kNL):
    def func(k):
        return (10000 / 9) * k**2 / kNL**4

    return func


def N_vv_2_0_0(theta, phi):
    return 3 * np.cos(theta)


def M_vv_2_2_0(kNL):
    def func(k):
        return (10000 / 9) * k**2 / kNL**4

    return func


def N_vv_2_2_0(theta, phi):
    return (9 / 2) * np.cos(2 * phi) + (3 / 2) * np.cos(theta)


dictionary_terms = {"vv": ["0", "1", "2"]}
dictionary_lmax = {"vv": [2, 2, 2]}
dictionary_subterms = {
    "vv_0_0": 1,
    "vv_0_1": 0,
    "vv_0_2": 1,
    "vv_1_0": 1,
    "vv_1_1": 0,
    "vv_1_2": 1,
    "vv_2_0": 1,
    "vv_2_1": 0,
    "vv_2_2": 1,
}
multi_index_model = False
regularize_M_terms = None
