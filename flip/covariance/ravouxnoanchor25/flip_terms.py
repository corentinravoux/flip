import mpmath
import numpy
import scipy

from flip.covariance.ravouxnoanchor25.coefficients import D1_function


def set_backend(module):
    global np, erf
    if module == "numpy":
        np = numpy
        erf = scipy.special.erf
    elif module == "mpmath":
        np = mpmath.mp
        erf = mpmath.erf


set_backend("numpy")


def M_vv_0_0_0(*args):
    def func(k):
        return (10000 / 9) / k**2

    return func


def N_vv_0_0_0(theta, phi):
    return 3 * np.cos(theta)


def M_vv_0_2_0(*args):
    def func(k):
        return (10000 / 9) / k**2

    return func


def N_vv_0_2_0(theta, phi):
    return (9 / 2) * np.cos(2 * phi) + (3 / 2) * np.cos(theta)


def Z_vv_0(k, redshift_1, redshift_2, Omega_m0, knl):
    D1_z1 = D1_function(redshift_1, Omega_m0)
    D1_z2 = D1_function(redshift_2, Omega_m0)
    return np.exp(np.outer((D1_z1 - D1_z2) ** 2, -((k / knl) ** 2)))


dictionary_terms = {"vv": ["0"]}
dictionary_lmax = {"vv": [2]}
dictionary_subterms = {"vv_0_0": 1, "vv_0_1": 0, "vv_0_2": 1}
multi_index_model = False
redshift_dependent_model = True
regularize_M_terms = None
