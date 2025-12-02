import numpy as np
from scipy.differentiate import derivative
from scipy.special import hyp2f1


def get_coefficients(
    model_kind,
    parameter_values_dict,
    variant=None,
    covariance_prefactor_dict=None,
):
    H0 = parameter_values_dict["H0"]
    Omega_m0 = parameter_values_dict["Omega_m0"]

    redshift_velocity = covariance_prefactor_dict["redshift_velocity"]

    D1_z = D1_function(redshift_velocity, Omega_m0)
    D1_0 = D1_function(0, Omega_m0)
    H_z = H(redshift_velocity, H0, Omega_m0)
    f_z = f(redshift_velocity, H0, Omega_m0)

    coefficient_v = (D1_z * H_z * f_z) / (D1_0 * (1 + redshift_velocity))
    coefficients_dict = {}
    coefficients_dict["vv"] = [np.outer(coefficient_v, coefficient_v)]
    return coefficients_dict


def Omega_m(z, Omega_m0):
    return Omega_m0 * (1 + z) ** 3 / (Omega_m0 * (1 + z) ** 3 + (1 - Omega_m0))


def H(z, H0, Omega_m0):
    return H0 * np.sqrt(Omega_m0 * (1 + z) ** 3 + (1 - Omega_m0))


def f(z, H0, Omega_m0):
    D1_derivative = derivative(D1_function, z, args=(Omega_m0,))
    return D1_derivative.df / (D1_function(z, Omega_m0) * H(z, H0, Omega_m0))


def D1_function(z, Omega_m0):
    prefactor = 1 / (5 * (1 + z) * Omega_m0)
    hyp2f1_value = hyp2f1(1 / 3, 1, 11 / 6, 1 - (1 / Omega_m(z, Omega_m0)))
    return prefactor * hyp2f1_value


def get_diagonal_coefficients(model_kind, parameter_values_dict):
    coefficients_dict = {}
    coefficients_dict["vv"] = parameter_values_dict["sigv"] ** 2
    return coefficients_dict
