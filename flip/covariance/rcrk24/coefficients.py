import numpy as np
import scipy.integrate as integrate
from astropy.cosmology import FlatLambdaCDM
from astropy.cosmology import Planck18 as cosmo_background

a_cmb = 1 / (1 + 1089.92)
lna_cmb = np.log(a_cmb)
s8_cmb = 0.832 * 0.001176774706956903  # ref. PDG O0=0.3 and gamma=0.5


# The Omega_M0 parameter is decoupled from the
# cosmological background expansion.
def aH(a):
    return a * cosmo_background.H(1 / a - 1) / cosmo_background.H0


# Omega_M(a)
def Om(a, Om0):
    numerator = Om0 * a ** (-3)
    denominator = numerator + 1 - Om0
    return numerator / denominator


def dOmdOm0(a, Om0):
    numerator = Om0 * a ** (-3)
    denominator = numerator + 1 - Om0
    return a ** (-3) / denominator - numerator / denominator**2 * (a ** (-3) - 1)


def f(a, Om0, gamma):
    return Om(a, Om0) ** gamma


def dfdOm0(a, Om0, gamma, f_values=None, dOmdOm0_values=None):
    if f_values is None:
        x = Om(a, Om0) ** (gamma - 1)
    else:
        x = f_values / Om(a, Om0)
    if dOmdOm0_values is None:
        y = dOmdOm0(a, Om0)
    else:
        y = dOmdOm0_values
    return gamma * x * y


def dfdgamma(a, Om0, gamma):
    Om_value = Om(a, Om0)
    return np.log(Om_value) * Om_value**gamma


def s8_objective(lna, Om0, gamma):
    cosmo = FlatLambdaCDM(H0=100, Om0=Om0)
    z = 1 / np.exp(lna) - 1
    return cosmo.Om(z) ** gamma


def s8_O0_objective(lna, Om0, gamma):
    cosmo = FlatLambdaCDM(H0=100, Om0=Om0)
    a = np.exp(lna)
    z = 1 / a - 1
    Om = cosmo.Om(z)
    return gamma * Om ** (gamma - 1) * dOmdOm0(a, Om0)


def s8_gamma_objective(lna, Om0, gamma):
    cosmo = FlatLambdaCDM(H0=100, Om0=Om0)
    a = np.exp(lna)
    z = 1 / a - 1
    Om = cosmo.Om(z)
    return np.log(Om) * Om**gamma


def s8_exact(r, Om0, gamma):
    r = np.asarray(r)
    scalar_input = False
    if r.ndim == 0:
        r = r[None]  # Makes x 1D
        scalar_input = True

    ret = []
    a = 1 / (1 + r)
    for _a in a:
        ret.append(
            integrate.quad(s8_objective, lna_cmb, np.log(_a), args=(Om0, gamma))[0]
        )

    if scalar_input:
        return np.squeeze(s8_cmb * np.exp(ret))

    return s8_cmb * np.exp(ret)


# Partials are
def ds8dO0_exact(r, Om0, gamma, s8_values=None):
    r = np.asarray(r)
    scalar_input = False
    if r.ndim == 0:
        r = r[None]  # Makes x 1D
        scalar_input = True

    ret = []
    a = 1 / (1 + r)
    for _a in a:
        ret.append(
            integrate.quad(s8_O0_objective, lna_cmb, np.log(_a), args=(Om0, gamma))[0]
        )
    ret = np.array(ret)

    if s8_values is None:
        s8_values = s8_exact(r, Om0, gamma)

    if scalar_input:
        return s8_values * np.squeeze(ret)
    return s8_values * ret


def ds8dgamma_exact(r, Om0, gamma, s8_values=None):
    r = np.asarray(r)
    scalar_input = False
    if r.ndim == 0:
        r = r[None]  # Makes x 1D
        scalar_input = True

    a = 1 / (1 + r)
    ret = []
    for _a in a:
        ret.append(
            integrate.quad(
                s8_gamma_objective,
                lna_cmb,
                np.log(_a),
                args=(Om0, gamma),
            )[0]
        )
    ret = np.array(ret)

    if s8_values is None:
        s8_values = s8_exact(r, Om0, gamma)

    if scalar_input:
        return s8_values * np.squeeze(ret)
    return s8_values * ret


# First order expansion of scale factor and its deriviatves in (1-a)
def lnD_approx(a, Om0, gamma):
    f0 = Om0**gamma
    return np.log(a) * (f0 + f0 * 3 * gamma * (1 - Om0)) + (1 - a) * f0 * 3 * gamma * (
        1 - Om0
    )


def dlnDdOm0_approx(a, Om0, gamma):
    return (
        gamma
        * Om0 ** (gamma - 1)
        * (
            3 * (a - 1) * (gamma * (Om0 - 1) + Om0)
            + np.log(a) * (-3 * gamma * (Om0 - 1) - 3 * Om0 + 1)
        )
    )


def dlnDdgamma_approx(a, Om0, gamma):
    f0 = Om0**gamma
    return (
        3 * (1 - a) * (1 - Om0) * f0
        + 3 * (1 - a) * gamma * (1 - Om0) * f0 * np.log(Om0)
        + np.log(a)
        * (
            3 * (1 - Om0) * f0
            + 3 * gamma * (1 - Om0) * f0 * np.log(Om0)
            + f0 * np.log(Om0)
        )
    )


def s8_approx(r, Om0, gamma):
    a = 1 / (1 + r)
    zero = integrate.quad(s8_objective, lna_cmb, 0, args=(Om0, gamma))[0]
    return s8_cmb * np.exp(zero + lnD_approx(a, Om0, gamma))


# Partials are
def ds8dO0_approx(r, Om0, gamma, s8_values=None):
    a = 1 / (1 + r)
    zero = integrate.quad(s8_O0_objective, lna_cmb, 0, args=(Om0, gamma))[0]
    if s8_values is None:
        s8_values = s8_approx(r, Om0, gamma)

    return s8_values * (zero + dlnDdOm0_approx(a, Om0, gamma))


def ds8dgamma_approx(r, Om0, gamma, s8_values=None):
    a = 1 / (1 + r)
    zero = integrate.quad(s8_gamma_objective, lna_cmb, 0, args=(Om0, gamma))[0]
    if s8_values is None:
        s8_values = s8_approx(r, Om0, gamma)

    return s8_values * (zero + dlnDdgamma_approx(a, Om0, gamma))


def get_coefficients(
    parameter_values_dict,
    model_kind,
    variant=None,
    covariance_prefactor_dict=None,
):
    redshift_velocity = covariance_prefactor_dict["redshift_velocity"]
    a = 1 / (1 + redshift_velocity)

    coefficients_dict = {}
    if variant == "growth_index":
        # Omega - gamma parameterization

        # vv
        #      P=(a H O**g s8)(a H O**g s8) (P_fid/s8^2_fid)

        Om0 = parameter_values_dict["Om0"]
        gamma = parameter_values_dict["gamma"]

        coefficient_vector = (
            aH(1 / (1 + redshift_velocity))
            * f(a, Om0, gamma)
            * s8_approx(redshift_velocity, Om0, gamma)
        )

        coefficients_dict["vv"] = [np.outer(coefficient_vector, coefficient_vector)]
    elif variant == "growth_rate":
        # fs8 parameterization

        # vv
        #      P = (aHfs8)(aHfs8) (P_fid/s8^2_fid)

        fs8 = parameter_values_dict["fs8"]
        coefficient_vector = aH(1 / (1 + redshift_velocity)) * fs8

        coefficients_dict["vv"] = [np.outer(coefficient_vector, coefficient_vector)]
    else:
        raise ValueError(
            "For the rcrk24 model, "
            "you need to chose variant between growth_index and growth_rate "
            "when you initialize the covariance matrix "
        )
    return coefficients_dict


def get_diagonal_coefficients(parameter_values_dict, model_kind):
    coefficients_dict = {}
    coefficients_dict["vv"] = parameter_values_dict["sigv"] ** 2
    return coefficients_dict
