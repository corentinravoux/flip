import numpy as np

from flip import utils
from flip.utils import create_log

log = create_log()


def load_density_vector(data):
    density, density_error = None, None
    if "density" in data.keys():
        density = data["density"]
        if "density_error" in data.keys():
            density_error = data["density_error"]
        else:
            log.add("""No density error in data, loading a null density error""")
            density_error = np.zeros_like(data["density"])
    else:
        raise ValueError("""The data does not contains a density field""")
    return density, density_error


def load_velocity_vector(
    data,
    parameter_values_dict,
    velocity_type="direct",
    velocity_estimator="full",
    **kwargs,
):
    velocity, velocity_error = None, None
    if velocity_type == "direct":
        if "velocity" in data.keys():
            velocity = data["velocity"]
            if "velocity_error" in data.keys():
                velocity_error = data["velocity_error"]
            else:
                log.add("""No velocity error in data, loading a null velocity error""")
                velocity_error = np.zeros_like(data["velocity"])
        else:
            raise ValueError(
                """The data does not contains a velocity field."""
                """Add it or choose a different velocity_type"""
            )

    elif velocity_type == "salt":
        if "mb" in data.keys():
            velocity, velocity_error = get_velocity_salt_model(
                data,
                parameter_values_dict,
                velocity_estimator,
                **kwargs,
            )
        else:
            raise ValueError(
                """The data does not contains parameters for salt velocity estimate."""
                """Add it or choose a different velocity_type"""
            )

    return velocity, velocity_error


def get_velocity_salt_model(
    data,
    parameters,
    velocity_estimator,
    q_0=None,
    j_0=None,
):
    alpha = parameters["alpha"]
    beta = parameters["beta"]
    M_0 = parameters["M_0"]
    sigma_M = parameters["sigma_M"]

    mu, variance_mu = compute_observed_distance_modulus(
        data,
        alpha,
        beta,
        M_0,
    )
    variance_mu += sigma_M**2

    muth = 5 * np.log10((1 + data["redshift_obs"]) * data["r_cosmo"]) + 25
    dmu = mu - muth

    redshift_dependence = redshift_dependence_velocity(
        data,
        velocity_estimator,
        q_0=q_0,
        j_0=j_0,
    )

    velocity = -redshift_dependence * dmu
    velocity_error = redshift_dependence * np.sqrt(variance_mu)

    return velocity, velocity_error


def compute_observed_distance_modulus(
    data,
    alpha,
    beta,
    M0,
):
    mu = data["mb"] + alpha * data["x1"] - beta * data["c"] - M0

    variance_mu = (
        data["e_mb"] ** 2
        + alpha**2 * data["e_x1"] ** 2
        + beta**2 * data["e_c"] ** 2
    )
    variance_mu += (
        2 * alpha * data["cov_mb_x1"]
        - 2 * beta * data["cov_mb_c"]
        - 2 * alpha * beta * data["cov_x1_c"]
    )
    return mu, variance_mu


def redshift_dependence_velocity(
    data,
    velocity_estimator,
    q_0=None,
    j_0=None,
):
    pfct = utils._C_LIGHT_KMS_ * np.log(10) / 5
    if velocity_estimator == "watkins":
        redshift_dependence = pfct * data["redshift"] / (1 + data["redshift"])
    elif velocity_estimator == "lowz":
        redshift_dependence = pfct / ((1 + data["redshift"]) / data["redshift"] - 1.0)
    elif velocity_estimator == "hubblehighorder":
        redshift_mod = data["redshift"] * (
            1
            + (1 / 2) * (1 - q_0) * data["redshift"]
            - (1 / 6) * (1 - q_0 - 3 * q_0**2 + j_0) * data["redshift"] ** 2
        )
        redshift_dependence = pfct * redshift_mod / (1 + data["redshift"])

    elif velocity_estimator == "full":
        redshift_dependence = pfct / (
            (1 + data["redshift"])
            * utils._C_LIGHT_KMS_
            / (data["hubble"] * data["r_cosmo"])
            - 1.0
        )

    return redshift_dependence
