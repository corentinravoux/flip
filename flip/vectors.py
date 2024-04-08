import numpy as np

from flip import utils
from flip.utils import create_log

log = create_log()

_avail_velocity_type = ["direct", "saltfit"]
_avail_velocity_estimator = ["watkins", "lowz", "hubblehighorder", "full"]


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
                """The data does not contains a "velocity" field."""
                f"""Add it or choose a different velocity_type among: {_avail_velocity_type}"""
            )

    elif velocity_type == "saltfit":
        key_to_verify = (
            "mb",
            "x1",
            "c",
            "e_mb",
            "e_x1",
            "e_c",
            "cov_mb_x1",
            "cov_mb_c",
            "cov_x1_c",
            "zobs",
            "rcom_zobs",
        )
        if all(k in data for k in key_to_verify):
            velocity, velocity_error = get_velocity_from_salt_fit(
                data,
                parameter_values_dict,
                velocity_estimator,
                **kwargs,
            )
        else:
            raise ValueError(
                f"""The data does not contains parameters for saltfit velocity estimate."""
                f"""Add all the following values: {key_to_verify}"""
                f"""Or choose a different velocity_type among {_avail_velocity_type}"""
            )
    else:
        raise ValueError(
            f"""Please choose a velocity_type among {_avail_velocity_type}"""
        )

    if "vmean" in parameter_values_dict:
        velocity = velocity - parameter_values_dict["vmean"]

    return velocity, velocity_error


def get_velocity_from_salt_fit(
    data,
    parameter_values_dict,
    velocity_estimator,
    q_0=None,
    j_0=None,
):
    alpha = parameter_values_dict["alpha"]
    beta = parameter_values_dict["beta"]
    M_0 = parameter_values_dict["M_0"]
    sigma_M = parameter_values_dict["sigma_M"]

    mu, variance_mu = compute_observed_distance_modulus(
        data,
        alpha,
        beta,
        M_0,
    )
    variance_mu += sigma_M**2

    muth = 5 * np.log10((1 + data["zobs"]) * data["rcom_zobs"]) + 25
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
    prefactor = utils._C_LIGHT_KMS_ * np.log(10) / 5
    redshift_obs = data["zobs"]

    if velocity_estimator == "watkins":
        redshift_dependence = prefactor * redshift_obs / (1 + redshift_obs)
    elif velocity_estimator == "lowz":
        redshift_dependence = prefactor / ((1 + redshift_obs) / redshift_obs - 1.0)
    elif velocity_estimator == "hubblehighorder":
        redshift_mod = redshift_obs * (
            1
            + (1 / 2) * (1 - q_0) * redshift_obs
            - (1 / 6) * (1 - q_0 - 3 * q_0**2 + j_0) * redshift_obs**2
        )
        redshift_dependence = prefactor * redshift_mod / (1 + redshift_obs)

    elif velocity_estimator == "full":
        # hubble_norm = H(z)/h = 100 E(z) with h = H0/100
        if "hubble_norm" not in data:
            raise ValueError(
                """ The "hubble_norm" field is not present in the data"""
                """ Please add it or choose a different velocity_estimator from salt fit among {_avail_velocity_estimator}"""
            )

        redshift_dependence = prefactor / (
            (1 + redshift_obs)
            * utils._C_LIGHT_KMS_
            / (data["hubble_norm"] * data["rcom_zobs"])
            - 1.0
        )

    else:
        raise ValueError(
            f"""Please choose a velocity_estimator from salt fit among {_avail_velocity_estimator}"""
        )

    return redshift_dependence
