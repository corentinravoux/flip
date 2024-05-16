import numpy as np

from flip import utils
from flip.utils import create_log

log = create_log()

_avail_velocity_type = ["direct", "scatter", "saltfit"]
_avail_velocity_estimator = ["watkins", "lowz", "hubblehighorder", "full"]


def load_density_error(data):
    if "density_error" in data.keys():
        density_error = data["density_error"]
    else:
        log.add("""No density error in data, loading a null density error""")
        density_error = np.zeros_like(data["density"])
    return density_error


def load_density_vectors(data):
    if "density" in data.keys():
        density = data["density"]
        density_error = load_density_error(data)
    else:
        raise ValueError("""The data does not contains a density field""")
    return density, density_error


def load_velocity_error(
    data,
    parameter_values_dict,
    velocity_type="direct",
    velocity_estimator="full",
):
    if velocity_type == "direct":
        velocity_error = get_velocity_error_directly(data)

    elif velocity_type == "scatter":

        if "zobs" in data.keys():
            velocity_error = get_velocity_error_from_scatter(
                data,
                parameter_values_dict,
                velocity_estimator,
            )
        else:
            raise ValueError(
                """The data does not contains a "zobs" field."""
                f"""Add it or choose a different velocity_type among: {_avail_velocity_type}"""
            )

    elif velocity_type == "saltfit":
        key_to_verify = (
            "e_mb",
            "e_x1",
            "e_c",
            "cov_mb_x1",
            "cov_mb_c",
            "cov_x1_c",
            "zobs",
        )
        if all(k in data for k in key_to_verify):
            velocity_error = get_velocity_error_from_salt_fit(
                data,
                parameter_values_dict,
                velocity_estimator,
            )
        else:
            raise ValueError(
                """The data does not contains parameters for saltfit velocity error estimate."""
                f"""Add all the following values: {key_to_verify}"""
                f"""Or choose a different velocity_type among {_avail_velocity_type}"""
            )
    else:
        raise ValueError(
            f"""Please choose a velocity_type among {_avail_velocity_type}"""
        )

    return velocity_error


def load_velocity_vectors(
    data,
    parameter_values_dict,
    velocity_type="direct",
    velocity_estimator="full",
):
    if velocity_type == "direct":
        if "velocity" in data.keys():
            velocity, velocity_error = get_velocity_directly(data)
        else:
            raise ValueError(
                """The data does not contains a "velocity" field."""
                f"""Add it or choose a different velocity_type among: {_avail_velocity_type}"""
            )
    elif velocity_type == "scatter":
        key_to_verify = (
            "velocity",
            "zobs",
        )
        if all(k in data for k in key_to_verify):
            velocity, velocity_error = get_velocity_from_scatter(
                data,
                parameter_values_dict,
                velocity_estimator,
            )
        else:
            raise ValueError(
                """The data does not contains parameters for scatter velocity estimate."""
                f"""Add all the following values: {key_to_verify}"""
                f"""Or choose a different velocity_type among {_avail_velocity_type}"""
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
            )
        else:
            raise ValueError(
                """The data does not contains parameters for saltfit velocity estimate."""
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


def get_velocity_error_directly(data):
    if "velocity_error" in data.keys():
        velocity_error = data["velocity_error"]
    else:
        log.add("""No velocity error in data, loading a null velocity error""")
        velocity_error = np.zeros_like(data["velocity"])
    return velocity_error


def get_velocity_directly(data):
    velocity = data["velocity"]
    velocity_error = get_velocity_error_directly(data)
    return velocity, velocity_error


def get_velocity_error_from_scatter(
    data,
    parameter_values_dict,
    velocity_estimator,
):
    sigma_M = parameter_values_dict["sigma_M"]

    redshift_dependence = redshift_dependence_velocity(
        data,
        parameter_values_dict,
        velocity_estimator,
    )

    velocity_error = redshift_dependence * sigma_M
    return velocity_error


def get_velocity_from_scatter(
    data,
    parameter_values_dict,
    velocity_estimator,
):
    velocity = data["velocity"]

    velocity_error = get_velocity_error_from_scatter(
        data,
        parameter_values_dict,
        velocity_estimator,
    )

    return velocity, velocity_error


def get_velocity_error_from_salt_fit(
    data,
    parameter_values_dict,
    velocity_estimator,
):

    variance_mu = compute_observed_distance_modulus_error(
        data,
        parameter_values_dict["alpha"],
        parameter_values_dict["beta"],
    )
    variance_mu += parameter_values_dict["sigma_M"] ** 2

    redshift_dependence = redshift_dependence_velocity(
        data,
        parameter_values_dict,
        velocity_estimator,
    )

    velocity_error = redshift_dependence * np.sqrt(variance_mu)

    return velocity_error, redshift_dependence


def get_velocity_from_salt_fit(
    data,
    parameter_values_dict,
    velocity_estimator,
):

    velocity_error, redshift_dependence = get_velocity_error_from_salt_fit(
        data,
        parameter_values_dict,
        velocity_estimator,
    )

    mu = compute_observed_distance_modulus(
        data,
        parameter_values_dict["alpha"],
        parameter_values_dict["beta"],
        parameter_values_dict["M_0"],
    )
    muth = 5 * np.log10((1 + data["zobs"]) * data["rcom_zobs"]) + 25
    dmu = mu - muth

    velocity = -redshift_dependence * dmu

    return velocity, velocity_error


def compute_observed_distance_modulus(
    data,
    alpha,
    beta,
    M0,
):
    mu = data["mb"] + alpha * data["x1"] - beta * data["c"] - M0

    return mu


def compute_observed_distance_modulus_error(
    data,
    alpha,
    beta,
):
    variance_mu = (
        data["e_mb"] ** 2 + alpha**2 * data["e_x1"] ** 2 + beta**2 * data["e_c"] ** 2
    )
    variance_mu += (
        2 * alpha * data["cov_mb_x1"]
        - 2 * beta * data["cov_mb_c"]
        - 2 * alpha * beta * data["cov_x1_c"]
    )
    return variance_mu


def redshift_dependence_velocity(
    data,
    parameter_values_dict,
    velocity_estimator,
):
    prefactor = utils._C_LIGHT_KMS_ * np.log(10) / 5
    redshift_obs = data["zobs"]

    if velocity_estimator == "watkins":
        redshift_dependence = prefactor * redshift_obs / (1 + redshift_obs)
    elif velocity_estimator == "lowz":
        redshift_dependence = prefactor / ((1 + redshift_obs) / redshift_obs - 1.0)
    elif velocity_estimator == "hubblehighorder":
        if ("q_0" not in parameter_values_dict) & ("j_0" not in parameter_values_dict):
            raise ValueError(
                """ The "q_0" and "j_0" parameters are not present in the parameter_values_dict"""
                f""" Please add it or choose a different velocity_estimator among {_avail_velocity_estimator}"""
            )
        q_0 = parameter_values_dict["q_0"]
        j_0 = parameter_values_dict["j_0"]
        redshift_mod = redshift_obs * (
            1
            + (1 / 2) * (1 - q_0) * redshift_obs
            - (1 / 6) * (1 - q_0 - 3 * q_0**2 + j_0) * redshift_obs**2
        )
        redshift_dependence = prefactor * redshift_mod / (1 + redshift_obs)

    elif velocity_estimator == "full":
        if ("hubble_norm" not in data) | ("rcom_zobs" not in data):
            raise ValueError(
                """ The "hubble_norm" (H(z)/h = 100 E(z)) or "rcom_zobs" (Dm(z)) fields are not present in the data"""
                f""" Please add it or choose a different velocity_estimator among {_avail_velocity_estimator}"""
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
