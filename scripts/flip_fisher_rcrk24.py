import os

import numpy as np
import pandas as pd
from flip.covariance import covariance, fisher
from pkg_resources import resource_filename

from flip import data_vector, utils


def main(parameter_dict=None, variant="growth_rate"):
    flip_base = resource_filename("flip", ".")
    data_path = os.path.join(flip_base, "data")

    ### Load data
    sn_data = pd.read_parquet(os.path.join(data_path, "data_velocity.parquet"))

    coordinates_velocity = np.array(
        [sn_data["ra"], sn_data["dec"], sn_data["rcom_zobs"]]
    )

    data_velocity = sn_data.to_dict("list")
    for key in data_velocity.keys():
        data_velocity[key] = np.array(data_velocity[key])
    data_velocity["velocity"] = data_velocity.pop("vpec")
    data_velocity["velocity_error"] = np.zeros_like(data_velocity["velocity"])

    data_velocity_object = data_vector.DirectVel(data_velocity)

    ktt, ptt = np.loadtxt(os.path.join(data_path, "power_spectrum_tt.txt"))

    sigmau_fiducial = 15

    power_spectrum_dict = {"vv": [[ktt, ptt * utils.Du(ktt, sigmau_fiducial) ** 2]]}

    size_batch = 10_000
    number_worker = 16

    covariance_fit = covariance.CovMatrix.init_from_flip(
        "rcrk24",
        "velocity",
        power_spectrum_dict,
        coordinates_velocity=coordinates_velocity,
        size_batch=size_batch,
        number_worker=number_worker,
        variant=variant,
    )

    ### Load fitter

    fisher_properties = {
        "inversion_method": "inverse",
        "velocity_type": "scatter",
    }

    Fisher = fisher.FisherMatrix.init_from_covariance(
        covariance_fit,
        data_velocity_object,
        parameter_dict,
        fisher_properties=fisher_properties,
        covariance_prefactor_dict={"redshift_velocity": data_velocity["zobs"]},
    )

    parameter_name_list, fisher_matrix = Fisher.compute_fisher_matrix(
        covariance_prefactor_dict={"redshift_velocity": np.array(data_velocity["zobs"])}
    )
    return parameter_name_list, fisher_matrix


def lnD(a, parameter_values_dict):
    f0 = parameter_values_dict["Om0"] ** parameter_values_dict["gamma"]
    return np.log(a) * (
        f0
        + f0 * 3 * parameter_values_dict["gamma"] * (1 - parameter_values_dict["Om0"])
    ) + (1 - a) * f0 * 3 * parameter_values_dict["gamma"] * (
        1 - parameter_values_dict["Om0"]
    )


def dlnDdOm0(a, parameter_values_dict):
    lna = np.log(a)
    return (
        parameter_values_dict["gamma"]
        * parameter_values_dict["Om0"] ** (parameter_values_dict["gamma"] - 1)
        * (
            3
            * parameter_values_dict["gamma"]
            * (parameter_values_dict["Om0"] - 1)
            * (a - lna - 1)
            + 3 * (a - 1) * parameter_values_dict["Om0"]
            - 3 * np.log(a) * parameter_values_dict["Om0"]
            + lna
        )
    )


def dlnDdgamma(a, parameter_values_dict):
    lna = np.log(a)
    f0 = parameter_values_dict["Om0"] ** parameter_values_dict["gamma"]
    return f0 * (
        np.log(parameter_values_dict["Om0"])
        * (
            3
            * parameter_values_dict["gamma"]
            * (parameter_values_dict["Om0"] - 1)
            * (a - lna - 1)
            + lna
        )
        + 3 * (parameter_values_dict["Om0"] - 1) * (a - lna - 1)
    )


if __name__ == "__main__":

    # dictionary that describes the fiducial power spectrum
    # should be specified by the
    #
    # parameters of the model
    # s8 of the fiducial power spectrum
    # z of the fiduccial power spectrum

    # for now the dictionaries conflate the two models.
    # Really should be for one model

    parameter_dict = {
        "gamma": 0.55,
        "Om0": 0.3,
        "s80": 0.832,
        "fs8": 0.3**0.55 * 0.832,
        "s8_cmb": 0.832 * 0.001176774706956903,
        "sigv": 200,
        "sigma_M": 0.12,
    }

    Om0_prior = 0.1

    # growth index
    parameter_name_list, fisher_matrix = main(
        parameter_dict=parameter_dict, variant="growth_index"
    )
    cov = np.linalg.inv(fisher_matrix + np.array([[1 / Om0_prior**2, 0], [0, 0]]))
    partials = parameter_dict["s80"] * np.array(
        [
            parameter_dict["gamma"]
            * parameter_dict["Om0"] ** (parameter_dict["gamma"] - 1),
            np.log(parameter_dict["Om0"])
            * parameter_dict["Om0"] ** parameter_dict["gamma"],
        ]
    )
    partials = partials + parameter_dict["Om0"] ** parameter_dict[
        "gamma"
    ] * parameter_dict["s80"] * np.array(
        [dlnDdOm0(1.0, parameter_dict), dlnDdgamma(1.0, parameter_dict)]
    )
    print(
        parameter_dict["Om0"] ** parameter_dict["gamma"] * parameter_dict["s80"],
        np.sqrt(partials.T @ cov @ partials),
    )

    # growth rate
    parameter_name_list, fisher_matrix = main(
        parameter_dict=parameter_dict, variant="growth_rate"
    )
    print(parameter_dict["fs8"], 1 / np.sqrt(fisher_matrix))
