import os

import numpy as np
import pandas as pd
from pkg_resources import resource_filename

from flip import fisher, utils
from flip.covariance import covariance
from astropy.cosmology import FlatLambdaCDM

import matplotlib.pyplot as plt

from flip.covariance.rcrk24.flip_terms import *

def main():
    flip_base = resource_filename("flip", ".")
    data_path = os.path.join(flip_base, "data")

    ### Load data
    sn_data = pd.read_parquet(os.path.join(data_path, "velocity_data.parquet"))

    sn_data = sn_data[np.array(sn_data["status"]) != False]
    sn_data = sn_data[np.array(sn_data["status"]) != None]
    coordinates_velocity = np.array([sn_data["ra"], sn_data["dec"], sn_data["rcom_zobs"], sn_data["zobs"]])

    # plt.hist(sn_data["zobs"])
    # plt.show()

    data_velocity = sn_data.to_dict("list")
    for key in data_velocity.keys():
        data_velocity[key] = np.array(data_velocity[key])
    data_velocity["velocity"] = data_velocity.pop("vpec")
    data_velocity["velocity_error"] = np.zeros_like(data_velocity["velocity"])


    ktt, ptt = np.loadtxt(os.path.join(data_path, "power_spectrum_tt.txt"))
    kmt, pmt = np.loadtxt(os.path.join(data_path, "power_spectrum_mt.txt"))
    kmm, pmm = np.loadtxt(os.path.join(data_path, "power_spectrum_mm.txt"))

    sigmau_fiducial = 15

    power_spectrum_dict = {"vv": [[ktt, ptt * utils.Du(ktt, sigmau_fiducial) ** 2]]}

    ### Compute covariance
    size_batch = 10_000
    number_worker = 16
    variant = "growth_rate"  # can be replaced by growth_index

    parameter_dict = {
        "fs8": 0.45570516784429815,
        "Om0": 0.3,
        "gamma": 0.55,        
        "sigv": 200,
        "sigma_M": 0.12,
    }

    covariance_fit = covariance.CovMatrix.init_from_flip(
        "rcrk24",
        # "agk24"
        # 'carreres23',
        "velocity",
        power_spectrum_dict,
        coordinates_velocity=coordinates_velocity,
        size_batch=size_batch,
        number_worker=number_worker,
        power_spectrum_amplitude_function=power_spectrum_amplitude_function_growth_rate,
        variant=variant,
    )

    ### Load fitter

    fisher_properties = {
        "inversion_method": "inverse",
        "velocity_type": "scatter",
    }

    Fisher = fisher.FisherMatrix.init_from_covariance(
        covariance_fit,
        data_velocity,
        parameter_dict,
        fisher_properties=fisher_properties,
    )

    parameter_name_list, fisher_matrix = Fisher.compute_fisher_matrix(
        parameter_dict, variant=variant
    )
    return parameter_name_list, fisher_matrix

def lnD(a, parameter_values_dict):
        f0 = parameter_values_dict["Om0"]**parameter_values_dict["gamma"]
        return np.log(a)*(f0+f0*3*parameter_values_dict["gamma"]*(1-parameter_values_dict["Om0"])) \
            + (1-a)*f0*3*parameter_values_dict["gamma"]*(1-parameter_values_dict["Om0"])

def dlnDdOm0(a, parameter_values_dict):
    lna=np.log(a)
    return (
        parameter_values_dict["gamma"] * parameter_values_dict["Om0"]**(parameter_values_dict["gamma"]-1) *
            (
                3 * parameter_values_dict["gamma"] * (parameter_values_dict["Om0"]-1) * (a - lna -1) +
                3 * (a-1) * parameter_values_dict["Om0"] -
                3 * np.log(a) * parameter_values_dict["Om0"] +  lna
            )
        )

def dlnDdgamma(a, parameter_values_dict):
    lna=np.log(a)
    f0 = parameter_values_dict["Om0"]**parameter_values_dict["gamma"]
    return (
            f0 *
                (
                    np.log(parameter_values_dict["Om0"]) *
                        (
                            3 * parameter_values_dict["gamma"] * (parameter_values_dict["Om0"]-1) * (a - lna -1) + lna
                        ) +
                    3 * (parameter_values_dict["Om0"]-1) * (a - lna -1)
                )
        )

if __name__ == "__main__":

    parameter_dict = {
        # "fs8": 0.45570516784429815,
        "f": 0.45570516784429815/0.832,
        "Om0": 0.3,
        "gamma": 0.55,        
        "sigv": 200,
        "sigma_M": 0.12,
    }


    z_mn=0.05

    cosmo = FlatLambdaCDM(H0=100, Om0=parameter_dict["Om0"])
    Om=cosmo.Om(z_mn)

    parameter_name_list, fisher_matrix = main()
    print(fisher_matrix)
    cov = np.linalg.inv(fisher_matrix[0:2,0:2] +np.array([[1/0.1**2,0],[0,0]]))
    s80=0.832
    partials = s80*np.array([parameter_dict['gamma']*parameter_dict['Om0']**(parameter_dict['gamma']-1),np.log(parameter_dict['Om0'])*parameter_dict['Om0']**parameter_dict['gamma']])
    partials = partials + parameter_dict['Om0']**parameter_dict['gamma'] *s80 * np.array([dlnDdOm0(1., parameter_dict), dlnDdgamma(1., parameter_dict)])
    print(parameter_dict["Om0"]**parameter_dict['gamma'] * s80, np.sqrt(partials.T @ cov[0:2,0:2] @ partials))
    # print((Om/parameter_dict["Om0"])**parameter_dict['gamma']*np.exp(lnD(1/(1+z_mn), parameter_dict)))
    # print(parameter_dict["fs8"], 1/np.sqrt(fisher_matrix[2,2]))