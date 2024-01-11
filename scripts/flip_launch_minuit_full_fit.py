import os

import numpy as np
import pandas as pd
from pkg_resources import resource_filename

from flip import fitter, utils
from flip.covariance import covariance

flip_base = resource_filename("flip", ".")
data_path = os.path.join(flip_base, "data")

### Load data
grid = pd.read_parquet(os.path.join(data_path, "density_data.parquet"))
grid_window = pd.read_parquet(os.path.join(data_path, "grid_window_m.parquet"))
coordinates_density = np.array([grid["ra"], grid["dec"], grid["rcom"]])
data_density = {
    "density": np.array(grid["density"]),
    "density_error": np.array(grid["density_err"]),
}


sn_data = pd.read_parquet(os.path.join(data_path, "velocity_data.parquet"))

sn_data = sn_data[np.array(sn_data["status"]) != False]
sn_data = sn_data[np.array(sn_data["status"]) != None]

coordinates_velocity = np.array([sn_data["ra"], sn_data["dec"], sn_data["como_dist"]])
data_velocity = sn_data.to_dict("list")
for key in data_velocity.keys():
    data_velocity[key] = np.array(data_velocity[key])
data_velocity["velocity"] = data_velocity.pop("vpec")
data_velocity["velocity_error"] = np.zeros_like(data_velocity["velocity"])

data_full = {}
data_full.update(data_density)
data_full.update(data_velocity)


sigmau_fiducial = 15.0
sigmag_fiducial = 3.0

ktt, ptt = np.loadtxt(os.path.join(data_path, "power_spectrum_tt.txt"))
kmt, pmt = np.loadtxt(os.path.join(data_path, "power_spectrum_mt.txt"))
kmm, pmm = np.loadtxt(os.path.join(data_path, "power_spectrum_mm.txt"))
power_spectrum_dict_bias = {
    "gg": [[kmm, pmm * np.array(grid_window["window_mm"]) ** 2]]
}
power_spectrum_dict = {
    "gg": [
        [kmm, pmm * np.array(grid_window["window_mm"]) ** 2],
        [kmt, pmt * np.array(grid_window["window_mt"])],
        [ktt, ptt],
    ],
    "gv": [
        [
            kmt,
            pmt * np.array(grid_window["window_mt"]) * utils.Du(kmt, sigmau_fiducial),
        ],
        [ktt, ptt * utils.Du(kmt, sigmau_fiducial)],
    ],
    "vv": [[ktt, ptt * utils.Du(ktt, sigmau_fiducial) ** 2]],
}


### Fit bias

size_batch = 10_000
number_worker = 1

covariance_bias = covariance.CovMatrix.init_from_flip(
    "adamsblake17plane",
    "density",
    power_spectrum_dict_bias,
    coordinates_density=coordinates_density,
    size_batch=size_batch,
    number_worker=number_worker,
)

likelihood_type = "multivariate_gaussian"
likelihood_properties = {"inversion_method": "cholesky"}


parameter_dict = {
    "bs8": {
        "value": 1.0,
        "limit_low": 0.0,
        "limit_up": 20.0,
        "fixed": False,
    },
}


minuit_fitter_bias = fitter.FitMinuit.init_from_covariance(
    covariance_bias,
    data_density,
    parameter_dict,
    likelihood_type=likelihood_type,
    likelihood_properties=likelihood_properties,
)

minuit_fitter_bias.run()

### Compute covariance
size_batch = 10_000
number_worker = 8

covariance_fit = covariance.CovMatrix.init_from_flip(
    "adamsblake20",
    "full",
    power_spectrum_dict,
    coordinates_density=coordinates_density,
    coordinates_velocity=coordinates_velocity,
    size_batch=size_batch,
    number_worker=number_worker,
    additional_parameters_values=(sigmag_fiducial,),
)

### Load fitter
likelihood_type = "multivariate_gaussian"
likelihood_properties = {"inversion_method": "inverse", "velocity_type": "direct"}


parameter_dict = {
    "bs8": {
        "value": minuit_fitter_bias.minuit.values["bs8"],
        "fixed": True,
    },
    "fs8": {
        "value": 0.4,
        "limit_low": 0.0,
        "limit_up": 1.0,
        "fixed": False,
    },
    "sigv": {
        "value": 100,
        "limit_low": None,
        "limit_up": None,
        "fixed": False,
    },
}


minuit_fitter = fitter.FitMinuit.init_from_covariance(
    covariance_fit,
    data_full,
    parameter_dict,
    likelihood_type=likelihood_type,
    likelihood_properties=likelihood_properties,
)


### Fit
minuit_fitter.run()
