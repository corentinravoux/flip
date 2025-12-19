import os

import numpy as np
import pandas as pd
from flip.covariance import covariance, fitter
from pkg_resources import resource_filename

flip_base = resource_filename("flip", ".")
data_path = os.path.join(flip_base, "data")

### Load data
grid = pd.read_parquet(os.path.join(data_path, "data_density.parquet"))
grid_window = pd.read_parquet(os.path.join(data_path, "grid_window_m.parquet"))

coordinates_density = np.array([grid["ra"], grid["dec"], grid["rcom"]])
data_density = {
    "density": np.array(grid["density"]),
    "density_error": np.array(grid["density_err"]),
}


ktt, ptt = np.loadtxt(os.path.join(data_path, "power_spectrum_tt.txt"))
kmt, pmt = np.loadtxt(os.path.join(data_path, "power_spectrum_mt.txt"))
kmm, pmm = np.loadtxt(os.path.join(data_path, "power_spectrum_mm.txt"))

sigmag_fiducial = 3.0


power_spectrum_dict_bias = {
    "gg": [[kmm, pmm * np.array(grid_window["window_mm"]) ** 2]]
}
power_spectrum_dict = {
    "gg": [
        [kmm, pmm * np.array(grid_window["window_mm"]) ** 2],
        [kmt, pmt * np.array(grid_window["window_mt"])],
        [ktt, ptt],
    ]
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

likelihood_type_bias = "multivariate_gaussian"
likelihood_properties_bias = {"inversion_method": "cholesky"}


parameter_dict_bias = {
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
    parameter_dict_bias,
    likelihood_type=likelihood_type_bias,
    likelihood_properties=likelihood_properties_bias,
)

minuit_fitter_bias.run()

### Compute covariance
size_batch = 10_000
number_worker = 8

covariance_fit = covariance.CovMatrix.init_from_flip(
    "adamsblake20",
    "density",
    power_spectrum_dict,
    coordinates_density=coordinates_density,
    size_batch=size_batch,
    number_worker=number_worker,
    additional_parameters_values=(sigmag_fiducial,),
)


### Load fitter
likelihood_type = "multivariate_gaussian"
likelihood_properties = {"inversion_method": "inverse"}

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
}


minuit_fitter = fitter.FitMinuit.init_from_covariance(
    covariance_fit,
    data_density,
    parameter_dict,
    likelihood_type=likelihood_type,
    likelihood_properties=likelihood_properties,
)


### Fit
minuit_fitter.run()
