import os

import numpy as np
import pandas as pd
from flip.covariance import covariance, fitter
from pkg_resources import resource_filename

from flip import data_vector

flip_base = resource_filename("flip", ".")
data_path = os.path.join(flip_base, "data")

### Load data
grid = pd.read_parquet(os.path.join(data_path, "data_density.parquet"))
grid_window = pd.read_parquet(os.path.join(data_path, "data_window_density.parquet"))

coordinates_density = np.array([grid["ra"], grid["dec"], grid["rcom_zobs"]])
data_density = {
    "density": np.array(grid["density"]),
    "density_error": np.array(grid["density_error"]),
}

data_density_object = data_vector.Dens(data_density)


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


### Compute covariance
size_batch = 500_000
number_worker = 8

covariance_fit = covariance.CovMatrix.init_from_flip(
    "adamsblake20",
    "density",
    power_spectrum_dict,
    coordinates_density=coordinates_density,
    size_batch=size_batch,
    number_worker=number_worker,
    additional_parameters_values=(sigmag_fiducial,),
    variant="nobeta",
)


### Load fitter
likelihood_type = "multivariate_gaussian"
likelihood_properties = {"inversion_method": "cholesky_inverse"}

parameter_dict = {
    "bs8": {
        "value": 1.0,
        "limit_low": 0.0,
        "limit_up": 3.0,
        "fixed": False,
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
    data_density_object,
    parameter_dict,
    likelihood_type=likelihood_type,
    likelihood_properties=likelihood_properties,
)


### Fit
minuit_fitter.run()
