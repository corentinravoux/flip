import os

import numpy as np
import pandas as pd
from flip.covariance import covariance, fitter
from pkg_resources import resource_filename

from flip import data_vector, utils

flip_base = resource_filename("flip", ".")
data_path = os.path.join(flip_base, "data")

### Load data
sn_data = pd.read_parquet(os.path.join(data_path, "data_velocity.parquet"))


coordinates_velocity = np.array([sn_data["ra"], sn_data["dec"], sn_data["rcom_zobs"]])

data_velocity = sn_data.to_dict("list")
for key in data_velocity.keys():
    data_velocity[key] = np.array(data_velocity[key])
data_velocity["velocity"] = data_velocity.pop("vpec")
data_velocity["velocity_error"] = np.zeros_like(data_velocity["velocity"])

data_velocity_object = data_vector.DirectVel(data_velocity)

ktt, ptt = np.loadtxt(os.path.join(data_path, "power_spectrum_tt.txt"))
kmt, pmt = np.loadtxt(os.path.join(data_path, "power_spectrum_mt.txt"))
kmm, pmm = np.loadtxt(os.path.join(data_path, "power_spectrum_mm.txt"))

sigmau_fiducial = 15

power_spectrum_dict = {"vv": [[ktt, ptt * utils.Du(ktt, sigmau_fiducial) ** 2]]}

### Compute covariance
size_batch = 10_000
number_worker = 16

covariance_fit = covariance.CovMatrix.init_from_flip(
    "carreres23",
    "velocity",
    power_spectrum_dict,
    coordinates_velocity=coordinates_velocity,
    size_batch=size_batch,
    number_worker=number_worker,
)


### Load fitter
likelihood_type = "multivariate_gaussian"
likelihood_properties = {"inversion_method": "cholesky_inverse"}


parameter_dict = {
    "fs8": {
        "value": 0.4,
        "limit_low": 0.0,
        "fixed": False,
    },
    "sigv": {
        "value": 200,
        "limit_low": 0.0,
        "fixed": False,
    },
}


minuit_fitter = fitter.FitMinuit.init_from_covariance(
    covariance_fit,
    data_velocity_object,
    parameter_dict,
    likelihood_type=likelihood_type,
    likelihood_properties=likelihood_properties,
)


### Fit
minuit_fitter.run()
