import os

import numpy as np
from pkg_resources import resource_filename

from flip import fitter, utils
from flip.covariance import covariance

flip_base = resource_filename("flip", ".")
data_files = os.path.join(flip_base, "data")

sigmau = 15
power_spectrum_file_name = "./data/power_spectrum_tt_log.txt"

model_name = "carreres23"
model_type = "velocity"

number_worker = 16

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


if __name__ == "__main__":
    coordinates_sn = [ra_sn, dec_sn, rcom_sn]
    data_sn = {"velocity": vpec_sn, "velocity_err": np.zeros(len(vpec_sn))}

    wavenumber = np.loadtxt(power_spectrum_file_name)[0]
    power_spectrum = np.loadtxt(power_spectrum_file_name)[1]

    power_spectrum_dict = {
        "vv": [[wavenumber, power_spectrum * utils.Du(wavenumber, sigmau) ** 2]]
    }

    cov_carreres23 = covariance.CovMatrix.init_from_flip(
        model_name,
        model_type,
        power_spectrum_dict,
        coordinates_velocity=coordinates_sn,
        number_worker=number_worker,
    )

    minuit_fitter = fitter.FitMinuit.init_from_covariance(
        cov_carreres23, data_sn, parameter_dict
    )

    minuit_fitter.run()
    minuit = minuit_fitter.minuit
