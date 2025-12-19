import os

import numpy as np
from flip.covariance import contraction
from pkg_resources import resource_filename
from scipy.interpolate import RegularGridInterpolator

from flip import utils

flip_base = resource_filename("flip", ".")
data_path = os.path.join(flip_base, "data")

sigmau_fiducial = 15.0
sigmag_fiducial = 3.0

r_array = np.linspace(10, 200, 100)
mu_array = np.linspace(-1, 1, 100)

r_reference = 10_000
mu_reference = 0


ktt, ptt = np.loadtxt(os.path.join(data_path, "power_spectrum_tt.txt"))
kmt, pmt = np.loadtxt(os.path.join(data_path, "power_spectrum_mt.txt"))
kmm, pmm = np.loadtxt(os.path.join(data_path, "power_spectrum_mm.txt"))


power_spectrum_dict = {
    "gg": [[kmm, pmm], [kmt, pmt], [ktt, ptt]],
    "gv": [
        [kmt, pmt * utils.Du(kmt, sigmau_fiducial)],
        [ktt, ptt * utils.Du(kmt, sigmau_fiducial)],
    ],
    "vv": [[ktt, ptt * utils.Du(ktt, sigmau_fiducial) ** 2]],
}


correlation = contraction.Contraction.init_from_flip(
    "adamsblake20",
    "full",
    power_spectrum_dict,
    r_array,
    mu_array,
    r_reference,
    mu_reference,
    additional_parameters_values=(sigmag_fiducial,),
    coordinate_type="rmu",
    basis_definition="bisector",
    number_worker=8,
    hankel=True,
    variant="nobeta",
)


parameter_dict = {
    "fs8": 0.4,
    "bs8": 1.8,
}


contraction_sum = correlation.compute_contraction_sum(parameter_dict)
xi_vv_interp = RegularGridInterpolator(
    (r_array, mu_array), contraction_sum["vv"], method="cubic"
)
xi_vg_interp = RegularGridInterpolator(
    (r_array, mu_array), contraction_sum["gv"], method="cubic"
)
xi_gg_interp = RegularGridInterpolator(
    (r_array, mu_array), contraction_sum["gg"], method="cubic"
)
