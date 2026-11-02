"""Velocity covariance of a generated SN sample with the minimal flip.

From the root of a minimal clone:
    PYTHONPATH=. python scripts/flip_minimal_velocity_covariance.py
"""

import os

os.environ["FLIP_MINIMAL"] = "1"

import numpy as np

from flip import utils
from flip.covariance import CovMatrix

### Generated sample: 200 SNe, uniform on the sky, 50-150 Mpc/h
rng = np.random.default_rng(42)
number_sn = 200
ra = rng.uniform(0, 2 * np.pi, number_sn)
dec = np.arcsin(rng.uniform(-1, 1, number_sn))
rcom = rng.uniform(50, 150, number_sn)
coordinates_velocity = np.array([ra, dec, rcom])
velocity_error = np.full(number_sn, 300.0)

### Toy power spectrum: BBKS shape, Gamma = 0.2 h/Mpc, sigma8 ~ 0.8
k = np.logspace(-4, 0, 1000)
q = k / 0.2
transfer = (
    np.log(1 + 2.34 * q)
    / (2.34 * q)
    * (1 + 3.89 * q + (16.1 * q) ** 2 + (5.46 * q) ** 3 + (6.71 * q) ** 4) ** -0.25
)
ptt = 1.79e6 * k * transfer**2

sigmau_fiducial = 15
power_spectrum_dict = {"vv": [[k, ptt * utils.Du(k, sigmau_fiducial) ** 2]]}

### Compute covariance
covariance = CovMatrix.init_from_flip(
    "carreres23",
    "velocity",
    power_spectrum_dict,
    coordinates_velocity=coordinates_velocity,
    size_batch=10_000,
    number_worker=1,
)

covariance_matrix = covariance.compute_covariance_sum(
    {"fs8": 0.45, "sigv": 200.0}, velocity_error**2
)
np.save("velocity_covariance.npy", covariance_matrix)
print(covariance_matrix.shape)
