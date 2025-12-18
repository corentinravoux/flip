import os
from pathlib import Path

import numpy as np
import pandas as pd

from flip import __flip_dir_path__, utils

flip_base = Path(__flip_dir_path__)
data_path = flip_base / "data"


def load_power_spectra():
    kmm, pmm = np.loadtxt(os.path.join(data_path, "power_spectrum_mm.txt"))
    kmt, pmt = np.loadtxt(os.path.join(data_path, "power_spectrum_mt.txt"))
    ktt, ptt = np.loadtxt(os.path.join(data_path, "power_spectrum_tt.txt"))
    return (kmm, pmm), (kmt, pmt), (ktt, ptt)


def load_grid_windows():
    data_window_density = pd.read_parquet(
        os.path.join(data_path, "data_window_density.parquet")
    )
    return data_window_density


def load_power_spectrum_dict(
    sigmau_fiducial=15.0,
):
    (kmm, pmm), (kmt, pmt), (ktt, ptt) = load_power_spectra()
    power_spectrum_dict = {
        "gg": [[kmm, pmm]],
        "vv": [[ktt, ptt]],
        "gv": [[kmt, pmt]],
    }

    window_density = load_grid_windows()

    power_spectrum_dict = {
        "gg": [
            [kmm, pmm * np.array(window_density["window_mm"]) ** 2],
            [kmt, pmt * np.array(window_density["window_mt"])],
            [ktt, ptt],
        ],
        "gv": [
            [
                kmt,
                pmt
                * np.array(window_density["window_mt"])
                * utils.Du(kmt, sigmau_fiducial),
            ],
            [ktt, ptt * utils.Du(kmt, sigmau_fiducial)],
        ],
        "vv": [[ktt, ptt * utils.Du(ktt, sigmau_fiducial) ** 2]],
    }

    return power_spectrum_dict


def load_density_data(subsample=None):
    density = pd.read_parquet(os.path.join(data_path, "data_density.parquet"))
    if subsample is not None:
        density = density.iloc[:subsample]
    coordinates_density = np.array([density["ra"], density["dec"], density["rcom"]])

    return coordinates_density, density


def load_velocity_data(subsample=None):
    velocity = pd.read_parquet(os.path.join(data_path, "data_velocity.parquet"))
    if subsample is not None:
        velocity = velocity.iloc[:subsample]
    coordinates_velocity = np.array(
        [velocity["ra"], velocity["dec"], velocity["rcom_zobs"]]
    )

    return coordinates_velocity, velocity
