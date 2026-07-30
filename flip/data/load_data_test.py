"""Loaders for the packaged example / test datasets.

Convenience functions that read the small example datasets shipped inside
``flip/data`` -- pre-computed mm/mt/tt power spectra, the density and velocity
example catalogs, grid windows, and the reference values used by the
covariance/end-to-end tests. Handy both for the test suite and for reproducing
the tutorial notebooks without external data.
"""

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd


def get_data_path():
    """Return the path to the packaged ``flip/data`` directory."""
    from flip import __flip_dir_path__

    flip_base = Path(__flip_dir_path__)
    data_path = flip_base / "data"
    return data_path


def load_power_spectra():
    """Load the example mm/mt/tt power spectra.

    Returns:
        tuple: ``((kmm, pmm), (kmt, pmt), (ktt, ptt))`` wavenumber/spectrum pairs
        for the matter-matter, matter-velocity and velocity-velocity spectra.
    """
    data_path = get_data_path()
    kmm, pmm = np.loadtxt(os.path.join(data_path, "power_spectrum_mm.txt"))
    kmt, pmt = np.loadtxt(os.path.join(data_path, "power_spectrum_mt.txt"))
    ktt, ptt = np.loadtxt(os.path.join(data_path, "power_spectrum_tt.txt"))
    return (kmm, pmm), (kmt, pmt), (ktt, ptt)


def load_grid_windows():
    """Load the example density grid-window table.

    Returns:
        pandas.DataFrame: Window functions (``window_mm``, ``window_mt``) used to
        deconvolve the gridding of the example density field.
    """
    data_path = get_data_path()
    data_window_density = pd.read_parquet(
        os.path.join(data_path, "data_window_density.parquet")
    )
    return data_window_density


def load_power_spectrum_dict(
    sigmau_fiducial=15.0,
):
    """Assemble the ``power_spectrum_dict`` for the example data.

    Builds the nested ``gg``/``gv``/``vv`` power-spectrum dictionary expected by
    the covariance models from the example spectra, applying the density grid
    windows and the small-scale velocity damping ``Du(k, sigma_u)``.

    Args:
        sigmau_fiducial (float): Fiducial small-scale velocity dispersion
            ``sigma_u`` (Mpc/h) used in the velocity damping.

    Returns:
        dict: Power-spectrum dictionary keyed by ``gg``, ``gv``, ``vv``.
    """
    from flip import utils

    (kmm, pmm), (kmt, pmt), (ktt, ptt) = load_power_spectra()

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
    """Load the example density catalog.

    Args:
        subsample (int|None): If given, keep only the first ``subsample`` rows.

    Returns:
        tuple: ``(coordinates_density, density)`` where ``coordinates_density`` is
        the ``(3, N)`` ``(ra, dec, rcom_zobs)`` array and ``density`` is the data
        dict.
    """
    data_path = get_data_path()
    density = pd.read_parquet(os.path.join(data_path, "data_density.parquet"))
    if subsample is not None:
        density = density.iloc[:subsample]
    coordinates_density = np.array(
        [density["ra"], density["dec"], density["rcom_zobs"]]
    )
    density = density.to_dict(orient="list")

    return coordinates_density, density


def load_velocity_data(subsample=None):
    """Load the example velocity catalog.

    Args:
        subsample (int|None): If given, keep only the first ``subsample`` rows.

    Returns:
        tuple: ``(coordinates_velocity, velocity)`` where ``coordinates_velocity``
        is the ``(3, N)`` ``(ra, dec, rcom_zobs)`` array and ``velocity`` is the
        data dict.
    """
    data_path = get_data_path()
    velocity = pd.read_parquet(os.path.join(data_path, "data_velocity.parquet"))
    if subsample is not None:
        velocity = velocity.iloc[:subsample]
    coordinates_velocity = np.array(
        [velocity["ra"], velocity["dec"], velocity["rcom_zobs"]]
    )
    velocity = velocity.to_dict(orient="list")

    return coordinates_velocity, velocity


def load_e2e_test_reference_values():
    """Load the reference values for the end-to-end tests (JSON)."""
    data_path = get_data_path()
    with open(os.path.join(data_path, "test_e2e_reference_values.json"), "r") as f:
        reference_values = json.load(f)
    return reference_values


def load_covariance_test_reference_values():
    """Load the reference values for the covariance tests (JSON)."""
    data_path = get_data_path()
    with open(
        os.path.join(data_path, "test_covariance_reference_values.json"), "r"
    ) as f:
        reference_values = json.load(f)
    return reference_values
