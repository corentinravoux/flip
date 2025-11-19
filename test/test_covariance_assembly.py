from pathlib import Path

import numpy as np
import pandas as pd

from flip import __flip_dir_path__, data_vector, utils


def test_density_velocity_block_diagonal_no_gv():
    base_path = Path(__flip_dir_path__)
    data_path = base_path / "data"

    # Build small density and velocity data vectors from packaged parquet
    grid = pd.read_parquet(data_path / "density_data.parquet")
    grid = grid.rename(columns={"density_err": "density_error", "rcom": "rcom_zobs"})
    grid = grid.iloc[:5]
    density_data_vector = data_vector.Dens(grid.to_dict(orient="list"))

    velocity_df = pd.read_parquet(data_path / "velocity_data.parquet")
    velocity_sample = velocity_df.rename(columns={"vpec": "velocity"}).iloc[:7]
    velocity_dict = velocity_sample.to_dict(orient="list")
    velocity_dict["velocity_error"] = np.zeros(len(velocity_sample))
    velocity_data_vector = data_vector.DirectVel(velocity_dict)

    density_velocity_data_vector = data_vector.DensVel(
        density_data_vector, velocity_data_vector
    )

    # Use packaged power spectra (no gv term) and init_from_flip through DataVector
    kmm, pmm = np.loadtxt(data_path / "power_spectrum_mm.txt")
    ktt, ptt = np.loadtxt(data_path / "power_spectrum_tt.txt")
    # DensVel.compute_covariance always builds a 'full' covariance; provide a zero gv term to enforce block-diagonal
    kmt, pmt = np.loadtxt(data_path / "power_spectrum_mt.txt")
    ps = {"gg": [[kmm, pmm]], "vv": [[ktt, ptt]], "gv": [[kmt, 0 * pmt]]}

    covariance = density_velocity_data_vector.compute_covariance(
        "adamsblake17plane", ps, size_batch=1000, number_worker=1
    )

    # Assemble total covariance with simple coefficients
    coefficients = {"bs8": 1.0, "fs8": 1.0, "sigv": 0.0}
    data_values, data_variance = density_velocity_data_vector.give_data_and_variance()
    covariance_matrix = covariance.compute_covariance_sum(coefficients, data_variance)

    # Check shapes and that cross-block is exactly zero (no gv provided)
    n_density = len(density_data_vector.data["density"])
    n_velocity = len(velocity_data_vector.data["velocity"])
    assert covariance_matrix.shape == (n_density + n_velocity, n_density + n_velocity)
    covariance_density_density = covariance_matrix[:n_density, :n_density]
    covariance_velocity_velocity = covariance_matrix[n_density:, n_density:]
    covariance_density_velocity = covariance_matrix[:n_density, n_density:]
    np.testing.assert_allclose(covariance_density_velocity, 0.0, atol=0.0)
    assert np.all(np.diag(covariance_density_density) > 0)
    assert np.all(np.diag(covariance_velocity_velocity) > 0)


def test_full_with_gv_nonzero():
    base_path = Path(__flip_dir_path__)
    data_path = base_path / "data"

    grid = pd.read_parquet(data_path / "density_data.parquet")
    grid = grid.rename(columns={"density_err": "density_error", "rcom": "rcom_zobs"})
    grid = grid.iloc[:4]
    density_data_vector = data_vector.Dens(grid.to_dict(orient="list"))

    velocity_df = pd.read_parquet(data_path / "velocity_data.parquet")
    velocity_sample = velocity_df.rename(columns={"vpec": "velocity"}).iloc[:4]
    velocity_dict = velocity_sample.to_dict(orient="list")
    velocity_dict["velocity_error"] = np.zeros(len(velocity_sample))
    velocity_data_vector = data_vector.DirectVel(velocity_dict)

    density_velocity_data_vector = data_vector.DensVel(
        density_data_vector, velocity_data_vector
    )

    kmm, pmm = np.loadtxt(data_path / "power_spectrum_mm.txt")
    kmt, pmt = np.loadtxt(data_path / "power_spectrum_mt.txt")
    ktt, ptt = np.loadtxt(data_path / "power_spectrum_tt.txt")
    ps = {"gg": [[kmm, pmm]], "gv": [[kmt, pmt]], "vv": [[ktt, ptt]]}

    covariance = density_velocity_data_vector.compute_covariance(
        "adamsblake17plane", ps, size_batch=1000, number_worker=1
    )

    coefficients = {"bs8": 1.0, "fs8": 1.0, "sigv": 0.0}
    data_values, data_variance = density_velocity_data_vector.give_data_and_variance()
    covariance_matrix = covariance.compute_covariance_sum(coefficients, data_variance)

    # Cross block should be non-zero in general
    n_density = len(density_data_vector.data["density"])
    covariance_density_velocity = covariance_matrix[:n_density, n_density:]
    assert np.any(np.abs(covariance_density_velocity) > 0)
