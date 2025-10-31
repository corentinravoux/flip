import importlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from flip import __flip_dir_path__, data_vector, utils


def load_power_spectra(data_path: Path):
    kmm, pmm = np.loadtxt(data_path / "power_spectrum_mm.txt")
    kmt, pmt = np.loadtxt(data_path / "power_spectrum_mt.txt")
    ktt, ptt = np.loadtxt(data_path / "power_spectrum_tt.txt")
    return (kmm, pmm), (kmt, pmt), (ktt, ptt)


def small_density(data_path: Path, n: int = 16):
    grid = pd.read_parquet(data_path / "density_data.parquet")
    grid = grid.rename(columns={"density_err": "density_error", "rcom": "rcom_zobs"})
    grid = grid.iloc[:n]
    return data_vector.Dens(grid.to_dict(orient="list"))


def small_velocity_true(data_path: Path, n: int = 16):
    velocity_df = pd.read_parquet(data_path / "velocity_data.parquet")
    velocity_sample = velocity_df.rename(columns={"vpec": "velocity"}).iloc[:n]
    velocity_dict = velocity_sample.to_dict(orient="list")
    velocity_dict["velocity_error"] = np.zeros(len(velocity_sample))
    return data_vector.DirectVel(velocity_dict)


@pytest.mark.parametrize(
    "model,kind",
    [
        ("carreres23", "velocity"),
        ("adamsblake17plane", "density"),
        ("adamsblake17plane", "velocity"),
        ("adamsblake17plane", "full"),
        ("adamsblake20", "density"),
        ("adamsblake20", "velocity"),
        ("adamsblake20", "full"),
        ("lai22", "density"),
        # rcrk24 is velocity-only (no gg/gv); skip density/full here
        # ravouxcarreres depends on flip_terms; include density/velocity quickly
        ("ravouxcarreres", "density"),
        ("ravouxcarreres", "velocity"),
    ],
)
def test_covariance_generation_shapes(model, kind):
    base_path = Path(__flip_dir_path__)
    data_path = base_path / "data"
    (kmm, pmm), (kmt, pmt), (ktt, ptt) = load_power_spectra(data_path)

    sigma_u = 15.0
    # Some models (adamsblake20, ravouxcarreres) require an extra smoothing parameter sigma_g
    needs_sigma_g = model in {"adamsblake20", "ravouxcarreres"}
    additional_parameters = (sigma_u,) if needs_sigma_g else ()

    # Inspect the model's required number of PS terms per block
    flip_terms_module = importlib.import_module(f"flip.covariance.{model}.flip_terms")
    dictionary_terms = getattr(flip_terms_module, "dictionary_terms", {})

    def repeat_specification(specification, count):
        # Return a list of count copies of a [k, Pk] pair
        return [[specification[0], specification[1]] for _ in range(max(count, 0))]

    power_spectra = {}
    if kind in ["density", "full", "density_velocity"]:
        if "gg" not in dictionary_terms:
            pytest.xfail(
                f"Model {model} does not define gg terms (density not supported)"
            )
        power_spectra["gg"] = repeat_specification(
            [kmm, pmm], len(dictionary_terms["gg"])
        )
    if kind in ["velocity", "full", "density_velocity"]:
        if "vv" not in dictionary_terms:
            pytest.xfail(
                f"Model {model} does not define vv terms (velocity not supported)"
            )
        power_spectra["vv"] = repeat_specification(
            [ktt, ptt * utils.Du(ktt, sigma_u) ** 2], len(dictionary_terms["vv"])
        )
    if kind == "full":
        if "gv" not in dictionary_terms:
            pytest.xfail(f"Model {model} does not define gv terms (full not supported)")
        power_spectra["gv"] = repeat_specification(
            [kmt, pmt], len(dictionary_terms["gv"])
        )

    if kind == "density":
        data_vector_obj = small_density(data_path)
    elif kind == "velocity":
        data_vector_obj = small_velocity_true(data_path)
    else:
        data_vector_obj = data_vector.DensVel(
            small_density(data_path), small_velocity_true(data_path)
        )

    covariance = data_vector_obj.compute_covariance(
        model,
        power_spectra,
        size_batch=2000,
        number_worker=1,
        additional_parameters_values=additional_parameters,
    )

    # Build a parameter dict with reasonable values
    parameters = {"bs8": 1.0, "fs8": 0.5, "sigv": 0.0}
    if model in {"adamsblake20", "ravouxcarreres"}:
        # Link beta_f consistently to fs8/bs8 so that default and nobeta branches agree
        parameters["beta_f"] = parameters["fs8"] / max(parameters["bs8"], 1e-6)
    if model == "lai22":
        parameters["sigg"] = 1.0
        # lai22 coefficients expect beta_f unless variant 'nobeta' is used
        parameters["beta_f"] = parameters["fs8"] / max(parameters["bs8"], 1e-6)
    data_values, data_variance = data_vector_obj.give_data_and_variance(
        parameters if "M_0" in data_vector_obj.free_par else {}
    )
    covariance_matrix = covariance.compute_covariance_sum(parameters, data_variance)

    # Check symmetry & finite diagonal
    np.testing.assert_allclose(covariance_matrix, covariance_matrix.T, atol=1e-10)
    assert np.all(np.isfinite(np.diag(covariance_matrix)))
    assert covariance_matrix.shape[0] == data_values.shape[0]
