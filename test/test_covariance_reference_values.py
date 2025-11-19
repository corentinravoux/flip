import importlib
import json
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


def build_power_spectra(model: str, kind: str, data_path: Path):
    (kmm, pmm), (kmt, pmt), (ktt, ptt) = load_power_spectra(data_path)
    sigma_u = 15.0
    flip_terms_module = importlib.import_module(f"flip.covariance.{model}.flip_terms")
    dictionary_terms = getattr(flip_terms_module, "dictionary_terms", {})

    def repeat(specification, count):
        return [[specification[0], specification[1]] for _ in range(max(count, 0))]

    power_spectra = {}
    if kind in ["density", "full", "density_velocity"] and "gg" in dictionary_terms:
        power_spectra["gg"] = repeat([kmm, pmm], len(dictionary_terms.get("gg", [])))
    if kind in ["velocity", "full", "density_velocity"] and "vv" in dictionary_terms:
        power_spectra["vv"] = repeat(
            [ktt, ptt * utils.Du(ktt, sigma_u) ** 2],
            len(dictionary_terms.get("vv", [])),
        )
    if kind == "full" and "gv" in dictionary_terms:
        power_spectra["gv"] = repeat([kmt, pmt], len(dictionary_terms.get("gv", [])))
    return power_spectra


def compute_covariance_metrics(model: str, kind: str):
    base_path = Path(__flip_dir_path__)
    data_path = base_path / "data"

    if kind == "density":
        data_vector_obj = small_density(data_path)
    elif kind == "velocity":
        data_vector_obj = small_velocity_true(data_path)
    else:
        data_vector_obj = data_vector.DensVel(
            small_density(data_path), small_velocity_true(data_path)
        )

    additional_parameters = ()
    parameters = {"bs8": 1.0, "fs8": 0.5, "sigv": 0.0}
    if model in {"adamsblake20", "ravouxcarreres"}:
        parameters["beta_f"] = parameters["fs8"] / max(parameters["bs8"], 1e-6)
        additional_parameters = (15.0,)  # sigma_g
    if model == "lai22":
        parameters["sigg"] = 1.0
        parameters["beta_f"] = parameters["fs8"] / max(parameters["bs8"], 1e-6)

    power_spectra = build_power_spectra(model, kind, data_path)
    covariance = data_vector_obj.compute_covariance(
        model,
        power_spectra,
        size_batch=2000,
        number_worker=1,
        additional_parameters_values=additional_parameters,
    )
    data_values, data_variance = data_vector_obj.give_data_and_variance(
        parameters if "M_0" in data_vector_obj.free_par else {}
    )
    covariance_matrix = covariance.compute_covariance_sum(parameters, data_variance)
    return {
        "shape": [int(covariance_matrix.shape[0]), int(covariance_matrix.shape[1])],
        "trace": float(np.trace(covariance_matrix)),
        "diag_mean": float(np.mean(np.diag(covariance_matrix))),
        "entry_0_0": float(covariance_matrix[0, 0]),
        "entry_0_-1": float(covariance_matrix[0, -1]),
        "entry_mid_mid": float(
            covariance_matrix[
                covariance_matrix.shape[0] // 2, covariance_matrix.shape[1] // 2
            ]
        ),
    }


@pytest.mark.parametrize(
    "model,kind",
    [
        ("carreres23", "velocity"),
        ("adamsblake17plane", "density"),
        ("adamsblake17plane", "full"),
        ("adamsblake20", "density"),
        ("ravouxcarreres", "velocity"),
        ("ravouxcarreres", "density"),
        ("lai22", "density"),
    ],
)
def test_covariance_reference_metrics(model, kind):
    # Load stored reference metrics
    ref_path = Path(__flip_dir_path__) / "data" / "test_cov_refs.json"
    with open(ref_path, "r") as f:
        refs = json.load(f)

    key = f"{model}:{kind}"
    assert key in refs, f"Missing reference metrics for {key}"
    ref = refs[key]

    got = compute_covariance_metrics(model, kind)

    # Compare metrics with reasonable tolerances
    assert got["shape"] == ref["shape"]
    np.testing.assert_allclose(got["trace"], ref["trace"], rtol=1e-6, atol=1e-8)
    np.testing.assert_allclose(got["diag_mean"], ref["diag_mean"], rtol=1e-6, atol=1e-8)
    np.testing.assert_allclose(got["entry_0_0"], ref["entry_0_0"], rtol=1e-6, atol=1e-8)
    np.testing.assert_allclose(
        got["entry_0_-1"], ref["entry_0_-1"], rtol=1e-6, atol=1e-8
    )
    np.testing.assert_allclose(
        got["entry_mid_mid"], ref["entry_mid_mid"], rtol=1e-6, atol=1e-8
    )
