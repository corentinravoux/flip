"""Minimal mode (``FLIP_MINIMAL=1``): velocity covariance from flip/covariance only.

Each check runs in a fresh interpreter, since the flag is read at import.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]

# Imported by the full flip.covariance, never needed for a velocity covariance.
NOT_NEEDED = ["sympy", "mpmath", "emcee", "iminuit"]

SCRIPT = """
import json, sys
import numpy as np
from flip import utils
from flip.covariance import CovMatrix
from flip.data import load_data_test

(_, _, (ktt, ptt)) = load_data_test.load_power_spectra()
coordinates_velocity, velocity = load_data_test.load_velocity_data(subsample=50)
power_spectrum_dict = {"vv": [[ktt, ptt * utils.Du(ktt, 15.0) ** 2]]}

model, kwargs, parameters = json.loads(sys.argv[1])
if "additional_parameters_values" in kwargs:
    kwargs["additional_parameters_values"] = tuple(kwargs["additional_parameters_values"])
cov = CovMatrix.init_from_flip(
    model,
    "velocity",
    power_spectrum_dict,
    coordinates_velocity=coordinates_velocity,
    size_batch=50_000,
    number_worker=1,
    **kwargs,
)
C = np.asarray(
    cov.compute_covariance_sum(parameters, np.asarray(velocity["velocity_error"]) ** 2)
)
import flip
print(json.dumps({
    "flip_file": flip.__file__,
    "loaded": sorted(m for m in sys.modules if m.split(".")[0] in {NOT_NEEDED}
                     or m.startswith("flip.covariance.")),
    "trace": float(np.trace(C)),
    "entry_0_0": float(C[0, 0]),
    "entry_0_-1": float(C[0, -1]),
}))
""".replace("{NOT_NEEDED}", repr(set(NOT_NEEDED)))

CASES = [
    ("carreres23", {}, {"fs8": 0.5, "sigv": 0.0}),
    (
        "adamsblake20",
        {"variant": "nobeta", "additional_parameters_values": [15.0]},
        {"fs8": 0.5, "bs8": 1.0, "sigv": 0.0, "beta_f": 0.5},
    ),
]


def run_minimal(case):
    env = dict(
        os.environ,
        FLIP_MINIMAL="1",
        FLIP_USE_JAX="0",
        FLIP_QUIET="1",
        PYTHONPATH=str(REPO),
    )
    result = subprocess.run(
        [sys.executable, "-c", SCRIPT, json.dumps(case)],
        cwd=REPO,
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(result.stdout.strip().splitlines()[-1])


@pytest.mark.parametrize("case", CASES, ids=[c[0] for c in CASES])
def test_minimal_velocity_covariance(case):
    got = run_minimal(case)
    model = case[0]

    assert Path(got["flip_file"]).resolve().is_relative_to(REPO)
    assert not [m for m in got["loaded"] if m.split(".")[0] in NOT_NEEDED]
    for module in ["fitter", "likelihood", "fisher", "emulators", "symbolic"]:
        assert f"flip.covariance.{module}" not in got["loaded"]
    # Only the requested model is imported.
    models = {m.split(".")[3] for m in got["loaded"] if m.count(".") >= 3}
    assert models == {model}

    reference = json.loads(
        (REPO / "flip" / "data" / "test_covariance_reference_values.json").read_text()
    )[f"{model}:velocity"]
    for key in ["trace", "entry_0_0", "entry_0_-1"]:
        assert got[key] == pytest.approx(reference[key], rel=1e-10)


def test_banner_minimal():
    env = dict(os.environ, FLIP_MINIMAL="1", FLIP_USE_JAX="0", PYTHONPATH=str(REPO))
    env.pop("FLIP_QUIET", None)
    out = subprocess.run(
        [sys.executable, "-c", "import flip"],
        cwd=REPO,
        env=env,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    assert "minimal mode" in out
    assert "data_vector" not in out
