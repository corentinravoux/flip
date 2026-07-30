<img src="https://github.com/corentinravoux/flip/blob/main/docs/_static/flip_logo.webp?raw=true" width=350>

# flip: Field Level Inference Package

[![Documentation Status](https://readthedocs.org/projects/flip/badge/?version=latest)](https://flip.readthedocs.io/en/latest/?badge=latest)
[![PyPI](https://img.shields.io/pypi/v/flipcosmo.svg)](https://pypi.org/project/flipcosmo/)
[![Python](https://img.shields.io/pypi/pyversions/flipcosmo.svg)](https://pypi.org/project/flipcosmo/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/corentinravoux/flip/blob/main/LICENSE)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21642058.svg)](https://doi.org/10.5281/zenodo.21642058)

**flip** fits the **growth rate of structure** (`fσ₈`) from the **peculiar-velocity**
and **density** fields by maximum likelihood, at the field level
(Ravoux et al. 2025, [arXiv:2501.16852](https://arxiv.org/abs/2501.16852);
building on Carreres et al. 2023, [arXiv:2303.01198](https://arxiv.org/abs/2303.01198)).

It is organised as a three-stage pipeline:

1. **Data representation** — wrap velocity and/or density measurements in a
   `DataVector` (direct velocities, Hubble-diagram residuals, SN Ia SALT2 fits,
   Tully-Fisher, Fundamental Plane, log-distance ratios, gridded fields, ...).
2. **Covariance computation** — build the model covariance from a linear power
   spectrum and the object coordinates. This works for any linear power-spectrum
   model — velocities, densities and their cross-term — and is optimised with a
   Hankel (FFTLog) transform. A dozen analytical models are shipped, plus
   Gaussian-process / neural-network covariance emulators.
3. **Statistical inference** — form a Gaussian likelihood and fit `fσ₈` with the
   integrated Minuit / MCMC (emcee) fitters, or forecast it with a Fisher matrix.

## Installation

From PyPI:

```bash
pip install flipcosmo
```

From source (latest development version):

```bash
git clone https://github.com/corentinravoux/flip.git
cd flip
pip install .
```

The import name is `flip` (the distribution is named `flipcosmo`); Python ≥ 3.10.

## Requirements

**Mandatory:** `numpy`, `scipy >= 1.12`, `pandas`, `matplotlib`, `astropy`,
`emcee`, `iminuit`, `mpmath`, `importlib-metadata`.

**Optional:**

| Package | Enables |
|---|---|
| `jax` | JAX-accelerated, JIT-compiled covariance and likelihood paths |
| `classy` (CLASS) / `pyccl` / [`cosmoprimo`](https://github.com/adematti/cosmoprimo) | power-spectrum engines (cosmoprimo recommended) |
| `GPy` / `scikit-learn` | Gaussian-process covariance emulators |
| `torch` | neural-network covariance emulator |
| `pmesh` | painting catalogs onto a mesh for the `*Mesh` data vectors |

`import flip` prints a banner reporting which subpackages are usable and which
optional dependencies are present (set `FLIP_QUIET=1` to silence it); subpackages
are imported lazily, so a missing optional dependency only matters once you use
the part that needs it.

## Quick start

```python
from flip import data_vector
from flip.covariance import covariance, fitter
from flip.data import load_data_test

# 1. Data: a direct peculiar-velocity sample (packaged example data)
coordinates_velocity, velocity = load_data_test.load_velocity_data()
data = data_vector.DirectVel(velocity)

# 2. Covariance: pick a model and a power-spectrum dictionary
power_spectrum_dict = load_data_test.load_power_spectrum_dict(sigmau_fiducial=15.0)
covariance_fit = covariance.CovMatrix.init_from_flip(
    "carreres23", "velocity", power_spectrum_dict,
    coordinates_velocity=coordinates_velocity,
)

# 3. Likelihood + Minuit fit for the growth rate fs8 and the dispersion sigv
parameter_dict = {
    "fs8":  {"value": 0.4,   "limit_low": 0.0, "fixed": False},
    "sigv": {"value": 200.0, "limit_low": 0.0, "fixed": False},
}
minuit_fitter = fitter.FitMinuit.init_from_covariance(
    covariance_fit, data, parameter_dict,
)
minuit_fitter.run()
```

## Examples

Runnable notebooks on Google Colab:

* Velocity fit: <a target="_blank" href="https://colab.research.google.com/github/corentinravoux/flip/blob/main/notebook/fit_velocity.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
* Density fit: <a target="_blank" href="https://colab.research.google.com/github/corentinravoux/flip/blob/main/notebook/fit_density.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
* Joint fit: <a target="_blank" href="https://colab.research.google.com/github/corentinravoux/flip/blob/main/notebook/fit_joint.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

## Documentation

Full documentation — installation, user guide and API reference — is available at
[flip.readthedocs.io](https://flip.readthedocs.io/).

## How to cite

If you use flip, please cite the two papers that describe it:

* Ravoux et al. 2025 — [arXiv:2501.16852](https://arxiv.org/abs/2501.16852)
* Carreres et al. 2023 — [arXiv:2303.01198](https://arxiv.org/abs/2303.01198)
  (this package builds on the earlier work of
  [@bastiencarreres](https://github.com/bastiencarreres))

and the software itself via the metadata in
[`CITATION.cff`](https://github.com/corentinravoux/flip/blob/main/CITATION.cff) or
the archived release ([DOI 10.5281/zenodo.21642058](https://doi.org/10.5281/zenodo.21642058)).

## License

flip is released under the [MIT License](https://github.com/corentinravoux/flip/blob/main/LICENSE).
