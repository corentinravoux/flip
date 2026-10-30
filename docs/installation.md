# Installation

## From source

Clone the repository from GitHub and install with pip.

```bash
git clone https://github.com/corentinravoux/flip.git
cd flip
pip install .
```

For a development install (editable, with the documentation tooling):

```bash
pip install -e ".[docs]"
```

## Requirements

**Mandatory:** `pandas`, `numpy`, `scipy >= 1.12`, `matplotlib`, `importlib-metadata`, `emcee`, `iminuit`,
`astropy`, `mpmath`.

**Optional**, enabling extra subpackages or acceleration:

| Package | Enables |
| --- | --- |
| `jax` | JIT-compiled, accelerated `data_vector` and `covariance` code paths |
| `sympy` | required by `flip.covariance` (symbolic term generation) |
| `classy` (CLASS) | the CLASS power-spectrum engine |
| `pyccl` | the pyCCL power-spectrum engine |
| [`cosmoprimo`](https://github.com/cosmodesi/cosmoprimo) | the cosmoprimo power-spectrum engine |
| `scikit-learn` | the Gaussian-process covariance emulator |
| `torch` | the neural-network covariance emulator |
| `jax-cosmo`, `tensorflow-probability` | required by `flip.forward` |

`flip.covariance` additionally requires `sympy`, `mpmath`, `emcee`, `iminuit` and `pandas` to be importable
(they're already part of the mandatory install above).

## Checking your environment

`import flip` prints a banner reporting which subpackages are usable and which optional dependencies are
present. Set `FLIP_QUIET=1` to suppress it.

```python
import flip           # prints the availability banner
print(flip.__version__)
```

Subpackages (`data_vector`, `covariance`, `power_spectra`, `comparison`, `forward`) are loaded lazily: `import
flip` only probes dependencies, it does not import any subpackage, so a missing optional dependency only
matters once you actually use the subpackage that needs it — accessing `flip.covariance` for the first time
imports only `flip.covariance` and its dependencies.

## The JAX flag

FLIP auto-detects JAX and enables the accelerated code paths when it is importable. To force the NumPy/SciPy
backend, set the flag **before** importing any subpackage:

```python
import flip._config
flip._config.__use_jax__ = False
```

## Minimal flip: velocity covariance only

To compute the velocity covariance of a peculiar-velocity sample (e.g. type Ia supernovae) without the
rest of flip, clone only the covariance code and the models you need, then set `FLIP_MINIMAL=1` before
`import flip`.

```bash
git clone --filter=blob:none --depth 1 --sparse \
    https://github.com/corentinravoux/flip.git flip_minimal
git -C flip_minimal sparse-checkout set flip/covariance/analytical/carreres23
pip install -r flip_minimal/requirements-minimal.txt   # numpy, scipy, astropy, matplotlib
```

Add more models to the `sparse-checkout set` line (e.g. `flip/covariance/analytical/adamsblake20`), and
`flip/data` for the test data. The clone holds the top-level `flip/*.py`, the `flip/covariance/*.py`
modules and the chosen models, a few hundred kilobytes.

```python
import os
import sys

os.environ["FLIP_MINIMAL"] = "1"
sys.path.insert(0, "flip_minimal")

import numpy as np
from flip import utils
from flip.covariance import CovMatrix

# coordinates_velocity = np.array([ra, dec, rcom])  -- radians, radians, Mpc/h
# k, p_tt: velocity-divergence power spectrum, h/Mpc and (Mpc/h)^3
cov = CovMatrix.init_from_flip(
    "carreres23",
    "velocity",
    {"vv": [[k, p_tt * utils.Du(k, 15.0) ** 2]]},
    coordinates_velocity=coordinates_velocity,
)
C = cov.compute_covariance_sum({"fs8": 0.45, "sigv": 200.0}, velocity_error**2)
```

In minimal mode `flip.covariance` imports only `analytical`, `cov_utils`, `generator` and `CovMatrix`; no
model is imported until requested, and `sympy`, `mpmath`, `emcee` and `iminuit` are not needed. `mpmath`
is imported only by models that request mpmath regularization, which no velocity (`vv`) term does.
