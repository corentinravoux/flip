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
| [`cosmoprimo`](https://github.com/cosmodesi/cosmoprimo) | the cosmoprimo power-spectrum engine, and a faster Hankel-transform backend |
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
