# Installation

## From source

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

**Mandatory:** `numpy`, `scipy >= 1.12`, `pandas`, `matplotlib`, `astropy`,
`emcee`, `iminuit`, `mpmath`, `importlib-metadata`.

**Optional** (enable extra features / speed):

| Package | Enables |
|---|---|
| `jax` | JAX-accelerated, JIT-compiled covariance and likelihood paths |
| `classy` (CLASS) | the CLASS power-spectrum engine |
| `pyccl` | the pyCCL power-spectrum engine |
| [`cosmoprimo`](https://github.com/adematti/cosmoprimo) | the recommended power-spectrum engine and Hankel backend |
| `GPy` / `scikit-learn` | Gaussian-process covariance emulators |
| `torch` | the neural-network covariance emulator |
| `pmesh` | painting catalogs onto a mesh for the `*Mesh` data vectors |

## Checking your environment

`import flip` prints a banner reporting which subpackages are usable and which
optional dependencies are present. Set `FLIP_QUIET=1` to suppress it.

```python
import flip           # prints the availability banner
print(flip.__version__)
```

Subpackages are loaded lazily: `import flip` probes dependencies but does not
import any subpackage, so a missing optional dependency only matters once you
actually use the subpackage that needs it.

## The JAX flag

flip auto-detects JAX and enables the accelerated code paths when it is
importable. To force the NumPy/SciPy backend, set the flag **before** importing
any subpackage:

```python
import flip._config
flip._config.__use_jax__ = False
```
