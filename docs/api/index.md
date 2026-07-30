# API reference

Auto-generated from the in-code docstrings via
[mkdocstrings](https://mkdocstrings.github.io/). The pages follow flip's three
stages — data representation, covariance computation and statistical inference —
plus the shared utilities.

| Page | Covers |
|---|---|
| [`data_vector`](data_vector.md) | `DataVector` classes: velocities, densities, SN Ia, distance indicators, meshes, and the example-data loaders |
| [`power_spectra`](power_spectra.md) | power-spectrum generation, models and cosmology engines |
| [`covariance`](covariance.md) | `CovMatrix`, generation, Hankel transform, likelihoods, fitters, Fisher, plotting |
| [`covariance` models & emulators](covariance-models.md) | the analytical model registry and the covariance emulators |
| [`utils`](utils.md) | logger, coordinate transforms, priors, subpackage probing |

!!! note
    The generated `flip_terms.py`, `coefficients.py` and `fisher_terms.py`
    kernel files inside each covariance model are intentionally omitted — they
    are produced offline by `flip.covariance.symbolic` and are not part of the
    everyday API.
