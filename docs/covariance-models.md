# Covariance models

The heart of flip is the **covariance matrix** of the density and velocity
fields. Given a [power spectrum](power-spectra.md) and the object coordinates,
flip builds the density-density (`gg`), velocity-velocity (`vv`) and cross
(`gv`) blocks analytically, for any linear power-spectrum model, and accelerates
the computation with a Hankel (FFTLog) transform.

## Building a covariance

The central object is [`CovMatrix`](api/covariance.md). The usual entry point is
`init_from_flip`, which takes the model name, the field *kind*
(`"velocity"`, `"density"` or `"full"`), the power-spectrum dictionary and the
coordinates of each tracer:

```python
from flip.covariance import covariance

covariance_fit = covariance.CovMatrix.init_from_flip(
    "carreres23",                       # model name
    "velocity",                         # kind: velocity / density / full
    power_spectrum_dict,
    coordinates_velocity=coordinates_velocity,   # (3, N): ra, dec, rcom_zobs
    size_batch=10_000,                  # pair-batch size (parallelization)
    number_worker=16,
)
```

Alternatively, a [data vector](data-vectors.md) can build its own covariance
directly with `data.compute_covariance(model, power_spectrum_dict, ...)`.

!!! note "Coordinate convention"
    Inside flip, RA/Dec are in **radians**, comoving distances in $\mathrm{Mpc}\,h^{-1}$,
    velocities in km/s and wavenumbers in $h\,\mathrm{Mpc}^{-1}$.

## Available models

Each model lives in its own module under
[`flip.covariance.analytical`](api/covariance-models.md) and declares its free
parameters, variants and the coordinate keys it needs.

| Model | Fields | Notes |
|---|---|---|
| `adamsblake17` | density + velocity | Adams & Blake (2017), wide-angle |
| `adamsblake17plane` | density + velocity | plane-parallel (flat-sky) variant |
| `adamsblake20` | density + velocity | **recommended default**, redshift-dependent (RSD $\beta$) |
| `carreres23` | velocity | Carreres et al. (2023), arXiv:2303.01198 |
| `lai22` | density + velocity | Lai et al. (2022) |
| `rcrk24` | velocity | growth-index / growth-rate parameterization |
| `ravouxcarreres` | density + velocity | combined Ravoux/Carreres model |
| `ravouxnoanchor25` | velocity | Ravoux et al. (2025), arXiv:2501.16852 — fits `H0` (no anchor) |
| `ravouxqin26` | density + velocity | latest; non-linear damping + full bias expansion |
| `genericzdep` | template | starting point for a new redshift-dependent model |

### Variants and free parameters

Most models expose **variants** (selected with the `variant` keyword) that toggle
which parameters are free — for example `adamsblake20` offers `baseline`
(with the RSD parameter `beta_f`) and `nobeta`. The free parameters and the
`variant`/`_free_par`/`_coordinate_keys` maps are documented on each model's
[API page](api/covariance-models.md); the free-parameter names (`fs8`, `bs8`,
`sigv`, ...) are exactly the parameters you fit or forecast.

## Under the hood

- [`generator`](api/covariance.md) turns the model's k-space multipole kernels
  and the input power spectra into the real-space covariance blocks, using either
  direct quadrature or the fast Hankel transform.
- [`hankel`](api/covariance.md) implements the FFTLog transform between $P(k)$
  and $\xi_\ell(r)$ (adapted from cosmoprimo).
- [`cov_utils`](api/covariance.md) provides the pair-separation geometry under
  several line-of-sight conventions.
- [`contraction`](api/covariance.md) predicts the model covariance as a function
  of separation (handy for plotting and validating a model without a catalog).

The `flip_terms.py`, `coefficients.py` and `fisher_terms.py` files inside each
model are **generated** offline by
[`symbolic`](api/covariance-models.md) and are not part of the everyday API.

## Emulators

For expensive models, the covariance can be replaced by a fast surrogate trained
over the model parameters. flip ships Gaussian-process
([GPy](api/covariance-models.md) and
[scikit-learn](api/covariance-models.md)) and neural-network
([PyTorch](api/covariance-models.md)) backends, driven through
`CovMatrix.init_from_emulator`. See
[Covariance models & emulators](api/covariance-models.md) in the API reference.
