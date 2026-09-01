# Covariance models

The heart of FLIP is the **covariance matrix** of the density and velocity fields. Given a
[power spectrum](power_spectra.md) and the object coordinates, FLIP builds the density-density (`gg`),
velocity-velocity (`vv`) and cross (`gv`) blocks analytically, for any linear power-spectrum model, and
accelerates the computation with a Hankel (FFTLog) transform.

## Building a covariance

The central object is [`CovMatrix`][flip.covariance.covariance.CovMatrix]. The usual entry point is
`init_from_flip`, which takes the model name, the field *kind* (`"velocity"`, `"density"`, `"density_velocity"`
or `"full"`), the power-spectrum dictionary and the coordinates of each tracer:

```python
from flip.covariance import covariance

covariance_fit = covariance.CovMatrix.init_from_flip(
    "carreres23",                       # model name
    "velocity",                         # kind: velocity / density / density_velocity / full
    power_spectrum_dict,
    coordinates_velocity=coordinates_velocity,   # (3, N): ra, dec, rcom_zobs
    size_batch=10_000,                  # pair-batch size (parallelization)
    number_worker=16,
)
```

Alternatively, a [data vector](DataVector.md) can build its own covariance directly with
`data.compute_covariance(model, power_spectrum_dict, ...)`.

!!! note
    Inside FLIP, RA/Dec are in **radians**, comoving distances in $\mathrm{Mpc}\,h^{-1}$, velocities in km/s
    and wavenumbers in $h\,\mathrm{Mpc}^{-1}$.

## Available models

Each model lives in its own module under `flip.covariance.analytical` and declares its free parameters,
variants and the coordinate keys it needs.

| Model | Fields | Notes |
| --- | --- | --- |
| `adamsblake17` | density + velocity | Adams & Blake (2017), wide-angle |
| `adamsblake17plane` | density + velocity | plane-parallel (flat-sky) variant |
| `adamsblake20` | density + velocity | redshift-dependent (RSD $\beta$) |
| `carreres23` | velocity | Carreres et al. (2023), [arXiv:2303.01198](https://arxiv.org/abs/2303.01198) |
| `lai22` | density + velocity | Lai et al. (2022) |
| `rcrk24` | velocity | growth-index (`Om0`, `gamma`) or growth-rate (`fs8`) parameterization |
| `ravouxcarreres` | density + velocity | Ravoux & Carreres |
| `ravouxnoanchor25` | velocity | no-anchor variant — fits `H0` instead of an anchor |
| `ravouxqin26` | density + velocity | non-linear damping + full bias expansion, several variants |
| `genericzdep` | template | starting point for a new redshift-dependent model |

### Variants and free parameters

Most models expose **variants** (selected with the `variant` keyword) that toggle which parameters are free.
For example `adamsblake20`, `lai22` and `ravouxcarreres` all offer `None` (default), `"baseline"` (with the
RSD parameter `beta_f`) and `"nobeta"`; `rcrk24` offers `"growth_index"` and `"growth_rate"`; `ravouxqin26`
offers five variants toggling a non-linear damping term (`sigvv1sq`/`sigvv2sq`) and a bias-linking scheme. The
free-parameter names (`fs8`, `bs8`, `sigv`, ...) are exactly the parameters you fit or forecast, and are
documented per model in the [API reference](api/covariance.md).

## Under the hood

- [`flip.covariance.generator`][] turns the model's k-space multipole kernels and the input power spectra
  into the real-space covariance blocks, using a Hankel (FFTLog) transform
  ([`flip.covariance.hankel`][]) or direct quadrature.
- [`flip.covariance.cov_utils`][] provides the pair-separation geometry under several line-of-sight
  conventions.
- [`flip.covariance.contraction`][] predicts the model covariance as a function of separation (handy for
  plotting and validating a model without a catalog).
- The `flip_terms.py` and `coefficients.py` files inside each model are **generated** offline by
  [`flip.covariance.symbolic`][] and are not part of the everyday API.

## Emulators

For expensive models, the covariance can be replaced by a fast surrogate trained over the model parameters.
FLIP ships Gaussian-process (`flip.covariance.emulators.gpmatrix`, requiring `GPy`, and
`flip.covariance.emulators.skgpmatrix`, requiring `scikit-learn`) and neural-network
(`flip.covariance.emulators.nnmatrix`, requiring `torch`) backends, driven through
`CovMatrix.init_from_emulator`.
