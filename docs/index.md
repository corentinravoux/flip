# Welcome to FLIP's documentation!

![FLIP logo](_static/flip_logo.svg)

**FLIP** (Field Level Inference Package) fits the **growth rate of structure** from the **peculiar-velocity**
and **density** fields with a maximum-likelihood, field-level method. It is organised as a three-stage
pipeline:

1. **Data representation** — wrap velocity and/or density measurements in a [`DataVector`](DataVector.md),
   which turns raw observables (direct velocities, Hubble-diagram residuals, SN Ia SALT2 fits, Tully-Fisher,
   Fundamental Plane, log-distance ratios, gridded fields, ...) into a data vector and its variance.
2. **Covariance computation** — build the model [covariance matrix](covariance_models.md) from a linear
   [power spectrum](power_spectra.md) and the object coordinates, for velocities, densities and their
   cross-term.
3. **Statistical inference** — combine the covariance and the data into a [likelihood](likelihoods.md), and
   [fit or forecast](fitting.md) the growth rate with the integrated Minuit / MCMC fitters, or a Fisher matrix.

`import flip` only probes which of its subpackages (`data_vector`, `covariance`, `power_spectra`, `comparison`,
`forward`) are usable in your environment — it prints an availability banner (set `FLIP_QUIET=1` to silence
it) and loads each subpackage lazily, only when you actually access it.

## Quick look

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
    "carreres23",
    "velocity",
    power_spectrum_dict,
    coordinates_velocity=coordinates_velocity,
)

# 3. Likelihood + Minuit fit for the growth rate fs8 and the dispersion sigv
parameter_dict = {
    "fs8": {"value": 0.4, "limit_low": 0.0, "fixed": False},
    "sigv": {"value": 200.0, "limit_low": 0.0, "fixed": False},
}
minuit_fitter = fitter.FitMinuit.init_from_covariance(
    covariance_fit, data, parameter_dict,
)
minuit_fitter.run()
```

## Contents

- [Installation](installation.md)
- [Getting started](basicusage.md)
- [DataVector](DataVector.md)
- [Velocity estimators](vel_estimators.md)
- [Power spectra](power_spectra.md)
- [Covariance models](covariance_models.md)
- [Likelihoods](likelihoods.md)
- [Fitting & forecasts](fitting.md)
- [API reference](api/index.md)
