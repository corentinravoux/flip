# Getting started

The fastest way to see FLIP in action is the set of example notebooks, which run end-to-end on the small
datasets packaged inside `flip/data` (no external data needed). They are available on Google Colab:

- Velocity fit: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/corentinravoux/flip/blob/main/notebook/fit_velocity.ipynb)
- Density fit: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/corentinravoux/flip/blob/main/notebook/fit_density.ipynb)
- Joint fit: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/corentinravoux/flip/blob/main/notebook/fit_joint.ipynb)
- Velocity fit with an emulated covariance: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/corentinravoux/flip/blob/main/notebook/fit_velocity-emulator.ipynb)
- Fisher forecast (density): [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/corentinravoux/flip/blob/main/notebook/fisher_density.ipynb)
- Fisher forecast (velocity): [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/corentinravoux/flip/blob/main/notebook/fisher_velocity.ipynb)

Two more notebooks cover the more experimental subpackages: `flip_mesh.ipynb` (gridding catalogs onto a mesh
with the `*Mesh` data vectors) and `flip_forward.ipynb` (the `flip.forward` field-level forward model, which
needs `jax-cosmo` and `tensorflow-probability`).

## The three steps

Every FLIP analysis follows the same three steps. The [quick look](index.md#quick-look) on the home page
shows the compact version; the pages below cover each step in detail.

### 1. Build a data vector

Wrap your measurements in the appropriate [`DataVector`](DataVector.md) subclass. For directly measured
peculiar velocities:

```python
from flip import data_vector
from flip.data import load_data_test

_, velocity = load_data_test.load_velocity_data()
data = data_vector.DirectVel(velocity)
```

### 2. Compute a covariance

Provide a [power-spectrum dictionary](power_spectra.md) and choose a [covariance model](covariance_models.md):

```python
from flip.covariance import covariance
from flip.data import load_data_test

_, velocity = load_data_test.load_velocity_data()
coordinates_velocity = [velocity["ra"], velocity["dec"], velocity["rcom_zobs"]]
power_spectrum_dict = load_data_test.load_power_spectrum_dict(sigmau_fiducial=15.0)

covariance_fit = covariance.CovMatrix.init_from_flip(
    "carreres23", "velocity", power_spectrum_dict,
    coordinates_velocity=coordinates_velocity,
)
```

### 3. Fit or forecast

Run a [Minuit / MCMC fit or a Fisher forecast](fitting.md):

```python
from flip.covariance import fitter

parameter_dict = {
    "fs8": {"value": 0.4, "limit_low": 0.0, "fixed": False},
    "sigv": {"value": 200.0, "limit_low": 0.0, "fixed": False},
}
minuit_fitter = fitter.FitMinuit.init_from_covariance(
    covariance_fit, data, parameter_dict,
)
minuit_fitter.run()
```

## Command-line scripts

The `scripts/` directory ships ready-to-run examples (invoke directly, they have no pip entry points):

```bash
python scripts/flip_compute_power_spectra.py
python scripts/flip_launch_minuit_velocity_fit.py
python scripts/flip_launch_minuit_density_fit.py
python scripts/flip_launch_minuit_full_fit.py
python scripts/flip_fisher_forecast_velocity.py
```
