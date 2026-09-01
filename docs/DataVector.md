# Data vectors

The first step of any FLIP analysis is to wrap your measurements in a **DataVector**. A
[`DataVector`][flip.data_vector.basic.DataVector] knows how to turn raw observables into

- a **data array** and its **variance / covariance**
  ([`give_data_and_variance`][flip.data_vector.basic.DataVector.give_data_and_variance]), and
- a model **covariance matrix** for a chosen model
  ([`compute_covariance`][flip.data_vector.basic.DataVector.compute_covariance], see
  [Covariance models](covariance_models.md)).

All concrete data vectors derive from the [`DataVector`][flip.data_vector.basic.DataVector] abstract base
class, which handles key validation, optional JAX acceleration, host grouping and observation-covariance
masking. Each subclass declares the fields it needs (`needed_keys`) and the model parameters it introduces
(`free_par`). `DataVector` cannot be instantiated directly — use one of its subclasses below.

```python
data, var_data = data.give_data_and_variance(parameter_values_dict)   # data + variance/covariance

Cov = data.compute_covariance(
    model,                         # covariance model name, e.g. "carreres23"
    power_spectrum_dict,
    size_batch=size_batch,         # parallelization
    number_worker=number_worker,
    additional_parameters_values=None,
)
```

## Choosing a data vector

| Kind | Class | Input observable | Module |
| --- | --- | --- | --- |
| density | [`Dens`][flip.data_vector.basic.Dens] | density contrast `density` (+ error) | `data_vector.basic` |
| density | [`DensMesh`][flip.data_vector.basic.DensMesh] | catalog gridded onto a mesh | `data_vector.basic` |
| density | [`GWDensMesh`][flip.data_vector.gw_vectors.GWDensMesh] | gravitational-wave localization kernels | `data_vector.gw_vectors` |
| velocity | [`DirectVel`][flip.data_vector.basic.DirectVel] | directly measured velocities `velocity` | `data_vector.basic` |
| velocity | [`DirectVelMesh`][flip.data_vector.basic.DirectVelMesh] | velocity catalog gridded onto a mesh | `data_vector.basic` |
| velocity | [`VelFromHDres`][flip.data_vector.basic.VelFromHDres] | Hubble-diagram residuals `dmu` | `data_vector.basic` |
| velocity | [`VelFromIntrinsicScatter`][flip.data_vector.basic.VelFromIntrinsicScatter] | intrinsic magnitude scatter `sigma_M` | `data_vector.basic` |
| velocity | [`VelTrippRelation`][flip.data_vector.snia_vectors.VelTrippRelation] | SN Ia SALT2 params (`mb`, `x1`, `c`) | `data_vector.snia_vectors` |
| velocity | [`VelCandleStandardized`][flip.data_vector.snia_vectors.VelCandleStandardized] | standardized candle magnitude `mb` | `data_vector.snia_vectors` |
| velocity | [`VelFromLogDist`][flip.data_vector.galaxypv_vectors.VelFromLogDist] | log-distance ratio `eta` | `data_vector.galaxypv_vectors` |
| velocity | [`VelFromTullyFisher`][flip.data_vector.galaxypv_vectors.VelFromTullyFisher] | Tully-Fisher (`logW`, `m_mean`) | `data_vector.galaxypv_vectors` |
| velocity | [`VelFromFundamentalPlane`][flip.data_vector.galaxypv_vectors.VelFromFundamentalPlane] | Fundamental Plane (`logRe`, `logsig`, `logI`) | `data_vector.galaxypv_vectors` |
| cross | [`DensVel`][flip.data_vector.basic.DensVel] | a density **and** a velocity vector | `data_vector.basic` |

`VelFromSALTfit` is kept as a backward-compatible alias of `VelTrippRelation`.

## Density

The [`Dens`][flip.data_vector.basic.Dens] class wraps a measured density field:

```python
import pandas as pd
from flip import data_vector

grid = pd.read_parquet("flip/data/data_density.parquet")
grid.rename(columns={"density_err": "density_error", "rcom": "rcom_zobs"}, inplace=True)
data_density = data_vector.Dens(grid.to_dict(orient="list"))
```

[`DensMesh`][flip.data_vector.basic.DensMesh] additionally builds the density field by gridding an object
catalog onto a Cartesian mesh (`DensMesh.init_from_catalog`), and
[`GWDensMesh`][flip.data_vector.gw_vectors.GWDensMesh] grids probabilistic gravitational-wave localizations.

## Velocity

### Direct velocities

The [`DirectVel`][flip.data_vector.basic.DirectVel] class takes peculiar velocities (km/s) directly:

```python
import numpy as np
import pandas as pd
from flip import data_vector

data_velocity = pd.read_parquet("flip/data/data_velocity.parquet")
data_velocity.rename(columns={"vpec": "velocity"}, inplace=True)
data_velocity["velocity_error"] = np.zeros(len(data_velocity))

data_true_vel = data_vector.DirectVel(data_velocity.to_dict(orient="list"))
```

If the data contain a `host_group_id` column, velocities sharing a host are averaged automatically (useful
e.g. for several SNe Ia in the same galaxy).

### Velocities from Hubble-diagram residuals

[`VelFromHDres`][flip.data_vector.basic.VelFromHDres] converts distance-modulus residuals `dmu` ($\Delta\mu$)
into velocities with a redshift-dependent estimator, minus an `M_0` offset that is a free parameter fit at
evaluation time. The estimators are described in [Velocity estimators](vel_estimators.md).

```python
from flip import data_vector

data_vel = data_vector.VelFromHDres(data, velocity_estimator=estimator_name)
```

### Velocities from SN Ia SALT2 parameters

[`VelTrippRelation`][flip.data_vector.snia_vectors.VelTrippRelation] standardizes the SALT2 fit parameters
`mb`, `x1`, `c` (with their errors and covariances) through the Tripp relation and always requires the
`rcom_zobs` field and a `h` (little-h) argument to convert it from Mpc/h to Mpc:

$$
\Delta\mu = m_b + \alpha x_1 - \beta c - M_0 - 5\log_{10}\left[(1+z)\,r(z)/h\right] - 25
$$

If the data contain a `host_logmass` field, an additional mass-step free parameter `gamma` is added (split at
`host_logmass > 10`).

```python
import pandas as pd
from flip import data_vector

data_velocity = pd.read_parquet("flip/data/data_velocity.parquet")
data_vel = data_vector.snia_vectors.VelTrippRelation(
    data_velocity.to_dict(orient="list"),
    h=0.7,
    velocity_estimator="full",
)

mu = data_vel.compute_observed_distance_modulus(test_parameters)
variance_mu = data_vel.compute_observed_distance_modulus_variance(test_parameters)
```

To get the velocities and their variance, use `give_data_and_variance` with the SN Ia standardization
parameters:

```python
test_parameters = {
    "alpha": 0.14,
    "beta": 3.1,
    "M_0": -19.133,
    "sigma_M": 0.12,
}
velocity, velocity_variance = data_vel.give_data_and_variance(test_parameters)
```

[`VelCandleStandardized`][flip.data_vector.snia_vectors.VelCandleStandardized] is a simpler variant for a
magnitude `mb` already standardized to a fixed absolute magnitude (only needs `M_0` and `sigma_M`).

### Other distance indicators

- [`VelFromLogDist`][flip.data_vector.galaxypv_vectors.VelFromLogDist] — from log-distance ratios `eta`.
- [`VelFromTullyFisher`][flip.data_vector.galaxypv_vectors.VelFromTullyFisher] — from the Tully-Fisher
  relation.
- [`VelFromFundamentalPlane`][flip.data_vector.galaxypv_vectors.VelFromFundamentalPlane] — from the
  Fundamental Plane.
- [`VelFromIntrinsicScatter`][flip.data_vector.basic.VelFromIntrinsicScatter] — the velocity noise induced by
  the intrinsic magnitude scatter `sigma_M`.

## Density x velocity

[`DensVel`][flip.data_vector.basic.DensVel] combines a density vector and a velocity vector so a single fit
uses the density-density, velocity-velocity **and** density-velocity cross covariance blocks:

```python
from flip import data_vector

dens_cross_vel = data_vector.DensVel(data_density, data_true_vel)
```

`DensVel` does not support a velocity vector built with its own observation covariance yet (raises
`NotImplementedError`).
