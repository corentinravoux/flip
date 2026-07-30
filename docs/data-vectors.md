# Data vectors

The first step of any flip analysis is to wrap your measurements in a
**`DataVector`**. A `DataVector` knows how to turn raw observables into

- a **data array** and its **variance / covariance**
  (`give_data_and_variance`), and
- a model **covariance matrix** for a chosen model
  (`compute_covariance`, see [Covariance models](covariance-models.md)).

All concrete data vectors derive from the
[`DataVector`](api/data_vector.md) abstract base class, which handles key
validation, optional JAX acceleration, host grouping and observation-covariance
masking. Each subclass declares the fields it needs (`needed_keys`) and the model
parameters it introduces (`free_par`).

```python
data, var_data = DataVector(parameter_dic)   # data + variance

Cov = DataVector.compute_covariance(
    model_name,
    power_spectrum_dict,
    size_batch=size_batch,        # parallelization
    number_worker=number_worker,
    additional_parameters_values=(),
)
```

## Choosing a data vector

| Kind | Class | Input observable | Module |
|---|---|---|---|
| density | `Dens` | density contrast `density` (+ error) | `data_vector.basic` |
| density | `DensMesh` | catalog gridded onto a mesh | `data_vector.basic` |
| density | `GWDensMesh` | gravitational-wave localization kernels | `data_vector.gw_vectors` |
| velocity | `DirectVel` | directly measured velocities `velocity` | `data_vector.basic` |
| velocity | `DirectVelMesh` | velocity catalog gridded onto a mesh | `data_vector.basic` |
| velocity | `VelFromHDres` | Hubble-diagram residuals `dmu` | `data_vector.basic` |
| velocity | `VelFromIntrinsicScatter` | intrinsic magnitude scatter `sigma_M` | `data_vector.basic` |
| velocity | `VelTrippRelation` | SN Ia SALT2 params (`mb`, `x1`, `c`) | `data_vector.snia_vectors` |
| velocity | `VelCandleStandardized` | standardized candle magnitude `mb` | `data_vector.snia_vectors` |
| velocity | `VelFromLogDist` | log-distance ratio `eta` | `data_vector.galaxypv_vectors` |
| velocity | `VelFromTullyFisher` | Tully-Fisher (`logW`, `m_mean`) | `data_vector.galaxypv_vectors` |
| velocity | `VelFromFundamentalPlane` | Fundamental Plane (`logRe`, `logsig`, `logI`) | `data_vector.galaxypv_vectors` |
| cross | `DensVel` | a density **and** a velocity vector | `data_vector.basic` |

## Density

The [`Dens`](api/data_vector.md) class wraps a measured density field:

```python
import pandas as pd
from flip import data_vector

grid = pd.read_parquet("flip/data/data_density.parquet")
grid.rename(columns={"density_err": "density_error", "rcom": "rcom_zobs"},
            inplace=True)
data_density = data_vector.Dens(grid.to_dict(orient="list"))
```

`DensMesh` additionally builds the density field by gridding an object catalog
onto a Cartesian mesh (`DensMesh.init_from_catalog`), and `GWDensMesh` grids
probabilistic gravitational-wave localizations.

## Velocity

### Direct velocities

The [`DirectVel`](api/data_vector.md) class takes peculiar velocities (km/s)
directly:

```python
import numpy as np
import pandas as pd
from flip import data_vector

data_velocity = pd.read_parquet("flip/data/data_velocity.parquet")
data_velocity.rename(columns={"vpec": "velocity"}, inplace=True)
data_velocity["velocity_error"] = np.zeros(len(data_velocity))

data_true_vel = data_vector.DirectVel(data_velocity.to_dict(orient="list"))
```

If the data contain a `host_group_id` column, velocities sharing a host are
averaged automatically (useful e.g. for several SNe Ia in the same galaxy).

### Velocities from Hubble-diagram residuals

[`VelFromHDres`](api/data_vector.md) converts distance-modulus residuals
`dmu` ($\Delta\mu$) into velocities with a redshift-dependent estimator
$\hat{v} = J(z)\,(\Delta\mu - M_0)$. The estimators are described in
[Velocity estimators](velocity-estimators.md).

```python
from flip import data_vector

data_vel = data_vector.VelFromHDres(
    data, velocity_estimator=estimator_name, **kwargs,
)
```

### Velocities from SN Ia SALT2 parameters

[`VelTrippRelation`](api/data_vector.md) standardizes the SALT2 fit
parameters `mb`, `x1`, `c` (with their errors and covariances) through the Tripp
relation and always requires the `rcom_zobs` field:

$$
\Delta\mu = m_b + \alpha x_1 - \beta c - M_0 - 5\log_{10}\!\left[(1+z)\,r(z)\right] - 25
$$

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

When the object is *called*, the SN Ia standardization parameters are passed as a
dictionary:

```python
test_parameters = {
    "alpha": 0.14,
    "beta": 3.1,
    "M_0": -19.133,
    "sigma_M": 0.12,
}
velocity, velocity_error = data_vel(test_parameters)
```

!!! note
    `VelFromSALTfit` is kept as a backward-compatible alias of
    `VelTrippRelation`.

### Other distance indicators

- [`VelFromLogDist`](api/data_vector.md) — from log-distance ratios `eta`.
- [`VelFromTullyFisher`](api/data_vector.md) — from the Tully-Fisher relation.
- [`VelFromFundamentalPlane`](api/data_vector.md) — from the Fundamental Plane.
- [`VelFromIntrinsicScatter`](api/data_vector.md) — the velocity noise induced
  by the intrinsic magnitude scatter `sigma_M`.

## Density × velocity

[`DensVel`](api/data_vector.md) combines a density vector and a velocity
vector so a single fit uses the density-density, velocity-velocity **and**
density-velocity cross covariance blocks:

```python
from flip import data_vector

dens_cross_vel = data_vector.DensVel(data_density, data_true_vel)
```

See the [API reference](api/data_vector.md) for the full signatures.
