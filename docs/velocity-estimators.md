# Velocity estimators

The velocity data vectors (e.g. [`VelFromHDres`](api/data_vector.md),
`VelTrippRelation`, `VelFromTullyFisher`, ...) convert a Hubble-diagram residual
$\Delta\mu$ into a peculiar velocity through a redshift-dependent coefficient
$J(z)$:

$$
\hat{v} = J(z)\,\Delta\mu .
$$

The coefficient is computed by
[`redshift_dependence_velocity`](api/data_vector.md) and selected with the
`velocity_estimator` argument. The available keys are `"watkins"`, `"lowz"`,
`"hubblehighorder"`, `"full"` and `"full_lcdm"`.

## Watkins

The estimator of
[Watkins & Feldman (2015)](http://academic.oup.com/mnras/article/450/2/1868/980317/An-unbiased-estimator-of-peculiar-velocity-with):

$$
J(z) = \frac{c\ln 10}{5}\,\frac{z}{1+z} .
$$

## Low-z

$$
J(z) = \frac{c\ln 10}{5}\,z .
$$

## Hubble high-order (`hubblehighorder`)

A third-order expansion in $z$ of the Hubble law:

$$
J(z) = \frac{c\ln 10}{5}\,\frac{z}{1+z}
\left[\,1 + \tfrac{1}{2}(1 - q_0)\,z
       - \tfrac{1}{6}(1 - q_0 - 3q_0^2 + j_0)\,z^2 \right] .
$$

This estimator requires the deceleration $q_0$ and jerk $j_0$ parameters, passed
as keyword arguments:

```python
from flip import data_vector

data_vel = data_vector.VelFromHDres(
    data, velocity_estimator="hubblehighorder", q0=-0.55, j0=-1,
)
```

## Full

The `full` estimator assumes a cosmology through the data:

$$
J(z) = \frac{c\ln 10}{5}
\left(c\,\frac{1+z}{r(z)\,H(z)} - 1\right)^{-1} .
$$

Your data must then contain the `hubble_norm` and `rcom_zobs` fields, where
`hubble_norm` is $h(z) = H(z)/100$ and `rcom_zobs` is the comoving distance in
$\mathrm{Mpc}\,h^{-1}$.

## Full ΛCDM (`full_lcdm`)

Same functional form as `full`, but $r(z)$ and $H(z)$ are computed on the fly
from a flat ΛCDM cosmology; pass `H0` and `Omega_m0` as keyword arguments instead
of pre-computing `hubble_norm` / `rcom_zobs`:

```python
data_vel = data_vector.VelFromHDres(
    data, velocity_estimator="full_lcdm", H0=70.0, Omega_m0=0.3,
)
```
