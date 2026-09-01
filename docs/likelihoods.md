# Likelihoods

Once a [covariance](covariance_models.md) and a [data vector](DataVector.md) are in hand, FLIP forms a
**multivariate Gaussian likelihood** and hands it to a [fitter](fitting.md). The likelihood classes live in
[`flip.covariance.likelihood`][].

## The Gaussian likelihood

For a data vector $\mathbf{d}$ (velocities and/or densities) with total covariance $C(\theta)$, the
log-likelihood is

$$
-2\ln\mathcal{L}(\theta) = \mathbf{d}^{\mathsf T} C(\theta)^{-1}\mathbf{d}
                         + \ln\lvert C(\theta)\rvert
                         + N\ln(2\pi) ,
$$

where the total covariance combines the model covariance with the data variance, e.g. for velocities

$$
C = C_{vv} + \sigma_v^2\,\mathbb{I}_N .
$$

FLIP minimizes $-\ln\mathcal{L}$ (or its negative, controlled by the `negative_log_likelihood` property).

## Constructing a likelihood

A likelihood is usually built from a covariance and data with `init_from_covariance`; in practice the
[fitters](fitting.md) do this for you (passing a full `parameter_dict` of values, errors and limits), but you
can also build one directly by giving the ordered list of free-parameter names:

```python
from flip.covariance.likelihood import MultivariateGaussianLikelihood

likelihood = MultivariateGaussianLikelihood.init_from_covariance(
    covariance_fit,
    data,
    parameter_names=["fs8", "sigv"],
    likelihood_properties={"inversion_method": "cholesky_inverse"},
)

# the likelihood is callable on a parameter vector (ordered as parameter_names)
minus_log_like = likelihood([0.4, 200.0])
```

## Likelihood types

| Type key | Class | Use |
| --- | --- | --- |
| `multivariate_gaussian` | [`MultivariateGaussianLikelihood`][flip.covariance.likelihood.MultivariateGaussianLikelihood] | standard Gaussian likelihood |
| `multivariate_gaussian_interp1d` | [`MultivariateGaussianLikelihoodInterpolate1D`][flip.covariance.likelihood.MultivariateGaussianLikelihoodInterpolate1D] | interpolate the covariance over **one** parameter (e.g. `sigma_g`) |
| `multivariate_gaussian_interp2d` | [`MultivariateGaussianLikelihoodInterpolate2D`][flip.covariance.likelihood.MultivariateGaussianLikelihoodInterpolate2D] | interpolate over **two** parameters (deprecated — relies on the removed `scipy.interpolate.interp2d`) |

The interpolating variant pre-computes the covariance on a grid of one nuisance parameter and interpolates at
fit time, which is much faster when that parameter would otherwise force a full covariance rebuild at every
step.

## Covariance inversion

The `inversion_method` likelihood property trades speed against numerical stability when applying $C^{-1}$:

- `inverse` — explicit matrix inverse;
- `solve` — solve the linear system instead of inverting;
- `cholesky` — Cholesky-based factorization;
- `cholesky_inverse` — Cholesky, falling back to the explicit inverse if the matrix isn't positive-definite;
- `cholesky_regularized` — Cholesky after clipping negative eigenvalues to their absolute value.

Each method has a JAX-jitted variant used automatically when JAX is enabled and the `use_jit` likelihood
property is set.

## Priors

Per-parameter priors are attached through `likelihood_properties["prior"]`, a dict mapping parameter name to
prior settings, and evaluated alongside the Gaussian term. FLIP provides
[`GaussianPrior`][flip.utils.GaussianPrior] (`mean`, `standard_deviation`),
[`PositivePrior`][flip.utils.PositivePrior] and [`UniformPrior`][flip.utils.UniformPrior] (`range`), built via
[`return_prior`][flip.utils.return_prior]:

```python
likelihood_properties = {
    "prior": {
        "sigma_M": {"type": "gaussian", "mean": 0.1, "standard_deviation": 0.02},
    },
}
```
