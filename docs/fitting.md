# Fitting & forecasts

The final step is to infer the growth rate (and nuisance parameters) from the
[likelihood](likelihoods.md). flip ships integrated **maximum-likelihood** and
**MCMC** fitters, plus a **Fisher** forecaster. They all live in
[`flip.covariance.fitter`](api/covariance.md) and
[`flip.covariance.fisher`](api/covariance.md).

## Maximum likelihood (Minuit)

[`FitMinuit`](api/covariance.md) wraps `iminuit`. Build it from a covariance,
a data vector and a `parameter_dict` describing each parameter's starting value,
limits and whether it is fixed, then call `run`:

```python
from flip.covariance import fitter

parameter_dict = {
    "fs8":  {"value": 0.4,   "limit_low": 0.0, "fixed": False},
    "sigv": {"value": 200.0, "limit_low": 0.0, "fixed": False},
}

minuit_fitter = fitter.FitMinuit.init_from_covariance(
    covariance_fit,
    data,
    parameter_dict,
    likelihood_type="multivariate_gaussian",
    likelihood_properties={"inversion_method": "cholesky_inverse"},
)
minuit_fitter.run()          # migrad (+ optional hesse / minos)

print(minuit_fitter.minuit.values, minuit_fitter.minuit.errors)
```

`run(migrad=True, hesse=False, minos=False, n_iter=1)` exposes the usual Minuit
stages.

## MCMC (emcee)

[`FitMCMC`](api/covariance.md) drives an `emcee` ensemble sampler through the
same `init_from_covariance` interface, then samples with `run_chains`:

```python
from flip.covariance import fitter

mcmc_fitter = fitter.FitMCMC.init_from_covariance(
    covariance_fit, data, parameter_dict,
)
mcmc_fitter.run_chains(nsteps=5000)
```

The underlying [`EMCEESampler`](api/covariance.md) supports an HDF5 backend for
checkpointing and resuming long chains, and a run-until-convergence helper.

## Fisher forecasts

[`FisherMatrix`](api/covariance.md) predicts parameter uncertainties without a
fit, from the Gaussian Fisher information

$$
F_{ij} = \tfrac{1}{2}\,\mathrm{Tr}\!\left[
    C^{-1}\,\partial_i C\; C^{-1}\,\partial_j C \right].
$$

Here the `parameter_dict` is a plain mapping of fiducial values:

```python
from flip.covariance import fisher

parameter_dict = {"fs8": 0.4, "sigv": 200.0, "sigma_M": 0.12}

Fisher = fisher.FisherMatrix.init_from_covariance(
    covariance_fit, data, parameter_dict,
    fisher_properties={"inversion_method": "inverse"},
)
parameter_name_list, fisher_matrix = Fisher.compute_fisher_matrix()
```

The parameter covariance is then $F^{-1}$, and the $1\sigma$ forecast on each
parameter is $\sqrt{(F^{-1})_{ii}}$.

## Ready-made recipes

For the common cases, [`fit_utils`](api/covariance.md) bundles the covariance,
data and likelihood setup into one-call Minuit drivers — density-only,
true-velocity and estimated-velocity fits, each with an optional variant that
interpolates the likelihood over the small-scale dispersion parameter for speed.

## Diagnostics

[`plot_utils`](api/covariance.md) provides diagnostic plots for the covariance
(1D/2D model contractions, the correlation implied by a likelihood) and for
batches of fits (individual and mean best-fit parameters).

## Emulated covariances

When the covariance itself is expensive, train a fast surrogate over the model
parameters and fit against it via `CovMatrix.init_from_emulator` — see
[Covariance models](covariance-models.md#emulators). The fitter and Fisher
interfaces are unchanged.
