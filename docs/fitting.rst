Fitting & forecasts
===================

The final step is to infer the growth rate (and nuisance parameters) from the
:doc:`likelihood <likelihoods>`. flip ships integrated **maximum-likelihood** and
**MCMC** fitters, plus a **Fisher** forecaster. They all live in
:py:mod:`flip.covariance.fitter` and :py:mod:`flip.covariance.fisher`.

Maximum likelihood (Minuit)
---------------------------

:py:class:`~flip.covariance.fitter.FitMinuit` wraps ``iminuit``. Build it from a
covariance, a data vector and a ``parameter_dict`` describing each parameter's
starting value, limits and whether it is fixed, then call ``run``:

.. code-block:: python

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

``run(migrad=True, hesse=False, minos=False, n_iter=1)`` exposes the usual Minuit
stages.

MCMC (emcee)
------------

:py:class:`~flip.covariance.fitter.FitMCMC` drives an ``emcee`` ensemble sampler
through the same ``init_from_covariance`` interface, then samples with
``run_chains``:

.. code-block:: python

   from flip.covariance import fitter

   mcmc_fitter = fitter.FitMCMC.init_from_covariance(
       covariance_fit, data, parameter_dict,
   )
   mcmc_fitter.run_chains(nsteps=5000)

The underlying :py:class:`~flip.covariance.fitter.EMCEESampler` supports an HDF5
backend for checkpointing and resuming long chains, and a run-until-convergence
helper.

Fisher forecasts
----------------

:py:class:`~flip.covariance.fisher.FisherMatrix` predicts parameter
uncertainties without a fit, from the Gaussian Fisher information

.. math::

   F_{ij} = \tfrac{1}{2}\,\mathrm{Tr}\!\left[
       C^{-1}\,\partial_i C\; C^{-1}\,\partial_j C \right].

Here the ``parameter_dict`` is a plain mapping of fiducial values:

.. code-block:: python

   from flip.covariance import fisher

   parameter_dict = {"fs8": 0.4, "sigv": 200.0, "sigma_M": 0.12}

   Fisher = fisher.FisherMatrix.init_from_covariance(
       covariance_fit, data, parameter_dict,
       fisher_properties={"inversion_method": "inverse"},
   )
   parameter_name_list, fisher_matrix = Fisher.compute_fisher_matrix()

The parameter covariance is then :math:`F^{-1}`, and the :math:`1\sigma` forecast
on each parameter is :math:`\sqrt{(F^{-1})_{ii}}`.

Ready-made recipes
------------------

For the common cases, :py:mod:`flip.covariance.fit_utils` bundles the covariance,
data and likelihood setup into one-call Minuit drivers — density-only,
true-velocity and estimated-velocity fits, each with an optional variant that
interpolates the likelihood over the small-scale dispersion parameter for speed.

Diagnostics
-----------

:py:mod:`flip.covariance.plot_utils` provides diagnostic plots for the covariance
(1D/2D model contractions, the correlation implied by a likelihood) and for
batches of fits (individual and mean best-fit parameters).

Emulated covariances
--------------------

When the covariance itself is expensive, train a fast surrogate over the model
parameters and fit against it via ``CovMatrix.init_from_emulator`` — see
:doc:`covariance_models`. The fitter and Fisher interfaces are unchanged.
