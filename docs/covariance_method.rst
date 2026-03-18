Covariance method
=================

The covariance method is the core approach implemented in **flip**. It is a likelihood-based field-level inference method
for fitting the growth rate of structures based on velocity and density fields. The full description is given in
`Ravoux et al. 2025 <https://arxiv.org/abs/2501.16852>`_.

Overview
--------

The method works in three steps:

1. **Covariance matrix computation**: A model covariance matrix is computed from a model power spectrum and the
   considered object coordinates. This part is generalized to work for any linear power spectrum model, both for
   velocities, densities, and cross-terms. It is optimized with Hankel transforms for any model.

2. **Likelihood construction**: The covariance is used to build a likelihood by combining it with velocity or density
   data vectors. See `Likelihoods <likelihoods.html>`_.

3. **Parameter fitting**: The package includes integrated fitters such as Minuit and MCMC (with emcee) to fit the
   growth rate of structures.


Subpackage contents
-------------------

The :py:mod:`flip.covariance` subpackage contains:

- :py:mod:`flip.covariance.covariance` — The :py:class:`~flip.covariance.covariance.CovMatrix` class for building and manipulating covariance matrices.
- :py:mod:`flip.covariance.likelihood` — Likelihood functions (multivariate Gaussian with various inversion methods).
- :py:mod:`flip.covariance.fitter` — Fitter classes wrapping Minuit and MCMC samplers.
- :py:mod:`flip.covariance.fisher` — Fisher matrix computation from covariance derivatives.
- :py:mod:`flip.covariance.analytical` — Analytical covariance models (e.g., Adams & Blake 2017, Carreres 2023, Lai 2022, RCRK 2024).
- :py:mod:`flip.covariance.emulators` — Covariance emulators using Gaussian processes and neural networks.
- :py:mod:`flip.covariance.generator` — Covariance matrix generation utilities.
- :py:mod:`flip.covariance.symbolic` — Symbolic computation of covariance terms.

Analytical models
~~~~~~~~~~~~~~~~~

Several analytical covariance models are available:

- **adamsblake17** / **adamsblake17plane**: Adams & Blake 2017 models.
- **adamsblake20**: Adams & Blake 2020 model.
- **carreres23**: Carreres et al. 2023 model.
- **lai22**: Lai et al. 2022 model.
- **rcrk24**: RCRK 2024 model.
- **genericzdep**: Generic redshift-dependent model.
- **ravouxcarreres**, **ravouxnoanchor25**, **ravouxqin26**: Additional models.

Usage
-----

The typical workflow involves:

1. Computing or loading power spectra (see `Power Spectra <power_spectra.html>`_).
2. Building data vectors (see `Data Vectors <DataVector.html>`_).
3. Computing the covariance matrix using :py:class:`~flip.covariance.covariance.CovMatrix`.
4. Running the fitter to constrain cosmological parameters.

A minimal example:

.. code-block:: python

    from flip import covariance, data_vector, power_spectra

    # 1. Load or compute power spectra
    power_spectrum_dict = ...

    # 2. Build the data vector
    data = data_vector.DirectVel(velocity_data)

    # 3. Compute covariance matrix
    cov = data.compute_covariance(
        model_name="carreres23",
        power_spectrum_dict=power_spectrum_dict,
    )

    # 4. Fit using Minuit
    fitter = covariance.fitter.MinuitFitter.init_from_covariance(
        covariance=cov,
        data=data,
    )
    result = fitter.minimize(parameter_dict)

For complete examples, see `Getting started <basicusage.html>`_.
