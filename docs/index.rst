
Welcome to flip's documentation!
=================================

**flip** (Field Level Inference Package) fits the **growth rate of structure**
from the **peculiar-velocity** and **density** fields with a maximum-likelihood,
field-level method (Ravoux et al. 2025, `arXiv:2501.16852
<https://arxiv.org/abs/2501.16852>`_; building on Carreres et al. 2023,
`arXiv:2303.01198 <https://arxiv.org/abs/2303.01198>`_).

It is organised as a three-stage pipeline:

#. **Data representation** — wrap velocity and/or density measurements in a
   :py:class:`~flip.data_vector.basic.DataVector`, which turns raw observables
   (direct velocities, Hubble-diagram residuals, SN Ia SALT2 fits, Tully-Fisher,
   Fundamental Plane, log-distance ratios, gridded fields, ...) into a data
   vector and its variance.
#. **Covariance computation** — build the model covariance matrix
   (:py:class:`~flip.covariance.covariance.CovMatrix`) from a linear power
   spectrum and the object coordinates. This step works for any linear
   power-spectrum model — for velocities, densities and their cross-term — and
   is optimised with a Hankel (FFTLog) transform.
#. **Statistical inference** — multiply the covariance by the data to form a
   likelihood and fit the growth rate with the integrated Minuit / MCMC fitters,
   or forecast it with a Fisher matrix.

Quick install
-------------

.. code:: bash

   git clone https://github.com/corentinravoux/flip.git
   cd flip
   pip install .

Quick look
----------

.. code-block:: python

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

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation.rst
   basicusage.rst
   DataVector.rst
   vel_estimators.rst
   power_spectra.rst
   covariance_models.rst
   likelihoods.rst
   fitting.rst

The complete **API reference** — auto-generated from the in-code docstrings for
every mature subpackage — is listed under **API Reference** in the sidebar.


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
