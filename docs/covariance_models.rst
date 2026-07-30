Covariance models
==================

The heart of flip is the **covariance matrix** of the density and velocity
fields. Given a :doc:`power spectrum <power_spectra>` and the object
coordinates, flip builds the density-density (``gg``), velocity-velocity
(``vv``) and cross (``gv``) blocks analytically, for any linear power-spectrum
model, and accelerates the computation with a Hankel (FFTLog) transform.

Building a covariance
---------------------

The central object is :py:class:`~flip.covariance.covariance.CovMatrix`. The
usual entry point is ``init_from_flip``, which takes the model name, the field
*kind* (``"velocity"``, ``"density"`` or ``"full"``), the power-spectrum
dictionary and the coordinates of each tracer:

.. code-block:: python

   from flip.covariance import covariance

   covariance_fit = covariance.CovMatrix.init_from_flip(
       "carreres23",                       # model name
       "velocity",                         # kind: velocity / density / full
       power_spectrum_dict,
       coordinates_velocity=coordinates_velocity,   # (3, N): ra, dec, rcom_zobs
       size_batch=10_000,                  # pair-batch size (parallelization)
       number_worker=16,
   )

Alternatively, a :doc:`data vector <DataVector>` can build its own covariance
directly with ``data.compute_covariance(model, power_spectrum_dict, ...)``.

.. note::

   Inside flip, RA/Dec are in **radians**, comoving distances in
   :math:`\mathrm{Mpc}\,h^{-1}`, velocities in km/s and wavenumbers in
   :math:`h\,\mathrm{Mpc}^{-1}`.

Available models
----------------

Each model lives in its own module under :py:mod:`flip.covariance.analytical`
and declares its free parameters, variants and the coordinate keys it needs.

.. list-table::
   :header-rows: 1
   :widths: 22 24 54

   * - Model
     - Fields
     - Notes
   * - ``adamsblake17``
     - density + velocity
     - Adams & Blake (2017), wide-angle
   * - ``adamsblake17plane``
     - density + velocity
     - plane-parallel (flat-sky) variant
   * - ``adamsblake20``
     - density + velocity
     - redshift-dependent (RSD :math:`\beta`)
   * - ``carreres23``
     - velocity
     - **recommended default**; Carreres et al. (2023), arXiv:2303.01198
   * - ``lai22``
     - density + velocity
     - Lai et al. (2022)
   * - ``rcrk24``
     - velocity
     - growth-index / growth-rate parameterization
   * - ``ravouxcarreres``
     - density + velocity
     - **Ravoux et al. (2025), arXiv:2501.16852** — flip flagship model
   * - ``ravouxnoanchor25``
     - velocity
     - no-anchor variant — fits ``H0`` instead of an anchor
   * - ``ravouxqin26``
     - density + velocity
     - latest; non-linear damping + full bias expansion
   * - ``genericzdep``
     - template
     - starting point for a new redshift-dependent model

Variants and free parameters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Most models expose **variants** (selected with the ``variant`` keyword) that
toggle which parameters are free — for example ``adamsblake20`` offers
``baseline`` (with the RSD parameter ``beta_f``) and ``nobeta``. The free
parameters and the ``_variant`` / ``_free_par`` / ``_coordinate_keys`` maps are
documented on each model module in the API reference; the free-parameter names
(``fs8``, ``bs8``, ``sigv``, ...) are exactly the parameters you fit or forecast.

Under the hood
--------------

* :py:mod:`flip.covariance.generator` turns the model's k-space multipole
  kernels and the input power spectra into the real-space covariance blocks,
  using either direct quadrature or the fast Hankel transform.
* :py:mod:`flip.covariance.hankel` implements the FFTLog transform between
  :math:`P(k)` and :math:`\xi_\ell(r)` (adapted from cosmoprimo).
* :py:mod:`flip.covariance.cov_utils` provides the pair-separation geometry
  under several line-of-sight conventions.
* :py:mod:`flip.covariance.contraction` predicts the model covariance as a
  function of separation (handy for plotting and validating a model without a
  catalog).

The ``flip_terms.py``, ``coefficients.py`` and ``fisher_terms.py`` files inside
each model are **generated** offline by :py:mod:`flip.covariance.symbolic` and
are not part of the everyday API.

Emulators
---------

For expensive models, the covariance can be replaced by a fast surrogate trained
over the model parameters. flip ships Gaussian-process
(:py:mod:`flip.covariance.emulators.gpmatrix` and
:py:mod:`flip.covariance.emulators.skgpmatrix`) and neural-network
(:py:mod:`flip.covariance.emulators.nnmatrix`) backends, driven through
``CovMatrix.init_from_emulator``.
