Likelihoods
===========

The likelihoods are implemented as part of the `covariance method <covariance_method.html>`_
in the :py:mod:`flip.covariance.likelihood` module.


Velocity alone
--------------

Standard velocity likelihood:

.. math::

    \mathcal{L} = (2\pi)^{-\frac{N}{2}}|C|^{-\frac{1}{2}} \exp\left[-\frac{1}{2} \mathbf{v}^T C^{-1}\mathbf{v}\right]

where :math:`C` is such that:

.. math::

    C = C_\mathrm{vv} + \sigma_v^2 \mathbb{I}_n


Inversion methods
-----------------

Several methods are available for computing the log-likelihood, differing in how the covariance matrix is inverted:

- **inverse** — Explicit matrix inversion.
- **solve** — Linear solver (avoids explicit inversion).
- **cholesky** — Cholesky decomposition.
- **cholesky_regularized** — Regularized Cholesky decomposition.
- **cholesky_inverse** — Cholesky-based inversion.
