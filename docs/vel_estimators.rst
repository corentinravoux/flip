Velocity estimators
===================

The velocity data vectors (e.g. :py:class:`~flip.data_vector.basic.VelFromHDres`,
:py:class:`~flip.data_vector.snia_vectors.VelTrippRelation`,
:py:class:`~flip.data_vector.galaxypv_vectors.VelFromTullyFisher`, ...) convert a
Hubble-diagram residual :math:`\Delta\mu` into a peculiar velocity through a
redshift-dependent coefficient :math:`J(z)`:

.. math::

   \hat{v} = J(z)\,\Delta\mu .

The coefficient is computed by
:py:func:`~flip.data_vector.vector_utils.redshift_dependence_velocity` and
selected with the ``velocity_estimator`` argument. The available keys are
``"watkins"``, ``"lowz"``, ``"hubblehighorder"``, ``"full"``, ``"full_lcdm"`` and
``"empty_universe"``.

Watkins
-------

The estimator of `Watkins & Feldman 2015
<http://academic.oup.com/mnras/article/450/2/1868/980317/An-unbiased-estimator-of-peculiar-velocity-with>`_:

.. math::

   J(z) = \frac{c\ln 10}{5}\,\frac{z}{1+z} .

Low-z
-----

.. math::

   J(z) = \frac{c\ln 10}{5}\,z .

Hubble high-order (``hubblehighorder``)
---------------------------------------

A third-order expansion in :math:`z` of the Hubble law:

.. math::

   J(z) = \frac{c\ln 10}{5}\,\frac{z}{1+z}
   \left[\,1 + \tfrac{1}{2}(1 - q_0)\,z
          - \tfrac{1}{6}(1 - q_0 - 3q_0^2 + j_0)\,z^2 \right] .

This estimator requires the deceleration :math:`q_0` and jerk :math:`j_0`
parameters, passed as keyword arguments:

.. code-block:: python

   from flip import data_vector

   data_vel = data_vector.VelFromHDres(
       data, velocity_estimator="hubblehighorder", q0=-0.55, j0=-1,
   )

Full
----

The ``full`` estimator assumes a cosmology through the data:

.. math::

   J(z) = \frac{c\ln 10}{5}
   \left(c\,\frac{1+z}{r(z)\,H(z)} - 1\right)^{-1} .

Your data must then contain the ``hubble_norm`` and ``rcom_zobs`` fields, where
``hubble_norm`` is :math:`h(z) = H(z)/100` and ``rcom_zobs`` is the comoving
distance in :math:`\mathrm{Mpc}\,h^{-1}`.

Full LambdaCDM (``full_lcdm``)
------------------------------

Same functional form as ``full``, but :math:`r(z)` and :math:`H(z)` are computed
on the fly from a flat :math:`\Lambda\mathrm{CDM}` cosmology; pass ``H0`` and
``Omega_m0`` as keyword arguments instead of pre-computing ``hubble_norm`` /
``rcom_zobs``:

.. code-block:: python

   data_vel = data_vector.VelFromHDres(
       data, velocity_estimator="full_lcdm", H0=70.0, Omega_m0=0.3,
   )

Empty universe (``empty_universe``)
-----------------------------------

An empty-universe (coasting) approximation — eq. 4 of
`arXiv:1610.04677 <https://arxiv.org/pdf/1610.04677>`_ — which needs no extra
parameters:

.. math::

   J(z) = \frac{c\ln 10}{5}\,\frac{z\,(1 + z/2)}{(1+z)^2} .
