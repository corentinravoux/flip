Simulation method
=================

The simulation method implements a forward modeling approach for velocity and density fields.
This method is part of the :py:mod:`flip.simulation` subpackage.

.. note::

   This method is currently **under construction**. The subpackage structure is in place but the
   implementation is being developed in collaboration with other contributors.

Overview
--------

The forward modeling simulation approach provides an alternative to the covariance-based method
by directly simulating velocity and density fields. This allows for more flexible modeling of
non-linear effects and complex survey geometries.

The simulation method shares the common modules of the **flip** package:

- :py:mod:`flip.power_spectra` for power spectrum computation.
- :py:mod:`flip.data_vector` for data vector handling.
- :py:mod:`flip.data` for test data.

Optional dependencies
---------------------

The simulation method requires additional dependencies that can be installed via:

.. code-block:: bash

   pip install flipcosmo[simulation]

This installs JAX, JAXpm, JAXopt, jax-cosmo, and diffrax for differentiable simulation support.
