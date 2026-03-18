Package overview
================

**flip** (Field Level Inference Package) is a Python package designed for the analysis of velocity and density fields in cosmology.
The package provides tools for fitting the growth rate of structures using different methodological approaches.

The full description of the core concepts of this package is given in `Ravoux et al. 2025 <https://arxiv.org/abs/2501.16852>`_.

Package structure
-----------------

The package is decomposed into several subpackages. Some are **common modules** shared across all methods,
while others implement specific **methodological approaches** to deal with velocity and density fields.

Common modules
~~~~~~~~~~~~~~

These modules are shared across all methods:

- :py:mod:`flip.power_spectra` — Power spectrum generation using various Boltzmann solver engines (CLASS, cosmoprimo, pyccl).
  See `Power Spectra <power_spectra.html>`_.

- :py:mod:`flip.data_vector` — Data vector classes for handling velocity, density, and joint data.
  See `Data Vectors <DataVector.html>`_.

- :py:mod:`flip.data` — Built-in test data and loading utilities.


Methods
~~~~~~~

The package currently supports or plans to support the following approaches:

1. **Likelihood-based field-level inference** (:py:mod:`flip.covariance`) — The core method of the package, using covariance matrices
   built from model power spectra and object coordinates. See `Covariance method <covariance_method.html>`_.

2. **Forward modeling simulation** (:py:mod:`flip.simulation`) — A simulation-based approach currently under construction.
   See `Simulation method <simulation.html>`_.

3. **Field comparison** — A direct field comparison method, not yet implemented.
   See `Field comparison method <field_comparison.html>`_.


How to cite
-----------

The full description of the core concepts of this package is given in `Ravoux et al. 2025 <https://arxiv.org/abs/2501.16852>`_.
This package was started on the previous work of `Carreres et al. 2023 <https://arxiv.org/abs/2303.01198>`_.
Please cite both papers when using the package.
