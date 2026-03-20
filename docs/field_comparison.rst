Field comparison method
=======================

The field comparison method provides a direct comparison approach for velocity and density fields.

.. note::

   This method is **not yet implemented**. It is planned for a future release of the package.

Overview
--------

The field comparison method will offer an alternative approach to the covariance-based likelihood inference
and the forward modeling simulation. It will enable direct comparison of observed fields with theoretical
predictions.

Like the other methods, the field comparison approach will share the common modules of the **flip** package:

- :py:mod:`flip.power_spectra` for power spectrum computation.
- :py:mod:`flip.data_vector` for data vector handling.
- :py:mod:`flip.data` for test data.
