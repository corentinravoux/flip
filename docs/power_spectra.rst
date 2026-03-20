Power Spectra Generator
=======================

The :py:mod:`flip.power_spectra` module provides tools to compute matter, momentum, and velocity power spectra
using various Boltzmann solver engines. This is a common module shared across all methods in the package.


Computing power spectra
-----------------------

You can use the :py:func:`flip.power_spectra.compute_power_spectra` function to compute power spectra:

.. code-block:: python

    from flip.power_spectra import compute_power_spectra

    k, pk_dict = compute_power_spectra(
        power_spectrum_engine,
        power_spectrum_settings,
        redshift,
        minimal_wavenumber,
        maximal_wavenumber,
        number_points,
        logspace=True,
        normalize_power_spectrum=True,
        power_spectrum_non_linear_model=None,
        power_spectrum_model="linearbel",
        save_path=None,
    )


Available engines
-----------------

The following Boltzmann solver engines are supported:

- **cosmoprimo** — via :py:mod:`flip.power_spectra.cosmoprimo_engine`
- **CLASS** — via :py:mod:`flip.power_spectra.class_engine`
- **pyccl** — via :py:mod:`flip.power_spectra.pyccl_engine`


Power spectrum models
---------------------

Several models are available for decomposing the matter power spectrum into velocity-related components:

- **linearbel** — Linear power spectrum with BEL damping model applied to compute :math:`P_{m\theta}(k)` and :math:`P_{\theta\theta}(k)` from the linear :math:`P_{mm}(k)`.
- **nonlinearbel** — Uses an external non-linear :math:`P_{mm}(k)` combined with BEL damping for :math:`P_{m\theta}(k)` and :math:`P_{\theta\theta}(k)`.
- **linear** — Direct linear power spectrum without BEL correction.

The BEL model coefficients are parameterized as a function of :math:`\sigma_8` following Bel et al.
