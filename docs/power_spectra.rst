Power Spectra Generator
=======================

You can use the `flip.power_spectra` module to compute your power spectra using the `flip.power_spectra.compute_power_spectra` function:

```
k, pdd, pdt, ptt = compute_power_spectra(
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
python```
