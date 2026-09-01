# Power spectra

The covariance models are built from a set of **model power spectra**: the matter-matter ($P_{gg}$ / mm), the
cross matter-velocity-divergence ($P_{gv}$ / mt) and the velocity-divergence ($P_{vv}$ / tt) spectra. FLIP can
compute these for you from a cosmology engine, and expects them packed into a `power_spectrum_dict` when
constructing a covariance.

## Computing spectra

Use [`compute_power_spectra`][flip.power_spectra.generator.compute_power_spectra] to generate the mm / mt / tt
spectra from a chosen engine and model:

```python
from flip.power_spectra import generator

k, pmm, pmt, ptt = generator.compute_power_spectra(
    power_spectrum_engine,          # "class_engine", "cosmoprimo_engine" or "pyccl_engine"
    power_spectrum_settings,        # engine-specific cosmology settings
    redshift,
    minimal_wavenumber,             # kmin  [h/Mpc]
    maximal_wavenumber,             # kmax  [h/Mpc]
    number_points,                  # number of k samples
    logspace=True,
    normalization_power_spectrum="no_normalization",
    power_spectrum_non_linear_model=None,
    power_spectrum_model="linearbel",
    save_path=None,                 # optionally write the spectra to disk
)
```

### Engines

| Engine | Backend |
| --- | --- |
| `class_engine` | CLASS via `classy` |
| `cosmoprimo_engine` | [cosmoprimo](https://github.com/cosmodesi/cosmoprimo) |
| `pyccl_engine` | the Core Cosmology Library `pyccl` |

### Models

The `power_spectrum_model` prescription turns the engine output into the mm / mt / tt spectra:

- `linear` / `nonlinear` — return the (non-)linear matter spectrum for all three terms;
- `linearbel` — linear matter spectrum with the Bel et al. non-linear damping applied to the mt and tt terms
  (the default);
- `nonlinearbel` — same damping, but with an external non-linear $P_{mm}$ for the mm term.

### Normalization

`normalization_power_spectrum` rescales the spectra:

- `no_normalization` — leave the spectra as returned;
- `growth_rate` — factor the growth rate $f\sigma_8$ out of the mt / tt terms;
- `growth_amplitude` — divide all terms by $\sigma_8^2$.

## The power-spectrum dictionary

Covariance construction consumes a `power_spectrum_dict` keyed by the covariance blocks, each a list of
`[wavenumber, power]` pairs. For a velocity-only fit only the `vv` block is needed; for a full fit all three
appear. The small-scale velocity damping $D_u(k, \sigma_u) = \mathrm{sinc}(k\,\sigma_u)$
([`Du`][flip.utils.Du]) is typically applied to the velocity terms:

```python
from flip import utils

sigmau_fiducial = 15.0
power_spectrum_dict = {
    "gg": [[kmm, pmm * window_mm**2],
           [kmt, pmt * window_mt],
           [ktt, ptt]],
    "gv": [[kmt, pmt * window_mt * utils.Du(kmt, sigmau_fiducial)],
           [ktt, ptt * utils.Du(kmt, sigmau_fiducial)]],
    "vv": [[ktt, ptt * utils.Du(ktt, sigmau_fiducial) ** 2]],
}
```

The packaged example builds exactly this structure via
[`load_data_test.load_power_spectrum_dict`][flip.data.load_data_test.load_power_spectrum_dict] — a convenient
starting point:

```python
from flip.data import load_data_test
power_spectrum_dict = load_data_test.load_power_spectrum_dict(sigmau_fiducial=15.0)
```
