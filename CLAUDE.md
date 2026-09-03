# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

`flip` (PyPI: `flipcosmo`) — field-level inference of the growth rate from peculiar-velocity and density
fields. Pipeline: model power spectrum + object coordinates → analytical covariance matrix → Gaussian
likelihood → Minuit/MCMC fit or Fisher forecast.

## Commands

```bash
pip install .                  # or pip install -e .
pytest                         # full suite (run from repo root; no conftest, no pytest config)
pytest test/test_e2e_velocity.py::test_e2e_velocity
python scripts/flip_launch_minuit_velocity_fit.py   # runnable examples, no console entry points

pip install .[docs] && zensical build     # docs -> site/ (mkdocs.yml is the config; RTD uses zensical)
```

Optional heavy deps gate parts of the code: `sympy`/`mpmath` (symbolic term generation), `cosmoprimo`,
`classy`, `pyccl` (power-spectrum engines), `jax` + `jax-cosmo` + `tensorflow-probability`
(`flip.forward`), `GPy`/`scikit-learn`/`torch` (emulators).

Version in `flip/__init__.py` is bumped by CI (bump2version on push to `main`) — do not edit it by hand.
Use `[bump minor]` / `[bump major]` / `[no bump]` in the commit message to control it.

## Architecture

### Three-step user flow
1. **Data vector** — `flip.data_vector`: a `DataVector` subclass wraps the measurements and declares
   `_kind` (`velocity`/`density`/`cross`), `_needed_keys`, `_free_par` and the observational-covariance
   parameters. `basic.py` holds the core ones (`Dens`, `DirectVel`, `DensMesh`, `DensVel`, …); the
   tracer-specific ones live in `snia_vectors.py` (`VelTrippRelation`, …), `galaxypv_vectors.py`
   (Tully-Fisher, fundamental plane, log-distance) and `gw_vectors.py`.
2. **Covariance** — `flip.covariance.CovMatrix`, built via `init_from_flip` (model name, kind, power
   spectrum dict, coordinates), `init_from_generator`, `init_from_emulator` or `init_from_file`.
   `data.compute_covariance(...)` is the shortcut.
3. **Likelihood + fit** — `flip.covariance.likelihood` (`MultivariateGaussianLikelihood` and 1D/2D
   covariance-interpolating variants) then `flip.covariance.fitter` (`FitMinuit`, `FitMCMC`) or
   `flip.covariance.fisher` for forecasts.

Conventions inside flip: RA/Dec in **radians**, distances in Mpc/h, velocities in km/s, k in h/Mpc.
Kind strings are `"velocity"`, `"density"`, `"density_velocity"`, `"full"`.

### Covariance model layout (the part that needs reading several files)
Each model is a package under `flip/covariance/analytical/<model>/`:
- `__init__.py` — declares `_free_par`, a dict `param -> ["<kind>@<variant>", ...]`. The `@` syntax is
  parsed by `_read_free_par` in `covariance/covariance.py`: `kind` may be `all` or match the requested
  kind (`full` matches everything), `variant` may be `all` or the requested variant, default variant is
  `"baseline"`. This is what determines the fittable parameter names (`fs8`, `bs8`, `sigv`, `beta_f`, …).
- `flip_terms.py` — the k-space `M_ab_i_l_j` kernels and angular `N_ab_i_l_j` terms. **Generated**, not
  hand-written. `set_backend("numpy"|"mpmath")` switches precision backend.
- `coefficients.py` — `get_coefficients(parameter_values_dict, model_kind, variant=...)` maps fit
  parameters onto the multiplicative coefficient of each term, per block (`gg`/`gv`/`vv`), plus
  `get_diagonal_coefficients` for the diagonal (e.g. `sigv**2`).
- `fisher_terms.py` — generated partial-derivative coefficient dicts.
- some models add a `generator.py` for a bespoke (non-Hankel) covariance path.

Generated files come from `flip/covariance/symbolic.py` (`generate_files()`, `generate_fisher_files()`),
which uses sympy and writes with paths relative to `flip/covariance/` — **run it with that as cwd**.
Never hand-edit `flip_terms.py` / `fisher_terms.py`; change the symbolic generator and regenerate.

`covariance/generator.py` is the shared engine: it turns model kernels + input power spectra into
real-space blocks via Hankel/FFTLog (`hankel.py`) or direct quadrature, with optional regularization
(`None`, `"mpmath"`, `"savgol"`, `"lowk_asymptote"`). `cov_utils.py` holds pair-separation geometry for
the different line-of-sight conventions; `contraction.py` predicts the model covariance vs separation
without a catalog (useful for validating a new model).

### Other subpackages
- `flip.power_spectra` — engine wrappers (`class_engine`, `pyccl_engine`, `cosmoprimo_engine`) producing
  the `power_spectrum_dict` (`mm`/`mt`/`tt` terms) the covariance consumes.
- `flip.forward` — jax field-level forward model and sampler (`flox/`, `simulation/`); needs jax stack.
- `flip.comparison` — likelihood for comparing two fields on a grid.
- `flip.data` — small packaged test catalogs (`load_data_test`) plus the stored test reference values.

### Import machinery
`flip/__init__.py` loads subpackages **lazily** via `__getattr__`, and prints a dependency-availability
banner on `import flip` (silence with `FLIP_QUIET=1`). `flip/_subpackages.py` is the manifest of
required/optional deps per subpackage; each subpackage `__init__.py` calls `require(name)` so unmet deps
give one clear message. When adding a subpackage or a dependency, update `SUBPACKAGES` there.
`flip/_config.py` has `__use_jax__`; data vectors and likelihoods fall back to numpy when jax is absent
or the flag is off, and jitted likelihood variants are used only when jax + `use_jit` are on.

## Tests

`test/test_covariance_reference_values.py` and the three `test_e2e_*.py` compare against numbers stored in
`flip/data/test_covariance_reference_values.json` and `test_e2e_reference_values.json`. If a physics or
numerics change legitimately moves those values, regenerate with `python test/refresh_reference_values.py`
(run from `test/`; it writes the JSONs next to itself — copy them into `flip/data/`) and justify the diff.
`model_to_test` in `test_covariance_reference_values.py` is the model×kind matrix — extend it when adding
a model. `test_covariance_assembly.py`, `test_covariance_utils.py` and `test_likelihood_inversions.py` are
self-contained.
