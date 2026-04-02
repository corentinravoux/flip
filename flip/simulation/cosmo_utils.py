"""Utilities for bridging flip parameter dicts and jax_cosmo Cosmology objects.

Flip's inference parameters (fs8, bs8, sigv, …) are not the same as the full
cosmological set required by JaxPM/jax_cosmo. This module handles:
  - converting a flip parameter dict + fixed background cosmology → jax_cosmo.Cosmology
  - decomposing fs8 into f and σ₈ given a cosmological background
  - building JAX-compatible P(k) callables from tabulated arrays
"""

try:
    import jax.numpy as jnp
    import jax_cosmo as jc
    from jax_cosmo import background

    jaxcosmo_installed = True
except ImportError:
    import numpy as jnp

    jaxcosmo_installed = False


def flip_params_to_jaxcosmo(params_dict, base_cosmo_dict):
    """Build a jax_cosmo Cosmology from flip inference parameters + fixed background.

    Flip typically infers fs8, bs8, sigv (and optionally sigma8, H_0) while
    keeping the background cosmology fixed. This function merges the two,
    letting entries in ``params_dict`` override ``base_cosmo_dict``.

    ``base_cosmo_dict`` should use jax_cosmo key names (Omega_c, not Omega_m).
    If ``Omega_m`` is present and ``Omega_c`` is absent, the conversion
    ``Omega_c = Omega_m - Omega_b`` is applied automatically.

    Args:
        params_dict (dict): Flip inference parameters, e.g.
            ``{"fs8": 0.4, "sigma8": 0.8, "H_0": 67.0}``.
        base_cosmo_dict (dict): Fixed background cosmology with keys taken
            from ``{Omega_c, Omega_b, h, sigma8, n_s, w0, wa, Omega_k}``.

    Returns:
        jax_cosmo.Cosmology: Cosmology object ready for JaxPM.

    Raises:
        ImportError: If jax_cosmo is not installed.
    """
    if not jaxcosmo_installed:
        raise ImportError(
            "jax_cosmo is required for forward modeling. "
            "Install with: pip install jax-cosmo"
        )

    cosmo_dict = dict(base_cosmo_dict)

    # H_0 [km/s/Mpc] → h
    if "H_0" in params_dict:
        cosmo_dict["h"] = params_dict["H_0"] / 100.0

    # Override cosmological parameters when present in flip params
    for key in ("sigma8", "Omega_c", "Omega_b", "h", "n_s", "w0", "wa", "Omega_k"):
        if key in params_dict:
            cosmo_dict[key] = params_dict[key]

    # Omega_m → Omega_c (jax_cosmo convention)
    if "Omega_m" in cosmo_dict and "Omega_c" not in cosmo_dict:
        Omega_b = cosmo_dict.get("Omega_b", 0.0)
        cosmo_dict["Omega_c"] = cosmo_dict.pop("Omega_m") - Omega_b

    return jc.Cosmology(
        Omega_c=float(cosmo_dict.get("Omega_c", 0.25)),
        Omega_b=float(cosmo_dict.get("Omega_b", 0.05)),
        h=float(cosmo_dict.get("h", 0.67)),
        sigma8=float(cosmo_dict.get("sigma8", 0.8)),
        n_s=float(cosmo_dict.get("n_s", 0.96)),
        w0=float(cosmo_dict.get("w0", -1.0)),
        wa=float(cosmo_dict.get("wa", 0.0)),
        Omega_k=float(cosmo_dict.get("Omega_k", 0.0)),
    )


def sigma8_from_fs8(fs8, cosmo, a):
    """Infer σ₈ from fs8 = f(a) · σ₈ and the linear growth rate f(a).

    Useful when the forward model needs σ₈ (to set the amplitude of P(k))
    but the inference parameter is fs8.

    Args:
        fs8 (float): Growth rate × σ₈ (flip's primary velocity parameter).
        cosmo (jax_cosmo.Cosmology): Background cosmology (used for f(a)).
        a (float): Scale factor at which the decomposition is made.

    Returns:
        float: σ₈ = fs8 / f(a).

    Raises:
        ImportError: If jax_cosmo is not installed.
    """
    if not jaxcosmo_installed:
        raise ImportError("jax_cosmo is required.")

    f = background.growth_rate(cosmo, jnp.atleast_1d(a))[0]
    return fs8 / f


def cosmo_with_sigma8_from_fs8(fs8, cosmo, a):
    """Return a copy of cosmo with σ₈ replaced by σ₈ = fs8 / f(a).

    This is the typical call sequence when the inference parameter is fs8
    and the power spectrum needs a consistent σ₈.

    Args:
        fs8 (float): Growth rate × σ₈.
        cosmo (jax_cosmo.Cosmology): Base cosmology.
        a (float): Scale factor.

    Returns:
        jax_cosmo.Cosmology: New cosmology with updated sigma8.
    """
    if not jaxcosmo_installed:
        raise ImportError("jax_cosmo is required.")

    sigma8 = sigma8_from_fs8(fs8, cosmo, a)
    return jc.Cosmology(
        Omega_c=cosmo.Omega_c,
        Omega_b=cosmo.Omega_b,
        h=cosmo.h,
        sigma8=sigma8,
        n_s=cosmo.n_s,
        w0=cosmo.w0,
        wa=cosmo.wa,
        Omega_k=cosmo.Omega_k,
    )


def make_pk_callable(k_arr, pk_arr):
    """Build a JAX-differentiable P(k) callable from tabulated arrays.

    Uses log-log linear interpolation (power-law between nodes), which is
    both accurate for CDM power spectra and fully differentiable via
    ``jnp.interp``.

    Args:
        k_arr (array-like): Wavenumbers [h/Mpc], sorted ascending.
        pk_arr (array-like): Power spectrum values [(Mpc/h)³].

    Returns:
        Callable: ``pk(k)`` accepting JAX arrays, returning P(k) [(Mpc/h)³].
    """
    log_k = jnp.log(jnp.asarray(k_arr, dtype=jnp.float32))
    log_pk = jnp.log(jnp.asarray(pk_arr, dtype=jnp.float32))

    def pk_callable(k):
        return jnp.exp(jnp.interp(jnp.log(k), log_k, log_pk))

    return pk_callable


def make_pk_callable_from_dict(power_spectrum_dict, kind="vv"):
    """Build a P(k) callable from a flip-style power_spectrum_dict.

    Flip's ``power_spectrum_dict`` has keys ``"gg"``, ``"vv"``, ``"gv"``,
    each mapping to a 2-column array ``[[k, P(k)], …]`` or a list of such
    arrays (one per redshift bin). This function wraps the first entry.

    Args:
        power_spectrum_dict (dict): Flip power spectrum dict.
        kind (str): Which spectrum to extract: ``"gg"``, ``"vv"``, or ``"gv"``.
            Defaults to ``"vv"`` (velocity–velocity, needed for PM simulations).

    Returns:
        Callable: JAX-differentiable ``pk(k)``.

    Raises:
        KeyError: If ``kind`` is not present in ``power_spectrum_dict``.
    """
    if kind not in power_spectrum_dict:
        raise KeyError(
            f"Key '{kind}' not found in power_spectrum_dict. "
            f"Available: {list(power_spectrum_dict.keys())}"
        )
    entry = power_spectrum_dict[kind]
    # Entry is either [[k, Pk], …] (2D array) or a list of such arrays
    arr = jnp.asarray(entry[0] if isinstance(entry, list) else entry)
    k_arr, pk_arr = arr[:, 0], arr[:, 1]
    return make_pk_callable(k_arr, pk_arr)
