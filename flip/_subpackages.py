"""Subpackage manifest and availability probing.

Drives the banner printed at ``import flip`` and validates which
subpackages can be used in the current environment without forcing
their import. Subpackages are loaded lazily on attribute access from
:mod:`flip` (see :mod:`flip.__init__`).
"""

import importlib.util


SUBPACKAGES = {
    "data_vector": {
        "required": ["numpy", "scipy", "astropy", "pandas"],
        "optional": {"jax (jit paths)": ["jax"]},
    },
    "comparison": {
        "required": [],
        "optional": {},
    },
    "forward": {
        "required": ["jax", "jax_cosmo", "tensorflow_probability"],
        "optional": {},
    },
    "power_spectra": {
        "required": ["numpy"],
        "optional": {
            "pyccl engine": ["pyccl"],
            "class engine": ["classy"],
            "cosmoprimo engine": ["cosmoprimo"],
        },
    },
    "simulation": {
        "required": [],
        "optional": {},
    },
    "covariance": {
        "required": [
            "numpy",
            "scipy",
            "sympy",
            "mpmath",
            "emcee",
            "iminuit",
            "pandas",
        ],
        "optional": {
            "jax (jit paths)": ["jax"],
            "hankel cosmoprimo backend": ["cosmoprimo"],
            "skgp emulator": ["sklearn"],
            "nn emulator": ["torch"],
            "SN fit utils": ["snsim", "snutils"],
        },
    },
}


_PIP_NAMES = {
    "jax_cosmo": "jax-cosmo",
    "tensorflow_probability": "tensorflow-probability",
    "sklearn": "scikit-learn",
}


def _has(pkg):
    return importlib.util.find_spec(pkg) is not None


def _pip_name(pkg):
    return _PIP_NAMES.get(pkg, pkg)


def require(name):
    """Raise ImportError listing missing required deps for subpackage ``name``.

    Called by each subpackage ``__init__.py`` so that an attempt to load
    a subpackage with unmet dependencies produces a single clear message
    with an install hint, instead of a deep ``ModuleNotFoundError``.
    """
    spec = SUBPACKAGES[name]
    missing = [p for p in spec["required"] if not _has(p)]
    if not missing:
        return
    pip_cmd = "pip install " + " ".join(_pip_name(p) for p in missing)
    raise ImportError(
        f"flip.{name} requires missing packages: {', '.join(missing)}. "
        f"Install with: {pip_cmd}"
    )


def probe(name):
    """Return availability dict for one subpackage."""
    spec = SUBPACKAGES[name]
    missing_required = [p for p in spec["required"] if not _has(p)]
    optional_status = {
        label: {
            "ok": all(_has(p) for p in pkgs),
            "missing": [p for p in pkgs if not _has(p)],
        }
        for label, pkgs in spec["optional"].items()
    }
    return {
        "available": not missing_required,
        "missing_required": missing_required,
        "optional": optional_status,
    }


def probe_all():
    return {name: probe(name) for name in SUBPACKAGES}


def format_banner(version):
    status = probe_all()
    lines = [f"flip {version} -- subpackage availability"]
    width = max(len(n) for n in SUBPACKAGES)
    for name, s in status.items():
        if not s["available"]:
            lines.append(
                f"  {name:<{width}}  UNAVAILABLE  missing required: "
                + ", ".join(s["missing_required"])
            )
            continue
        opt_missing = [
            f"{label} ({','.join(o['missing'])})"
            for label, o in s["optional"].items()
            if not o["ok"]
        ]
        opt_present = [label for label, o in s["optional"].items() if o["ok"]]
        extras = []
        if opt_present:
            extras.append("optional ok: " + ", ".join(opt_present))
        if opt_missing:
            extras.append("optional missing: " + "; ".join(opt_missing))
        suffix = ("  [" + " | ".join(extras) + "]") if extras else ""
        lines.append(f"  {name:<{width}}  OK{suffix}")
    return "\n".join(lines)
