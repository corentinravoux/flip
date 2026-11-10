"""Write the YAML configurations of the lai22 model, whose terms come from loops.

The YAML files are then turned into flip_terms.py and fisher_terms.py by
:func:`flip.covariance.analytical.symbolic.generate_flip_terms` and
:func:`flip.covariance.analytical.symbolic.generate_fisher_terms`.

Example::

    from flip.covariance.analytical import symbolic
    from flip.covariance.analytical.lai22 import yaml_generator

    yaml_generator.write_lai22_yaml()
    symbolic.generate_flip_terms("lai22/flip_terms.yml")
    symbolic.generate_fisher_terms("lai22/fisher_terms.yml")
"""

import os

import sympy as sy
import yaml

_HEADER_FLIP = (
    "# flip_terms generator configuration: {model}.\n"
    "# Written by flip.covariance.analytical.lai22.yaml_generator.{function}, do not edit by hand.\n"
    '# Regenerate flip_terms.py with symbolic.generate_flip_terms("flip_terms.yml").\n'
)
_HEADER_FISHER = (
    "# Fisher partial-derivative generator configuration: {model}.\n"
    "# Each variant lists its parameters (Fisher matrix order) and the coefficients w_ab,n,\n"
    "# keyed by the term indices n of flip_terms.yml. null = default variant.\n"
    "# Written by flip.covariance.analytical.lai22.yaml_generator.{function}, do not edit by hand.\n"
    '# Regenerate fisher_terms.py with symbolic.generate_fisher_terms("fisher_terms.yml").\n'
)


def _write_yaml(config, filename, header):
    with open(filename, "w") as f:
        f.write(header)
        yaml.safe_dump(config, f, sort_keys=False, width=1000)


def lai22_flip_terms_config(pmax_gg=3, qmax_gg=3, pmax_gv=3):
    """flip_terms configuration of the Lai et al. 2022 model.

    The damping exponentials are Taylor-expanded: the gg term (i, m) collects
    the orders p + q = m of the expansion in mu1 and mu2, multiplied by
    1, mu1^2 + mu2^2 or mu1^2 mu2^2 for i = 0, 1, 2. The gv term (i, m) is the
    order m in mu1, multiplied by 1 or mu1^2 for i = 0, 1.

    Args:
        pmax_gg: Maximal expansion order in mu1 for gg.
        qmax_gg: Maximal expansion order in mu2 for gg.
        pmax_gv: Maximal expansion order in mu1 for gv.

    Returns:
        dict: Configuration read by symbolic.generate_flip_terms.
    """
    mu1, mu2 = sy.symbols("mu1 mu2")
    k = sy.Symbol("k", positive=True, finite=True, real=True)

    gg = {}
    for i, add in enumerate([1, mu1**2 + mu2**2, mu1**2 * mu2**2]):
        for m in range(pmax_gg + qmax_gg + 1):
            F = 0
            for p in range(pmax_gg + 1):
                q = m - p
                if 0 <= q <= qmax_gg:
                    F += (
                        sy.Integer(-1) ** m
                        / (2**m * sy.factorial(p) * sy.factorial(q))
                        * k ** (2 * m)
                        * mu1 ** (2 * p)
                        * mu2 ** (2 * q)
                        * add
                    )
            # l1max and l2max of i = 2 use m + 1, as in the original lai22 model.
            order = m + min(i, 1)
            gg[f"{i}_{m}"] = {
                "F": str(F),
                "lmax": 2 * (m + i),
                "l1max": min(2 * order, 2 * (pmax_gg + min(i, 1))),
                "l2max": min(2 * order, 2 * (qmax_gg + min(i, 1))),
            }

    gv = {}
    for i, add in enumerate([1, mu1**2]):
        for m in range(pmax_gv + 1):
            F = (
                100
                * sy.Integer(-1) ** m
                / (2**m * sy.factorial(m))
                * k ** (2 * m - 1)
                * mu1 ** (2 * m)
                * mu2
                * add
            )
            gv[f"{i}_{m}"] = {
                "F": str(F),
                "lmax": 2 * m + 1 + 2 * i,
                "l1max": 2 * m + 2 * i,
                "l2max": 1,
            }

    vv = {"0_0": {"F": str(10000 * mu1 * mu2 / k**2), "lmax": 2, "l1max": 1, "l2max": 1}}

    return {
        "wide_angle": True,
        "additional_parameters": [],
        "multi_index_model": True,
        "regularize_M_terms": None,
        "terms": {"gg": gg, "gv": gv, "vv": vv},
    }


def lai22_fisher_terms_config(flip_terms_config):
    """Fisher configuration of the Lai et al. 2022 model.

    Each coefficient is the bias / growth amplitude of the term multiplied by
    sigg^(2m), with the term indices "i_m" of the flip_terms configuration.

    Args:
        flip_terms_config: Output of :func:`lai22_flip_terms_config`.

    Returns:
        dict: Configuration read by symbolic.generate_fisher_terms.
    """
    fs8_growth = "Omegam**gamma*s8"
    variants = {
        "growth_index": {
            "parameters": ["Omegam", "gamma", "s8", "bs8", "beta_f", "sigg"],
            "gg": ["bs8**2", "bs8**2*beta_f", "bs8**2*beta_f**2"],
            "gv": [f"bs8*{fs8_growth}", f"bs8*{fs8_growth}*beta_f"],
            "vv": f"({fs8_growth})**2",
        },
        "growth_index_nobeta": {
            "parameters": ["Omegam", "gamma", "s8", "bs8", "sigg"],
            "gg": ["bs8**2", f"bs8*{fs8_growth}", f"({fs8_growth})**2"],
            "gv": [f"bs8*{fs8_growth}", f"({fs8_growth})**2"],
            "vv": f"({fs8_growth})**2",
        },
        "nobeta": {
            "parameters": ["fs8", "bs8", "sigg"],
            "gg": ["bs8**2", "bs8*fs8", "fs8**2"],
            "gv": ["bs8*fs8", "fs8**2"],
            "vv": "fs8**2",
        },
        None: {
            "parameters": ["fs8", "bs8", "beta_f", "sigg"],
            "gg": ["bs8**2", "bs8**2*beta_f", "bs8**2*beta_f**2"],
            "gv": ["bs8*fs8", "bs8*fs8*beta_f"],
            "vv": "fs8**2",
        },
    }

    terms = flip_terms_config["terms"]
    config = {}
    for name, variant in variants.items():
        namespace = {p: sy.Symbol(p) for p in variant["parameters"]}
        sigg = namespace["sigg"]
        coefficients = {}
        for component in ["gg", "gv"]:
            coefficients[component] = {}
            for key in terms[component]:
                i, m = (int(index) for index in key.split("_"))
                amplitude = sy.sympify(variant[component][i], locals=namespace)
                coefficients[component][key] = str(amplitude * sigg ** (2 * m))
        coefficients["vv"] = {
            key: str(sy.sympify(variant["vv"], locals=namespace))
            for key in terms["vv"]
        }
        config[name] = {
            "parameters": variant["parameters"],
            "coefficients": coefficients,
        }
    return {"variants": config}


def write_lai22_yaml(directory=None, pmax_gg=3, qmax_gg=3, pmax_gv=3):
    """Write flip_terms.yml and fisher_terms.yml of the lai22 model.

    Args:
        directory: Output directory. Defaults to this lai22 directory.
        pmax_gg, qmax_gg, pmax_gv: Expansion orders, see lai22_flip_terms_config.
    """
    if directory is None:
        directory = os.path.dirname(os.path.abspath(__file__))
    flip_config = lai22_flip_terms_config(pmax_gg, qmax_gg, pmax_gv)
    names = dict(model="Lai et al. 2022", function="write_lai22_yaml")
    _write_yaml(
        flip_config,
        os.path.join(directory, "flip_terms.yml"),
        _HEADER_FLIP.format(**names),
    )
    _write_yaml(
        lai22_fisher_terms_config(flip_config),
        os.path.join(directory, "fisher_terms.yml"),
        _HEADER_FISHER.format(**names),
    )


if __name__ == "__main__":
    write_lai22_yaml()
