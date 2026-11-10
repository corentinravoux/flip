import glob
import multiprocessing as mp
import os

import sympy as sy
from sympy.physics import wigner
from sympy.polys.orthopolys import legendre_poly
from sympy.printing import pycode
from sympy.simplify.fu import TR8

from flip.utils import create_log

log = create_log()


def simplify_term(
    term,
    simplification_method="simplify_iteration",
    max_simplification=20,
):
    """
    The simplify_term function takes a sympy expression and simplifies it.
    It does so by applying the TR8 function to the term, then checks if this is different from the original term.
    If it is, then it applies TR8 again to this new simplified term and compares that with the previous one.
    This process continues until either no further simplification can be achieved or max_simplification iterations have been reached.

    Args:
        term: Pass the term to be simplified
        simplification_method: Choose between two different methods of simplification
        max_simplification: Limit the number of iterations
        : Set the method of simplification

    Returns:
        The simplified term

    """
    _avail_simplification_methods = "simplify_iteration", "tr8_iteration"

    if simplification_method == "simplify_iteration":
        term_simplified = sy.simplify(term)
        i = 0
        while (term_simplified != term) & (i < max_simplification):
            term_simplified, term = (
                sy.factor(TR8(term_simplified)),
                term_simplified,
            )
            i += 1
    elif simplification_method == "tr8_iteration":
        term_simplified = sy.factor(TR8(term))
        i = 0
        while (term_simplified != term) & (i < max_simplification):
            term_simplified, term = (
                sy.factor(TR8(term_simplified)),
                term_simplified,
            )
            i += 1
    else:
        log.add(
            f"Simplification method {simplification_method} not available."
            f"Choose between: {_avail_simplification_methods}"
        )

    return term_simplified


def generate_MN_ab_i_l_function_wide_angle(
    term_F,
    ell,
    l1,
    l2,
    term_N_prefactor,
):
    """
    The generate_MN_ab_i_l_function_wide_angle function generates the M and N terms for a given l, l', and l'' value.

    Args:
        term_F: Geometric term F_ab,n(k, mu1, mu2) of the power spectrum
            decomposition (Ravoux et al. 2025), expanded in Legendre
            polynomials of mu1 and mu2 to give M_ab,n^{l1,l2}(k)
        l: Determine the order of the legendre polynomial
        l1: Define the first legendre polynomial in the integral
        l2: Determine the order of the legendre polynomial in the second integral
        term_N_prefactor: The prefactor for the N term

    Returns:
        The terms m_l,l',l'' and n_l, l', l''

    """
    theta, phi = sy.symbols("theta phi")
    mu1, mu2 = sy.symbols("mu1 mu2")
    integral_mu1_M_l = sy.integrate(term_F * legendre_poly(l1, x=mu1), (mu1, -1, 1))
    integral_mu1_mu2_M_l = sy.integrate(
        integral_mu1_M_l * legendre_poly(l2, x=mu2), (mu2, -1, 1)
    )
    term_N_l_l1_l2 = 0
    for m in range(-ell, ell + 1):
        for m1 in range(-l1, l1 + 1):
            for m2 in range(-l2, l2 + 1):
                term_N_l_l1_l2_m_m1_m2 = wigner.gaunt(ell, l1, l2, m, m1, m2)
                # The two comments are for the following line, they are the results of an intense
                # head scratching and are quite important for all the modeling of flip.

                # The spherical harmonic terms are taken from Lai et al. 2022 matematica notebook
                # The sy.pi phase is put to l2 term instead of l1, to obtain the same results as Lai et al. 2022
                # This is understood by the asymetry term inside the F_ab,n modeling.
                # Changing this will have an impact on the modeling.
                # Be sure to accord F_ab,n terms with sperical harmonic definition here.

                # The sy.pi term with phi is directly linked to the definition of r chosen in flip.
                # For the order of term chosen, the sy.pi must be added here.
                # If not, it will give wrong results for cross-terms (gv).
                term_N_l_l1_l2_m_m1_m2 *= (
                    sy.Ynm(ell, m, sy.pi - phi, 0)
                    * sy.Ynm(l1, m1, theta / 2, 0)
                    * sy.Ynm(l2, m2, theta / 2, sy.pi)
                )
                term_N_l_l1_l2 = term_N_l_l1_l2 + term_N_l_l1_l2_m_m1_m2
    term_M_l_l1_l2 = (1 / sy.Rational(4)) * integral_mu1_mu2_M_l.expand(func=True)
    term_N_l_l1_l2 = (
        term_N_prefactor
        * (sy.Rational(4) * sy.pi) ** 2
        * term_N_l_l1_l2.expand(func=True)
    )
    term_M_l_l1_l2 = simplify_term(
        term_M_l_l1_l2,
        simplification_method="simplify_iteration",
    )
    term_N_l_l1_l2 = simplify_term(
        term_N_l_l1_l2,
        simplification_method="tr8_iteration",
    )

    return term_M_l_l1_l2, term_N_l_l1_l2


def generate_MN_ab_i_l_function_parallel_plane(term_F, ell):
    """
    The generate_MN_ab_i_l_function_parallel_plane function takes in a term_F and an l value.
    It then generates the M_l and N_l functions for that particular term F_ab,n, which is used to calculate the parallel plane integral.
    The function returns both M_l and N_l as sympy expressions.

    Args:
        term_F: Geometric term F_ab,n(k, mu) of the power spectrum
            decomposition (Ravoux et al. 2025), expanded in Legendre
            polynomials of mu to give M_ab,n^l(k)
        l: Determine the order of the legendre polynomial

    Returns:
        A tuple of

    """
    phi = sy.symbols("phi")
    mu = sy.symbols("mu")
    M_l = sy.Rational(1 / 2) * sy.integrate(
        term_F * legendre_poly(ell, x=mu), (mu, -1, 1)
    )
    # The sy.pi term is directly linked to the definition of r chosen in flip.
    # For the order of term chosen, the sy.pi must be added here.
    # If not, it will give wrong results for cross-terms (gv).
    N_l = sy.Rational(2 * ell + 1) * legendre_poly(ell, x=sy.cos(sy.pi - phi))
    M_l = simplify_term(
        M_l.expand(func=True),
        simplification_method="simplify_iteration",
    )
    N_l = simplify_term(
        N_l.expand(func=True),
        simplification_method="tr8_iteration",
    )
    return M_l, N_l


def write_output(
    filename,
    type_list,
    term_index_list,
    lmax_list,
    output_pool,
    index_pool,
    wide_angle,
    additional_parameters=None,
    l1max_list=None,
    l2max_list=None,
    multi_index_model=False,
    regularize_M_terms="None",
):
    """
    The write_output function takes the following arguments:
        filename (str): The name of the file to be written.
        type_list (list): A list of strings that are used as keys for each term in the model. For example, if you have a model with two terms, one called 'A' and one called 'B', then type_list = ['A','B'] would be appropriate.
        term_index_list (nested list): A nested list containing integers that correspond to each term in your model. For example, if you have a three-term model where terms 1 and 2 are both linear functions

    Args:
        filename: Specify the name of the output file
        type_list: Specify the type of term
        term_index_list: Determine the order of the terms in
        lmax_list: Specify the maximum l value for each term
        output_pool: Store the output of the function
        index_pool: Store the index of each term in the output_pool
        wide_angle: Determine if the wide angle approximation should be used
        additional_parameters: Pass additional parameters to the functions
        l1max_list: Specify the maximum l value for the first index of a wide-angle term
        l2max_list: Specify the maximum l2 value for each term in the wide-angle case
        multi_index_model: Distinguish between the two models
        : Determine whether the model is a multi-index model or not

    Returns:
        A python file with the functions m_ab_i_l and n_ab_i

    """
    f = open(filename, "w")
    f.write("import numpy\n")
    f.write("import scipy\n")
    f.write("\n")
    f.write("\n")
    f.write("def set_backend(module):\n")
    f.write("    global np, erf\n")
    f.write("""    if module == "numpy":\n""")
    f.write("        np = numpy\n")
    f.write("        erf = scipy.special.erf\n")
    f.write("""    elif module == "mpmath":\n""")
    f.write("        import mpmath\n")
    f.write("\n")
    f.write("        np = mpmath.mp\n")
    f.write("        erf = mpmath.erf\n")
    f.write("\n")
    f.write("""set_backend("numpy")\n""")
    f.write("\n")
    f.write("\n")
    dict_terms = {}
    dict_lmax = {}
    dict_j = {}
    for k, type in enumerate(type_list):
        dict_terms[f"{type}"] = term_index_list[k]
        dict_lmax[f"{type}"] = lmax_list[k]
        for i, t in enumerate(term_index_list[k]):
            for ell in range(lmax_list[k][i] + 1):
                list_M_ab_i_l = []
                list_N_ab_i_l = []
                if wide_angle:
                    for l1 in range(l1max_list[k][i] + 1):
                        for l2 in range(l2max_list[k][i] + 1):
                            M_ab_i_l_l1_l2, N_ab_i_l_l1_l2 = output_pool[
                                index_pool[f"{type}_{t}_{ell}_{l1}_{l2}"]
                            ]
                            if (M_ab_i_l_l1_l2 != 0) & (N_ab_i_l_l1_l2 != 0):
                                list_M_ab_i_l.append(M_ab_i_l_l1_l2)
                                list_N_ab_i_l.append(N_ab_i_l_l1_l2)
                else:
                    M_ab_i_l_l1_l2, N_ab_i_l_l1_l2 = output_pool[
                        index_pool[f"{type}_{t}_{ell}"]
                    ]
                    if (M_ab_i_l_l1_l2 != 0) & (N_ab_i_l_l1_l2 != 0):
                        list_M_ab_i_l.append(M_ab_i_l_l1_l2)
                        list_N_ab_i_l.append(N_ab_i_l_l1_l2)
                dict_j[f"{type}_{t}_{ell}"] = len(list_M_ab_i_l)
                for j in range(len(list_M_ab_i_l)):
                    M_ab_i_l_j = (
                        pycode(list_M_ab_i_l[j])
                        .replace("math.erf", "erf")
                        .replace("math.", "np.")
                    )
                    N_ab_i_l_j = (
                        pycode(list_N_ab_i_l[j])
                        .replace("math.erf", "erf")
                        .replace("math.", "np.")
                    )

                    additional_str = ""
                    if additional_parameters is not None:
                        for add in additional_parameters:
                            additional_str = additional_str + f"{add},"
                    additional_str = additional_str[:-1]
                    f.write(f"def M_{type}_{t}_{ell}_{j}({additional_str}):\n")
                    f.write("    def func(k):\n")
                    f.write(f"        return {M_ab_i_l_j}\n")
                    f.write("    return func\n")
                    f.write("\n")

                    f.write(f"def N_{type}_{t}_{ell}_{j}(theta,phi):\n")
                    f.write(f"    return({N_ab_i_l_j})\n")
                    f.write("\n")

    f.write("dictionary_terms = ")
    f.write(repr(dict_terms))
    f.write("\n")

    f.write("dictionary_lmax = ")
    f.write(repr(dict_lmax))
    f.write("\n")

    f.write("dictionary_subterms = ")
    f.write(repr(dict_j))
    f.write("\n")

    f.write(f"multi_index_model = {multi_index_model}")
    f.write("\n")

    f.write(f"regularize_M_terms = {regularize_M_terms}")
    f.write("\n")
    f.close()


def write_M_N_functions(
    filename,
    type_list,
    term_index_list,
    lmax_list,
    dict_F,
    additional_parameters=None,
    number_worker=1,
    wide_angle=False,
    l1max_list=None,
    l2max_list=None,
    multi_index_model=False,
    regularize_M_terms="None",
    dict_N_prefactors=None,
):
    """
    The write_M_N_functions function is used to generate the M_N functions for a given model.
    The function takes in the following arguments:
        filename (str): The name of the file that will be generated. This should end with .py, and should not include any path information. If no path information is included, then this file will be saved in your current working directory.

    Args:
        filename: Specify the name of the file to which the output is written
        type_list: Specify the type of terms that are to be included in the calculation
        term_index_list: Specify the index of each term in a given type
        lmax_list: Determine the maximum value of l for each term in the model
        dict_F: Geometric terms F_ab,n, keyed "F_{type}_{index}"
            (e.g. "F_vv_0"), following Ravoux et al. 2025 notation
        additional_parameters: Add additional parameters to the output file
        number_worker: Specify the number of processes to be used
        wide_angle: Determine if the function should be written for wide angle or not
        l1max_list: Define the maximum value of l_i in the wide angle approximation
        l2max_list: Specify the maximum l2 value for each term in the model
        multi_index_model: Tell the function whether to use a multi-index model or not
        : Write the output to a file

    Returns:
        The output_pool and index_pool

    """
    params_pool = []
    index_pool = {}
    index = 0
    for k, type in enumerate(type_list):
        for i, t in enumerate(term_index_list[k]):
            for ell in range(lmax_list[k][i] + 1):
                F_ab_n = dict_F[f"F_{type}_{t}"]
                if wide_angle:
                    for l1 in range(l1max_list[k][i] + 1):
                        for l2 in range(l2max_list[k][i] + 1):
                            if dict_N_prefactors is not None:
                                prefactor_N_ab_n = dict_N_prefactors[
                                    f"N_prefactor_{type}_{t}"
                                ]
                                params_pool.append(
                                    [F_ab_n, ell, l1, l2, prefactor_N_ab_n]
                                )
                            else:
                                params_pool.append([F_ab_n, ell, l1, l2, 1])
                            index_pool[f"{type}_{t}_{ell}_{l1}_{l2}"] = index
                            index = index + 1
                else:
                    params_pool.append([F_ab_n, ell])
                    index_pool[f"{type}_{t}_{ell}"] = index
                    index = index + 1

    if number_worker == 1:
        if wide_angle:
            output_pool = [
                generate_MN_ab_i_l_function_wide_angle(*param) for param in params_pool
            ]
        else:
            output_pool = [
                generate_MN_ab_i_l_function_parallel_plane(*param)
                for param in params_pool
            ]
    else:
        if wide_angle:
            with mp.Pool(number_worker) as pool:
                output_pool = pool.starmap(
                    generate_MN_ab_i_l_function_wide_angle, params_pool
                )
        else:
            with mp.Pool(number_worker) as pool:
                output_pool = pool.starmap(
                    generate_MN_ab_i_l_function_parallel_plane, params_pool
                )

    write_output(
        filename,
        type_list,
        term_index_list,
        lmax_list,
        output_pool,
        index_pool,
        wide_angle,
        additional_parameters=additional_parameters,
        l1max_list=l1max_list,
        l2max_list=l2max_list,
        multi_index_model=multi_index_model,
        regularize_M_terms=regularize_M_terms,
    )


def generate_flip_terms(config_file, filename=None, number_worker=8):
    """Generate a flip_terms.py file from a YAML model description.

    The YAML file lists the geometric terms F_ab,n of the power spectrum
    decomposition P_ab = sum_n w_ab,n F_ab,n(k, mu1, mu2) P_ab,n(k)
    (Ravoux et al. 2025). Example::

        wide_angle: true                 # F(k, mu1, mu2) if true, F(k, mu) if false
        additional_parameters: [sig_g]   # extra arguments of the M functions
        multi_index_model: false
        regularize_M_terms: null         # or {gg: mpmath, gv: mpmath, vv: null}
        terms:
          vv:                            # field pair (gg, gv, vv)
            '0':                         # term index n
              F: 10000*mu1*mu2/k**2      # F_ab,n as a sympy expression
              lmax: 2
              l1max: 1                   # wide angle only
              l2max: 1                   # wide angle only
              N_prefactor: cos(theta)    # optional, wide angle only

    Args:
        config_file: Path to the YAML file.
        filename: Output file. Defaults to flip_terms.py next to the YAML file.
        number_worker: Number of processes for the symbolic integration.
    """
    import yaml

    with open(config_file) as f:
        config = yaml.safe_load(f)
    if filename is None:
        filename = os.path.join(
            os.path.dirname(os.path.abspath(config_file)), "flip_terms.py"
        )

    wide_angle = config["wide_angle"]
    additional_parameters = config.get("additional_parameters") or []

    positive = dict(positive=True, finite=True, real=True)
    namespace = {name: sy.Symbol(name) for name in ["mu", "mu1", "mu2", "theta", "phi"]}
    namespace["k"] = sy.Symbol("k", **positive)
    for name in additional_parameters:
        if not name.startswith("*"):
            namespace[name] = sy.Symbol(name, **positive)

    type_list, term_index_list = [], []
    lmax_list, l1max_list, l2max_list = [], [], []
    dict_F, dict_N_prefactors = {}, {}
    for type, terms in config["terms"].items():
        type_list.append(type)
        term_index_list.append([str(t) for t in terms])
        lmax_list.append([term["lmax"] for term in terms.values()])
        l1max_list.append([term.get("l1max") for term in terms.values()])
        l2max_list.append([term.get("l2max") for term in terms.values()])
        for t, term in terms.items():
            if not wide_angle and "N_prefactor" in term:
                raise ValueError("N_prefactor is only available for wide_angle models")
            dict_F[f"F_{type}_{t}"] = sy.sympify(str(term["F"]), locals=namespace)
            dict_N_prefactors[f"N_prefactor_{type}_{t}"] = sy.sympify(
                str(term.get("N_prefactor", 1)), locals=namespace
            )

    write_M_N_functions(
        filename,
        type_list,
        term_index_list,
        lmax_list,
        dict_F,
        additional_parameters=additional_parameters or None,
        number_worker=number_worker,
        wide_angle=wide_angle,
        l1max_list=l1max_list if wide_angle else None,
        l2max_list=l2max_list if wide_angle else None,
        multi_index_model=config.get("multi_index_model", False),
        regularize_M_terms=repr(config.get("regularize_M_terms")),
        dict_N_prefactors=dict_N_prefactors if wide_angle else None,
    )


def _parameter_symbol(parameter):
    return sy.Symbol(
        f'parameter_values_dict["{parameter}"]', positive=True, finite=True, real=True
    )


def compute_partial_derivative_dictionnary(
    components,
    parameter_models,
    coefficient_models,
):
    """Differentiate the coefficients of each variant with respect to its parameters.

    Args:
        components: Covariance blocks to treat (e.g. ["gg", "gv", "vv"]).
        parameter_models: Parameter names of each variant.
        coefficient_models: For each variant, a dictionary mapping each block
            to the list of sympy coefficients w_ab,n.

    Returns:
        list: For each variant, a dictionary {parameter: {block: code string}}.
    """
    partial_derivative_dictionnary_list = []
    for parameters, coefficients in zip(parameter_models, coefficient_models):
        partial_derivative_dictionnary = {}
        for parameter in parameters:
            symbol = _parameter_symbol(parameter)
            partial_derivative_dictionnary[parameter] = {
                component: "["
                + "".join(
                    pycode(sy.diff(coefficient, symbol)) + ","
                    for coefficient in coefficients[component]
                )
                + "]"
                for component in components
            }
        partial_derivative_dictionnary_list.append(partial_derivative_dictionnary)
    return partial_derivative_dictionnary_list


def write_partial_derivatives(
    filename,
    name_models,
    components,
    parameter_models,
    coefficient_models,
):
    """Write a fisher_terms.py file with the partial derivatives of the coefficients."""
    f = open(filename, "w")
    f.write("import numpy as np\n")
    f.write("\n")
    f.write("\n")
    if len(components) == 1:

        partial_derivative_dictionnary_list = compute_partial_derivative_dictionnary(
            components,
            parameter_models,
            coefficient_models,
        )

        f.write(
            "def get_partial_derivative_coefficients(model_kind,parameter_values_dict,variant=None,covariance_prefactor_dict=None,):\n"
        )
        write_one_function(
            f,
            name_models,
            components,
            parameter_models,
            partial_derivative_dictionnary_list,
        )
    else:
        f.write(
            "def get_partial_derivative_coefficients(model_kind,parameter_values_dict,variant=None,covariance_prefactor_dict=None,):\n"
        )
        f.write("    if model_kind == 'density':\n")
        f.write(
            "        return get_partial_derivative_coefficients_density(parameter_values_dict,variant=variant,)\n"
        )
        f.write("    elif model_kind == 'velocity':\n")
        f.write(
            "        return get_partial_derivative_coefficients_velocity(parameter_values_dict,variant=variant,)\n"
        )
        f.write("    elif model_kind == 'density_velocity':\n")
        f.write(
            "        return get_partial_derivative_coefficients_density_velocity(parameter_values_dict,variant=variant,)\n"
        )
        f.write("    elif model_kind == 'full':\n")
        f.write(
            "        return get_partial_derivative_coefficients_full(parameter_values_dict,variant=variant,)\n"
        )
        f.write("\n")

        components_to_treat = [["vv"], ["gg"], ["gg", "vv"], ["gg", "gv", "vv"]]
        component_names = ["velocity", "density", "density_velocity", "full"]
        for i in range(len(components_to_treat)):
            partial_derivative_dictionnary_list = (
                compute_partial_derivative_dictionnary(
                    components_to_treat[i],
                    parameter_models,
                    coefficient_models,
                )
            )

            f.write(
                f"def get_partial_derivative_coefficients_{component_names[i]}(parameter_values_dict,variant=None,):\n"
            )
            write_one_function(
                f,
                name_models,
                components_to_treat[i],
                parameter_models,
                partial_derivative_dictionnary_list,
            )

    f.write("\n")
    f.close()


def write_one_function(
    f,
    name_models,
    components,
    parameter_models,
    partial_derivative_dictionnary_list,
):
    if len(name_models) == 1:
        f.write("    partial_coefficients_dict = {\n")
        for parameter in parameter_models[0]:
            f.write("'" + parameter + "': {\n")
            for component in components:
                f.write(
                    "'"
                    + component
                    + "':"
                    + partial_derivative_dictionnary_list[0][parameter][
                        component
                    ].replace("math.", "np.")
                    + ",\n"
                )
            f.write("},\n")
        f.write("}\n")
    else:
        for i_model, name in enumerate(name_models):
            if name is None:
                f.write("    else:\n")
            elif i_model > 0:
                f.write(f"    elif variant == '{name}':\n")
            else:
                f.write(f"    if variant == '{name}':\n")
            f.write("        partial_coefficients_dict = {\n")
            for parameter in parameter_models[i_model]:
                f.write("'" + parameter + "': {\n")
                for component in components:
                    f.write(
                        "'"
                        + component
                        + "':"
                        + partial_derivative_dictionnary_list[i_model][parameter][
                            component
                        ].replace("math.", "np.")
                        + ",\n"
                    )
                f.write("},\n")
            f.write("}\n")
    f.write("    return partial_coefficients_dict")
    f.write("\n")


def generate_fisher_terms(config_file, filename=None):
    """Generate a fisher_terms.py file from a YAML model description.

    The YAML file gives, for each variant, the fitted parameters (in the order
    of the Fisher matrix) and the coefficients w_ab,n of the covariance
    decomposition C_ab = sum_n w_ab,n C_ab,n (Ravoux et al. 2025), keyed by the
    term indices n of the model flip_terms.yml. Example::

        variants:
          growth_index:                    # variant name, null = default
            parameters: [Omegam, gamma, s8]
            coefficients:
              vv: {'0': Omegam**(2*gamma)*s8**2}
          null:
            parameters: [fs8]
            coefficients:
              vv: {'0': fs8**2}

    Args:
        config_file: Path to the YAML file.
        filename: Output file. Defaults to fisher_terms.py next to the YAML file.
    """
    import yaml

    directory = os.path.dirname(os.path.abspath(config_file))
    with open(config_file) as f:
        config = yaml.safe_load(f)
    if filename is None:
        filename = os.path.join(directory, "fisher_terms.py")

    flip_terms_config = os.path.join(directory, "flip_terms.yml")
    terms = None
    if os.path.exists(flip_terms_config):
        with open(flip_terms_config) as f:
            terms = yaml.safe_load(f)["terms"]

    name_models, parameter_models, coefficient_models = [], [], []
    for name, variant in config["variants"].items():
        parameters = variant["parameters"]
        namespace = {parameter: _parameter_symbol(parameter) for parameter in parameters}
        coefficients = {}
        for component, coefficient_terms in variant["coefficients"].items():
            if terms is not None and list(coefficient_terms) != list(terms[component]):
                raise ValueError(
                    f"Variant {name}, block {component}: term indices differ from flip_terms.yml"
                )
            coefficients[component] = []
            for coefficient in coefficient_terms.values():
                expression = sy.sympify(str(coefficient), locals=namespace)
                unknown = expression.free_symbols - set(namespace.values())
                if unknown:
                    raise ValueError(
                        f"Variant {name}, block {component}: {unknown} not in parameters"
                    )
                coefficients[component].append(expression)
        name_models.append(name)
        parameter_models.append(parameters)
        coefficient_models.append(coefficients)

    write_partial_derivatives(
        filename,
        name_models,
        list(coefficient_models[0]),
        parameter_models,
        coefficient_models,
    )


def generate_files(number_worker=8):
    """Regenerate flip_terms.py for every model directory holding a flip_terms.yml."""
    root = os.path.dirname(os.path.abspath(__file__))
    for config_file in sorted(glob.glob(os.path.join(root, "*", "flip_terms.yml"))):
        generate_flip_terms(config_file, number_worker=number_worker)


def generate_fisher_files():
    """Regenerate fisher_terms.py for every model directory holding a fisher_terms.yml."""
    root = os.path.dirname(os.path.abspath(__file__))
    for config_file in sorted(glob.glob(os.path.join(root, "*", "fisher_terms.yml"))):
        generate_fisher_terms(config_file)
