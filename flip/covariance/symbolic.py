import itertools
import multiprocessing as mp

import numpy as np
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
    term_B,
    ell,
    l1,
    l2,
):
    """
    The generate_MN_ab_i_l_function_wide_angle function generates the M and N terms for a given l, l', and l'' value.

    Args:
        term_B: Define the integrand of the integral over mu_2
        l: Determine the order of the legendre polynomial
        l1: Define the first legendre polynomial in the integral
        l2: Determine the order of the legendre polynomial in the second integral
        : Determine the number of terms in the sum

    Returns:
        The terms m_l,l',l'' and n_l, l', l''

    """
    theta, phi = sy.symbols("theta phi")
    mu1, mu2 = sy.symbols("mu1 mu2")
    integral_mu1_M_l = sy.integrate(term_B * legendre_poly(l1, x=mu1), (mu1, -1, 1))
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
                # This is understood by the asymetry term inside the B_ab modeling.
                # Changing this will have an impact on the modeling.
                # Be sure to accord B_ab terms with sperical harmonic definition here.

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
    term_N_l_l1_l2 = (sy.Rational(4) * sy.pi) ** 2 * term_N_l_l1_l2.expand(func=True)
    term_M_l_l1_l2 = simplify_term(
        term_M_l_l1_l2,
        simplification_method="simplify_iteration",
    )
    term_N_l_l1_l2 = simplify_term(
        term_N_l_l1_l2,
        simplification_method="tr8_iteration",
    )

    return term_M_l_l1_l2, term_N_l_l1_l2


def generate_MN_ab_i_l_function_parallel_plane(term_B, ell):
    """
    The generate_MN_ab_i_l_function_parallel_plane function takes in a term_B and an l value.
    It then generates the M_l and N_l functions for that particular term B, which is used to calculate the parallel plane integral.
    The function returns both M_l and N_l as sympy expressions.

    Args:
        term_B: Define the term b in the equation for m_l and n_l
        l: Determine the order of the legendre polynomial

    Returns:
        A tuple of

    """
    phi = sy.symbols("phi")
    mu = sy.symbols("mu")
    M_l = sy.Rational(1 / 2) * sy.integrate(
        term_B * legendre_poly(ell, x=mu), (mu, -1, 1)
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
    f.write("import mpmath\n")
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
    dict_B,
    additional_parameters=None,
    number_worker=1,
    wide_angle=False,
    l1max_list=None,
    l2max_list=None,
    multi_index_model=False,
    regularize_M_terms="None",
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
        dict_B: Store the b_ab_i values
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
                B_ab_i = dict_B[f"B_{type}_{t}"]
                if wide_angle:
                    for l1 in range(l1max_list[k][i] + 1):
                        for l2 in range(l2max_list[k][i] + 1):
                            params_pool.append([B_ab_i, ell, l1, l2])
                            index_pool[f"{type}_{t}_{ell}_{l1}_{l2}"] = index
                            index = index + 1
                else:
                    params_pool.append([B_ab_i, ell])
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


def generate_generalized_genericzdep_functions(
    filename="./analytical/genericzdep/flip_terms.py", number_worker=8
):
    """
    The generate_generalized_genericzdep_functions function generates the flip_terms.py file in the genericzdep directory, which contains functions that calculate M and N terms for a generalized version of Carreres' (2012) model 2 and 3.

    Args:
        filename: Specify the name of the file that will be generated
        number_worker: Determine the number of processes to use for multiprocessing

    Returns:
        A list of functions,

    """
    mu1, mu2 = sy.symbols("mu1 mu2")
    k = sy.symbols("k", positive=True, finite=True, real=True)
    kNL = sy.symbols("kNL", positive=True, finite=True, real=True)
    additional_parameters = ["kNL"]

    type_list = ["vv"]
    term_index_list = [["0", "1", "2"]]
    lmax_list = [[2, 2, 2]]
    l1max_list = [[1, 1, 1]]
    l2max_list = [[1, 1, 1]]
    dict_B = {
        "B_vv_0": mu1 * mu2 / k**2,
        "B_vv_1": mu1 * mu2 / kNL**2,
        "B_vv_2": mu1 * mu2 * k**2 / kNL**4,
    }

    regularize_M_terms = "None"
    write_M_N_functions(
        filename,
        type_list,
        term_index_list,
        lmax_list,
        dict_B,
        number_worker=number_worker,
        wide_angle=True,
        l1max_list=l1max_list,
        l2max_list=l2max_list,
        regularize_M_terms=regularize_M_terms,
        additional_parameters=additional_parameters,
    )


def generate_generalized_adamsblake17plane_functions(
    filename="./analytical/adamsblake17plane/flip_terms.py", number_worker=8
):
    mu = sy.symbols("mu")
    k = sy.symbols("k", positive=True, finite=True, real=True)
    type_list = ["gg", "gv", "vv"]
    term_index_list = [["0"], ["0"], ["0"]]
    lmax_list = [[0], [1], [2]]
    dict_B = {
        "B_gg_0": 1,
        "B_gv_0": 100 * (mu / k),
        "B_vv_0": 100**2 * mu**2 / k**2,
    }
    regularize_M_terms = "None"

    write_M_N_functions(
        filename,
        type_list,
        term_index_list,
        lmax_list,
        dict_B,
        number_worker=number_worker,
        wide_angle=False,
        regularize_M_terms=regularize_M_terms,
    )


def generate_generalized_adamsblake17_functions(
    filename="./analytical/adamsblake17/flip_terms.py", number_worker=8
):

    mu1, mu2 = sy.symbols("mu1 mu2")
    k = sy.symbols("k", positive=True, finite=True, real=True)
    type_list = ["gg", "gv", "vv"]
    term_index_list = [["0"], ["0"], ["0"]]
    lmax_list = [[2], [2], [2]]
    l1max_list = [[2], [2], [2]]
    l2max_list = [[2], [2], [2]]
    dict_B = {
        "B_gg_0": 1,
        "B_gv_0": 100 * (mu2 / k),
        "B_vv_0": 100**2 * mu1 * mu2 / k**2,
    }

    regularize_M_terms = "None"
    write_M_N_functions(
        filename,
        type_list,
        term_index_list,
        lmax_list,
        dict_B,
        number_worker=number_worker,
        wide_angle=True,
        l1max_list=l1max_list,
        l2max_list=l2max_list,
        regularize_M_terms=regularize_M_terms,
    )


def generate_generalized_adamsblake20_functions(
    filename="./analytical/adamsblake20/flip_terms.py", number_worker=8
):
    """
    The generate_generalized_adamsblake20_functions function generates the functions needed to compute the M and N matrices for a generalized version of Adams, Blake &amp; Kitching (2020).

    Args:
        filename: Specify the name of the file where the functions will be written
        number_worker: Parallelize the computation of the m and n functions

    Returns:
        A dictionary of functions

    """
    mu = sy.symbols("mu")
    k = sy.symbols("k", positive=True, finite=True, real=True)
    sig_g = sy.symbols("sig_g", positive=True, finite=True, real=True)
    type_list = ["gg", "gv", "vv"]
    term_index_list = [["0", "1", "2"], ["0", "1"], ["0"]]
    lmax_list = [[4, 4, 4], [3, 3], [2]]  # lmax list to stick to AD20
    # lmax_list = [[6, 6, 6], [5, 5], [2]]  # lmax list to generate lmax + 1
    dict_B = {
        "B_gg_0": sy.exp(-((k * sig_g * mu) ** 2)),
        "B_gg_1": 2 * mu**2 * sy.exp(-((k * sig_g * mu) ** 2)),
        "B_gg_2": mu**4 * sy.exp(-((k * sig_g * mu) ** 2)),
        "B_gv_0": 100 * (mu / k) * sy.exp(-((k * sig_g * mu) ** 2) / 2),
        "B_gv_1": 100 * (mu**3 / k) * sy.exp(-((k * sig_g * mu) ** 2) / 2),
        "B_vv_0": 100**2 * mu**2 / k**2,
    }
    regularize_M_terms = """{"gg": "mpmath", "gv": "mpmath", "vv": None}"""

    write_M_N_functions(
        filename,
        type_list,
        term_index_list,
        lmax_list,
        dict_B,
        additional_parameters=["sig_g"],
        number_worker=number_worker,
        wide_angle=False,
        regularize_M_terms=regularize_M_terms,
    )


def generate_generalized_lai22_functions(
    filename="./analytical/lai22/flip_terms.py", number_worker=8
):
    """
    The generate_generalized_lai22_functions function generates the functions for calculating the M and N terms in
    the generalized Lai22 model. The function is called by running:

    Args:
        filename: Specify the name of the file where all functions will be written
        number_worker: Determine the number of processes that will be used to generate the functions

    Returns:
        A set of functions that are used in the calculation of the scattering matrix

    """
    mu1, mu2 = sy.symbols("mu1 mu2")
    k = sy.symbols("k", positive=True, finite=True, real=True)

    # gg
    term_index_list_gg = []
    lmax_list_gg = []
    l1max_list_gg = []
    l2max_list_gg = []

    pmax_gg, qmax_gg = 3, 3
    p_index = np.arange(pmax_gg + 1)
    q_index = np.arange(qmax_gg + 1)
    m_index_gg = np.arange(0, (qmax_gg + pmax_gg) + 1)
    iter_pq = np.array(list(itertools.product(p_index, q_index)))
    sum_iter_pq = np.sum(iter_pq, axis=1)
    dict_B = {}
    for i in ["0", "1", "2"]:
        for m_value in m_index_gg:
            pq_index = iter_pq[sum_iter_pq == m_value]
            term_index_list_gg.append(f"{i}_{m_value}")
            if i == "0":
                lmax_list_gg.append(2 * m_value)
                l1max_list_gg.append(min(2 * m_value, 2 * pmax_gg))
                l2max_list_gg.append(min(2 * m_value, 2 * qmax_gg))
            elif i == "1":
                lmax_list_gg.append(2 * (m_value + 1))
                l1max_list_gg.append(min(2 * (m_value + 1), 2 * (pmax_gg + 1)))
                l2max_list_gg.append(min(2 * (m_value + 1), 2 * (qmax_gg + 1)))
            elif i == "2":
                lmax_list_gg.append(2 * (m_value + 2))
                l1max_list_gg.append(min(2 * (m_value + 1), 2 * (pmax_gg + 1)))
                l2max_list_gg.append(min(2 * (m_value + 1), 2 * (qmax_gg + 1)))

            B_gg_i = 0
            for pq in pq_index:
                p, q = pq[0], pq[1]

                if i == "0":
                    add = 1
                elif i == "1":
                    add = mu1**2 + mu2**2
                elif i == "2":
                    add = mu1**2 * mu2**2
                B_gg_i = (
                    B_gg_i
                    + (
                        (-1) ** (p + q)
                        / (2 ** (p + q) * sy.factorial(p) * sy.factorial(q))
                    )
                    * k ** (2 * (p + q))
                    * mu1 ** (2 * p)
                    * mu2 ** (2 * q)
                    * add
                )
                dict_B[f"B_gg_{i}_{m_value}"] = B_gg_i

    # gv
    term_index_list_gv = []
    lmax_list_gv = []
    l1max_list_gv = []
    l2max_list_gv = []

    pmax_gv = 3
    m_index_gv = np.arange(pmax_gv + 1)

    for i in ["0", "1"]:
        for m_value in m_index_gv:
            term_index_list_gv.append(f"{i}_{m_value}")
            l2max_list_gv.append(1)
            B_gv_i = 0
            if i == "0":
                add = 1
                lmax_list_gv.append(2 * m_value + 1)
                l1max_list_gv.append(2 * m_value)
            elif i == "1":
                add = mu1**2
                lmax_list_gv.append(2 * m_value + 3)
                l1max_list_gv.append(2 * m_value + 2)
            B_gv_i = (
                B_gv_i
                + ((-1) ** (m_value) / (2 ** (m_value) * sy.factorial(m_value)))
                * k ** (2 * m_value - 1)
                * mu1 ** (2 * m_value)
                * mu2
                * add
            )
            dict_B[f"B_gv_{i}_{m_value}"] = 100 * B_gv_i

    # vv
    lmax_list_vv = [2]
    l1max_list_vv = [1]
    l2max_list_vv = [1]
    term_index_list_vv = ["0_0"]
    dict_B["B_vv_0_0"] = 100**2 * mu1 * mu2 / k**2

    type_list = ["gg", "gv", "vv"]
    term_index_list = [term_index_list_gg, term_index_list_gv, term_index_list_vv]
    lmax_list = [lmax_list_gg, lmax_list_gv, lmax_list_vv]
    l1max_list = [l1max_list_gg, l1max_list_gv, l1max_list_vv]
    l2max_list = [l2max_list_gg, l2max_list_gv, l2max_list_vv]

    regularize_M_terms = "None"
    write_M_N_functions(
        filename,
        type_list,
        term_index_list,
        lmax_list,
        dict_B,
        additional_parameters=None,
        number_worker=number_worker,
        wide_angle=True,
        l1max_list=l1max_list,
        l2max_list=l2max_list,
        multi_index_model=True,
        regularize_M_terms=regularize_M_terms,
    )


def generate_generalized_carreres23_functions(
    filename="./analytical/carreres23/flip_terms.py", number_worker=8
):
    """
    The generate_generalized_carreres23_functions function generates the flip_terms.py file in the carreres23 directory, which contains functions that calculate M and N terms for a generalized version of Carreres' (2012) model 2 and 3.

    Args:
        filename: Specify the name of the file that will be generated
        number_worker: Determine the number of processes to use for multiprocessing

    Returns:
        A list of functions,

    """
    mu1, mu2 = sy.symbols("mu1 mu2")
    k = sy.symbols("k", positive=True, finite=True, real=True)
    type_list = ["vv"]
    term_index_list = [["0"]]
    lmax_list = [[2]]
    l1max_list = [[1]]
    l2max_list = [[1]]
    dict_B = {"B_vv_0": 100**2 * mu1 * mu2 / k**2}

    regularize_M_terms = "None"
    write_M_N_functions(
        filename,
        type_list,
        term_index_list,
        lmax_list,
        dict_B,
        number_worker=number_worker,
        wide_angle=True,
        l1max_list=l1max_list,
        l2max_list=l2max_list,
        regularize_M_terms=regularize_M_terms,
    )


def generate_generalized_ravouxcarreres_functions(
    filename="./analytical/ravouxcarreres/flip_terms.py", number_worker=8
):
    """
    The generate_generalized_ravouxcarreres_functions function generates the functions needed to compute the generalized Ravoux-Carreres model.

    Args:
        filename: Specify the name of the file where you want to write your functions
        number_worker: Specify the number of cores to use for multiprocessing

    Returns:
        A dictionary of functions

    """
    mu1, mu2 = sy.symbols("mu1 mu2")
    k = sy.symbols("k", positive=True, finite=True, real=True)
    sig_g = sy.symbols("sig_g", positive=True, finite=True, real=True)
    type_list = ["gg", "gv", "vv"]
    term_index_list = [["0", "1", "2"], ["0", "1"], ["0"]]
    lmax_list = [[4, 4, 4], [3, 3], [2]]  # Lists to follow the logic of AD20
    l1max_list = [[2, 2, 2], [2, 2], [1]]
    l2max_list = [[2, 2, 2], [1, 1], [1]]
    # lmax_list = [[6, 6, 6] , [5, 5] , [2]] # Lists to generate the model lmax + 1
    # l1max_list = [[4, 4, 4] , [4, 4] , [1]]
    # l2max_list = [[4, 4, 4] , [1, 1] , [1]]
    dict_B = {
        "B_gg_0": sy.exp(-((k * sig_g) ** 2) * (mu1**2 + mu2**2) / 2),
        "B_gg_1": (mu1**2 + mu2**2)
        * sy.exp(-((k * sig_g) ** 2) * (mu1**2 + mu2**2) / 2),
        "B_gg_2": (mu1**2 * mu2**2)
        * sy.exp(-((k * sig_g) ** 2) * (mu1**2 + mu2**2) / 2),
        "B_gv_0": 100 * (mu2 / k) * sy.exp(-((k * sig_g * mu1) ** 2) / 2),
        "B_gv_1": 100 * (mu2 * mu1**2 / k) * sy.exp(-((k * sig_g * mu1) ** 2) / 2),
        "B_vv_0": 100**2 * mu1 * mu2 / k**2,
    }

    regularize_M_terms = """{"gg": "mpmath", "gv": "mpmath", "vv": None}"""
    write_M_N_functions(
        filename,
        type_list,
        term_index_list,
        lmax_list,
        dict_B,
        additional_parameters=["sig_g"],
        number_worker=number_worker,
        wide_angle=True,
        l1max_list=l1max_list,
        l2max_list=l2max_list,
        regularize_M_terms=regularize_M_terms,
    )


def generate_generalized_rcrk24_functions(
    filename="./analytical/rcrk24/flip_terms.py", number_worker=8
):
    """
    The generate_generalized_rcrk24_functions function generates the flip_terms.py file in the carreres23 directory, which contains functions that calculate M and N terms for a generalized version of Carreres' (2012) model 2 and 3.

    Args:
        filename: Specify the name of the file that will be generated
        number_worker: Determine the number of processes to use for multiprocessing

    Returns:
        A list of functions,

    """
    mu1, mu2 = sy.symbols("mu1 mu2")
    k = sy.symbols("k", positive=True, finite=True, real=True)
    type_list = ["vv"]
    term_index_list = [["0"]]
    lmax_list = [[2]]
    l1max_list = [[1]]
    l2max_list = [[1]]
    dict_B = {"B_vv_0": 100**2 * mu1 * mu2 / k**2}

    regularize_M_terms = "None"
    write_M_N_functions(
        filename,
        type_list,
        term_index_list,
        lmax_list,
        dict_B,
        number_worker=number_worker,
        wide_angle=True,
        l1max_list=l1max_list,
        l2max_list=l2max_list,
        regularize_M_terms=regularize_M_terms,
    )


def compute_partial_derivative_dictionnary(
    name_models,
    components,
    parameter_models,
    all_parameters,
    coefficient_models,
):
    """
    The compute_partial_derivatiove_dictionnary function computes the partial derivatives of each component of the model with respect to each parameter.
    The output is a list containing dictionnaries, one for each model. Each dictionnary contains as keys all parameters and as values another dictionnary which contains as keys all components and as values lists containing the partial derivatives.

    Args:
        name_models: Name the models
        parameter_models: Get the parameters of each model
        components: Create the keys of the dictionnaries in compute_partial_derivatiove_dictionnary
        coefficient_models: Compute the partial derivatives of each component with respect to each parameter
        : Create the keys of the dictionnaries in compute_partial_derivatiove_dictionnary

    Returns:
        A list containing dictionnaries, one for each model

    Doc Author:
        Trelent
    """

    parameter_symbolic_dict = {}
    for param in all_parameters:
        parameter_symbolic_dict[param] = sy.symbols(
            f'parameter_values_dict["{param}"]', positive=True, finite=True, real=True
        )

    partial_derivative_dictionnary_list = []
    for i in range(len(name_models)):
        partial_derivative_dictionnary = {}
        for parameter in parameter_models[i]:
            partial_derivative_dictionnary[parameter] = {}
            for component in components:
                derivative = "["
                model = eval(coefficient_models[i][component])
                for j in range(len(model)):
                    derivative = (
                        derivative
                        + pycode(sy.diff(model[j], parameter_symbolic_dict[parameter]))
                        + ","
                    )
                derivative = derivative + "]"
                partial_derivative_dictionnary[parameter][component] = derivative
        partial_derivative_dictionnary_list.append(partial_derivative_dictionnary)

    return partial_derivative_dictionnary_list


def write_partial_derivatives(
    filename,
    name_models,
    components,
    parameter_models,
    all_parameters,
    coefficient_models,
):

    f = open(filename, "w")
    f.write("import numpy as np\n")
    f.write("\n")
    f.write("\n")
    if len(components) == 1:

        partial_derivative_dictionnary_list = compute_partial_derivative_dictionnary(
            name_models,
            components,
            parameter_models,
            all_parameters,
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
                    name_models,
                    components_to_treat[i],
                    parameter_models,
                    all_parameters,
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
        for parameter in parameter_models:
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


def generate_fisher_coefficients_dictionnary_carreres23(
    filename="./analytical/carreres23/fisher_terms.py",
):

    name_models = ["growth_index", None]
    components = ["vv"]
    parameter_models = [["Omegam", "gamma", "s8"], ["fs8"]]
    all_parameters = ["Omegam", "gamma", "s8", "fs8"]
    coefficient_models = [{"vv": "[(Omegam**gamma * s8)**2]"}, {"vv": "[fs8**2]"}]

    write_partial_derivatives(
        filename,
        name_models,
        components,
        parameter_models,
        all_parameters,
        coefficient_models,
    )


def generate_fisher_coefficients_dictionnary_adamsblake17(
    filename="./analytical/adamsblake17/fisher_terms.py",
):

    name_models = ["growth_index", None]
    components = ["gg", "gv", "vv"]
    parameter_models = [
        ["Omegam", "gamma", "s8", "bs8"],
        ["fs8", "bs8"],
    ]
    all_parameters = ["Omegam", "gamma", "s8", "fs8", "bs8"]
    coefficient_models = [
        {
            "gg": "[bs8**2]",
            "gv": "[bs8*Omegam**gamma*s8]",
            "vv": "[(Omegam**gamma*s8)**2]",
        },
        {
            "gg": "[bs8**2]",
            "gv": "[bs8*fs8]",
            "vv": "[fs8**2]",
        },
    ]

    write_partial_derivatives(
        filename,
        name_models,
        components,
        parameter_models,
        all_parameters,
        coefficient_models,
    )


def generate_fisher_coefficients_dictionnary_adamsblake17plane(
    filename="./analytical/adamsblake17plane/fisher_terms.py",
):

    name_models = ["growth_index", None]
    components = ["gg", "gv", "vv"]
    parameter_models = [
        ["Omegam", "gamma", "s8", "bs8"],
        ["fs8", "bs8"],
    ]
    all_parameters = ["Omegam", "gamma", "s8", "fs8", "bs8"]
    coefficient_models = [
        {
            "gg": "[bs8**2]",
            "gv": "[bs8*Omegam**gamma*s8]",
            "vv": "[(Omegam**gamma*s8)**2]",
        },
        {
            "gg": "[bs8**2]",
            "gv": "[bs8*fs8]",
            "vv": "[fs8**2]",
        },
    ]

    write_partial_derivatives(
        filename,
        name_models,
        components,
        parameter_models,
        all_parameters,
        coefficient_models,
    )


def generate_fisher_coefficients_dictionnary_full_nosigmag(
    filename,
):

    name_models = ["growth_index", "growth_index_nobeta", "nobeta", None]
    components = ["gg", "gv", "vv"]
    parameter_models = [
        ["Omegam", "gamma", "s8", "bs8", "beta_f"],
        ["Omegam", "gamma", "s8", "bs8"],
        ["fs8", "bs8"],
        ["fs8", "bs8", "beta_f"],
    ]
    all_parameters = ["Omegam", "gamma", "s8", "fs8", "bs8", "beta_f"]
    coefficient_models = [
        {
            "gg": "[bs8**2, bs8**2*beta_f, bs8**2*beta_f**2]",
            "gv": "[bs8*Omegam**gamma*s8, bs8*Omegam**gamma*s8*beta_f]",
            "vv": "[(Omegam**gamma*s8)**2]",
        },
        {
            "gg": "[bs8**2, bs8*Omegam**gamma*s8, (Omegam**gamma*s8)**2]",
            "gv": "[bs8*Omegam**gamma*s8, (Omegam**gamma*s8)**2]",
            "vv": "[(Omegam**gamma*s8)**2]",
        },
        {
            "gg": "[bs8**2, bs8*fs8, fs8**2]",
            "gv": "[bs8*fs8, fs8**2]",
            "vv": "[fs8**2]",
        },
        {
            "gg": "[bs8**2, bs8**2*beta_f, bs8**2*beta_f**2]",
            "gv": "[bs8*fs8, bs8*fs8*beta_f]",
            "vv": "[fs8**2]",
        },
    ]

    write_partial_derivatives(
        filename,
        name_models,
        components,
        parameter_models,
        all_parameters,
        coefficient_models,
    )


def generate_fisher_coefficients_dictionnary_lai22(
    filename="./analytical/lai22/fisher_terms.py",
):
    from flip.covariance.analytical.lai22.flip_terms import dictionary_terms

    name_models = ["growth_index", "growth_index_nobeta", "nobeta", None]
    components = ["gg", "gv", "vv"]
    parameter_models = [
        ["Omegam", "gamma", "s8", "bs8", "beta_f", "sigg"],
        ["Omegam", "gamma", "s8", "bs8", "sigg"],
        ["fs8", "bs8", "sigg"],
        ["fs8", "bs8", "beta_f", "sigg"],
    ]
    all_parameters = ["Omegam", "gamma", "s8", "fs8", "bs8", "beta_f", "sigg"]

    gg_terms = dictionary_terms["gg"]
    gv_terms = dictionary_terms["gv"]

    coefficient_models = []

    coefficient_model_gg = "["
    for gg_term in gg_terms:
        term_index, m_index = np.array(gg_term.split("_")).astype(int)
        if term_index == 0:
            coefficient_model_gg = coefficient_model_gg + f"bs8**2*sigg**({2*m_index}),"
        elif term_index == 1:
            coefficient_model_gg = (
                coefficient_model_gg + f"bs8**2*beta_f*sigg**({2*m_index}),"
            )
        elif term_index == 2:
            coefficient_model_gg = (
                coefficient_model_gg + f"bs8**2*beta_f**2*sigg**({2*m_index}),"
            )
    coefficient_model_gg = coefficient_model_gg + "]"

    coefficient_model_gv = "["
    for gv_term in gv_terms:
        term_index, m_index = np.array(gv_term.split("_")).astype(int)
        if term_index == 0:
            coefficient_model_gv = (
                coefficient_model_gv + f"bs8*(Omegam**gamma*s8)*sigg**({2*m_index}),"
            )
        elif term_index == 1:
            coefficient_model_gv = (
                coefficient_model_gv + f"bs8**2*beta_f**2*sigg**({2*m_index}),"
            )
    coefficient_model_gv = coefficient_model_gv + "]"
    coefficient_models.append(
        {
            "gg": coefficient_model_gg,
            "gv": coefficient_model_gv,
            "vv": "[(Omegam**gamma*s8)**2]",
        }
    )

    coefficient_model_gg = "["
    for gg_term in gg_terms:
        term_index, m_index = np.array(gg_term.split("_")).astype(int)
        if term_index == 0:
            coefficient_model_gg = coefficient_model_gg + f"bs8**2*sigg**({2*m_index}),"
        elif term_index == 1:
            coefficient_model_gg = (
                coefficient_model_gg + f"bs8*(Omegam**gamma*s8)*sigg**({2*m_index}),"
            )
        elif term_index == 2:
            coefficient_model_gg = (
                coefficient_model_gg + f"(Omegam**gamma*s8)**2*sigg**({2*m_index}),"
            )
    coefficient_model_gg = coefficient_model_gg + "]"

    coefficient_model_gv = "["
    for gv_term in gv_terms:
        term_index, m_index = np.array(gv_term.split("_")).astype(int)
        if term_index == 0:
            coefficient_model_gv = (
                coefficient_model_gv + f"bs8*(Omegam**gamma*s8)*sigg**({2*m_index}),"
            )
        elif term_index == 1:
            coefficient_model_gv = (
                coefficient_model_gv + f"(Omegam**gamma*s8)**2*sigg**({2*m_index}),"
            )
    coefficient_model_gv = coefficient_model_gv + "]"
    coefficient_models.append(
        {
            "gg": coefficient_model_gg,
            "gv": coefficient_model_gv,
            "vv": "[(Omegam**gamma*s8)**2]",
        }
    )

    coefficient_model_gg = "["
    for gg_term in gg_terms:
        term_index, m_index = np.array(gg_term.split("_")).astype(int)
        if term_index == 0:
            coefficient_model_gg = coefficient_model_gg + f"bs8**2*sigg**({2*m_index}),"
        elif term_index == 1:
            coefficient_model_gg = (
                coefficient_model_gg + f"bs8*fs8*sigg**({2*m_index}),"
            )
        elif term_index == 2:
            coefficient_model_gg = coefficient_model_gg + f"fs8**2*sigg**({2*m_index}),"
    coefficient_model_gg = coefficient_model_gg + "]"

    coefficient_model_gv = "["
    for gv_term in gv_terms:
        term_index, m_index = np.array(gv_term.split("_")).astype(int)
        if term_index == 0:
            coefficient_model_gv = (
                coefficient_model_gv + f"bs8*fs8*sigg**({2*m_index}),"
            )
        elif term_index == 1:
            coefficient_model_gv = coefficient_model_gv + f"fs8**2*sigg**({2*m_index}),"
    coefficient_model_gv = coefficient_model_gv + "]"
    coefficient_models.append(
        {
            "gg": coefficient_model_gg,
            "gv": coefficient_model_gv,
            "vv": "[fs8**2]",
        }
    )

    coefficient_model_gg = "["
    for gg_term in gg_terms:
        term_index, m_index = np.array(gg_term.split("_")).astype(int)
        if term_index == 0:
            coefficient_model_gg = coefficient_model_gg + f"bs8**2*sigg**({2*m_index}),"
        elif term_index == 1:
            coefficient_model_gg = (
                coefficient_model_gg + f"bs8**2*beta_f*sigg**({2*m_index}),"
            )
        elif term_index == 2:
            coefficient_model_gg = (
                coefficient_model_gg + f"bs8**2*beta_f**2*sigg**({2*m_index}),"
            )
    coefficient_model_gg = coefficient_model_gg + "]"

    coefficient_model_gv = "["
    for gv_term in gv_terms:
        term_index, m_index = np.array(gv_term.split("_")).astype(int)
        if term_index == 0:
            coefficient_model_gv = (
                coefficient_model_gv + f"bs8*fs8*sigg**({2*m_index}),"
            )
        elif term_index == 1:
            coefficient_model_gv = (
                coefficient_model_gv + f"bs8**2*beta_f**2*sigg**({2*m_index}),"
            )
    coefficient_model_gv = coefficient_model_gv + "]"
    coefficient_models.append(
        {
            "gg": coefficient_model_gg,
            "gv": coefficient_model_gv,
            "vv": "[fs8**2]",
        }
    )

    write_partial_derivatives(
        filename,
        name_models,
        components,
        parameter_models,
        all_parameters,
        coefficient_models,
    )


def generate_files():
    """
    The generate_files function generates the following files:
        - generalized_carreres23.py
        - generalized_adamsblake20.py
        - generalized_lai22.py
        - generalized_ravouxcarreres.py

    Args:

    Returns:
        A list of all the functions that have been generated

    """
    generate_generalized_carreres23_functions()
    generate_generalized_adamsblake17_functions()
    generate_generalized_adamsblake17plane_functions()
    generate_generalized_adamsblake20_functions()
    generate_generalized_lai22_functions()
    generate_generalized_ravouxcarreres_functions()
    generate_generalized_rcrk24_functions()


def generate_fisher_files():
    """Generate Fisher coefficient modules across supported covariance models.

    Writes `fisher_terms.py` files for each model with partial derivative
    coefficient dictionaries tailored to their parameterizations.
    """
    generate_fisher_coefficients_dictionnary_carreres23()
    generate_fisher_coefficients_dictionnary_adamsblake17()
    generate_fisher_coefficients_dictionnary_adamsblake17plane()
    generate_fisher_coefficients_dictionnary_full_nosigmag(
        "./analytical/adamsblake20/fisher_terms.py"
    )
    generate_fisher_coefficients_dictionnary_full_nosigmag(
        "./analytical/ravouxcarreres/fisher_terms.py"
    )
    generate_fisher_coefficients_dictionnary_lai22()
