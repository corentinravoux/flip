import sympy as sy
from sympy.printing import pycode
from sympy.physics import wigner
from sympy.simplify.fu import TR8
from sympy.polys.orthopolys import legendre_poly
import multiprocessing as mp


import sympy as sy
from sympy.printing import pycode
from sympy.physics import wigner
from sympy.simplify.fu import TR8
from sympy.polys.orthopolys import legendre_poly
import multiprocessing as mp


def simplify_term(
    term,
    max_simplification=20,
):
    """simplification tests with https://docs.sympy.org/latest/tutorials/intro-tutorial/simplification.html
    https://docs.sympy.org/latest/modules/simplify/simplify.html#ratsimpmodprime
    https://docs.sympy.org/dev/modules/simplify/fu.html#sympy.simplify.fu
    """
    term_simplified = sy.factor(TR8(term))
    i = 0
    while (term_simplified != term) & (i < max_simplification):
        term_simplified, term = sy.factor(TR8(term_simplified)), term_simplified
        i += 1
    return term_simplified


def generate_K_ab_i_l_function_wide_angle(
    B_ab_i,
    l,
    l1max,
    l2max,
):
    theta, phi = sy.symbols("theta phi")
    mu1, mu2 = sy.symbols("mu1 mu2")
    K_l = 0
    for l1 in range(l1max + 1):
        integral_mu1 = sy.integrate(B_ab_i * legendre_poly(l1, x=mu1), (mu1, -1, 1))
        if integral_mu1 == 0:
            continue
        for l2 in range(l2max + 1):
            integral_mu1_mu2 = sy.integrate(
                integral_mu1 * legendre_poly(l2, x=mu2), (mu2, -1, 1)
            )
            if integral_mu1_mu2 == 0:
                continue
            K_l_l1_l2 = 0
            for m in range(-l, l + 1):
                for m1 in range(-l1, l1 + 1):
                    for m2 in range(-l2, l2 + 1):
                        K_l_l1_l2_m_m1_m2 = wigner.gaunt(l, l1, l2, m, m1, m2)
                        K_l_l1_l2_m_m1_m2 *= (
                            sy.Ynm_c(l, m, sy.pi - phi, 0)
                            * sy.Ynm_c(l1, m1, theta / 2, sy.pi)
                            * sy.Ynm_c(l2, m2, theta / 2, 0)
                        )
                        K_l_l1_l2 = K_l_l1_l2 + K_l_l1_l2_m_m1_m2

            K_l = K_l + integral_mu1_mu2 * K_l_l1_l2

    prefactor = (sy.Rational(4) * sy.pi) ** 2 * (1 / sy.Rational(4))
    K_l = K_l * prefactor
    K_l = simplify_term(K_l.expand(func=True))

    return K_l


def write_K_functions_wide_angle(
    filename,
    type_list,
    term_index_list,
    l_list,
    l1max_list,
    l2max_list,
    B_dict,
    additional_parameters=None,
    number_worker=1,
):
    params_pool = []
    index_pool = {}
    index = 0
    for i, t in enumerate(term_index_list):
        for j, l in enumerate(l_list):
            B_ab_i = B_dict[f"B_{type_list[i]}_{t}"]
            params_pool.append([B_ab_i, l, l1max_list[j], l2max_list[j]])
            index_pool[f"{type_list[i]}_{t}_{l}"] = index
            index = index + 1

    if number_worker == 1:
        output_K = [
            generate_K_ab_i_l_function_wide_angle(*param) for param in params_pool
        ]
    else:
        with mp.Pool(number_worker) as pool:
            output_K = pool.starmap(generate_K_ab_i_l_function_wide_angle, params_pool)

    f = open(filename, "w")
    f.write("import numpy as np\n")
    f.write("\n")
    f.write("\n")

    for i, t in enumerate(term_index_list):
        for _, l in enumerate(l_list):
            K_ab_i_l = pycode(output_K[index_pool[f"{type_list[i]}_{t}_{l}"]]).replace(
                "math.", "np."
            )

            additional_str = ""
            if additional_parameters is not None:
                for add in additional_parameters:
                    additional_str = additional_str + f",{add}"
            f.write(f"def K_{type_list[i]}_{i}_{l}(theta,phi,k{additional_str}):\n")
            f.write(f"    def func(k):\n")
            f.write(f"        return({K_ab_i_l})\n")
            f.write(f"    return(func)\n")
            f.write("\n")
    f.write("\n")
    f.close()


def generate_K_ab_i_l_function_parallel_plane(B_ab_i_l, l):
    theta, phi = sy.symbols("theta phi")
    mu = sy.symbols("mu")
    K_l = (
        sy.Rational((2 * l + 1) / 2)
        * legendre_poly(l, x=sy.cos(phi))
        * sy.integrate(B_ab_i_l * legendre_poly(l, x=mu), (mu, -1, 1))
    )
    K_l = simplify_term(K_l.expand(func=True))
    return K_l


def write_K_functions_parallel_plane(
    filename,
    type_list,
    term_index_list,
    l_list,
    B_dict,
    additional_parameters=None,
    number_worker=1,
):
    params_pool = []
    index_pool = {}
    index = 0
    for i, t in enumerate(term_index_list):
        for _, l in enumerate(l_list):
            B_ab_i = B_dict[f"B_{type_list[i]}_{t}"]
            params_pool.append([B_ab_i, l])
            index_pool[f"{type_list[i]}_{t}_{l}"] = index
            index = index + 1

    if number_worker == 1:
        output_K = [
            generate_K_ab_i_l_function_parallel_plane(*param) for param in params_pool
        ]
    else:
        with mp.Pool(number_worker) as pool:
            output_K = pool.starmap(
                generate_K_ab_i_l_function_parallel_plane, params_pool
            )

    f = open(filename, "w")
    f.write("import numpy as np\n")
    f.write("\n")
    f.write("\n")

    for i, t in enumerate(term_index_list):
        for _, l in enumerate(l_list):
            K_ab_i_l = pycode(output_K[index_pool[f"{type_list[i]}_{t}_{l}"]]).replace(
                "math.", "np."
            )
            additional_str = ""
            if additional_parameters is not None:
                for add in additional_parameters:
                    additional_str = additional_str + f",{add}"
            f.write(f"def K_{type_list[i]}_{i}_{l}(theta,phi,k{additional_str}):\n")
            f.write(f"    def func(k):\n")
            f.write(f"        return({K_ab_i_l})\n")
            f.write(f"    return(func)\n")
            f.write("\n")
    f.write("\n")
    f.close()
    return output_K


def generate_generalized_carreres23_functions(
    filename="./carreres23/flip_terms.py", number_worker=8
):
    mu1, mu2 = sy.symbols("mu1 mu2")
    k = sy.symbols("k", nonzero=True, finite=True)
    type_list = ["vv"]
    term_index_list = ["0"]
    l_list = [0, 2]
    l1max_list = [1, 2]
    l2max_list = [1, 2]
    B_dict = {"B_vv_0": mu1 * mu2 / k**2}
    write_K_functions_wide_angle(
        filename,
        type_list,
        term_index_list,
        l_list,
        l1max_list,
        l2max_list,
        B_dict,
        number_worker=number_worker,
    )


def generate_generalized_adamsblake20_functions(
    filename="./adamsblake20/flip_terms.py", number_worker=8
):
    mu = sy.symbols("mu")
    k = sy.symbols("k", nonzero=True, finite=True)
    sig_g = sy.symbols("sig_g", nonzero=True, finite=True)
    type_list = ["gg", "gg", "gg"] + ["gv", "gv"] + ["vv"]
    term_index_list = ["0", "1", "2"] + ["0", "1"] + ["0"]
    l_list = [0, 1, 2, 3, 4]
    B_dict = {
        "B_gg_0": sy.exp(-((k * sig_g * mu) ** 2)),
        "B_gg_1": mu**2 * sy.exp(-((k * sig_g * mu) ** 2)),
        "B_gg_2": mu**4 * sy.exp(-((k * sig_g * mu) ** 2)),
        "B_gv_0": (mu / k) * sy.exp(-((k * sig_g * mu) ** 2) / 2),
        "B_gv_1": (mu**3 / k) * sy.exp(-((k * sig_g * mu) ** 2) / 2),
        "B_vv_0": mu**2 / k**2,
    }

    write_K_functions_parallel_plane(
        filename,
        type_list,
        term_index_list,
        l_list,
        B_dict,
        additional_parameters=["sig_g"],
        number_worker=number_worker,
    )


def generate_generalized_ravouxcarreres_functions(
    filename="./ravouxcarreres/flip_terms.py", number_worker=8
):
    mu1, mu2 = sy.symbols("mu1 mu2")
    k = sy.symbols("k", nonzero=True, finite=True)
    sig_g = sy.symbols("sig_g", nonzero=True, finite=True)
    type_list = ["gg", "gg", "gg"] + ["gv", "gv"] + ["vv"]
    term_index_list = ["0", "1", "2"] + ["0", "1"] + ["0"]
    l_list = [0, 1, 2, 3, 4]
    l1max_list = [1, 2, 3, 4, 5, 6]
    l2max_list = [1, 2, 3, 4, 5, 6]

    B_dict = {
        "B_gg_0": sy.exp(-((k * sig_g) ** 2) * (mu1**2 + mu2**2) / 2),
        "B_gg_1": (mu1**2 + mu2**2)
        * sy.exp(-((k * sig_g) ** 2) * (mu1**2 + mu2**2) / 2),
        "B_gg_2": mu1**2
        * mu2**2
        * sy.exp(-((k * sig_g) ** 2) * (mu1**2 + mu2**2) / 2),
        "B_gv_0": (mu1 / k) * sy.exp(-((k * sig_g * mu2) ** 2) / 2),
        "B_gv_1": (mu1 * mu2**2 / k) * sy.exp(-((k * sig_g * mu2) ** 2) / 2),
        "B_vv_0": mu1 * mu2 / k**2,
    }

    write_K_functions_wide_angle(
        filename,
        type_list,
        term_index_list,
        l_list,
        l1max_list,
        l2max_list,
        B_dict,
        additional_parameters=["sig_g"],
        number_worker=number_worker,
    )


def generate_files():
    generate_generalized_carreres23_functions()
    generate_generalized_adamsblake20_functions()
    generate_generalized_ravouxcarreres_functions()


generate_generalized_adamsblake20_functions()
