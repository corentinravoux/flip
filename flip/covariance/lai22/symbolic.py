import multiprocessing as mp

import sympy as sy
from sympy.physics import wigner
from sympy.polys.orthopolys import legendre_poly
from sympy.printing import pycode
from sympy.simplify.fu import TR8


def simplify_h(H, max_simplification=20):
    """Iteratively simplify a symbolic expression for H terms.

    Applies ``TR8`` and factoring repeatedly up to ``max_simplification`` times
    or until convergence. See SymPy docs on simplification strategies.

    Args:
        H: SymPy expression representing an H term.
        max_simplification: Maximum number of iterations to apply.

    Returns:
        Simplified SymPy expression.
    """
    H_simplified = sy.factor(TR8(H))
    i = 0
    while (H_simplified != H) & (i < max_simplification):
        H_simplified, H = sy.factor(TR8(H_simplified)), H_simplified
        i += 1
    return H_simplified


def generate_h_term(ell, p, q):
    """Generate the H term for Lai22 wide-angle coefficients.

    Constructs the angular function ``H_{ℓ}^{p,q}(θ,ϕ)`` using Gaunt integrals
    and Legendre polynomials, then simplifies the symbolic result.

    Args:
        ell: Multipole order ℓ (integer).
        p: Half-integer or integer index for point 1 (interpreted as rational).
        q: Half-integer or integer index for point 2 (interpreted as rational).

    Returns:
        SymPy expression of ``H(θ, ϕ)``.
    """
    theta, phi = sy.symbols("theta phi")
    mu = sy.symbols("mu")
    l1max = int(2 * p + 1)
    l2max = int(2 * q + 1)
    p = sy.Rational(p)
    q = sy.Rational(q)
    H = 0
    for l1 in range(l1max):
        for l2 in range(l2max):
            h_term = (sy.Rational(4) * sy.pi) ** 2 / sy.Rational(
                ((2 * l1 + 1) * (2 * l2 + 1))
            )

            legendre_int_l1_2p = sy.integrate(
                mu ** (2 * p) * legendre_poly(l1, x=mu), mu
            )
            legendre_int_l2_2q = sy.integrate(
                mu ** (2 * q) * legendre_poly(l2, x=mu), mu
            )

            a_l1_2p = sy.Rational(((2 * l1 + 1) / 2)) * (
                legendre_int_l1_2p.subs(mu, 1) - legendre_int_l1_2p.subs(mu, -1)
            )
            a_l2_2q = sy.Rational(((2 * l2 + 1) / 2)) * (
                legendre_int_l2_2q.subs(mu, 1) - legendre_int_l2_2q.subs(mu, -1)
            )

            h_sum = 0

            for m in range(-ell, ell + 1):
                for m1 in range(-l1, l1 + 1):
                    for m2 in range(-l2, l2 + 1):
                        sum_term = wigner.gaunt(ell, l1, l2, m, m1, m2)
                        sum_term *= (
                            sy.Ynm(ell, m, sy.pi - phi, 0)
                            * sy.Ynm(l1, m1, theta / 2, 0)
                            * sy.Ynm(l2, m2, theta / 2, sy.pi)
                        )
                        sum_term *= a_l1_2p * a_l2_2q
                        h_sum = h_sum + sum_term

            H = H + h_term * h_sum
    H = simplify_h(H.expand(func=True))
    return H


def write_h_terms(pmax, qmax, filename="./h_terms.py", number_worker=1):
    """Generate and write H-term Python functions to a file.

    Produces three sets of functions: gg with even ℓ up to ``2(p+q+1)``, gv with
    odd ℓ up to ``2(p+1)``, and vv with even ℓ up to 2. The functions are
    emitted as NumPy-evaluable code via ``pycode``.

    Args:
        pmax: Maximum p index.
        qmax: Maximum q index.
        filename: Output Python file path for the generated H-term functions.
        number_worker: Number of workers for parallel generation.

    Returns:
        None. Writes functions to ``filename``.
    """
    params_pool = []
    for p in range(pmax + 1):
        for q in range(qmax + 1):
            lmax = 2 * (p + q + 1)
            for ell in range(0, lmax + 1, 2):
                params_pool.append([ell, p, q])

    if number_worker == 1:
        output_H_gg = [generate_h_term(*param) for param in params_pool]
    else:
        with mp.Pool(number_worker) as pool:
            output_H_gg = pool.starmap(generate_h_term, params_pool)

    params_pool = []
    for p in range(pmax + 1):
        lmax = 2 * (p + 1)
        for ell in range(1, lmax + 1, 2):
            params_pool.append([ell, p, 1 / 2])

    if number_worker == 1:
        output_H_gv = [generate_h_term(*param) for param in params_pool]
    else:
        with mp.Pool(number_worker) as pool:
            output_H_gv = pool.starmap(generate_h_term, params_pool)

    params_pool = []
    lmax = 2
    for ell in range(0, lmax + 1, 2):
        params_pool.append([ell, 1 / 2, 1 / 2])

    if number_worker == 1:
        output_H_vv = [generate_h_term(*param) for param in params_pool]
    else:
        with mp.Pool(number_worker) as pool:
            output_H_vv = pool.starmap(generate_h_term, params_pool)

    f = open(filename, "w")
    f.write("import numpy as np\n")
    f.write("\n")
    f.write("\n")
    f.write("# Density-Density \n")
    f.write("\n")
    i = 0
    for p in range(pmax + 1):
        for q in range(qmax + 1):
            lmax = 2 * (p + q + 1)
            for ell in range(0, lmax + 1, 2):
                H_txt = pycode(output_H_gg[i]).replace("math.", "np.")
                i = i + 1
                f.write(f"def H_gg_l{ell}_p{p}_q{q}(theta,phi):\n")
                f.write(f"    return({H_txt})\n")
                f.write("\n")

    f.write("\n")
    f.write("# Density-Velocity \n")
    i = 0
    for p in range(pmax + 1):
        lmax = 2 * (p + 1)
        for ell in range(1, lmax + 1, 2):
            H_txt = pycode(output_H_gv[i]).replace("math.", "np.")
            i = i + 1
            f.write(f"def H_gv_l{ell}_p{p}(theta,phi):\n")
            f.write(f"    return({H_txt})\n")
            f.write("\n")

    f.write("\n")
    f.write("# Velocity-Velocity \n")
    lmax = 2
    i = 0
    for ell in range(0, lmax + 1, 2):
        H_txt = pycode(output_H_vv[i]).replace("math.", "np.")
        i = i + 1
        f.write(f"def H_vv_l{ell}(theta,phi):\n")
        f.write(f"    return({H_txt})\n")
        f.write("\n")

    f.write("\n")
    f.close()
