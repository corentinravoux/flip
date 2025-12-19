import numpy as np
from flip.covariance.likelihood import (
    log_likelihood_gaussian_cholesky,
    log_likelihood_gaussian_cholesky_inverse,
    log_likelihood_gaussian_inverse,
    log_likelihood_gaussian_solve,
)


def rand_spd(n, seed=0):
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((n, n))
    C = A @ A.T + np.eye(n) * 1e-6
    return C


def test_inversion_methods_agree():
    n = 12
    rng = np.random.default_rng(1)
    C = rand_spd(n)
    x = rng.standard_normal(n)

    vals = [
        log_likelihood_gaussian_inverse(x, C),
        log_likelihood_gaussian_solve(x, C),
        log_likelihood_gaussian_cholesky(x, C),
        log_likelihood_gaussian_cholesky_inverse(x, C),
    ]

    for i in range(1, len(vals)):
        np.testing.assert_allclose(vals[i], vals[0], rtol=1e-10, atol=1e-10)
