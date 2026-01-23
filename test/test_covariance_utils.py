import numpy as np
from flip.covariance import cov_utils


def test_flat_matrix_roundtrip_small():
    # Covariance flattening stores a single variance value (shared diagonal)
    # followed by upper-triangular off-diagonals. Build a matrix consistent with this contract.
    rng = np.random.default_rng(0)
    n = 5
    off = rng.standard_normal((n, n))
    off = np.triu(off, k=1)
    off = off + off.T
    var = 2.3
    cov = off.copy()
    np.fill_diagonal(cov, var)

    flat = cov_utils.return_flat_covariance(cov)
    cov_back = cov_utils.return_matrix_covariance(flat)

    assert cov_back.shape == cov.shape
    np.testing.assert_allclose(cov_back, cov, rtol=0, atol=1e-12)


def test_flat_cross_roundtrip_shape():
    # Cross-covariance flattening/reshaping
    g, v = 3, 4
    rng = np.random.default_rng(1)
    cross = rng.standard_normal((g, v))

    flat = cov_utils.return_flat_cross_cov(cross)
    back = cov_utils.return_matrix_covariance_cross(flat, g, v)

    assert back.shape == (g, v)
    np.testing.assert_allclose(back, cross)
