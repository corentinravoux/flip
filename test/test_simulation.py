"""Tests for flip.simulation — forward modeling via JaxPM.

Covers:
  - Field shapes and value ranges from ForwardModel.get_fields
  - CIC interpolation correctness (_cic_read)
  - LOS velocity projection
  - Sky ↔ Cartesian coordinate conversion
  - Gaussian log-likelihood sign and shape
  - Gradient flow: jax.grad through the full pipeline (sigma8 → fields)
  - VelocityFieldLikelihood evaluation
  - ForwardModel.from_survey_geometry constructor

All tests use a tiny mesh (16³) to keep wall-clock time short.
"""

import pytest

pytest.importorskip("jaxpm", reason="jaxpm not installed")
pytest.importorskip("jax_cosmo", reason="jax_cosmo not installed")

import jax
import jax.numpy as jnp
import numpy as np

from flip.simulation.generator import ForwardModel, _run_lpt, get_cosmology
from flip.simulation.likelihood import VelocityFieldLikelihood, log_likelihood_gaussian
from flip.simulation.painter import (
    _cic_read,
    cartesian_to_box_frame,
    compute_los_velocity,
    interpolate_velocity_to_positions,
    sky_to_cartesian,
)

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

MESH = (16, 16, 16)
BOX = (256.0, 256.0, 256.0)
SEED = jax.random.PRNGKey(42)


@pytest.fixture(scope="module")
def cosmo():
    return get_cosmology(omega_m=0.3, sigma8=0.8)


@pytest.fixture(scope="module")
def model():
    return ForwardModel(mesh_shape=MESH, box_size=BOX, a_final=1.0, lpt_only=True)


@pytest.fixture(scope="module")
def fields(model, cosmo):
    return model.get_fields(cosmo, seed=SEED)


# ---------------------------------------------------------------------------
# Generator tests
# ---------------------------------------------------------------------------


def test_get_cosmology_attributes():
    cosmo = get_cosmology(omega_m=0.3, sigma8=0.8)
    assert hasattr(cosmo, "sigma8")
    assert hasattr(cosmo, "Omega_c")
    # Omega_c should be Omega_m - Omega_b
    np.testing.assert_allclose(float(cosmo.Omega_c) + float(cosmo.Omega_b), 0.3, atol=1e-5)


def test_density_field_shape(fields):
    density, _ = fields
    assert density.shape == MESH


def test_velocity_field_shape(fields):
    _, velocity = fields
    assert velocity.shape == (*MESH, 3)


def test_density_field_mean_near_zero(fields):
    """Mean of density contrast should be close to zero (by definition of δ)."""
    density, _ = fields
    assert abs(float(jnp.mean(density))) < 0.5


def test_velocity_field_finite(fields):
    _, velocity = fields
    assert jnp.all(jnp.isfinite(velocity))


def test_density_field_finite(fields):
    density, _ = fields
    assert jnp.all(jnp.isfinite(density))


def test_velocity_magnitude_reasonable(fields):
    """Typical peculiar velocities at z=0 should be O(100–1000) km/s."""
    _, velocity = fields
    rms = float(jnp.sqrt(jnp.mean(velocity**2)))
    assert 10.0 < rms < 5000.0, f"Unexpected RMS velocity: {rms:.1f} km/s"


def test_reproducibility(model, cosmo):
    """Same seed must produce identical fields."""
    d1, v1 = model.get_fields(cosmo, seed=SEED)
    d2, v2 = model.get_fields(cosmo, seed=SEED)
    np.testing.assert_array_equal(np.array(d1), np.array(d2))
    np.testing.assert_array_equal(np.array(v1), np.array(v2))


def test_different_seeds_differ(model, cosmo):
    """Different seeds must produce different fields."""
    d1, _ = model.get_fields(cosmo, seed=jax.random.PRNGKey(0))
    d2, _ = model.get_fields(cosmo, seed=jax.random.PRNGKey(1))
    assert not jnp.allclose(d1, d2), "Fields with different seeds should differ"


def test_sigma8_increases_amplitude(cosmo):
    """Higher σ₈ should produce a larger density contrast variance."""
    model = ForwardModel(mesh_shape=MESH, box_size=BOX, lpt_only=True)
    cosmo_lo = get_cosmology(omega_m=0.3, sigma8=0.6)
    cosmo_hi = get_cosmology(omega_m=0.3, sigma8=1.2)
    d_lo, _ = model.get_fields(cosmo_lo, seed=SEED)
    d_hi, _ = model.get_fields(cosmo_hi, seed=SEED)
    assert float(jnp.std(d_hi)) > float(jnp.std(d_lo))


def test_from_survey_geometry():
    """ForwardModel.from_survey_geometry returns a valid instance."""
    model = ForwardModel.from_survey_geometry(
        rcom_max=200.0, cell_size_mpc=10.0, z_survey=0.05
    )
    assert len(model.mesh_shape) == 3
    assert all(n > 0 for n in model.mesh_shape)
    assert model.a_final == pytest.approx(1.0 / 1.05, rel=1e-4)


def test_integer_seed(model, cosmo):
    """ForwardModel should accept plain integer seeds."""
    d, v = model.get_fields(cosmo, seed=0)
    assert d.shape == MESH


# ---------------------------------------------------------------------------
# Gradient tests
# ---------------------------------------------------------------------------


def test_grad_sigma8(model):
    """jax.grad w.r.t. sigma8 should be finite and non-zero."""
    def loss(sigma8):
        c = get_cosmology(omega_m=0.3, sigma8=sigma8)
        d, _ = model.get_fields(c, seed=SEED)
        return jnp.mean(d**2)

    g = jax.grad(loss)(0.8)
    assert jnp.isfinite(g), f"Gradient w.r.t. sigma8 is not finite: {g}"
    assert abs(float(g)) > 0.0, "Gradient w.r.t. sigma8 should be non-zero"


def test_grad_omega_m(model):
    """jax.grad w.r.t. omega_m should be finite."""
    def loss(omega_m):
        c = get_cosmology(omega_m=omega_m, sigma8=0.8)
        _, v = model.get_fields(c, seed=SEED)
        return jnp.mean(v**2)

    g = jax.grad(loss)(0.3)
    assert jnp.isfinite(g), f"Gradient w.r.t. omega_m is not finite: {g}"


# ---------------------------------------------------------------------------
# Painter tests
# ---------------------------------------------------------------------------


def test_cic_read_grid_points():
    """CIC at exact cell centres should return the cell value."""
    Nx, Ny, Nz = 8, 8, 8
    grid = jnp.arange(Nx * Ny * Nz, dtype=jnp.float32).reshape(Nx, Ny, Nz)
    # Query at integer positions (cell centres)
    pos = jnp.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 2.0, 3.0]])
    vals = _cic_read(grid, pos)
    expected = jnp.array([grid[0, 0, 0], grid[1, 0, 0], grid[0, 2, 3]])
    np.testing.assert_allclose(np.array(vals), np.array(expected), atol=1e-5)


def test_cic_read_periodic():
    """CIC should wrap periodically at box boundaries."""
    grid = jnp.ones((8, 8, 8))
    pos = jnp.array([[7.9, 0.0, 0.0]])  # near boundary
    val = _cic_read(grid, pos)
    np.testing.assert_allclose(float(val[0]), 1.0, atol=1e-5)


def test_compute_los_velocity_radial():
    """For a purely radial velocity, v_los = |v|."""
    positions = jnp.array([[100.0, 0.0, 0.0], [0.0, 50.0, 0.0]])
    velocities = jnp.array([[300.0, 0.0, 0.0], [0.0, -200.0, 0.0]])
    v_los = compute_los_velocity(velocities, positions)
    np.testing.assert_allclose(np.array(v_los), np.array([300.0, -200.0]), atol=1e-3)


def test_compute_los_velocity_transverse_zero():
    """Velocity perpendicular to LOS should give v_los ≈ 0."""
    positions = jnp.array([[100.0, 0.0, 0.0]])
    velocities = jnp.array([[0.0, 500.0, 0.0]])  # transverse to x-LOS
    v_los = compute_los_velocity(velocities, positions)
    np.testing.assert_allclose(float(v_los[0]), 0.0, atol=1e-3)


def test_sky_to_cartesian_on_axis():
    """RA=0, Dec=0 should give purely x-direction."""
    ra = jnp.array([0.0])
    dec = jnp.array([0.0])
    r = jnp.array([100.0])
    xyz = sky_to_cartesian(ra, dec, r)
    np.testing.assert_allclose(float(xyz[0, 0]), 100.0, atol=1e-4)
    np.testing.assert_allclose(float(xyz[0, 1]), 0.0, atol=1e-4)
    np.testing.assert_allclose(float(xyz[0, 2]), 0.0, atol=1e-4)


def test_cartesian_to_box_frame_default_center():
    """Centering at box centre should shift by box/2."""
    positions = jnp.array([[0.0, 0.0, 0.0]])
    box = [256.0, 256.0, 256.0]
    shifted = cartesian_to_box_frame(positions, box)
    np.testing.assert_allclose(
        np.array(shifted[0]), np.array([128.0, 128.0, 128.0]), atol=1e-4
    )


def test_interpolate_velocity_shape(fields):
    _, velocity = fields
    N_gal = 20
    pos = jnp.ones((N_gal, 3)) * 100.0  # all inside box
    v_gal = interpolate_velocity_to_positions(velocity, pos, BOX, MESH)
    assert v_gal.shape == (N_gal, 3)


# ---------------------------------------------------------------------------
# Likelihood tests
# ---------------------------------------------------------------------------


def test_log_likelihood_gaussian_diagonal():
    """Diagonal log-likelihood should be finite for matching arrays."""
    v_sim = jnp.zeros(10)
    v_obs = jnp.zeros(10)
    var = jnp.ones(10)
    ll = log_likelihood_gaussian(v_sim, v_obs, var)
    assert jnp.isfinite(ll)


def test_log_likelihood_gaussian_perfect_fit():
    """Log-likelihood at perfect fit (residual=0) should be maximum."""
    v = jnp.ones(5) * 100.0
    var = jnp.ones(5) * 25.0
    ll_perfect = log_likelihood_gaussian(v, v, var)
    ll_shifted = log_likelihood_gaussian(v + 10.0, v, var)
    assert float(ll_perfect) > float(ll_shifted)


def test_log_likelihood_gaussian_full_covariance():
    """Full (2D) covariance branch should return finite scalar."""
    N = 5
    v_sim = jnp.zeros(N)
    v_obs = jnp.zeros(N)
    C = jnp.eye(N) * 100.0
    ll = log_likelihood_gaussian(v_sim, v_obs, C)
    assert jnp.isfinite(ll)


class _MockDataVector:
    """Minimal stub data vector for likelihood tests."""
    def give_data_and_variance(self, params=None):
        N = 20
        return np.random.default_rng(0).normal(0, 200, N), np.ones(N) * 200.0**2


def test_velocity_field_likelihood_evaluates(model, cosmo):
    """VelocityFieldLikelihood should return a finite scalar."""
    N_gal = 20
    rng = np.random.default_rng(1)
    xyz = rng.uniform(10.0, 240.0, (N_gal, 3))

    lik = VelocityFieldLikelihood(
        forward_model=model,
        data_vector=_MockDataVector(),
        positions_cartesian=xyz,
        seed=SEED,
    )
    neg_ll = lik({"omega_m": 0.3, "sigma8": 0.8})
    assert jnp.isfinite(neg_ll), f"Likelihood returned non-finite value: {neg_ll}"


def test_velocity_field_likelihood_gradient(model):
    """jax.grad through VelocityFieldLikelihood w.r.t. sigma8 should be finite."""
    N_gal = 20
    rng = np.random.default_rng(2)
    xyz = jnp.array(rng.uniform(10.0, 240.0, (N_gal, 3)))

    lik = VelocityFieldLikelihood(
        forward_model=model,
        data_vector=_MockDataVector(),
        positions_cartesian=xyz,
        seed=SEED,
    )

    def loss(sigma8):
        return lik({"omega_m": 0.3, "sigma8": sigma8})

    g = jax.grad(loss)(0.8)
    assert jnp.isfinite(g), f"Likelihood gradient w.r.t. sigma8 not finite: {g}"
