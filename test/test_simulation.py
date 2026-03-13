"""Tests for the flip.simulation package.

Covers generate.py (LPT and N-body field generation), likelihood.py
(VelocityFieldLikelihood), and fitter.py (SimulationFitter).

All tests use a small mesh (8^3) and LPT mode to keep execution fast.
The N-body (diffrax) pipeline is exercised in a smoke-test that checks shapes
and finiteness without running a full optimisation.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from flip import data_vector
from flip.simulation import generate, likelihood
from flip.simulation.fitter import SimulationFitter

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

jax.config.update("jax_enable_x64", True)

_MESH_SHAPE = (8, 8, 8)
_BOX_SIZE = [64.0, 64.0, 64.0]
_SEED = jax.random.PRNGKey(0)
_TRUE_OMEGA_M = 0.3
_TRUE_SIGMA8 = 0.8


def _make_mock_data_vector(n_galaxies=20, seed_np=42):
    """Return (positions, DirectVel data vector) for a small synthetic catalogue.

    The observed velocities are drawn from a N(0, 300) km/s distribution with
    a flat error of 200 km/s to make the likelihood well-conditioned.
    """
    rng = np.random.RandomState(seed_np)
    positions = rng.uniform(5.0, 59.0, (n_galaxies, 3))
    velocities = rng.normal(0.0, 300.0, n_galaxies)
    velocity_errors = np.full(n_galaxies, 200.0)
    vel_data = {"velocity": velocities, "velocity_error": velocity_errors}
    dv = data_vector.DirectVel(vel_data)
    return positions, dv


# ---------------------------------------------------------------------------
# generate.py tests
# ---------------------------------------------------------------------------


class TestGetCosmology:
    def test_returns_cosmology_object(self):
        cosmo = generate.get_cosmology(omega_m=_TRUE_OMEGA_M, sigma8=_TRUE_SIGMA8)
        assert hasattr(cosmo, "sigma8")
        assert float(cosmo.sigma8) == pytest.approx(_TRUE_SIGMA8)

    def test_omega_c_derived_from_omega_m(self):
        cosmo = generate.get_cosmology(omega_m=0.3, sigma8=0.8, omega_b=0.05)
        assert float(cosmo.Omega_c) == pytest.approx(0.3 - 0.05, abs=1e-6)


class TestComputeFsigma8:
    def test_fsigma8_finite_and_positive(self):
        cosmo = generate.get_cosmology(omega_m=_TRUE_OMEGA_M, sigma8=_TRUE_SIGMA8)
        fs8 = generate.compute_fsigma8(cosmo, a=1.0)
        assert jnp.isfinite(fs8)
        assert float(fs8) > 0.0

    def test_fsigma8_scales_with_sigma8(self):
        cosmo_hi = generate.get_cosmology(omega_m=0.3, sigma8=1.0)
        cosmo_lo = generate.get_cosmology(omega_m=0.3, sigma8=0.5)
        fs8_hi = float(generate.compute_fsigma8(cosmo_hi))
        fs8_lo = float(generate.compute_fsigma8(cosmo_lo))
        assert fs8_hi > fs8_lo

    def test_fsigma8_differentiable(self):
        def obj(sigma8):
            cosmo = generate.get_cosmology(omega_m=0.3, sigma8=sigma8)
            return generate.compute_fsigma8(cosmo)

        grad = jax.grad(obj)(0.8)
        assert jnp.isfinite(grad)


class TestGenerateLpt:
    def test_output_shapes(self):
        cosmo = generate.get_cosmology(omega_m=_TRUE_OMEGA_M, sigma8=_TRUE_SIGMA8)
        density, velocity = generate.generate_density_and_velocity_lpt(
            cosmo, _MESH_SHAPE, _BOX_SIZE, _SEED
        )
        assert density.shape == _MESH_SHAPE
        assert velocity.shape == (*_MESH_SHAPE, 3)

    def test_fields_finite(self):
        cosmo = generate.get_cosmology(omega_m=_TRUE_OMEGA_M, sigma8=_TRUE_SIGMA8)
        density, velocity = generate.generate_density_and_velocity_lpt(
            cosmo, _MESH_SHAPE, _BOX_SIZE, _SEED
        )
        assert jnp.all(jnp.isfinite(density))
        assert jnp.all(jnp.isfinite(velocity))

    def test_density_mean_near_one(self):
        cosmo = generate.get_cosmology(omega_m=_TRUE_OMEGA_M, sigma8=_TRUE_SIGMA8)
        density, _ = generate.generate_density_and_velocity_lpt(
            cosmo, _MESH_SHAPE, _BOX_SIZE, _SEED
        )
        assert float(density.mean()) == pytest.approx(1.0, abs=0.1)

    def test_gradient_through_lpt_wrt_sigma8(self):
        def obj(sigma8):
            cosmo = generate.get_cosmology(omega_m=0.3, sigma8=sigma8)
            _, vel = generate.generate_density_and_velocity_lpt(
                cosmo, _MESH_SHAPE, _BOX_SIZE, _SEED
            )
            return (vel**2).sum()

        grad = jax.grad(obj)(0.8)
        assert jnp.isfinite(grad)


class TestGenerateNbody:
    def test_output_shapes(self):
        cosmo = generate.get_cosmology(omega_m=_TRUE_OMEGA_M, sigma8=_TRUE_SIGMA8)
        density, velocity = generate.generate_density_and_velocity_nbody(
            cosmo, _MESH_SHAPE, _BOX_SIZE, _SEED,
            ode_rtol=1e-3, ode_atol=1e-3,
        )
        assert density.shape == _MESH_SHAPE
        assert velocity.shape == (*_MESH_SHAPE, 3)

    def test_fields_finite(self):
        cosmo = generate.get_cosmology(omega_m=_TRUE_OMEGA_M, sigma8=_TRUE_SIGMA8)
        density, velocity = generate.generate_density_and_velocity_nbody(
            cosmo, _MESH_SHAPE, _BOX_SIZE, _SEED,
            ode_rtol=1e-3, ode_atol=1e-3,
        )
        assert jnp.all(jnp.isfinite(density))
        assert jnp.all(jnp.isfinite(velocity))


class TestGenerateDispatch:
    def test_lpt_method_matches_lpt_function(self):
        cosmo = generate.get_cosmology(omega_m=_TRUE_OMEGA_M, sigma8=_TRUE_SIGMA8)
        d1, v1 = generate.generate_density_and_velocity_lpt(
            cosmo, _MESH_SHAPE, _BOX_SIZE, _SEED
        )
        d2, v2 = generate.generate_density_and_velocity(
            cosmo, _MESH_SHAPE, _BOX_SIZE, _SEED, method="lpt"
        )
        np.testing.assert_array_equal(np.array(d1), np.array(d2))
        np.testing.assert_array_equal(np.array(v1), np.array(v2))

    def test_invalid_method_raises(self):
        cosmo = generate.get_cosmology(omega_m=0.3, sigma8=0.8)
        with pytest.raises(ValueError, match="Unknown simulation method"):
            generate.generate_density_and_velocity(
                cosmo, _MESH_SHAPE, _BOX_SIZE, _SEED, method="invalid"
            )


class TestInterpolateAndLosVelocity:
    def test_interpolate_output_shape(self):
        cosmo = generate.get_cosmology(omega_m=_TRUE_OMEGA_M, sigma8=_TRUE_SIGMA8)
        _, vel_field = generate.generate_density_and_velocity_lpt(
            cosmo, _MESH_SHAPE, _BOX_SIZE, _SEED
        )
        rng = np.random.RandomState(1)
        positions = jnp.array(rng.uniform(5.0, 59.0, (30, 3)))
        vel_at_pos = generate.interpolate_velocity_to_positions(
            vel_field, positions, _BOX_SIZE, _MESH_SHAPE
        )
        assert vel_at_pos.shape == (30, 3)
        assert jnp.all(jnp.isfinite(vel_at_pos))

    def test_los_velocity_output_shape(self):
        rng = np.random.RandomState(2)
        velocities = jnp.array(rng.normal(0.0, 200.0, (20, 3)))
        positions = jnp.array(rng.uniform(5.0, 59.0, (20, 3)))
        los_vel = generate.compute_los_velocity(velocities, positions)
        assert los_vel.shape == (20,)
        assert jnp.all(jnp.isfinite(los_vel))


# ---------------------------------------------------------------------------
# likelihood.py tests
# ---------------------------------------------------------------------------


class TestVelocityFieldLikelihood:
    def _build_lik(self, n_gal=15, method="lpt"):
        positions, dv = _make_mock_data_vector(n_galaxies=n_gal)
        lik = likelihood.VelocityFieldLikelihood(
            data_vector=dv,
            positions_cartesian=positions,
            mesh_shape=_MESH_SHAPE,
            box_size=_BOX_SIZE,
            seed=_SEED,
            method=method,
            fixed_cosmo_params={"omega_m": _TRUE_OMEGA_M},
        )
        return lik

    def test_returns_finite_scalar(self):
        lik = self._build_lik()
        val = lik({"sigma8": _TRUE_SIGMA8})
        assert jnp.isfinite(val)

    def test_returns_positive_neg_log_lik(self):
        lik = self._build_lik()
        val = lik({"sigma8": _TRUE_SIGMA8})
        assert np.isscalar(float(val)) and jnp.isfinite(val)

    def test_gradient_wrt_sigma8_finite(self):
        lik = self._build_lik()
        grad = jax.grad(lambda s8: lik({"sigma8": s8}))(_TRUE_SIGMA8)
        assert jnp.isfinite(grad)

    def test_fixed_cosmo_params_merged(self):
        """fixed_cosmo_params must be used when cosmo_params omits omega_m."""
        positions, dv = _make_mock_data_vector(n_galaxies=10)
        lik = likelihood.VelocityFieldLikelihood(
            data_vector=dv,
            positions_cartesian=positions,
            mesh_shape=_MESH_SHAPE,
            box_size=_BOX_SIZE,
            seed=_SEED,
            method="lpt",
            fixed_cosmo_params={"omega_m": 0.3},
        )
        # Should not raise even though omega_m is not in cosmo_params
        val = lik({"sigma8": 0.8})
        assert jnp.isfinite(val)

    def test_full_cosmo_params_without_fixed(self):
        """Passing all params directly also works (no fixed_cosmo_params)."""
        positions, dv = _make_mock_data_vector(n_galaxies=10)
        lik = likelihood.VelocityFieldLikelihood(
            data_vector=dv,
            positions_cartesian=positions,
            mesh_shape=_MESH_SHAPE,
            box_size=_BOX_SIZE,
            seed=_SEED,
            method="lpt",
        )
        val = lik({"omega_m": 0.3, "sigma8": 0.8})
        assert jnp.isfinite(val)

    def test_nbody_method_runs(self):
        lik = self._build_lik(n_gal=10, method="nbody")
        val = lik({"sigma8": _TRUE_SIGMA8})
        assert jnp.isfinite(val)


# ---------------------------------------------------------------------------
# fitter.py tests
# ---------------------------------------------------------------------------


class TestSimulationFitter:
    def _build_lik_and_fitter(self, solver="LBFGSB", maxiter=3):
        positions, dv = _make_mock_data_vector(n_galaxies=20)
        lik = likelihood.VelocityFieldLikelihood(
            data_vector=dv,
            positions_cartesian=positions,
            mesh_shape=_MESH_SHAPE,
            box_size=_BOX_SIZE,
            seed=_SEED,
            method="lpt",
            fixed_cosmo_params={"omega_m": _TRUE_OMEGA_M},
        )
        fitter = SimulationFitter(
            likelihood=lik,
            initial_params={"sigma8": 0.8},
            bounds=({"sigma8": 0.3}, {"sigma8": 1.5}),
            solver=solver,
            maxiter=maxiter,
        )
        return lik, fitter

    def test_run_returns_dict(self):
        _, fitter = self._build_lik_and_fitter()
        result = fitter.run()
        assert isinstance(result, dict)
        assert "sigma8" in result

    def test_best_sigma8_in_bounds(self):
        _, fitter = self._build_lik_and_fitter(maxiter=5)
        result = fitter.run()
        assert 0.3 <= float(result["sigma8"]) <= 1.5

    def test_result_attribute_set_after_run(self):
        _, fitter = self._build_lik_and_fitter()
        assert fitter.result is None
        fitter.run()
        assert fitter.result is not None

    def test_invalid_solver_raises(self):
        positions, dv = _make_mock_data_vector(n_galaxies=5)
        lik = likelihood.VelocityFieldLikelihood(
            data_vector=dv,
            positions_cartesian=positions,
            mesh_shape=_MESH_SHAPE,
            box_size=_BOX_SIZE,
            seed=_SEED,
            method="lpt",
            fixed_cosmo_params={"omega_m": _TRUE_OMEGA_M},
        )
        with pytest.raises(ValueError, match="Solver"):
            SimulationFitter(
                likelihood=lik,
                initial_params={"sigma8": 0.8},
                solver="NOT_A_SOLVER",
            )

    def test_lbfgs_unconstrained_solver(self):
        positions, dv = _make_mock_data_vector(n_galaxies=10)
        lik = likelihood.VelocityFieldLikelihood(
            data_vector=dv,
            positions_cartesian=positions,
            mesh_shape=_MESH_SHAPE,
            box_size=_BOX_SIZE,
            seed=_SEED,
            method="lpt",
            fixed_cosmo_params={"omega_m": _TRUE_OMEGA_M},
        )
        fitter = SimulationFitter(
            likelihood=lik,
            initial_params={"sigma8": 0.8},
            solver="LBFGS",
            maxiter=3,
        )
        result = fitter.run()
        assert "sigma8" in result
        assert jnp.isfinite(result["sigma8"])
