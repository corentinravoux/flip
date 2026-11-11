import logging

import numpy as np
import pytest

from flip import data_vector, utils
from flip.covariance import cov_utils
from flip.data import load_data_test


def _flat_lcdm_comoving_distance_h(z, omega_m=0.3, n=2000):
    """Comoving distance in Mpc/h for flat LCDM."""
    out = []
    for zi in np.atleast_1d(z):
        zz = np.linspace(0, zi, n)
        ez = np.sqrt(omega_m * (1 + zz) ** 3 + 1 - omega_m)
        out.append(utils._C_LIGHT_KMS_ / 100 * np.trapezoid(1 / ez, zz))
    return np.array(out)


def test_angles_in_radians_do_not_warn(caplog):
    rng = np.random.default_rng(0)
    ra = rng.uniform(0, 2 * np.pi, 100)
    dec = rng.uniform(-np.pi / 2, np.pi / 2, 100)
    with caplog.at_level(logging.WARNING):
        assert not utils.check_angle_units(ra, dec)
        assert not utils.check_angle_units(ra - np.pi, dec)
    assert caplog.records == []


def test_angles_in_degrees_warn(caplog):
    rng = np.random.default_rng(0)
    ra = rng.uniform(0, 360, 100)
    dec = rng.uniform(-90, 90, 100)
    with caplog.at_level(logging.WARNING):
        assert utils.check_angle_units(ra, dec)
    assert "degrees" in caplog.text


@pytest.mark.parametrize("zmax", [0.05, 0.3, 0.8])
@pytest.mark.parametrize("h", [0.6, 0.7, 0.75])
def test_distance_units(zmax, h, caplog):
    z = np.linspace(0.01, zmax, 50)
    rcom_h = _flat_lcdm_comoving_distance_h(z)
    with caplog.at_level(logging.WARNING):
        assert not utils.check_distance_units(rcom_h, z)
        assert caplog.records == []
        assert utils.check_distance_units(rcom_h / h, z)
    assert "Mpc/h" in caplog.text


def test_distance_check_needs_redshift():
    assert utils.check_coordinates_units(rcom=np.array([100.0, 200.0])) == (
        False,
        False,
    )


def test_bundled_data_do_not_warn(caplog):
    with caplog.at_level(logging.WARNING):
        load_data_test.load_density_data()
        load_data_test.load_velocity_data()
    assert "degrees" not in caplog.text
    assert "Mpc/h" not in caplog.text


def test_data_vector_warns(caplog):
    z = np.linspace(0.01, 0.1, 20)
    data = {
        "velocity": np.zeros(20),
        "velocity_error": np.ones(20),
        "ra": np.linspace(0, 350, 20),
        "dec": np.linspace(-80, 80, 20),
        "zobs": z,
        "rcom_zobs": _flat_lcdm_comoving_distance_h(z) / 0.7,
    }
    with caplog.at_level(logging.WARNING):
        data_vector.DirectVel(data)
    assert "degrees" in caplog.text
    assert "Mpc/h" in caplog.text


def test_generator_coordinates_warn(caplog):
    coordinates = np.array(
        [np.linspace(0, 350, 20), np.linspace(-80, 80, 20), np.full(20, 100.0)]
    )
    with caplog.at_level(logging.WARNING):
        cov_utils.check_generator_need("velocity", None, coordinates)
    assert "degrees" in caplog.text
