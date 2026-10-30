import numpy as np
import pytest

pytest.importorskip("firecrown")

from firecrown.likelihood import NamedParameters
from firecrown.updatable import ParamsMap, get_default_params
from flip.covariance import covariance, firecrown_interface
from flip.covariance.likelihood import MultivariateGaussianLikelihood
from flip.data import load_data_test

from flip import data_vector

COSMOLOGY = {
    "Omega_c": 0.25,
    "Omega_b": 0.05,
    "h": 0.70,
    "n_s": 0.96,
    "sigma8": 0.819,
    "m_nu": 0.06,
}
POINT = {"fs8": 0.45, "sigv": 250.0}
TOOLS = {"ccl_creation_mode": "pure_ccl_mode", "mass_split": "single"}


@pytest.fixture(scope="module")
def flip_likelihood():
    coordinates_velocity, velocity_data = load_data_test.load_velocity_data(
        subsample=60
    )
    cov = covariance.CovMatrix.init_from_flip(
        "carreres23",
        "velocity",
        load_data_test.load_power_spectrum_dict(),
        coordinates_velocity=coordinates_velocity,
        size_batch=10_000,
        number_worker=1,
    )
    return MultivariateGaussianLikelihood(
        covariance=cov,
        data=data_vector.DirectVel(velocity_data),
        parameter_names=list(POINT),
        likelihood_properties={
            "inversion_method": "cholesky",
            "negative_log_likelihood": False,
        },
    )


def loglike(likelihood, tools, cosmology=COSMOLOGY):
    # One sampler step, as in the firecrown connectors.
    values = {**cosmology, **{f"flip_{name}": v for name, v in POINT.items()}}
    params = ParamsMap({**get_default_params(likelihood, tools), **values})
    likelihood.update(params)
    tools.update(params)
    tools.prepare()
    try:
        return likelihood.compute_loglike_for_sampling(tools)
    finally:
        likelihood.reset()
        tools.reset()


def test_registered_likelihood(flip_likelihood):
    reference = flip_likelihood(np.array(list(POINT.values())))
    firecrown_interface.set_flip_likelihood(flip_likelihood)
    likelihood, tools = firecrown_interface.build_likelihood(NamedParameters(TOOLS))
    assert loglike(likelihood, tools) == reference


def test_unregistered_likelihood():
    with pytest.raises(KeyError):
        firecrown_interface.build_likelihood(
            NamedParameters({"flip_likelihood": "missing", **TOOLS})
        )


def test_rebuild_on_cosmology_change(flip_likelihood):
    cosmologies = []

    def build(ccl_cosmology):
        cosmologies.append(ccl_cosmology)
        return flip_likelihood

    likelihood = firecrown_interface.FlipLikelihood(
        build, parameter_names=list(POINT), parameter_prefix="flip"
    )
    firecrown_interface.set_flip_likelihood(flip_likelihood)
    _, tools = firecrown_interface.build_likelihood(NamedParameters(TOOLS))
    loglike(likelihood, tools)
    loglike(likelihood, tools)
    assert len(cosmologies) == 1
    loglike(likelihood, tools, {**COSMOLOGY, "Omega_c": 0.27})
    assert len(cosmologies) == 2
