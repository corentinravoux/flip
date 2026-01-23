import numpy as np
import pytest
from flip.data import load_data_test

from flip import covariance, data_vector

model_to_test = [
    ("carreres23", "velocity"),
    ("adamsblake17plane", "density"),
    ("adamsblake17plane", "velocity"),
    ("adamsblake17plane", "full"),
    ("adamsblake20", "density"),
    ("adamsblake20", "velocity"),
    ("adamsblake20", "full"),
    ("ravouxcarreres", "velocity"),
    ("ravouxcarreres", "density"),
    ("ravouxcarreres", "full"),
    ("lai22", "density"),
    ("lai22", "velocity"),
    ("lai22", "full"),
]


def compute_covariance_metrics(
    model,
    model_type,
):

    if model_type == "density":
        coordinates_density, density_data = load_data_test.load_density_data(
            subsample=50
        )
        data_vector_obj = data_vector.Dens(density_data)
        coordinates_velocity = None

    elif model_type == "velocity":
        coordinates_velocity, velocity_data = load_data_test.load_velocity_data(
            subsample=50
        )
        data_vector_obj = data_vector.DirectVel(velocity_data)
        coordinates_density = None
    else:
        coordinates_velocity, velocity_data = load_data_test.load_velocity_data(
            subsample=50
        )
        coordinates_density, density_data = load_data_test.load_density_data(
            subsample=50
        )
        data_vector_obj = data_vector.DensVel(
            data_vector.Dens(density_data),
            data_vector.DirectVel(velocity_data),
        )

    power_spectrum_dict = load_data_test.load_power_spectrum_dict()

    additional_parameters = ()
    variant = None
    parameters = {"bs8": 1.0, "fs8": 0.5, "sigv": 0.0}
    if model in {"adamsblake20", "ravouxcarreres"}:
        parameters["beta_f"] = parameters["fs8"] / max(parameters["bs8"], 1e-6)
        additional_parameters = (15.0,)
        variant = "nobeta"
    if model == "lai22":
        parameters["sigg"] = 1.0
        parameters["beta_f"] = parameters["fs8"] / max(parameters["bs8"], 1e-6)
        variant = "nobeta"

    covariance_object = covariance.CovMatrix.init_from_flip(
        model,
        model_type,
        power_spectrum_dict,
        coordinates_density=coordinates_density,
        coordinates_velocity=coordinates_velocity,
        size_batch=50_000,
        number_worker=1,
        variant=variant,
        additional_parameters_values=additional_parameters,
    )

    _, data_variance = data_vector_obj.give_data_and_variance(
        parameters if "M_0" in data_vector_obj.free_par else {}
    )
    covariance_matrix = covariance_object.compute_covariance_sum(
        parameters, data_variance
    )
    return {
        "shape": [int(covariance_matrix.shape[0]), int(covariance_matrix.shape[1])],
        "trace": float(np.trace(covariance_matrix)),
        "diag_mean": float(np.mean(np.diag(covariance_matrix))),
        "entry_0_0": float(covariance_matrix[0, 0]),
        "entry_0_-1": float(covariance_matrix[0, -1]),
        "entry_mid_mid": float(
            covariance_matrix[
                covariance_matrix.shape[0] // 2, covariance_matrix.shape[1] // 2
            ]
        ),
    }


@pytest.mark.parametrize(
    "model,kind",
    model_to_test,
)
def test_covariance_reference_metrics(model, kind, debug_return=False):

    got = compute_covariance_metrics(model, kind)

    if debug_return:
        return got

    reference_values = load_data_test.load_covariance_test_reference_values()

    key = f"{model}:{kind}"
    assert key in reference_values, f"Missing reference metrics for {key}"
    ref = reference_values[key]
    assert got["shape"] == ref["shape"]
    np.testing.assert_allclose(got["trace"], ref["trace"], rtol=1e-6, atol=1e-8)
    np.testing.assert_allclose(got["diag_mean"], ref["diag_mean"], rtol=1e-6, atol=1e-8)
    np.testing.assert_allclose(got["entry_0_0"], ref["entry_0_0"], rtol=1e-6, atol=1e-8)
    np.testing.assert_allclose(
        got["entry_0_-1"], ref["entry_0_-1"], rtol=1e-6, atol=1e-8
    )
    np.testing.assert_allclose(
        got["entry_mid_mid"], ref["entry_mid_mid"], rtol=1e-6, atol=1e-8
    )
