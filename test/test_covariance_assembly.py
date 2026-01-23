import numpy as np
from flip.data import load_data_test

from flip import covariance, data_vector


def test_covariance_assembly_density_velocity():

    coordinates_density, density_data = load_data_test.load_density_data(subsample=50)
    density_data_vector = data_vector.Dens(density_data)

    coordinates_velocity, velocity_data = load_data_test.load_velocity_data(
        subsample=50
    )
    velocity_data_vector = data_vector.DirectVel(velocity_data)

    density_velocity_data_vector = data_vector.DensVel(
        density_data_vector, velocity_data_vector
    )

    power_spectrum_dict = load_data_test.load_power_spectrum_dict()

    model_name = "ravouxcarreres"
    model_type = "density_velocity"
    sigmag_fiducial = 5.0

    covariance_object = covariance.CovMatrix.init_from_flip(
        model_name,
        model_type,
        power_spectrum_dict,
        coordinates_density=coordinates_density,
        coordinates_velocity=coordinates_velocity,
        size_batch=50_000,
        number_worker=1,
        additional_parameters_values=(sigmag_fiducial,),
        variant="nobeta",
    )

    # Assemble total covariance with simple coefficients
    coefficients = {"bs8": 1.0, "fs8": 1.0, "sigv": 0.0}
    _, data_variance = density_velocity_data_vector.give_data_and_variance()
    covariance_matrix = covariance_object.compute_covariance_sum(
        coefficients, data_variance
    )

    # Check shapes and that cross-block is exactly zero (no gv provided)
    n_density = len(density_data_vector.data["density"])
    n_velocity = len(velocity_data_vector.data["velocity"])
    covariance_density_density = covariance_matrix[:n_density, :n_density]
    covariance_velocity_velocity = covariance_matrix[n_density:, n_density:]
    covariance_density_velocity = covariance_matrix[:n_density, n_density:]

    np.testing.assert_allclose(covariance_density_velocity, 0.0, atol=0.0)
    assert covariance_matrix.shape == (n_density + n_velocity, n_density + n_velocity)
    assert np.all(np.diag(covariance_density_density) > 0)
    assert np.all(np.diag(covariance_velocity_velocity) > 0)


def test_covariance_assembly_full():

    coordinates_density, density_data = load_data_test.load_density_data(subsample=50)
    density_data_vector = data_vector.Dens(density_data)

    coordinates_velocity, velocity_data = load_data_test.load_velocity_data(
        subsample=50
    )
    velocity_data_vector = data_vector.DirectVel(velocity_data)

    density_velocity_data_vector = data_vector.DensVel(
        density_data_vector, velocity_data_vector
    )

    power_spectrum_dict = load_data_test.load_power_spectrum_dict()

    model_name = "ravouxcarreres"
    model_type = "full"
    sigmag_fiducial = 5.0

    covariance_object = covariance.CovMatrix.init_from_flip(
        model_name,
        model_type,
        power_spectrum_dict,
        coordinates_density=coordinates_density,
        coordinates_velocity=coordinates_velocity,
        size_batch=50_000,
        number_worker=1,
        additional_parameters_values=(sigmag_fiducial,),
        variant="nobeta",
    )

    coefficients = {"bs8": 1.0, "fs8": 1.0, "sigv": 0.0}
    _, data_variance = density_velocity_data_vector.give_data_and_variance()
    covariance_matrix = covariance_object.compute_covariance_sum(
        coefficients, data_variance
    )

    n_density = len(density_data_vector.data["density"])
    covariance_density_velocity = covariance_matrix[:n_density, n_density:]

    print(np.min(np.abs(covariance_density_velocity)))

    assert np.any(np.abs(covariance_density_velocity) > 0)
