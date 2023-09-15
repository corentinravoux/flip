import numpy as np
from flip.utils import create_log
from flip.covariance.lai22 import generator as generator_lai22
from flip.covariance.carreres23 import generator as generator_carreres23
from flip.covariance import generator as generator_flip
from flip.covariance import cov_utils

log = create_log()


def generator_need(
    coordinates_density=None,
    coordinates_velocity=None,
):
    if coordinates_density is not False:
        if coordinates_density is None:
            log.add(
                f"The coordinates_density input is needed to proceed covariance generation, please provide it"
            )
            raise ValueError("Density coordinates not provided")
    if coordinates_velocity is not False:
        if coordinates_velocity is None:
            log.add(
                f"The coordinates_velocity input is needed to proceed covariance generation, please provide it"
            )
            raise ValueError("Velocity coordinates not provided")


def check_generator_need(model_type, coordinates_density, coordinates_velocity):
    if model_type == "density":
        generator_need(
            coordinates_density=coordinates_density,
            coordinates_velocity=False,
        )
    if model_type == "velocity":
        generator_need(
            coordinates_density=False,
            coordinates_velocity=coordinates_velocity,
        )
    if model_type in ["full", "density_velocity"]:
        generator_need(
            coordinates_density=coordinates_density,
            coordinates_velocity=coordinates_velocity,
        )


def generate_carreres23(
    model_type,
    power_spectrum_dict,
    coordinates_density=False,
    coordinates_velocity=None,
    **kwargs,
):
    assert model_type == "velocity"
    check_generator_need(
        model_type,
        coordinates_density,
        coordinates_velocity,
    )
    number_densities = None
    number_velocities = len(coordinates_velocity[0])
    cov_vv = generator_carreres23.covariance_vv(
        coordinates_velocity[0],
        coordinates_velocity[1],
        coordinates_velocity[2],
        power_spectrum_dict["vv"][0][0],
        power_spectrum_dict["vv"][0][1],
        **kwargs,
    )
    return {"vv": [cov_vv]}, number_densities, number_velocities


def generate_lai22(
    model_type,
    power_spectrum_dict,
    coordinates_velocity=None,
    coordinates_density=None,
    pmax=3,
    qmax=3,
    **kwargs,
):
    check_generator_need(
        model_type,
        coordinates_density,
        coordinates_velocity,
    )
    covariance_dict = {}

    if model_type in ["density", "full", "density_velocity"]:
        covariance_dict["gg"] = generator_lai22.compute_cov_gg(
            pmax,
            qmax,
            coordinates_density[0],
            coordinates_density[1],
            coordinates_density[2],
            power_spectrum_dict["gg"][0][0],
            power_spectrum_dict["gg"][1][0],
            power_spectrum_dict["gg"][2][0],
            power_spectrum_dict["gg"][0][1],
            power_spectrum_dict["gg"][1][1],
            power_spectrum_dict["gg"][2][1],
            **kwargs,
        )
        number_densities = len(coordinates_density[0])
    else:
        number_densities = None

    if model_type in ["velocity", "full", "density_velocity"]:
        covariance_dict["vv"] = generator_lai22.compute_cov_vv(
            coordinates_velocity[0],
            coordinates_velocity[1],
            coordinates_velocity[2],
            power_spectrum_dict["vv"][0][0],
            power_spectrum_dict["vv"][1][0],
            **kwargs,
        )
        number_velocities = len(coordinates_velocity[0])
    else:
        number_velocities = None

    if model_type == "full":
        covariance_dict["gv"] = generator_lai22.compute_cov_gv(
            pmax,
            coordinates_density[0],
            coordinates_density[1],
            coordinates_density[2],
            coordinates_velocity[0],
            coordinates_velocity[1],
            coordinates_velocity[2],
            power_spectrum_dict["gv"][0][0],
            power_spectrum_dict["gv"][1][0],
            power_spectrum_dict["gv"][0][1],
            power_spectrum_dict["gv"][1][1],
            **kwargs,
        )
    return covariance_dict, number_densities, number_velocities


def generate_flip(
    model_name,
    model_type,
    power_spectrum_dict,
    coordinates_velocity=None,
    coordinates_density=None,
    additional_parameters_values=None,
    size_batch=10_000,
    number_worker=8,
    hankel=True,
):
    check_generator_need(
        model_type,
        coordinates_density,
        coordinates_velocity,
    )
    covariance_dict = {}
    if model_type in ["density", "full", "density_velocity"]:
        covariance_dict["gg"] = generator_flip.compute_cov(
            model_name,
            "gg",
            power_spectrum_dict["gg"],
            coordinates_density=coordinates_density,
            coordinates_velocity=coordinates_velocity,
            additional_parameters_values=additional_parameters_values,
            size_batch=size_batch,
            number_worker=number_worker,
            hankel=hankel,
        )
        number_densities = len(coordinates_density[0])
    else:
        number_densities = None

    if model_type in ["velocity", "full", "density_velocity"]:
        covariance_dict["vv"] = generator_flip.compute_cov(
            model_name,
            "vv",
            power_spectrum_dict["vv"],
            coordinates_density=coordinates_density,
            coordinates_velocity=coordinates_velocity,
            additional_parameters_values=additional_parameters_values,
            size_batch=size_batch,
            number_worker=number_worker,
            hankel=hankel,
        )
        number_velocities = len(coordinates_velocity[0])
    else:
        number_velocities = None

    if model_type == "full":
        covariance_dict["gv"] = generator_flip.compute_cov(
            model_name,
            "gv",
            power_spectrum_dict["gv"],
            coordinates_density=coordinates_density,
            coordinates_velocity=coordinates_velocity,
            additional_parameters_values=additional_parameters_values,
            size_batch=size_batch,
            number_worker=number_worker,
            hankel=hankel,
        )
    return covariance_dict, number_densities, number_velocities


class CovMatrix:
    def __init__(
        self,
        model_name=None,
        model_type=None,
        covariance_dict=None,
        full_matrix=False,
        number_densities=None,
        number_velocities=None,
    ):
        self.model_name = model_name
        self.model_type = model_type
        self.covariance_dict = covariance_dict
        self.full_matrix = full_matrix
        self.number_densities = number_densities
        self.number_velocities = number_velocities

    @classmethod
    def init_from_flip(
        cls,
        model_name,
        model_type,
        power_spectrum_dict,
        coordinates_density=None,
        coordinates_velocity=None,
        additional_parameters_values=None,
        **kwargs,
    ):
        covariance_dict, number_densities, number_velocities = generate_flip(
            model_name,
            model_type,
            power_spectrum_dict,
            coordinates_density=coordinates_density,
            coordinates_velocity=coordinates_velocity,
            additional_parameters_values=additional_parameters_values,
            **kwargs,
        )
        log.add(f"Covariance matrix generated from flip with {model_name} model")
        return cls(
            model_name=model_name,
            model_type=model_type,
            covariance_dict=covariance_dict,
            number_densities=number_densities,
            number_velocities=number_velocities,
            full_matrix=False,
        )

    @classmethod
    def init_from_generator(
        cls,
        model_name,
        model_type,
        power_spectrum_dict,
        coordinates_velocity=None,
        coordinates_density=None,
        additional_parameters_values=None,
        **kwargs,
    ):
        covariance_dict, number_densities, number_velocities = eval(
            f"generate_{model_name}"
        )(
            model_type,
            power_spectrum_dict,
            coordinates_density=coordinates_density,
            coordinates_velocity=coordinates_velocity,
            **kwargs,
        )

        log.add(f"Covariance matrix generated from {model_name} model")
        return cls(
            model_name=model_name,
            model_type=model_type,
            covariance_dict=covariance_dict,
            number_densities=number_densities,
            number_velocities=number_velocities,
            full_matrix=False,
        )

    @classmethod
    def init_from_file(
        cls,
        model_name,
        model_type,
        filename,
    ):
        log.add(f"Reading from filename not implemented yet")

    @property
    def type(self):
        if self.model_type == "velocity":
            log.add("The covariance model is computed for velocity")
        elif self.model_type == "density":
            log.add("The covariance model is computed for density")
        elif self.model_type == "density_velocity":
            log.add(
                "The covariance model is computed for velocity and density, without cross-term"
            )
        elif self.model_type == "full":
            log.add(
                "The covariance model is computed for velocity and density, with cross-term"
            )
        return self.model_type

    @property
    def loaded(self):
        if self.model_type == "density":
            if "gg" in self.covariance_dict.keys():
                return True
            else:
                return False
        elif self.model_type == "velocity":
            if "vv" in self.covariance_dict.keys():
                return True
            else:
                return False
        elif self.model_type == "density_velocity":
            if ("vv" in self.covariance_dict.keys()) & (
                "gg" in self.covariance_dict.keys()
            ):
                return True
            else:
                return False
        elif self.model_type == "full":
            if (
                ("vv" in self.covariance_dict.keys())
                & ("gg" in self.covariance_dict.keys())
                & ("gv" in self.covariance_dict.keys())
            ):
                return True
            else:
                return False
        else:
            log.add("The model type was not found")
            return False

    def compute_full_matrix(self):
        if self.full_matrix is False:
            for key in ["gg", "vv", "gv"]:
                if key in self.covariance_dict.keys():
                    for i, _ in enumerate(self.covariance_dict[key]):
                        if key == "gv":
                            self.covariance_dict[key][
                                i
                            ] = cov_utils.return_full_cov_cross(
                                self.covariance_dict[key][i],
                                self.covariance_dict["gg"][0].shape[0],
                                self.covariance_dict["vv"][0].shape[0],
                            )
                        else:
                            self.covariance_dict[key][i] = cov_utils.return_full_cov(
                                self.covariance_dict[key][i]
                            )
            self.full_matrix = True

    def write(
        self,
        filename,
    ):
        np.savez(filename, **self.covariance_dict)
        log.add(f"Cov written in {filename}.")
