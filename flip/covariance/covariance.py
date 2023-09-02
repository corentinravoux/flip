from flip.utils import create_log
from flip.covariance.lai22 import generator as generator_lai22
from flip.covariance.carreres23 import generator as generator_carreres23
from flip.covariance import generator as generator_flip


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


def generate_carreres23(
    model_type,
    power_spectrum_dict,
    coordinates_velocity=None,
    **kwargs,
):
    generator_need(
        coordinates_density=False,
        coordinates_velocity=coordinates_velocity,
    )
    cov_vv = generator_carreres23.covariance_vv(
        coordinates_velocity[0],
        coordinates_velocity[1],
        coordinates_velocity[2],
        power_spectrum_dict["vv"][0][0],
        power_spectrum_dict["vv"][0][1],
        **kwargs,
    )
    return {"vv": cov_vv}


def generate_lai22(
    model_type,
    power_spectrum_dict,
    coordinates_velocity=None,
    coordinates_density=None,
    pmax=3,
    qmax=3,
    **kwargs,
):
    if model_type == "joint":
        generator_need(
            coordinates_density=coordinates_density,
            coordinates_velocity=coordinates_velocity,
        )
    elif model_type == "density":
        generator_need(
            coordinates_density=coordinates_density,
            coordinates_velocity=False,
        )
    elif model_type == "velocity":
        generator_need(
            coordinates_density=False,
            coordinates_velocity=coordinates_velocity,
        )
    covariance_dict = {}

    if model_type in ["joint", "density"]:
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

    if model_type == "joint":
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

    if model_type in ["joint", "velocity"]:
        covariance_dict["vv"] = generator_lai22.compute_cov_vv(
            coordinates_velocity[0],
            coordinates_velocity[1],
            coordinates_velocity[2],
            power_spectrum_dict["vv"][0][0],
            power_spectrum_dict["vv"][1][0],
            **kwargs,
        )

    return covariance_dict


def generate_flip(
    model_name,
    model_type,
    power_spectrum_dict,
    coordinates_velocity=None,
    coordinates_density=None,
    additional_parameters_values=None,
):
    covariance_dict = {}
    if model_type in ["density", "full"]:
        generator_need(
            coordinates_density=coordinates_density,
            coordinates_velocity=False,
        )
        covariance_dict["gg"] = generator_flip.compute_cov(
            model_name,
            "gg",
            power_spectrum_dict["gg"],
            coordinates_density=coordinates_density,
            coordinates_velocity=coordinates_velocity,
            additional_parameters_values=additional_parameters_values,
            size_batch=10_000,
            number_worker=8,
        )
    if model_type in ["full"]:
        generator_need(
            coordinates_density=coordinates_density,
            coordinates_velocity=coordinates_velocity,
        )
        covariance_dict["gv"] = generator_flip.compute_cov(
            model_name,
            "gv",
            power_spectrum_dict["gv"],
            coordinates_density=coordinates_density,
            coordinates_velocity=coordinates_velocity,
            additional_parameters_values=additional_parameters_values,
            size_batch=10_000,
            number_worker=8,
        )
    if model_type in ["velocity", "full"]:
        generator_need(
            coordinates_density=False,
            coordinates_velocity=coordinates_velocity,
        )
        covariance_dict["vv"] = generator_flip.compute_cov(
            model_name,
            "vv",
            power_spectrum_dict["vv"],
            coordinates_density=coordinates_density,
            coordinates_velocity=coordinates_velocity,
            additional_parameters_values=additional_parameters_values,
            size_batch=10_000,
            number_worker=8,
        )
    return covariance_dict


class CovMatrix:
    def __init__(
        self,
        model_name=None,
        model_type=None,
        covariance_dict=None,
    ):
        self.model_name = model_name
        self.model_type = model_type
        self.covariance_dict = covariance_dict

    @classmethod
    def init_from_flip(
        cls,
        model_name,
        model_type,
        power_spectrum_dict,
        coordinates_velocity=None,
        coordinates_density=None,
        additional_parameters_values=None,
    ):
        covariance_dict = generate_flip(
            model_name,
            model_type,
            power_spectrum_dict,
            coordinates_density=coordinates_density,
            coordinates_velocity=coordinates_velocity,
            additional_parameters_values=additional_parameters_values,
        )
        log.add(f"Covariance matrix generated from flip with {model_name} model")
        return cls(
            model_name=model_name,
            model_type=model_type,
            covariance_dict=covariance_dict,
        )

    @classmethod
    def init_from_generator(
        cls,
        model_name,
        model_type,
        power_spectrum_dict,
        coordinates_velocity=None,
        coordinates_density=None,
        **kwargs,
    ):
        covariance_dict = eval(f"generate_{model_name}")(
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

    def write(
        self,
        filename,
    ):
        log.add(f"Writing not implemented yet")
