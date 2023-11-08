import numpy as np
import time
from flip.utils import create_log
from flip.covariance.lai22 import generator as generator_lai22
from flip.covariance.carreres23 import generator as generator_carreres23

from flip.covariance.adamsblake20 import coefficients as coefficients_adamsblake20
from flip.covariance.lai22 import coefficients as coefficients_lai22
from flip.covariance.carreres23 import coefficients as coefficients_carreres23
from flip.covariance.ravouxcarreres import coefficients as coefficients_ravouxcarreres


from flip.covariance import generator as generator_flip
from flip.covariance import cov_utils


log = create_log()


def generator_need(
    coordinates_density=None,
    coordinates_velocity=None,
):
    """
    The generator_need function checks if the coordinates_density and coordinates_velocity inputs are provided.
    If they are not, it raises a ValueError exception.


    Args:
        coordinates_density: Generate the density covariance matrix
        coordinates_velocity: Generate the covariance matrix of the velocity field
        : Check if the coordinates are provided or not

    Returns:
        A list of the coordinates that are needed to proceed with covariance generation

    """
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
    """
    The check_generator_need function is used to check if the generator_need function
    is called with the correct arguments. The model type determines which coordinates are needed,
    and these are passed as arguments to generator_need.

    Args:
        model_type: Determine if the density, velocity or full model is being used
        coordinates_density: Check if the density coordinates are needed
        coordinates_velocity: Determine whether the velocity model is needed

    Returns:
        A boolean

    """
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
    """
    The generate_carreres23 function generates a covariance matrix for the velocity field.

    Args:
        model_type: Specify the type of model to generate
        power_spectrum_dict: Pass the power spectrum to the function
        coordinates_density: Specify the coordinates of the density field
        coordinates_velocity: Generate the covariance matrix
        **kwargs: Pass additional parameters to the function
        : Generate the covariance matrix for the velocity field

    Returns:
        A dictionary with a single key &quot;vv&quot;

    """
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
    """
    The generate_lai22 function generates the covariance matrix for a given model type.

    Args:
        model_type: Determine which covariance matrices to generate
        power_spectrum_dict: Pass the power spectrum of the density and velocity fields
        coordinates_velocity: Generate the velocity field
        coordinates_density: Pass the coordinates of the density field
        pmax: Determine the maximum order of the legendre polynomial used in the computation of
        qmax: Determine the maximum order of the legendre polynomials used to compute the covariance matrix
        **kwargs: Pass keyworded, variable-length argument list
        : Define the type of model to be generated

    Returns:
        A dictionary of covariance matrices,

    """
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
    """
    The generate_flip function computes the covariance matrix for a given model.

    Args:
        model_name: Select the model to use
        model_type: Determine the type of model to generate
        power_spectrum_dict: Store the power spectra of the different fields
        coordinates_velocity: Specify the coordinates of the velocity field
        coordinates_density: Specify the coordinates of the density field
        additional_parameters_values: Pass the values of the additional parameters to be used in the computation of covariance matrices
        size_batch: Split the computation of the covariance matrix into smaller batches
        number_worker: Specify the number of cores to use for computing the covariance matrix
        hankel: Decide whether to use the hankel transform or not
        : Define the number of workers to use for the computation

    Returns:
        A dictionary with the covariance matrices and their dimensions

    """
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


def contract_flip(
    model_name,
    model_type,
    power_spectrum_dict,
    r_perpendicular,
    r_parallel,
    r_reference,
    additional_parameters_values=None,
    number_worker=8,
    hankel=True,
):
    # CR - make it more general, not only one binning.

    coord_rper_rpar = np.array(
        np.meshgrid(r_perpendicular, r_parallel, indexing="ij")
    ).reshape((2, len(r_perpendicular) * len(r_parallel)))

    contraction_coordinates = np.zeros((3, len(r_perpendicular) * len(r_parallel)))
    contraction_coordinates[0, :] = np.sqrt(
        coord_rper_rpar[0, :] ** 2 + coord_rper_rpar[1, :] ** 2
    )
    contraction_coordinates[1, :] = np.arctan2(
        coord_rper_rpar[0, :], r_reference + coord_rper_rpar[1, :]
    )
    contraction_coordinates[2, :] = np.arcsin(
        np.clip(
            (
                (r_reference / contraction_coordinates[0, :])
                + (
                    coord_rper_rpar[0, :]
                    / (
                        contraction_coordinates[0, :]
                        * np.sin(contraction_coordinates[1, :])
                    )
                )
            )
            * np.sqrt((1 - np.cos(contraction_coordinates[1, :])) / 2),
            -1,
            1,
        )
    )

    contraction_covariance_dict = {}
    if model_type in ["density", "full", "density_velocity"]:
        contraction_covariance_dict["gg"] = generator_flip.compute_coeficient(
            [contraction_coordinates],
            model_name,
            "gg",
            power_spectrum_dict["gg"],
            additional_parameters_values=additional_parameters_values,
            number_worker=number_worker,
            hankel=hankel,
        )[:, 1:]

    if model_type in ["velocity", "full", "density_velocity"]:
        contraction_covariance_dict["vv"] = generator_flip.compute_coeficient(
            [contraction_coordinates],
            model_name,
            "vv",
            power_spectrum_dict["vv"],
            additional_parameters_values=additional_parameters_values,
            number_worker=number_worker,
            hankel=hankel,
        )[:, 1:]

    if model_type == "full":
        contraction_covariance_dict["gv"] = generator_flip.compute_coeficient(
            [contraction_coordinates],
            model_name,
            "gv",
            power_spectrum_dict["gv"],
            additional_parameters_values=additional_parameters_values,
            number_worker=number_worker,
            hankel=hankel,
        )[:, 1:]

    return contraction_covariance_dict, contraction_coordinates


class CovMatrix:
    def __init__(
        self,
        model_name=None,
        model_type=None,
        covariance_dict=None,
        full_matrix=False,
        number_densities=None,
        number_velocities=None,
        contraction_covariance_dict=None,
        contraction_coordinates=None,
    ):
        """
        The __init__ function is called when the class is instantiated.
        It sets up the instance of the class, and defines all of its attributes.


        Args:
            self: Represent the instance of the class
            model_name: Set the name of the model
            model_type: Determine which covariance matrix to use
            covariance_dict: Store the covariance matrix
            full_matrix: Determine whether the covariance matrix is stored as a full matrix or in compressed form
            number_densities: Store the number of density fields that are used in the covariance matrix
            number_velocities: Determine the number of velocity bins in the covariance matrix
            : Initialize the class

        Returns:
            The instance of the class

        """
        self.model_name = model_name
        self.model_type = model_type
        self.covariance_dict = covariance_dict
        self.full_matrix = full_matrix
        self.number_densities = number_densities
        self.number_velocities = number_velocities
        self.contraction_covariance_dict = contraction_covariance_dict
        self.contraction_coordinates = contraction_coordinates

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
        """
        The init_from_flip function is a function that initializes the covariance matrix from the flip code.
        It takes as input:
            - model_name: name of the model used to generate the covariance matrix (e.g., 'lai22')
            - model_type: type of data used to generate the covariance matrix (e.g., 'density' or 'velocity')
            - power_spectrum_dict: dictionary containing all information about power spectrum, including k and P(k) values, redshift, etc...
                It is generated by calling getPowerSpectrumDict() in

        Args:
            cls: Indicate that the function is a class method
            model_name: Determine which model to use for the covariance matrix
            model_type: Determine the type of model to be used
            power_spectrum_dict: Pass the power spectrum of the model
            coordinates_density: Specify the coordinates of the density field
            coordinates_velocity: Define the velocity coordinates of the covariance matrix
            additional_parameters_values: Pass the values of additional parameters to the flip code
            **kwargs: Pass a variable number of keyword arguments to the function
            : Generate the covariance matrix from a flip model

        Returns:
            A covariancematrix object

        """
        begin = time.time()
        covariance_dict, number_densities, number_velocities = generate_flip(
            model_name,
            model_type,
            power_spectrum_dict,
            coordinates_density=coordinates_density,
            coordinates_velocity=coordinates_velocity,
            additional_parameters_values=additional_parameters_values,
            **kwargs,
        )
        end = time.time()
        log.add(
            f"Covariance matrix generated from flip with {model_name} model in {'{:.2e}'.format(end - begin)} seconds"
        )
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
        """
        The init_from_generator function is a helper function that allows the user to initialize
        a Covariance object from a generator. The init_from_generator function takes in as arguments:
            - cls: the class of the object being initialized (Covariance)
            - model_name: name of covariance model used to generate covariance matrix (e.g., 'lai22')
            - model_type: type of covariance matrix generated ('density' or 'velocity')
            - power spectrum dictionary containing keys for each redshift bin and values corresponding to
                power spectra at those red

        Args:
            cls: Refer to the class itself
            model_name: Specify the type of model used to generate the covariance matrix
            model_type: Determine which model to use
            power_spectrum_dict: Pass the power spectrum to the generate_* functions
            coordinates_velocity: Generate the velocity covariance matrix
            coordinates_density: Generate the density field
            additional_parameters_values: Pass additional parameters to the generator function
            **kwargs: Pass a variable number of keyword arguments to the function
            : Generate the covariance matrix from a given model

        Returns:
            An object of the class covariancematrix

        """
        begin = time.time()
        covariance_dict, number_densities, number_velocities = eval(
            f"generate_{model_name}"
        )(
            model_type,
            power_spectrum_dict,
            coordinates_density=coordinates_density,
            coordinates_velocity=coordinates_velocity,
            **kwargs,
        )
        end = time.time()
        log.add(
            f"Covariance matrix generated from {model_name} model in {'{:.2e}'.format(end - begin)} seconds"
        )
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
        """
        The init_from_file function is used to initialize a model from a file.

        Args:
            cls: Create a new instance of the class
            model_name: Name the model
            model_type: Determine the type of model to be created
            filename: Specify the file to read from
            : Specify the name of the model

        Returns:
            A tuple of the model and a list of

        """
        log.add(f"Reading from filename not implemented yet")

    @classmethod
    def init_contraction_from_flip(
        cls,
        model_name,
        model_type,
        power_spectrum_dict,
        bin_centers_2d,
        r0,
        additional_parameters_values=None,
        **kwargs,
    ):
        """
        The init_contraction_from_flip function is a helper function that allows the user to initialize
        a Contraction object from a FLIP model. The contraction_covariance_dict and contraction_coordinates
        are calculated by contracting with the FLIP model, which is done in contract_flip. This function
        is called in __init__ of Contraction.

        Args:
            cls: Create a new instance of the class
            model_name: Determine which model to use for the contraction
            model_type: Determine the type of model to be used
            power_spectrum_dict: Store the power spectrum of the model
            bin_centers_2d: Pass the bin centers of the 2d correlation function
            r0: Define the size of the grid
            additional_parameters_values: Pass in the values of the additional parameters that are
            **kwargs: Pass keyworded, variable-length argument list
            : Set the model type

        Returns:
            An instance of the contraction class
        """

        contraction_covariance_dict, contraction_coordinates = contract_flip(
            model_name,
            model_type,
            power_spectrum_dict,
            bin_centers_2d,
            r0,
            additional_parameters_values=additional_parameters_values,
            **kwargs,
        )
        return cls(
            model_name=model_name,
            model_type=model_type,
            contraction_covariance_dict=contraction_covariance_dict,
            contraction_coordinates=contraction_coordinates,
        )

    @property
    def type(self):
        """
        The type function is used to determine the type of covariance model that will be computed.
        The options are:
            - velocity: The covariance model is computed for velocity only.
            - density: The covariance model is computed for density only.
            - density_velocity: The covariance model is computed for both velocity and density, without cross-term (i.e., the covariances between velocities and densities are zero). This option should be used when computing a full 3D tomography in which we want to compute a separate 1D tomography along each axis (x, y, z

        Args:
            self: Represent the instance of the class

        Returns:
            The type of the model

        """
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
        """
        The loaded function checks if the covariance matrix is loaded.

        Args:
            self: Refer to the object itself

        Returns:
            A boolean

        """
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

    def compute_covariance_sum(
        self,
        parameter_values_dict,
    ):
        """
        The compute_covariance_sum function computes the sum of all covariance matrices
            and adds the diagonal terms.

        Args:
            self: Access the attributes of the class
            parameter_values_dict: Pass the values of the parameters
            : Compute the covariance matrix

        Returns:
            The sum of the covariance matrices with their respective coefficients

        """
        coefficients_dict = eval(f"coefficients_{self.model_name}.get_coefficients")(
            self.model_type,
            parameter_values_dict,
        )
        coefficients_dict_diagonal = eval(
            f"coefficients_{self.model_name}.get_diagonal_coefficients"
        )(
            self.model_type,
            parameter_values_dict,
        )

        if self.model_type == "density":
            covariance_sum = np.sum(
                [
                    coefficients_dict["gg"][i] * cov
                    for i, cov in enumerate(self.covariance_dict["gg"])
                ],
                axis=0,
            )
            covariance_sum += np.diag(
                coefficients_dict_diagonal["gg"] + self.vector_err**2
            )

        elif self.model_type == "velocity":
            covariance_sum = np.sum(
                [
                    coefficients_dict["vv"][i] * cov
                    for i, cov in enumerate(self.covariance_dict["vv"])
                ],
                axis=0,
            )

            covariance_sum += np.diag(
                coefficients_dict_diagonal["vv"] + self.vector_err**2
            )

        elif self.model_type in ["density_velocity", "full"]:
            number_densities = self.number_densities
            number_velocities = self.number_velocities
            density_err = self.vector_err[:number_densities]
            velocity_err = self.vector_err[
                number_densities : number_densities + number_velocities
            ]

            if self.model_type == "density_velocity":
                covariance_sum_gv = np.zeros((number_densities, number_velocities))
            elif self.model_type == "full":
                covariance_sum_gv = np.sum(
                    [
                        coefficients_dict["gv"][i] * cov
                        for i, cov in enumerate(self.covariance_dict["gv"])
                    ],
                    axis=0,
                )
            covariance_sum_gg = np.sum(
                [
                    coefficients_dict["gg"][i] * cov
                    for i, cov in enumerate(self.covariance_dict["gg"])
                ],
                axis=0,
            )
            covariance_sum_gg += np.diag(
                coefficients_dict_diagonal["gg"] + density_err**2
            )

            covariance_sum_vv = np.sum(
                [
                    coefficients_dict["vv"][i] * cov
                    for i, cov in enumerate(self.covariance_dict["vv"])
                ],
                axis=0,
            )

            covariance_sum_vv += np.diag(
                coefficients_dict_diagonal["vv"] + velocity_err**2
            )

            covariance_sum = np.block(
                [
                    [covariance_sum_gg, covariance_sum_gv],
                    [covariance_sum_gv.T, covariance_sum_vv],
                ]
            )
        else:
            log.add(f"Wrong model type in the loaded covariance.")

        return covariance_sum

    def compute_contraction_sum(
        self,
        parameter_values_dict,
    ):
        """
        The compute_contraction_sum function computes the sum of all the contractions
            for a given model type and parameter values.

        Args:
            self: Make the function a method of the class
            parameter_values_dict: Get the coefficients for each of the covariances
            : Get the coefficients of the model

        Returns:
            A dictionary of contraction_covariance_sum
        """
        coefficients_dict = eval(f"coefficients_{self.model_name}.get_coefficients")(
            self.model_type,
            parameter_values_dict,
        )
        contraction_covariance_sum_dict = {}
        if self.model_type == "density":
            contraction_covariance_sum_dict["gg"] = np.sum(
                [
                    coefficients_dict["gg"][i] * cov
                    for i, cov in enumerate(self.contraction_covariance_dict["gg"])
                ],
                axis=0,
            )

        elif self.model_type == "velocity":
            contraction_covariance_sum_dict["vv"] = np.sum(
                [
                    coefficients_dict["vv"][i] * cov
                    for i, cov in enumerate(self.contraction_covariance_dict["vv"])
                ],
                axis=0,
            )

        elif self.model_type in ["density_velocity", "full"]:
            if self.model_type == "full":
                contraction_covariance_sum_dict["gv"] = np.sum(
                    [
                        coefficients_dict["gv"][i] * cov
                        for i, cov in enumerate(self.contraction_covariance_dict["gv"])
                    ],
                    axis=0,
                )
            contraction_covariance_sum_dict["gg"] = np.sum(
                [
                    coefficients_dict["gg"][i] * cov
                    for i, cov in enumerate(self.contraction_covariance_dict["gg"])
                ],
                axis=0,
            )

            contraction_covariance_sum_dict["vv"] = np.sum(
                [
                    coefficients_dict["vv"][i] * cov
                    for i, cov in enumerate(self.contraction_covariance_dict["vv"])
                ],
                axis=0,
            )
        else:
            log.add(f"Wrong model type in the loaded covariance.")

        return contraction_covariance_sum_dict

    def compute_full_matrix(self):
        """
        The compute_full_matrix function is used to convert the covariance matrices from sparse to full.
        This function is called in the compute_covariance_matrix function, which is called by all of the
        generator functions (e.g., generator_lai22). The compute_full_matrix function checks if a matrix has already been converted from sparse to full and skips it if so. If not, then it converts each matrix into a full one.

        Args:
            self: Bind the method to an object

        Returns:
            The full covariance matrix

        """
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
        """
        The write function writes the covariance matrix to a file.

        Args:
            self: Represent the instance of the class
            filename: Specify the name of the file to be written
            : Specify the name of the file in which we want to save our covariance matrix

        Returns:
            Nothing

        """
        np.savez(filename, **self.covariance_dict)
        log.add(f"Cov written in {filename}.")
