import importlib
import pickle
import time

import numpy as np

from flip.utils import create_log

try:
    import jax.numpy as jnp
    from jax import jit

    jax_installed = True
except:
    import numpy as jnp

    jax_installed = False
from flip.covariance import cov_utils

log = create_log()


def compute_covariance_sum_density(
    coefficients_dict,
    covariance_dict,
    coefficients_dict_diagonal,
    vector_err,
    number_densities,
    number_velocities,
):
    covariance_sum = jnp.sum(
        jnp.array(
            [
                coefficients_dict["gg"][i] * cov
                for i, cov in enumerate(covariance_dict["gg"])
            ]
        ),
        axis=0,
    )
    covariance_sum += jnp.diag(coefficients_dict_diagonal["gg"] + vector_err**2)

    return covariance_sum


def compute_covariance_sum_velocity(
    coefficients_dict,
    covariance_dict,
    coefficients_dict_diagonal,
    vector_err,
    number_densities,
    number_velocities,
):
    covariance_sum = jnp.sum(
        jnp.array(
            [
                coefficients_dict["vv"][i] * cov
                for i, cov in enumerate(covariance_dict["vv"])
            ]
        ),
        axis=0,
    )

    covariance_sum += jnp.diag(coefficients_dict_diagonal["vv"] + vector_err**2)

    return covariance_sum


def compute_covariance_sum_density_velocity(
    coefficients_dict,
    covariance_dict,
    coefficients_dict_diagonal,
    vector_err,
    number_densities,
    number_velocities,
):

    density_err = vector_err[:number_densities]
    velocity_err = vector_err[number_densities : number_densities + number_velocities]

    covariance_sum_gv = jnp.zeros((number_densities, number_velocities))
    covariance_sum_gg = jnp.sum(
        jnp.array(
            [
                coefficients_dict["gg"][i] * cov
                for i, cov in enumerate(covariance_dict["gg"])
            ]
        ),
        axis=0,
    )
    covariance_sum_gg += jnp.diag(coefficients_dict_diagonal["gg"] + density_err**2)

    covariance_sum_vv = jnp.sum(
        jnp.array(
            [
                coefficients_dict["vv"][i] * cov
                for i, cov in enumerate(covariance_dict["vv"])
            ]
        ),
        axis=0,
    )

    covariance_sum_vv += jnp.diag(coefficients_dict_diagonal["vv"] + velocity_err**2)

    covariance_sum_vg = -covariance_sum_gv.T

    covariance_sum = jnp.block(
        [
            [covariance_sum_gg, covariance_sum_gv],
            [covariance_sum_vg, covariance_sum_vv],
        ]
    )
    return covariance_sum


def compute_covariance_sum_full(
    coefficients_dict,
    covariance_dict,
    coefficients_dict_diagonal,
    vector_err,
    number_densities,
    number_velocities,
):

    density_err = vector_err[:number_densities]
    velocity_err = vector_err[number_densities : number_densities + number_velocities]

    covariance_sum_gv = jnp.sum(
        jnp.array(
            [
                coefficients_dict["gv"][i] * cov
                for i, cov in enumerate(covariance_dict["gv"])
            ]
        ),
        axis=0,
    )
    covariance_sum_gg = jnp.sum(
        jnp.array(
            [
                coefficients_dict["gg"][i] * cov
                for i, cov in enumerate(covariance_dict["gg"])
            ]
        ),
        axis=0,
    )
    covariance_sum_gg += jnp.diag(coefficients_dict_diagonal["gg"] + density_err**2)

    covariance_sum_vv = jnp.sum(
        jnp.array(
            [
                coefficients_dict["vv"][i] * cov
                for i, cov in enumerate(covariance_dict["vv"])
            ]
        ),
        axis=0,
    )

    covariance_sum_vv += jnp.diag(coefficients_dict_diagonal["vv"] + velocity_err**2)

    covariance_sum_vg = -covariance_sum_gv.T

    covariance_sum = jnp.block(
        [
            [covariance_sum_gg, covariance_sum_gv],
            [covariance_sum_vg, covariance_sum_vv],
        ]
    )
    return covariance_sum


class CovMatrix:
    def __init__(
        self,
        model_name=None,
        model_type=None,
        los_definition=None,
        covariance_dict=None,
        full_matrix=False,
        number_densities=None,
        number_velocities=None,
        variant=None,
    ):
        """
        The __init__ function is called when the class is instantiated.
        It sets up the instance of the class, and defines all of its attributes.


        Args:
            self: Represent the instance of the class
            model_name: Identify the model
            model_type: Define the type of model that is being used
            los_definition: Define the angle between two vectors
            covariance_dict: Store the covariance matrix
            full_matrix: Determine whether the covariance matrix is stored as a full matrix or in sparse form
            number_densities: Set the number of density variables in the model
            number_velocities: Set the number of velocities in the model
            variant: Name of the variation of the analysis

        Returns:
            An object of the class
        """

        self.model_name = model_name
        self.model_type = model_type
        self.los_definition = los_definition
        self.covariance_dict = covariance_dict
        self.full_matrix = full_matrix
        self.number_densities = number_densities
        self.number_velocities = number_velocities
        self.variant = variant

    @classmethod
    def init_from_flip(
        cls,
        model_name,
        model_type,
        power_spectrum_dict,
        coordinates_density=None,
        coordinates_velocity=None,
        additional_parameters_values=None,
        los_definition="bisector",
        variant=None,
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

        Returns:
            A covariancematrix object

        """
        begin = time.time()
        from flip.covariance import generator as generator_flip

        (
            covariance_dict,
            number_densities,
            number_velocities,
        ) = generator_flip.generate_covariance(
            model_name,
            model_type,
            power_spectrum_dict,
            coordinates_density=coordinates_density,
            coordinates_velocity=coordinates_velocity,
            additional_parameters_values=additional_parameters_values,
            los_definition=los_definition,
            **kwargs,
        )
        end = time.time()
        log.add(
            f"Covariance matrix generated from flip with {model_name} model in {'{:.2e}'.format(end - begin)} seconds"
        )
        return cls(
            model_name=model_name,
            model_type=model_type,
            los_definition=los_definition,
            covariance_dict=covariance_dict,
            full_matrix=False,
            number_densities=number_densities,
            number_velocities=number_velocities,
            variant=variant,
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
        variant=None,
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
        generator = importlib.import_module(f"flip.covariance.{model_name}.generator")

        (
            covariance_dict,
            number_densities,
            number_velocities,
            los_definition,
        ) = generator.generate_covariance(
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
            los_definition=los_definition,
            covariance_dict=covariance_dict,
            full_matrix=False,
            number_densities=number_densities,
            number_velocities=number_velocities,
            variant=variant,
        )

    @classmethod
    def init_from_file(
        cls,
        filename,
        file_format,
    ):
        if file_format == "parquet":
            raise NotImplementedError(f"Reading from parquet not implemented yet")
        if file_format == "pickle":
            with open(f"{filename}.pickle", "rb") as file_read:
                class_attrs_dictionary = pickle.load(file_read)
        if file_format == "npz":
            class_attrs_dictionary = dict(np.load(f"{filename}.npz", allow_pickle=True))
            for key in class_attrs_dictionary.keys():
                class_attrs_dictionary[key] = class_attrs_dictionary[key].item()
        return cls(**class_attrs_dictionary)

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
        vector_err,
        use_jax=False,
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
        coefficients = importlib.import_module(
            f"flip.covariance.{self.model_name}.coefficients"
        )

        coefficients_dict = coefficients.get_coefficients(
            self.model_type,
            parameter_values_dict,
            variant=self.variant,
        )
        coefficients_dict_diagonal = coefficients.get_diagonal_coefficients(
            self.model_type,
            parameter_values_dict,
        )

        covariance_sum_func = eval(f"compute_covariance_sum_{self.model_type}")
        if jax_installed & use_jax:
            covariance_sum_func = jit(covariance_sum_func)
        covariance_sum = covariance_sum_func(
            coefficients_dict,
            self.covariance_dict,
            coefficients_dict_diagonal,
            vector_err,
            self.number_densities,
            self.number_velocities,
        )
        return covariance_sum

    def compute_covariance_sum_eigenvalues(
        self,
        parameter_values_dict,
        vector_err,
    ):
        covariance_sum = self.compute_covariance_sum(
            parameter_values_dict,
            vector_err,
        )
        return np.linalg.eigvals(covariance_sum)

    def compute_full_matrix(self):
        """
        The compute_full_matrix function takes the covariance matrix and fills in all of the missing values.

        Args:
            self: Bind the method to the object

        Returns:
            A dictionary with the full covariance matrices for each redshift bin

        """
        if self.full_matrix is False:
            for key in ["gg", "vv", "gv"]:
                if key in self.covariance_dict.keys():
                    if key == "gg":
                        new_shape = (
                            self.covariance_dict[key].shape[0],
                            self.number_densities,
                            self.number_densities,
                        )
                    elif key == "gv":
                        new_shape = (
                            self.covariance_dict[key].shape[0],
                            self.number_densities,
                            self.number_velocities,
                        )
                    elif key == "vv":
                        new_shape = (
                            self.covariance_dict[key].shape[0],
                            self.number_velocities,
                            self.number_velocities,
                        )
                    new_cov = np.zeros(new_shape)
                    for i, _ in enumerate(self.covariance_dict[key]):
                        if key == "gv":
                            new_cov[i] = cov_utils.return_full_cov_cross(
                                self.covariance_dict[key][i],
                                self.number_densities,
                                self.number_velocities,
                            )
                        else:
                            new_cov[i] = cov_utils.return_full_cov(
                                self.covariance_dict[key][i]
                            )
                    self.covariance_dict[key] = new_cov

            self.full_matrix = True

    def write(
        self,
        filename,
        file_format,
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
        class_attrs_dictionary = {
            key: eval(f"self.{key}", {"self": self}) for key in vars(self)
        }
        if file_format == "parquet":
            import pyarrow

            metadata = {}
            for key in ["gg", "gv", "vv"]:
                if key in self.covariance_dict:
                    metadata[f"number_matrices_{key}"] = str(
                        self.covariance_dict[key].shape[0]
                    )

            class_attrs_dictionary.pop("covariance_dict")
            metadata.update(class_attrs_dictionary)
            for key in metadata:
                metadata[key] = str(metadata[key])
            covariance_dict_flatten = self.covariance_dict.copy()
            for key in covariance_dict_flatten:
                covariance_dict_flatten[key] = covariance_dict_flatten[key].ravel()
            pa_table = pyarrow.table(covariance_dict_flatten)
            pa_table = pa_table.replace_schema_metadata(metadata)
            pyarrow.parquet.write_table(pa_table, f"{filename}.parquet")

        elif file_format == "pickle":
            with open(f"{filename}.pickle", "wb") as file_save:
                pickle.dump(class_attrs_dictionary, file_save)

        elif file_format == "npz":
            np.savez(f"{filename}.npz", **class_attrs_dictionary)
