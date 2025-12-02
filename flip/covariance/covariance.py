import importlib
import inspect
import pickle
import time
from functools import partial

import numpy as np

from flip.covariance import cov_utils
from flip.utils import create_log

from ..config import __use_jax__

if __use_jax__:
    try:
        import jax.numpy as jnp
        from jax import jit

        jax_installed = True

    except ImportError:
        import numpy as jnp

        jax_installed = False
else:
    import numpy as jnp

    jax_installed = False


log = create_log()


def _read_free_par(
    model_name,
    model_kind,
    variant=None,
):
    _free_par = importlib.import_module(f"flip.covariance.{model_name}")._free_par
    model_kind = model_kind.split("_")

    if variant is None:
        variant = "baseline"

    free_par = []
    for k, val in _free_par.items():
        val = np.atleast_1d(val)
        for v in val:
            fp_def = v.split("@")
            fp_model, fp_variant = fp_def[0], fp_def[1:]
            if "full" in model_kind or fp_model == "all" or fp_model in model_kind:
                if "all" in fp_variant or variant in fp_variant:
                    free_par.append(k)
                    continue
    return list(set(free_par))


def compute_covariance_sum(
    covariance_dict,
    coefficients_dict,
    coefficients_dict_diagonal,
    vector_variance,
    cov_matrix_prefactor_dict=None,
    kind="full",
    parameter_values_dict=None,
):
    if kind == "density":
        keys = ["gg"]
    elif kind == "velocity":
        keys = ["vv"]
    elif kind == "density_velocity":
        keys = ["gg", "vv"]
    elif kind == "full":
        keys = ["gg", "gv", "vv"]

    covariance_sum_ = {}

    if cov_matrix_prefactor_dict is None:
        cov_matrix_prefactor_dict = {k: 1 for k in keys}

    for k in keys:
        if parameter_values_dict is not None:
            covariance_evaluated = np.array(
                [cov(parameter_values_dict) for i, cov in enumerate(covariance_dict[k])]
            )
        else:
            covariance_evaluated = covariance_dict[k]

        covariance_sum_[k] = jnp.sum(
            coefficients_dict[k][:, None, None]
            * cov_matrix_prefactor_dict[k]
            * covariance_evaluated,
            axis=0,
        )

        if k in coefficients_dict_diagonal:
            covariance_sum_[k] += coefficients_dict_diagonal[k] * jnp.eye(
                covariance_sum_[k].shape[0]
            )

    if len(keys) == 1:
        covariance_sum = covariance_sum_[keys[0]]
    else:
        # Assemble full covariance; handle density_velocity (no cross-term) vs full (with gv)
        if kind == "density_velocity":
            zeros_gv = jnp.zeros(
                (covariance_sum_["gg"].shape[0], covariance_sum_["vv"].shape[0])
            )
            covariance_sum = jnp.block(
                [
                    [covariance_sum_["gg"], zeros_gv],
                    [zeros_gv.T, covariance_sum_["vv"]],
                ]
            )
        else:
            covariance_sum = jnp.block(
                [
                    [covariance_sum_["gg"], covariance_sum_["gv"]],
                    [covariance_sum_["gv"].T, covariance_sum_["vv"]],
                ]
            )

    if len(vector_variance.shape) == 1:
        covariance_sum += jnp.diag(vector_variance)
    else:
        covariance_sum += vector_variance
    return covariance_sum


class CovMatrix:
    """CovMatrix is a class for managing and manipulating covariance matrices used in cosmological analyses, particularly for models involving density and velocity fields. It provides methods for initialization from various sources, conversion between flat and matrix forms, masking, and file I/O operations.

    Attributes:
        model_name (str): Name of the covariance model.
        model_kind (str): Type of covariance ('density', 'velocity', 'density_velocity', 'full').
        free_par (dict): Dictionary of free parameters for the model.
        los_definition (str): Definition of the line-of-sight.
        covariance_dict (dict): Dictionary containing covariance matrices.
        matrix_form (bool): Indicates if covariance is in matrix form.
        variant (str): Variant of the model.
        number_densities (int): Number of density bins.
        number_velocities (int): Number of velocity bins.
        coefficients (module): Imported module for model coefficients.
        compute_covariance_sum (callable): Function to compute covariance sum.
        compute_covariance_sum_jit (callable): JIT-compiled function to compute covariance sum (if JAX is installed).

    Methods:
        __init__: Initializes a CovMatrix instance.
        init_from_flip: Class method to initialize from the flip code generator.
        init_from_generator: Class method to initialize from a model-specific generator.
        init_from_file: Class method to initialize from a file (pickle, npz, parquet).
        kind: Property returning the kind of covariance model.
        loaded: Property indicating if the covariance matrix is loaded.
        init_compute_covariance_sum: Initializes functions for computing covariance sums.
        compute_covariance_sum_eigenvalues: Computes eigenvalues of the covariance sum.
        compute_matrix_covariance: Converts flat covariance to matrix form.
        compute_flat_covariance: Converts matrix covariance to flat form.
        write: Writes the covariance matrix to a file (pickle, npz, parquet).
        mask: Returns a masked CovMatrix instance based on provided masks for density and velocity.

    Usage:
        CovMatrix can be instantiated directly or via class methods for loading from generators or files. It supports conversion between flat and matrix forms, masking, and saving/loading to disk.

    Raises:
        ValueError: If variants or mask sizes are invalid.
        NotImplementedError: If reading from unsupported file formats."""

    def __init__(
        self,
        model_name=None,
        model_kind=None,
        free_par=None,
        los_definition=None,
        covariance_dict=None,
        matrix_form=False,
        variant=None,
        coefficients=None,
        number_densities=None,
        number_velocities=None,
        emulator_flag=False,
    ):
        """
        Initialize the covariance model with specified parameters.

        Args:
            model_name (str, optional): Name of the covariance model to use.
            model_kind (str, optional): Type or kind of the model.
            free_par (Any, optional): Free parameters for the model.
            los_definition (Any, optional): Definition of the line-of-sight.
            covariance_dict (dict, optional): Dictionary containing covariance data.
            matrix_form (bool, optional): If True, use matrix form for computations.
            variant (Any, optional): Variant of the model to use.
            number_densities (Any, optional): Number densities for the model.
            number_velocities (Any, optional): Number velocities for the model.

        Attributes:
            model_name (str): Name of the covariance model.
            model_kind (str): Type or kind of the model.
            free_par (Any): Free parameters for the model.
            los_definition (Any): Definition of the line-of-sight.
            covariance_dict (dict): Dictionary containing covariance data.
            matrix_form (bool): Indicates if matrix form is used.
            variant (Any): Variant of the model.
            coefficients (module): Imported coefficients module for the model.
            compute_covariance_sum (callable): Function to compute covariance sum.
            compute_covariance_sum_jit (callable): JIT-compiled function to compute covariance sum.
        """
        self.model_name = model_name
        self.model_kind = model_kind
        self.free_par = free_par
        self.los_definition = los_definition
        self.covariance_dict = covariance_dict
        self.matrix_form = matrix_form
        self.variant = variant
        self.coefficients = coefficients
        self.compute_covariance_sum = None
        self.compute_covariance_sum_jit = None
        self.number_densities = number_densities
        self.number_velocities = number_velocities
        self.emulator_flag = emulator_flag

        self.init_compute_covariance_sum()

    @classmethod
    def init_from_flip(
        cls,
        model_name,
        model_kind,
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
            - model_kind: kind of data used to generate the covariance matrix (e.g., 'density' or 'velocity')
            - power_spectrum_dict: dictionary containing all information about power spectrum, including k and P(k) values, redshift, etc...
                It is generated by calling getPowerSpectrumDict() in

        Args:
            cls: Indicate that the function is a class method
            model_name: Determine which model to use for the covariance matrix
            model_kind: Determine the kind of model to be used
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

        _available_variants = importlib.import_module(
            f"flip.covariance.{model_name}"
        )._variant
        if variant not in _available_variants:
            raise ValueError(
                f"Variant is not in available variants: {_available_variants}"
            )

        free_par = _read_free_par(model_name, model_kind, variant=variant)

        coefficients = importlib.import_module(
            f"flip.covariance.{model_name}.coefficients"
        )

        (
            covariance_dict,
            number_densities,
            number_velocities,
        ) = generator_flip.generate_covariance(
            model_name,
            model_kind,
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
            model_kind=model_kind,
            free_par=free_par,
            los_definition=los_definition,
            covariance_dict=covariance_dict,
            matrix_form=False,
            variant=variant,
            coefficients=coefficients,
            number_densities=number_densities,
            number_velocities=number_velocities,
        )

    @classmethod
    def init_from_generator(
        cls,
        model_name,
        model_kind,
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
            - model_kind: kind of covariance matrix generated ('density' or 'velocity')
            - power spectrum dictionary containing keys for each redshift bin and values corresponding to
                power spectra at those red

        Args:
            cls: Refer to the class itself
            model_name: Specify the kind of model used to generate the covariance matrix
            model_kind: Determine which model to use
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

        _available_variants = importlib.import_module(
            f"flip.covariance.{model_name}"
        )._variant
        if variant not in _available_variants:
            raise ValueError(
                f"Variant is not in available variants: {_available_variants}"
            )

        free_par = _read_free_par(model_name, model_kind, variant=variant)

        coefficients = importlib.import_module(
            f"flip.covariance.{model_name}.coefficients"
        )

        (
            covariance_dict,
            number_densities,
            number_velocities,
            los_definition,
        ) = generator.generate_covariance(
            model_kind,
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
            model_kind=model_kind,
            free_par=free_par,
            los_definition=los_definition,
            covariance_dict=covariance_dict,
            matrix_form=False,
            variant=variant,
            coefficients=coefficients,
            number_densities=number_densities,
            number_velocities=number_velocities,
        )

    @classmethod
    def init_from_emulator(
        cls,
        emulator_model_name,
        model_kind,
        covariance_list,
        emulator_parameter_values,
        parameter_names,
        **kwargs,
    ):
        begin = time.time()
        from flip.covariance.emulators import generator as generator_emulators

        emulator_covariance_dict = generator_emulators.generate_covariance(
            emulator_model_name,
            model_kind,
            covariance_list,
            emulator_parameter_values,
            parameter_names,
            **kwargs,
        )
        end = time.time()
        log.add(
            f"Covariance matrix generated from emulator with {emulator_model_name} model in {'{:.2e}'.format(end - begin)} seconds"
        )
        return cls(
            model_name=emulator_model_name,
            model_kind=model_kind,
            free_par=covariance_list[0].free_par,
            los_definition=covariance_list[0].los_definition,
            covariance_dict=emulator_covariance_dict,
            matrix_form=False,
            variant=covariance_list[0].variant,
            coefficients=covariance_list[0].coefficients,
            number_densities=covariance_list[0].number_densities,
            number_velocities=covariance_list[0].number_velocities,
            emulator_flag=True,
        )

    @classmethod
    def init_from_file(
        cls,
        filename,
        file_format,
    ):
        if file_format == "parquet":
            raise NotImplementedError("Reading from parquet not implemented yet")
        if file_format == "pickle":
            with open(f"{filename}.pickle", "rb") as file_read:
                class_attrs_dictionary = pickle.load(file_read)
        if file_format == "npz":
            class_attrs_dictionary = dict(np.load(f"{filename}.npz", allow_pickle=True))
            for key in class_attrs_dictionary.keys():
                if class_attrs_dictionary[key].size == 1:
                    class_attrs_dictionary[key] = class_attrs_dictionary[key].item()
                else:
                    class_attrs_dictionary[key] = list(class_attrs_dictionary[key])
        return cls(**class_attrs_dictionary)

    @property
    def kind(self):
        """
        The kind function is used to determine the kind of covariance model that will be computed.
        The options are:
            - velocity: The covariance model is computed for velocity only.
            - density: The covariance model is computed for density only.
            - density_velocity: The covariance model is computed for both velocity and density, without cross-term (i.e., the covariances between velocities and densities are zero). This option should be used when computing a full 3D tomography in which we want to compute a separate 1D tomography along each axis (x, y, z

        Args:
            self: Represent the instance of the class

        Returns:
            The kind of the model

        """
        if self.model_kind == "velocity":
            log.add("The covariance model is computed for velocity")
        elif self.model_kind == "density":
            log.add("The covariance model is computed for density")
        elif self.model_kind == "density_velocity":
            log.add(
                "The covariance model is computed for velocity and density, without cross-term"
            )
        elif self.model_kind == "full":
            log.add(
                "The covariance model is computed for velocity and density, with cross-term"
            )
        return self.model_kind

    @property
    def loaded(self):
        """
        The loaded function checks if the covariance matrix is loaded.

        Args:
            self: Refer to the object itself

        Returns:
            A boolean

        """
        if self.model_kind == "density":
            if "gg" in self.covariance_dict.keys():
                return True
            else:
                return False
        elif self.model_kind == "velocity":
            if "vv" in self.covariance_dict.keys():
                return True
            else:
                return False
        elif self.model_kind == "density_velocity":
            if ("vv" in self.covariance_dict.keys()) & (
                "gg" in self.covariance_dict.keys()
            ):
                return True
            else:
                return False
        elif self.model_kind == "full":
            if (
                ("vv" in self.covariance_dict.keys())
                & ("gg" in self.covariance_dict.keys())
                & ("gv" in self.covariance_dict.keys())
            ):
                return True
            else:
                return False
        else:
            log.add("The model kind was not found")
            return False

    def init_compute_covariance_sum(self):
        if not self.matrix_form and not self.emulator_flag:
            self.compute_matrix_covariance()

        get_coefficients = partial(
            self.coefficients.get_coefficients,
            model_kind=self.model_kind,
            variant=self.variant,
        )

        get_diagonal_coefficients = partial(
            self.coefficients.get_diagonal_coefficients,
            model_kind=self.model_kind,
        )

        compute_covariance_sum_fun = partial(
            compute_covariance_sum,
            covariance_dict=self.covariance_dict,
            kind=self.model_kind,
        )

        def _compute_covariance_sum(
            parameter_values_dict,
            vector_variance,
            covariance_prefactor_dict=None,
        ):
            coefficients_dict = {
                k: jnp.array(v)
                for k, v in get_coefficients(
                    parameter_values_dict=parameter_values_dict,
                    covariance_prefactor_dict=covariance_prefactor_dict,
                ).items()
            }

            coefficients_dict_diagonal = get_diagonal_coefficients(
                parameter_values_dict=parameter_values_dict,
            )

            covariance_sum = compute_covariance_sum_fun(
                coefficients_dict=coefficients_dict,
                coefficients_dict_diagonal=coefficients_dict_diagonal,
                vector_variance=vector_variance,
                parameter_values_dict=(
                    parameter_values_dict if self.emulator_flag else None
                ),
            )

            return covariance_sum

        self.compute_covariance_sum = _compute_covariance_sum

        if jax_installed:
            self.compute_covariance_sum_jit = jit(_compute_covariance_sum)

    def compute_covariance_sum_eigenvalues(
        self,
        parameter_values_dict,
        vector_variance,
        covariance_prefactor_dict=None,
    ):
        covariance_sum = self.compute_covariance_sum(
            parameter_values_dict,
            vector_variance,
            covariance_prefactor_dict=covariance_prefactor_dict,
        )
        return np.linalg.eigvals(covariance_sum)

    # CR - the two next functions should be more general (covariance_type[0] != covariance_type[1] case or not)

    def compute_matrix_covariance(self, verbose=True):
        """
        The compute_matrix_covariance function takes the covariance matrix and fills in all of the missing values.

        Args:
            self: Bind the method to the object

        Returns:
            A dictionary with the full covariance matrices for each redshift bin

        """
        if self.matrix_form:
            if verbose:
                log.add("Matrix covariance already computed")
            return

        for key in ["gg", "vv", "gv"]:
            if key not in self.covariance_dict:
                continue
            if key == "gg":
                number_densities = cov_utils.flatshape_to_fullshape(
                    self.covariance_dict[key].shape[1] - 1
                )
                new_shape = (
                    self.covariance_dict[key].shape[0],
                    number_densities,
                    number_densities,
                )
            elif key == "vv":
                number_velocities = cov_utils.flatshape_to_fullshape(
                    self.covariance_dict[key].shape[1] - 1
                )
                new_shape = (
                    self.covariance_dict[key].shape[0],
                    number_velocities,
                    number_velocities,
                )
            elif key == "gv":
                new_shape = (
                    self.covariance_dict[key].shape[0],
                    number_densities,
                    number_velocities,
                )

            new_cov = np.zeros(new_shape)
            for i, _ in enumerate(self.covariance_dict[key]):
                if key[0] != key[1]:
                    new_cov[i] = cov_utils.return_matrix_covariance_cross(
                        self.covariance_dict[key][i],
                        number_densities,
                        number_velocities,
                    )
                else:
                    new_cov[i] = cov_utils.return_matrix_covariance(
                        self.covariance_dict[key][i]
                    )
            self.covariance_dict[key] = jnp.array(new_cov)

            self.matrix_form = True

    def compute_flat_covariance(self, verbose=True):
        if not self.matrix_form:
            if verbose:
                log.add("Flat covariance already computed")
            return

        for key in self.covariance_dict.keys():
            if key == "gg":
                new_shape = (
                    self.covariance_dict[key].shape[0],
                    int(self.number_densities * (self.number_densities - 1) / 2) + 1,
                )
            elif key == "gv":
                new_shape = (
                    self.covariance_dict[key].shape[0],
                    self.number_densities * self.number_velocities + 1,
                )
            elif key == "vv":
                new_shape = (
                    self.covariance_dict[key].shape[0],
                    int(self.number_velocities * (self.number_velocities - 1) / 2) + 1,
                )
            else:
                log.warning(f"{key} != 'gg', 'gv' or 'vv' was ignored")
                continue

            new_cov = np.zeros(new_shape)
            for i, _ in enumerate(self.covariance_dict[key]):
                if key == "gv":
                    new_cov[i] = cov_utils.return_flat_cross_cov(
                        self.covariance_dict[key][i],
                    )
                else:
                    new_cov[i] = cov_utils.return_flat_covariance(
                        self.covariance_dict[key][i]
                    )
                    self.covariance_dict[key] = new_cov

            self.matrix_form = False

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

    def mask(self, mask_vel=None, mask_dens=None):
        Ng = self.number_densities
        Nv = self.number_velocities

        if mask_vel is None and mask_dens is None:
            raise ValueError("No mask set")

        masked_cov_dic = {}
        if mask_vel is not None:
            if len(mask_vel) != self.number_velocities:
                raise ValueError("Velocities mask size does not match vel cov size")

            if self.matrix_form:
                masked_cov_dic["vv"] = np.array(
                    [
                        cov[np.ix_(mask_vel, mask_vel)]
                        for cov in self.covariance_dict["vv"]
                    ]
                )
            else:
                cov_vv_mask = np.outer(mask_vel, mask_vel)[
                    np.triu_indices(self.number_velocities, k=1)
                ]
                cov_vv_mask = np.insert(cov_vv_mask, 0, True)

                masked_cov_dic["vv"] = np.array(
                    [cov[cov_vv_mask] for cov in self.covariance_dict["vv"]]
                )

        if mask_dens is not None:
            if len(mask_dens) != self.number_densities:
                raise ValueError("Densities mask size does not match density cov size")

            if self.matrix_form:
                masked_cov_dic["gg"] = np.array(
                    [
                        cov[np.ix_(mask_dens, mask_dens)]
                        for cov in self.covariance_dict["gg"]
                    ]
                )
            else:
                cov_gg_mask = np.outer(mask_dens, mask_dens)[
                    np.triu_indices(self.number_densities, k=1)
                ]
                cov_gg_mask = np.insert(cov_gg_mask, 0, True)
                masked_cov_dic["gg"] = np.array(
                    [cov[cov_gg_mask] for cov in self.covariance_dict["gg"]]
                )

        if self.number_densities is not None and self.number_velocities is not None:
            if mask_vel is None:
                mask_vel = np.ones(self.number_velocities, dtype=bool)
            elif mask_dens is None:
                mask_dens = np.ones(self.number_densities, dtype=bool)

            if self.matrix_form:
                masked_cov_dic["gv"] = np.array(
                    [
                        cov[np.ix_(mask_dens, mask_vel)]
                        for cov in self.covariance_dict["gv"]
                    ]
                )
            else:
                cov_gv_mask = np.outer(mask_dens, mask_vel).flatten()
                masked_cov_dic["gv"] = np.array(
                    [cov[cov_gv_mask] for cov in self.covariance_dict["gv"]]
                )

        for k in self.covariance_dict:
            if k not in masked_cov_dic:
                masked_cov_dic[k] = self.covariance_dict[k]

        return CovMatrix(
            model_name=self.model_name,
            model_kind=self.model_kind,
            free_par=self.free_par,
            los_definition=self.los_definition,
            covariance_dict=masked_cov_dic,
            matrix_form=self.matrix_form,
            number_densities=np.sum(mask_dens),
            number_velocities=np.sum(mask_vel),
            variant=self.variant,
        )
