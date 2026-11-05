import importlib
import time
from functools import partial

import numpy as np

from flip._config import __use_jax__
from flip.covariance import cov_utils
from flip.covariance import io as covariance_io
from flip.utils import create_log

if __use_jax__:
    try:
        import jax
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


_available_generation_methods = [
    "inverse",
    "cholesky",
]

# covariance blocks of each model kind; gv comes last (it needs the gg and vv sizes)
_blocks_per_kind = {
    "density": ["gg"],
    "velocity": ["vv"],
    "density_velocity": ["gg", "vv"],
    "full": ["gg", "vv", "gv"],
}


def _read_free_par(
    model_name,
    model_kind,
    variant=None,
):
    """Read free parameter names for a given model kind and variant.

    Args:
        model_name (str): Covariance model package name.
        model_kind (str): Kind string (`density`, `velocity`, `density_velocity`, `full`).
        variant (str, optional): Model variant; defaults to `baseline`.

    Returns:
        list[str]: Unique free parameter names used by the model and variant.
    """
    _free_par = importlib.import_module(
        f"flip.covariance.analytical.{model_name}"
    )._free_par
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


def _add_to_diagonal(matrix, diagonal):
    """Add a scalar or a vector to the diagonal of a square matrix.

    Avoids building a dense N x N identity or diagonal matrix. The NumPy
    branch works in place: only call it on a freshly built matrix.

    Args:
        matrix (array-like): Square matrix.
        diagonal (float or array-like): Value(s) added to the diagonal.

    Returns:
        array-like: Matrix with the updated diagonal.
    """
    if jax_installed:
        index = jnp.arange(matrix.shape[0])
        return matrix.at[index, index].add(diagonal)
    matrix[jnp.diag_indices_from(matrix)] += diagonal
    return matrix


def compute_covariance_sum(
    covariance_dict,
    coefficients_dict,
    coefficients_dict_diagonal,
    vector_variance,
    kind="full",
    parameter_values_dict=None,
):
    """Compose the total covariance matrix from blocks and coefficients.

    Assembles `gg`, `gv`, and `vv` blocks with optional diagonal terms and adds
    data variance. Supports special cases like `density_velocity` (no cross-term).

    Args:
        covariance_dict (dict): Covariance blocks keyed by `gg/gv/vv`.
        coefficients_dict (dict): Coefficient arrays per block.
        coefficients_dict_diagonal (dict): Diagonal noise terms per block.
        vector_variance (array-like): Data variance vector or matrix.
        kind (str): Model kind.
        parameter_values_dict (dict, optional): Values for callable covariances (emulator).

    Returns:
        array-like: Total covariance matrix `C`.
    """
    keys = _blocks_per_kind[kind]

    covariance_sum_ = {}
    number_tracers = []
    for k in keys:
        if parameter_values_dict is not None:
            if k == "gv":
                covariance_evaluated = np.array(
                    [
                        cov(parameter_values_dict, number_tracers[0], number_tracers[1])
                        for i, cov in enumerate(covariance_dict[k])
                    ]
                )
            else:
                covariance_evaluated = np.array(
                    [
                        cov(parameter_values_dict)
                        for i, cov in enumerate(covariance_dict[k])
                    ]
                )
                number_tracers.append(covariance_evaluated.shape[1])

        else:
            covariance_evaluated = covariance_dict[k]

        if coefficients_dict[k].ndim == 1:
            covariance_sum_[k] = jnp.tensordot(
                coefficients_dict[k], covariance_evaluated, axes=1
            )
        else:
            # Per-object coefficients (n_terms, N): scale rows of each term
            coefficient_3d = jnp.atleast_3d(jnp.atleast_2d(coefficients_dict[k].T).T)
            covariance_sum_[k] = jnp.sum(
                coefficient_3d * covariance_evaluated,
                axis=0,
            )

        if k in coefficients_dict_diagonal:
            covariance_sum_[k] = _add_to_diagonal(
                covariance_sum_[k], coefficients_dict_diagonal[k]
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
        covariance_sum = _add_to_diagonal(covariance_sum, vector_variance)
    else:
        covariance_sum += vector_variance
    return covariance_sum


def generate_field_inverse(covariance_sum, number_draws, seed=None):
    """Generate a random field from the covariance using inverse transform sampling.

    Args:
        covariance_sum (array-like): Total covariance matrix.
        number_draws (int): Number of random draws to generate.
        seed (int, optional): Random seed for reproducibility."""
    if seed is not None:
        if jax_installed:
            jax_random_key = jax.random.key(seed)
        else:
            jnp.random.seed(seed)
    else:
        if jax_installed:
            jax_random_key = jax.random.key(int(time.time() * 1e6))

    if jax_installed:
        return jax.random.multivariate_normal(
            jax_random_key,
            jnp.zeros(len(covariance_sum)),
            covariance_sum,
            (number_draws,),
        )
    else:
        return jnp.random.multivariate_normal(
            jnp.zeros(len(covariance_sum)),
            covariance_sum,
            (number_draws,),
        )


def generate_field_cholesky(covariance_sum, number_draws, seed=None):
    """Generate a random field from the covariance using Cholesky decomposition.

    Args:
        covariance_sum (array-like): Total covariance matrix.
        seed (int, optional): Random seed for reproducibility."""

    if seed is not None:
        if jax_installed:
            jax_random_key = jax.random.key(seed)
        else:
            jnp.random.seed(seed)
    else:
        if jax_installed:
            jax_random_key = jax.random.key(int(time.time() * 1e6))

    cholesky_matrix = jnp.linalg.cholesky(covariance_sum)
    if jax_installed:
        random_draws = jax.random.normal(
            jax_random_key,
            (number_draws, len(covariance_sum)),
        )
    else:
        random_draws = jnp.random.standard_normal((number_draws, len(covariance_sum)))

    return jnp.dot(random_draws, cholesky_matrix.T)


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
        covariance_prefactor_dict=None,
    ):
        """Initialize the covariance model.

        Args:
            model_name (str, optional): Name of the covariance model.
            model_kind (str, optional): Kind (`density`, `velocity`, `density_velocity`, `full`).
            free_par (list[str], optional): Free parameters names for the model.
            los_definition (str, optional): Line-of-sight definition.
            covariance_dict (dict, optional): Covariance blocks.
            matrix_form (bool, optional): Whether blocks are in matrix form.
            variant (str, optional): Model variant.
            coefficients (module, optional): Coefficients provider module.
            number_densities (int, optional): Density block size.
            number_velocities (int, optional): Velocity block size.
            emulator_flag (bool, optional): Whether covariances are emulator callables.
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
        self.covariance_prefactor_dict = covariance_prefactor_dict

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
        covariance_prefactor_dict=None,
        **kwargs,
    ):
        """Initialize covariance from flip code generator.

        Args:
            model_name (str): Covariance model package.
            model_kind (str): Kind (`density`, `velocity`, `density_velocity`, `full`).
            power_spectrum_dict (dict): Power spectrum inputs for the model.
            coordinates_density (array-like, optional): Density coordinates.
            coordinates_velocity (array-like, optional): Velocity coordinates.
            additional_parameters_values (tuple, optional): Extra parameters for generator.
            los_definition (str): LOS choice; defaults to `bisector`.
            variant (str): Model variant.
            **kwargs: Extra generator options.

        Returns:
            CovMatrix: Initialized covariance matrix in matrix form.
        """
        begin = time.time()
        from flip.covariance import generator as generator_flip

        _available_variants = importlib.import_module(
            f"flip.covariance.analytical.{model_name}"
        )._variant
        if variant not in _available_variants:
            raise ValueError(
                f"Variant is not in available variants: {_available_variants}"
            )

        free_par = _read_free_par(model_name, model_kind, variant=variant)

        coefficients = importlib.import_module(
            f"flip.covariance.analytical.{model_name}.coefficients"
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
            covariance_prefactor_dict=covariance_prefactor_dict,
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
        covariance_prefactor_dict=None,
        **kwargs,
    ):
        """Initialize covariance from a model-local generator implementation.

        Args:
            model_name (str): Covariance model package.
            model_kind (str): Kind (`density`, `velocity`, `density_velocity`, `full`).
            power_spectrum_dict (dict): Power spectrum inputs.
            coordinates_velocity (array-like, optional): Velocity coordinates.
            coordinates_density (array-like, optional): Density coordinates.
            additional_parameters_values (tuple, optional): Extra generator params.
            variant (str): Model variant.
            **kwargs: Extra generator options.

        Returns:
            CovMatrix: Initialized covariance matrix in matrix form.
        """
        begin = time.time()
        generator = importlib.import_module(
            f"flip.covariance.analytical.{model_name}.generator"
        )

        _available_variants = importlib.import_module(
            f"flip.covariance.analytical.{model_name}"
        )._variant
        if variant not in _available_variants:
            raise ValueError(
                f"Variant is not in available variants: {_available_variants}"
            )

        free_par = _read_free_par(model_name, model_kind, variant=variant)

        coefficients = importlib.import_module(
            f"flip.covariance.analytical.{model_name}.coefficients"
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
            covariance_prefactor_dict=covariance_prefactor_dict,
        )

    @classmethod
    def init_from_emulator(
        cls,
        emulator_model_name,
        model_kind,
        covariance_list,
        emulator_parameter_values,
        parameter_names,
        covariance_prefactor_dict=None,
        **kwargs,
    ):
        """Initialize covariance from an emulator over precomputed covariances.

        Args:
            emulator_model_name (str): Emulator model name.
            model_kind (str): Kind (`density`, `velocity`, `density_velocity`, `full`).
            covariance_list (list[CovMatrix]): Base covariances forming the grid.
            emulator_parameter_values (array-like): Emulator parameter vector.
            parameter_names (list[str]): Names aligned with emulator parameters.
            **kwargs: Extra emulator options.

        Returns:
            CovMatrix: Covariance matrix with emulator-backed blocks.
        """
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
            covariance_prefactor_dict=covariance_prefactor_dict,
        )

    @classmethod
    def init_from_file(
        cls,
        filename,
        file_format,
    ):
        """Load a CovMatrix written by `write` (see `flip.covariance.io`).

        The coefficients module is re-imported from `model_name` and the
        covariance-sum functions are rebuilt, as for any new CovMatrix. A
        covariance written in flat form is therefore returned in matrix form.

        Args:
            filename (str): Path without extension.
            file_format (str): One of `pickle`, `npz`, `parquet`.

        Returns:
            CovMatrix: Loaded covariance matrix.

        Raises:
            ValueError: For an unknown format or a file not written by flip.
        """
        init_arguments = covariance_io.read(filename, file_format)
        init_arguments["covariance_dict"] = {
            key: jnp.asarray(value)
            for key, value in init_arguments["covariance_dict"].items()
        }
        return cls(**init_arguments)

    @property
    def kind(self):
        """Return and log the covariance model kind.

        Returns:
            str: One of `velocity`, `density`, `density_velocity`, `full`.
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
        """Check that required covariance blocks are present for the kind.

        Returns:
            bool: True if the covariance has necessary blocks, False otherwise.
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
        """Prepare functions to compute covariance sums.

        Ensures matrix form, binds coefficient accessors, and defines (optionally
        JIT-compiled) functions combining blocks and data variance.
        """
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
        ):
            """Compute total covariance given parameters and data variance.

            Args:
                parameter_values_dict (dict): Parameter values for coefficients.
                vector_variance (array-like): Data variance vector or matrix.

            Returns:
                array-like: Total covariance matrix.
            """
            coefficients_dict = {
                k: jnp.array(v)
                for k, v in get_coefficients(
                    parameter_values_dict=parameter_values_dict,
                    covariance_prefactor_dict=self.covariance_prefactor_dict,
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
    ):
        """Return eigenvalues of the covariance sum for diagnostics.

        Args:
            parameter_values_dict (dict): Parameter values.
            vector_variance (array-like): Data variance.

        Returns:
            numpy.ndarray: Eigenvalues of `C`.
        """
        covariance_sum = self.compute_covariance_sum(
            parameter_values_dict,
            vector_variance,
        )
        return np.linalg.eigvalsh(covariance_sum)

    def generate_field_from_covariance(
        self,
        parameter_values_dict,
        vector_variance,
        number_draws=1,
        generation_method="cholesky",
        seed=None,
    ):

        covariance_sum = self.compute_covariance_sum(
            parameter_values_dict,
            vector_variance,
        )

        if generation_method == "inverse":
            return generate_field_inverse(
                covariance_sum,
                number_draws,
                seed=seed,
            )

        elif generation_method == "cholesky":
            return generate_field_cholesky(
                covariance_sum,
                number_draws,
                seed=seed,
            )
        else:
            raise ValueError(
                f"Generation method {generation_method} not recognized. "
                f"Choose from {_available_generation_methods}"
            )

    # CR - the two next functions should be more general (covariance_type[0] != covariance_type[1] case or not)

    def compute_matrix_covariance(self, verbose=True):
        """Convert flat covariance vectors to full matrix blocks.

        For each block (`gg`, `gv`, `vv`), reconstruct the 2D matrices from their
        flattened representation and set `matrix_form=True`.
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
        """Convert full matrix blocks back to flat vector representation.

        Updates `covariance_dict` blocks and sets `matrix_form=False`.
        """
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
                    self.number_densities * self.number_velocities,
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
            self.covariance_dict[key] = jnp.array(new_cov)

            self.matrix_form = False

    def write(
        self,
        filename,
        file_format,
    ):
        """Write the covariance matrix to disk (see `flip.covariance.io`).

        Saves the model attributes, the covariance blocks (in their current
        flat or matrix form) and `covariance_prefactor_dict` as plain arrays.
        The file only appears once the write has succeeded.

        Args:
            filename (str): Output path without extension.
            file_format (str): One of `pickle`, `npz`, `parquet`.

        Raises:
            ValueError: For an unknown format.
            NotImplementedError: For emulator covariances (callable blocks).
            TypeError: If a block or prefactor is not a numeric array.
        """
        covariance_io.write(self, filename, file_format)

    def mask(self, mask_vel=None, mask_dens=None):
        """Return a masked copy of the covariance restricting indices.

        Args:
            mask_vel (array-like, optional): Boolean mask for velocity indices.
            mask_dens (array-like, optional): Boolean mask for density indices.

        Returns:
            CovMatrix: Masked covariance object.

        Raises:
            ValueError: If no mask is provided or sizes mismatch.
        """
        if mask_vel is None and mask_dens is None:
            raise ValueError("No mask set")

        masked_cov_dic = {}
        if mask_vel is not None:
            if len(mask_vel) != self.number_velocities:
                raise ValueError("Velocities mask size does not match vel cov size")

            if self.matrix_form:
                index_vel = jnp.flatnonzero(mask_vel)
                masked_cov_dic["vv"] = self.covariance_dict["vv"][
                    :, index_vel[:, None], index_vel[None, :]
                ]
            else:
                cov_vv_mask = np.outer(mask_vel, mask_vel)[
                    np.triu_indices(self.number_velocities, k=1)
                ]
                cov_vv_mask = np.insert(cov_vv_mask, 0, True)

                masked_cov_dic["vv"] = self.covariance_dict["vv"][:, cov_vv_mask]

        if mask_dens is not None:
            if len(mask_dens) != self.number_densities:
                raise ValueError("Densities mask size does not match density cov size")

            if self.matrix_form:
                index_dens = jnp.flatnonzero(mask_dens)
                masked_cov_dic["gg"] = self.covariance_dict["gg"][
                    :, index_dens[:, None], index_dens[None, :]
                ]
            else:
                cov_gg_mask = np.outer(mask_dens, mask_dens)[
                    np.triu_indices(self.number_densities, k=1)
                ]
                cov_gg_mask = np.insert(cov_gg_mask, 0, True)
                masked_cov_dic["gg"] = self.covariance_dict["gg"][:, cov_gg_mask]

        if (
            self.number_densities is not None
            and self.number_velocities is not None
            and self.model_kind == "full"
        ):
            if mask_vel is None:
                mask_vel = np.ones(self.number_velocities, dtype=bool)
            elif mask_dens is None:
                mask_dens = np.ones(self.number_densities, dtype=bool)

            if self.matrix_form:

                masked_cov_dic["gv"] = self.covariance_dict["gv"][
                    :,
                    jnp.flatnonzero(mask_dens)[:, None],
                    jnp.flatnonzero(mask_vel)[None, :],
                ]
            else:
                cov_gv_mask = np.outer(mask_dens, mask_vel).flatten()
                masked_cov_dic["gv"] = self.covariance_dict["gv"][:, cov_gv_mask]

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
            coefficients=self.coefficients,
            emulator_flag=self.emulator_flag,
            covariance_prefactor_dict=self.covariance_prefactor_dict,
        )


class ProjectedCovariance:
    r"""Signal terms of a `CovMatrix` projected on a fixed basis.

    flip assembles $C(\theta) = \sum_n a_n(\theta) C_n + D(\theta)$, with $D$ the
    diagonal of data variance and diagonal coefficients, and factorises it densely at
    every call. Here the signal terms are projected once on a basis chosen at fiducial
    parameters, so that every call only involves $K \times K$ matrices:

    - whitened fiducial signal $D_0^{-1/2} S_0 D_0^{-1/2} = V \Lambda V^T$; the modes with
      signal-to-noise $\lambda_k > s$ are kept,
    - $\Phi = D_0^{1/2} V_K$ and $\Psi = D_0^{-1/2} V_K$ (so $\Psi^T \Phi = I$),
    - $W_n = \Psi^T C_n \Psi$ and the diagonal remainders
      $r_n = \mathrm{diag}(C_n - \Phi W_n \Phi^T)$, precomputed.

    Per call, $C(\theta) \simeq \hat D + \Phi W \Phi^T$ with $W = \sum_n a_n W_n$ and
    $\hat D = D + \max(\sum_n a_n r_n, 0)$, evaluated with the Woodbury identity in a
    form that never inverts $W$ (see
    `flip.covariance.likelihood.log_likelihood_gaussian_woodbury`).

    The projection is a local approximation: it is accurate for coefficients close to
    the fiducial ones, not for any linear combination of the terms. Supported: scalar
    coefficients, per-object coefficients $g g^T \circ M(\theta_0)$ (velocity models),
    a diagonal or a fixed 2D data variance, exact densities (`exact_density`), and a
    shared basis along an interpolation grid (`basis`). The derivation is in the note
    ``woodbury_maths.pdf`` (flip-woodbury feasibility report, 2026-11-03).

    Attributes:
        covariance (CovMatrix): Source covariance, in matrix form.
        blocks (list[str]): Covariance blocks of the model kind.
        terms (list[tuple[str, int]]): `(block, index)` of every term, in coefficient order.
        number_densities (int): Number of density entries.
        number_velocities (int): Number of velocity entries.
        eigenvalues (numpy.ndarray): Whitened S/N spectrum of the basis signal, decreasing.
        k (int): Number of kept modes.
        phi (array-like): Basis $\Phi$, shape `(N, K)`.
        w_terms (array-like): Projected terms $W_n$, shape `(n_terms, K, K)`.
        remainder_terms (array-like): Diagonal remainders $r_n$, shape `(n_terms, N)`.
        remainder (bool): Whether the diagonal remainder correction is applied.
        rotation (array-like or None): Rotation $Q$ of a fixed 2D data variance.
        rotated_variance (array-like or None): Its eigenvalues $\lambda$.
    """

    def __init__(
        self,
        covariance,
        parameter_values_fiducial,
        vector_variance_fiducial,
        snr_min=0.01,
        k=None,
        remainder=True,
        exact_density=False,
        basis=None,
    ):
        """Build the basis and project every term.

        Args:
            covariance (CovMatrix): Covariance with analytical terms (no emulator).
            parameter_values_fiducial (dict or list[dict]): Fiducial parameters. A list
                builds the basis from several design points: the whitening noise is
                the smallest diagonal noise over the points and the basis signal is
                the sum of their signals.
            vector_variance_fiducial (array-like or list): Data variance at the
                fiducial point(s): shape `(N,)`, or a fixed `(N, N)` matrix that does
                not depend on the parameters. One per design point if a list.
            snr_min (float): Keep the whitened modes with S/N above this threshold.
            k (int, optional): Number of kept modes; overrides `snr_min`.
            remainder (bool): Apply the diagonal remainder correction.
            exact_density (bool): Keep every density entry exactly and truncate only
                the velocity block (hybrid: density cells are signal dominated and do
                not compress). `snr_min`, `k` and `eigenvalues` then refer to the
                velocity modes only.
            basis (ProjectedCovariance, optional): Reuse the basis of another projection
                instead of building one (`snr_min`, `k` and `exact_density` are then
                ignored). Used along an interpolation grid.

        Raises:
            NotImplementedError: For emulated covariances, a 2D data variance coupling
                densities and velocities, or per-object (N x N) coefficients that are
                not of the form handled by `_init_matrix_coefficients`.
        """
        if covariance.emulator_flag:
            raise NotImplementedError("Projection of emulated covariances is not supported")
        if not covariance.matrix_form:
            covariance.compute_matrix_covariance()

        self.covariance = covariance
        self.blocks = _blocks_per_kind[covariance.model_kind]
        self.remainder = remainder
        self.number_densities = covariance.number_densities if "gg" in self.blocks else 0
        self.number_velocities = covariance.number_velocities if "vv" in self.blocks else 0
        self.terms = [
            (block, i)
            for block in self.blocks
            for i in range(np.shape(covariance.covariance_dict[block])[0])
        ]
        # (rows, columns) of each block in the full matrix
        density, velocity = slice(0, self.number_densities), slice(self.number_densities, self.size)
        self._block_slices = {"gg": (density, density), "vv": (velocity, velocity), "gv": (density, velocity)}

        if isinstance(parameter_values_fiducial, dict):
            parameter_values_fiducial = [parameter_values_fiducial]
            vector_variance_fiducial = [vector_variance_fiducial]
        vector_variance_fiducial = [np.asarray(v, dtype=float) for v in vector_variance_fiducial]

        # Per-object (N x N) coefficients: M_n(theta_0) is folded into the terms
        self._matrix_reference, self._scale_reference = None, None
        reference = self._coefficients_dict(parameter_values_fiducial[0])
        if any(reference[block].ndim > 1 for block in self.blocks):
            if vector_variance_fiducial[0].ndim == 2:
                raise NotImplementedError(
                    "Per-object coefficients with a 2D data variance are not supported"
                )
            self._init_matrix_coefficients(reference, parameter_values_fiducial[0])

        if basis is None:
            self._build_basis(
                parameter_values_fiducial, vector_variance_fiducial, snr_min, k, exact_density
            )
        else:
            # share the basis and frame of another projection (interpolation grid)
            for name in ("rotation", "rotated_variance", "eigenvalues", "k", "phi", "_whitening"):
                setattr(self, name, getattr(basis, name))
        self._project_terms()

    def _build_basis(
        self, parameter_values_fiducial, vector_variance_fiducial, snr_min, k, exact_density
    ):
        """Set `phi`, `k`, `eigenvalues`, the whitening and, for a 2D variance, the rotation."""
        # A fixed 2D data variance V = Q diag(lambda) Q^T: the projection is built in the
        # Q frame, where the noise lambda + diagonal coefficients is diagonal again
        self.rotation, self.rotated_variance = None, None
        if vector_variance_fiducial[0].ndim == 2:
            self.rotation, self.rotated_variance = self._diagonalise(vector_variance_fiducial[0])
            vector_variance_fiducial = [self.rotated_variance] * len(parameter_values_fiducial)

        # Whitening noise: the smallest diagonal noise over the design points
        noise = [
            vector_variance + np.asarray(self.diagonal(parameter_values_dict))
            for parameter_values_dict, vector_variance in zip(
                parameter_values_fiducial, vector_variance_fiducial
            )
        ]
        self._whitening = 1.0 / np.sqrt(np.min(noise, axis=0))

        # Basis: top whitened eigenmodes of the summed fiducial signal
        signal = np.zeros((self.size, self.size))
        for parameter_values_dict in parameter_values_fiducial:
            signal += np.asarray(self._signal(self._coefficients_dict(parameter_values_dict)))
        if self.rotation is not None:
            signal = self.rotation.T @ signal @ self.rotation
        signal *= self._whitening[:, None]
        signal *= self._whitening[None, :]
        if exact_density:
            if "gg" not in self.blocks or "vv" not in self.blocks:
                raise ValueError("exact_density needs a model with densities and velocities")
            velocity = self._block_slices["vv"][0]
            signal = signal[velocity, velocity]
        eigenvalues, eigenvectors = np.linalg.eigh(signal)
        del signal
        self.eigenvalues = eigenvalues[::-1]

        number_modes = int(np.sum(self.eigenvalues > snr_min)) if k is None else int(k)
        number_modes = max(number_modes, 1)
        if k is None and number_modes > self.eigenvalues.size / 10:
            log.add(
                f"Projected covariance keeps {number_modes} of {self.eigenvalues.size} modes: "
                "the data are signal dominated and the projection gains little."
            )
        basis = eigenvectors[:, ::-1][:, :number_modes]
        if exact_density:
            # [density unit vectors, velocity modes], in the whitened frame
            basis = np.block(
                [
                    [np.eye(self.number_densities), np.zeros((self.number_densities, number_modes))],
                    [np.zeros((self.number_velocities, self.number_densities)), basis],
                ]
            )
        self.k = basis.shape[1]
        self.phi = jnp.array(basis / self._whitening[:, None])
        if self.rotation is not None:
            self.rotation = jnp.array(self.rotation)
            self.rotated_variance = jnp.array(self.rotated_variance)

    def _project_terms(self):
        """Set the projected terms $W_n = \\Psi^T C_n \\Psi$ and their diagonal remainders."""
        phi = np.asarray(self.phi)
        psi = phi * self._whitening[:, None] ** 2
        rotation = None if self.rotation is None else np.asarray(self.rotation)
        # Psi in the original frame, to project the terms as stored
        psi_terms = psi if rotation is None else rotation @ psi

        w_terms, remainder_terms = [], []
        for block, i in self.terms:
            term = np.asarray(self.covariance.covariance_dict[block][i])
            if self._matrix_reference is not None:
                term = term * self._matrix_reference[i]
            rows, columns = self._block_slices[block]
            w_n = psi_terms[rows].T @ term @ psi_terms[columns]
            # gv enters as [[0, B], [B^T, 0]]: both off-diagonal blocks project
            w_n = w_n + w_n.T if block == "gv" else 0.5 * (w_n + w_n.T)
            w_terms.append(w_n)
            if self.remainder:
                diagonal_n = np.zeros(self.size)
                if block != "gv" and rotation is None:
                    diagonal_n[rows] = np.diag(term)
                elif block != "gv":
                    # diagonal of Q^T C_n Q on the block (Q is block diagonal)
                    rotation_block = rotation[rows, rows]
                    diagonal_n[rows] = np.sum(rotation_block * (term @ rotation_block), axis=0)
                remainder_terms.append(diagonal_n - np.sum((phi @ w_n) * phi, axis=1))

        self.w_terms = jnp.array(np.array(w_terms))
        self.remainder_terms = jnp.array(np.array(remainder_terms)) if self.remainder else None

    @property
    def size(self):
        """int: Size `N` of the data vector."""
        return self.number_densities + self.number_velocities

    def _diagonalise(self, vector_variance):
        """Eigendecomposition of a 2D data variance, block by block.

        Returns:
            tuple: Block-diagonal rotation `Q` of shape `(N, N)` and the eigenvalues
            `lambda`, with `vector_variance = Q diag(lambda) Q^T`.

        Raises:
            NotImplementedError: If the variance couples densities and velocities.
        """
        density, velocity = self._block_slices["gv"]
        if np.any(vector_variance[density, velocity]):
            raise NotImplementedError(
                "A 2D data variance coupling densities and velocities is not supported"
            )
        rotation = np.zeros_like(vector_variance)
        eigenvalues = np.zeros(self.size)
        for rows in (density, velocity):
            if rows.stop > rows.start:
                eigenvalues[rows], rotation[rows, rows] = np.linalg.eigh(
                    vector_variance[rows, rows]
                )
        return rotation, eigenvalues

    def to_frame(self, vector):
        """Data vector in the frame of the projection (rotated for a 2D data variance)."""
        return vector if self.rotation is None else self.rotation.T @ vector

    def _init_matrix_coefficients(self, reference, parameter_values_dict):
        """Per-object coefficients of the form $M_n(\\theta) = g g^T \\circ M_n(\\theta_0)$.

        $M_n(\\theta_0)$ is folded into the terms. If the coefficients depend on the
        parameters, $g(\\theta)$ (see `scale`) rescales the data at each call, which is
        exact: $C = \\mathrm{diag}(g)\\,(T + D/g^2)\\,\\mathrm{diag}(g)$. Velocity models only
        (rcrk24, genericzdep).

        Raises:
            NotImplementedError: For another model kind or another dependence.
        """
        if self.blocks != ["vv"]:
            raise NotImplementedError(
                "Per-object coefficients are supported for velocity models only"
            )
        self._matrix_reference = np.asarray(reference["vv"])
        perturbed = {k: 1.01 * v for k, v in parameter_values_dict.items()}
        coefficients = np.asarray(self._coefficients_dict(perturbed)["vv"])
        if np.allclose(coefficients, self._matrix_reference):
            return
        self._scale_reference = np.sqrt(np.diag(self._matrix_reference[0]))
        scale = np.sqrt(np.diag(coefficients[0])) / self._scale_reference
        if not np.allclose(coefficients, np.outer(scale, scale) * self._matrix_reference):
            raise NotImplementedError(
                "Per-object coefficients must be of the form g g^T * M(theta_0)"
            )

    def scale(self, parameter_values_dict):
        """Per-object scale $g(\\theta)$, or None when the coefficients are scalars or fixed.

        Args:
            parameter_values_dict (dict): Parameter values.

        Returns:
            array-like or None: $g = \\sqrt{\\mathrm{diag}\\,M_1(\\theta) / \\mathrm{diag}\\,M_1(\\theta_0)}$, shape `(N,)`.
        """
        if self._scale_reference is None:
            return None
        coefficient = self._coefficients_dict(parameter_values_dict)["vv"][0]
        return jnp.sqrt(jnp.diag(coefficient)) / self._scale_reference

    def _coefficients_dict(self, parameter_values_dict):
        """Coefficients per block, as returned by the model."""
        return {
            block: jnp.array(coefficient)
            for block, coefficient in self.covariance.coefficients.get_coefficients(
                parameter_values_dict,
                self.covariance.model_kind,
                variant=self.covariance.variant,
                covariance_prefactor_dict=self.covariance.covariance_prefactor_dict,
            ).items()
        }

    def _signal(self, coefficients_dict):
        """Dense signal $\\sum_n a_n C_n$ without noise, used once to build the basis."""
        return compute_covariance_sum(
            self.covariance.covariance_dict,
            coefficients_dict,
            {},
            np.zeros(self.size),
            kind=self.covariance.model_kind,
        )

    def coefficients(self, parameter_values_dict):
        """Scalar coefficients $a_n(\\theta)$ in the order of `terms`.

        Args:
            parameter_values_dict (dict): Parameter values.

        Returns:
            array-like: Coefficients, shape `(n_terms,)`. Ones for per-object
            coefficients, which are folded into the terms (see `scale`).
        """
        if self._matrix_reference is not None:
            return jnp.ones(len(self.terms))
        coefficients_dict = self.covariance.coefficients.get_coefficients(
            parameter_values_dict,
            self.covariance.model_kind,
            variant=self.covariance.variant,
            covariance_prefactor_dict=self.covariance.covariance_prefactor_dict,
        )
        return jnp.concatenate(
            [jnp.atleast_1d(jnp.asarray(coefficients_dict[block])) for block in self.blocks]
        )

    def diagonal(self, parameter_values_dict):
        """Diagonal coefficients (e.g. $\\sigma_v^2$ on vv) as a vector of size `N`.

        Args:
            parameter_values_dict (dict): Parameter values.

        Returns:
            array-like: Diagonal noise from the model, shape `(N,)`.
        """
        diagonal_dict = self.covariance.coefficients.get_diagonal_coefficients(
            parameter_values_dict, self.covariance.model_kind
        )
        return jnp.concatenate(
            [
                diagonal_dict.get("gg", 0.0) * jnp.ones(self.number_densities),
                diagonal_dict.get("vv", 0.0) * jnp.ones(self.number_velocities),
            ]
        )

    def compute_projected_covariance(self, parameter_values_dict, vector_variance):
        """Projected covariance at given parameters: $C \\simeq \\hat D + \\Phi W \\Phi^T$.

        Counterpart of `compute_covariance_sum` for the dense path: it returns the
        covariance, the likelihood is evaluated in `flip.covariance.likelihood`.

        Args:
            parameter_values_dict (dict): Parameter values.
            vector_variance (array-like): Diagonal data variance, shape `(N,)`. Not
                used for a fixed 2D data variance, which is stored at construction.

        Returns:
            tuple: `(w, noise, scale)`, with $W = \\sum_n a_n W_n$ of shape `(K, K)`,
            the diagonal noise $\\hat D$ (data variance, diagonal coefficients and, if
            `remainder`, the clipped diagonal remainder) of shape `(N,)`, and the
            per-object scale $g$ (None for scalar coefficients): in the frame of the
            projection, $C = \\mathrm{diag}(g)\\,(\\hat D + \\Phi W \\Phi^T)\\,\\mathrm{diag}(g)$.
        """
        if self.rotation is not None:
            # fixed 2D data variance, diagonal in the frame of the projection
            vector_variance = self.rotated_variance
        coefficients = self.coefficients(parameter_values_dict)
        w = jnp.tensordot(coefficients, self.w_terms, axes=1)
        noise = vector_variance + self.diagonal(parameter_values_dict)
        scale = self.scale(parameter_values_dict)
        if scale is not None:
            noise = noise / scale**2
        if self.remainder:
            noise = noise + jnp.clip(coefficients @ self.remainder_terms, 0.0, None)
        return w, noise, scale
