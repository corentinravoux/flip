import importlib

from flip.utils import create_log

from .._config import __use_jax__

if __use_jax__:
    try:
        import jax.numpy as jnp

        jax_installed = True

    except ImportError:
        import numpy as jnp

        jax_installed = False
else:
    import numpy as jnp

    jax_installed = False

log = create_log()


def inverse_covariance_inverse(covariance):
    """Return the explicit inverse of a covariance matrix.

    Args:
        covariance (array-like): Positive-definite covariance matrix of shape `(N, N)`.

    Returns:
        numpy.ndarray: The matrix inverse `covariance^{-1}`.

    Notes:
        For Fisher computations, explicit inversion is acceptable on modest sizes,
        but Cholesky-based solves are usually more stable for ill-conditioned matrices.
    """
    return jnp.linalg.inv(covariance)


def inverse_covariance_cholesky(covariance):
    """Return the explicit inverse of a covariance matrix.

    Args:
        covariance (array-like): Positive-definite covariance matrix of shape `(N, N)`.

    Returns:
        numpy.ndarray: The matrix inverse `covariance^{-1}`.

    Notes:
        For Fisher computations, explicit inversion is acceptable on modest sizes,
        but Cholesky-based solves are usually more stable for ill-conditioned matrices.
    """
    c = jnp.linalg.inv(jnp.linalg.cholesky(covariance))
    inverse = jnp.dot(c.T, c)
    return inverse


def inverse_covariance_cholesky_inverse(covariance):
    """Return the explicit inverse of a covariance matrix.

    Args:
        covariance (array-like): Positive-definite covariance matrix of shape `(N, N)`.

    Returns:
        numpy.ndarray: The matrix inverse `covariance^{-1}`.

    Notes:
        For Fisher computations, explicit inversion is acceptable on modest sizes,
        but Cholesky-based solves are usually more stable for ill-conditioned matrices.
    """
    c = jnp.linalg.inv(jnp.linalg.cholesky(covariance))
    inverse = jnp.dot(c.T, c)
    return inverse


class FisherMatrix:
    """Compute Fisher information matrix from covariance derivatives.

    Builds the Fisher matrix $F_{ij} = \tfrac{1}{2}\,\mathrm{Tr}[C^{-1} \partial_i C\, C^{-1} \partial_j C]$
    using model-specific derivative coefficients and a provided inverse covariance.

    Attributes:
        covariance (CovMatrix): Covariance model describing blocks gg/gv/vv.
        inverse_covariance_sum (numpy.ndarray): Precomputed `C^{-1}` used in traces.
        parameter_values_dict (dict): Parameter values for computing coefficients.
        free_par (list[str]): Combined free parameters from covariance and data.
    """

    _default_fisher_properties = {
        "inversion_method": "inverse",
        "negative_log_likelihood": True,
    }

    def __init__(
        self,
        covariance=None,
        inverse_covariance_sum=None,
        data_free_par=None,
        parameter_values_dict=None,
    ):

        self.covariance = covariance
        self.inverse_covariance_sum = inverse_covariance_sum
        self.parameter_values_dict = parameter_values_dict
        self.free_par = self.covariance.free_par[:]
        if data_free_par is not None:
            self.free_par += data_free_par

    @classmethod
    def init_from_covariance(
        cls,
        covariance,
        data,
        parameter_values_dict,
        fisher_properties={},
        covariance_prefactor_dict=None,
    ):
        """Initialize a FisherMatrix instance from a covariance and data model.

        Ensures the covariance is in matrix form, prepares the `compute_covariance_sum`
        functions, computes the data variance and the full covariance sum, and inverts
        it according to the selected method.

        Args:
            covariance (CovMatrix): Covariance model to use.
            data (object): Data model; called to obtain vector errors given parameters.
            parameter_values_dict (dict): Parameter values for coefficient evaluation.
            fisher_properties (dict, optional): Options including `inversion_method`.
            covariance_prefactor_dict (dict, optional): Prefactors per block (gg/gv/vv).

        Returns:
            FisherMatrix: Ready-to-use Fisher matrix builder instance.
        """
        if covariance.matrix_form is False and covariance.emulator_flag is False:
            covariance.compute_matrix_covariance()
        if (
            covariance.compute_covariance_sum is None
            or covariance.compute_covariance_sum_jit is None
        ):
            covariance.init_compute_covariance_sum()

        fisher_properties = {
            **cls._default_fisher_properties,
            **fisher_properties,
        }

        _, vector_variance = data.give_data_and_variance(
            parameter_values_dict,
        )
        covariance_sum = covariance.compute_covariance_sum(
            parameter_values_dict,
            vector_variance,
            covariance_prefactor_dict=covariance_prefactor_dict,
        )

        inverse_covariance_sum = eval(
            f"inverse_covariance_{fisher_properties['inversion_method']}"
        )(covariance_sum)

        return cls(
            covariance=covariance,
            inverse_covariance_sum=inverse_covariance_sum,
            data_free_par=data.free_par,
            parameter_values_dict=parameter_values_dict,
        )

    def compute_covariance_derivative(
        self,
        partial_coefficients_dict_param,
    ):
        """Assemble covariance derivative matrix for a single parameter.

        Uses model-kind-specific blocks and partial derivative coefficients to form
        $\partial_i C$ for the parameter `i`.

        Args:
            partial_coefficients_dict_param (dict): Coefficients for blocks `gg/gv/vv`.

        Returns:
            numpy.ndarray: Covariance derivative matrix $\partial_i C$.

        Raises:
            ValueError: If the covariance model kind is unsupported.
        """

        if self.covariance.model_kind == "density":
            covariance_derivative_sum = jnp.sum(
                jnp.array(
                    [
                        partial_coefficients_dict_param["gg"][i] * cov
                        for i, cov in enumerate(self.covariance.covariance_dict["gg"])
                    ]
                ),
                axis=0,
            )

        elif self.covariance.model_kind == "velocity":
            covariance_derivative_sum = jnp.sum(
                jnp.array(
                    [
                        partial_coefficients_dict_param["vv"][i] * cov
                        for i, cov in enumerate(self.covariance.covariance_dict["vv"])
                    ]
                ),
                axis=0,
            )

        elif self.covariance.model_kind in ["density_velocity", "full"]:
            number_densities = self.covariance.number_densities
            number_velocities = self.covariance.number_velocities

            if self.covariance.model_kind == "density_velocity":
                covariance_derivative_sum_gv = jnp.zeros(
                    (number_densities, number_velocities)
                )
            elif self.covariance.model_kind == "full":
                covariance_derivative_sum_gv = jnp.sum(
                    [
                        partial_coefficients_dict_param["gv"][i] * cov
                        for i, cov in enumerate(self.covariance.covariance_dict["gv"])
                    ],
                    axis=0,
                )
            covariance_derivative_sum_gg = jnp.sum(
                [
                    partial_coefficients_dict_param["gg"][i] * cov
                    for i, cov in enumerate(self.covariance.covariance_dict["gg"])
                ],
                axis=0,
            )

            covariance_derivative_sum_vv = jnp.sum(
                [
                    partial_coefficients_dict_param["vv"][i] * cov
                    for i, cov in enumerate(self.covariance.covariance_dict["vv"])
                ],
                axis=0,
            )
            covariance_derivative_sum_vg = covariance_derivative_sum_gv.T

            covariance_derivative_sum = jnp.block(
                [
                    [covariance_derivative_sum_gg, covariance_derivative_sum_gv],
                    [covariance_derivative_sum_vg, covariance_derivative_sum_vv],
                ]
            )
        else:
            log.add("Wrong model type in the loaded covariance.")

        return covariance_derivative_sum

    def compute_fisher_matrix(self, covariance_prefactor_dict=None):
        """Compute the Fisher matrix using covariance derivatives.

        Returns:
            tuple[list[str], numpy.ndarray]: Parameter names list and Fisher matrix
            of shape `(n_params, n_params)`.
        """
        coefficients = importlib.import_module(
            f"flip.covariance.analytical.{self.covariance.model_name}.fisher_terms"
        )
        partial_coefficients_dict = coefficients.get_partial_derivative_coefficients(
            self.covariance.model_kind,
            self.parameter_values_dict,
            variant=self.covariance.variant,
            covariance_prefactor_dict=covariance_prefactor_dict,
        )
        parameter_name_list = []
        covariance_derivative_sum_list = []

        for (
            parameter_name,
            partial_coefficients_dict_param,
        ) in partial_coefficients_dict.items():
            parameter_name_list.append(parameter_name)
            covariance_derivative_sum_list.append(
                jnp.dot(
                    self.inverse_covariance_sum,
                    self.compute_covariance_derivative(
                        partial_coefficients_dict_param,
                    ),
                )
            )

        fisher_matrix_size = len(partial_coefficients_dict.keys())
        fisher_matrix = jnp.zeros((fisher_matrix_size, fisher_matrix_size))
        tri_i, tri_j = jnp.triu_indices_from(fisher_matrix)

        for i, j in zip(tri_i, tri_j):
            if jax_installed:
                fisher_matrix = fisher_matrix.at[i, j].set(
                    0.5
                    * jnp.trace(
                        covariance_derivative_sum_list[i]
                        @ covariance_derivative_sum_list[j]
                    )
                )
                fisher_matrix = fisher_matrix.at[j, i].set(fisher_matrix[i, j])
            else:
                fisher_matrix[i][j] = 0.5 * jnp.trace(
                    covariance_derivative_sum_list[i]
                    @ covariance_derivative_sum_list[j]
                )
                fisher_matrix[j, i] = fisher_matrix[i, j]

        return parameter_name_list, fisher_matrix
