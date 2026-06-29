import abc
from functools import partial

from flip.data_vector.basic import DensMesh
from flip.utils import create_log, prior_sum, return_prior

from .._config import __use_jax__

if __use_jax__:
    try:
        import jax.numpy as jnp
        from jax import grad, jit

        jax_installed = True

    except ImportError:
        import numpy as jnp

        jax_installed = False
else:

    import numpy as jnp

    jax_installed = False


# CR - Very first idea of the likelihood class

log = create_log()


_available_field_expressions = ["interpolation", "remesh"]


def express_field_on_grid(
    scaling_catalog,
    parameter_values_dict,
    fixed_grid,
    method,
    **kwargs,
):
    if method not in _available_field_expressions:
        raise ValueError(
            f"Method {method} not recognized. Available methods: {_available_field_expressions}"
        )

    if method == "interpolation":
        if isinstance(scaling_catalog, DensMesh):
            log.warning(
                "Scaling catalog is a dict. Interpolation method may not be appropriate."
            )

            # CR - not implemented yet - need to add interpolation method to DensMesh class
            scaled_mesh = scaling_catalog.interpolate(
                parameter_values_dict,
                fixed_grid,
            )
        else:
            raise ValueError(
                "Interpolation method only implemented for DensMesh catalogs."
            )
    elif method == "remesh":

        scaling_catalog["rcom_zobs"] = (
            parameter_values_dict["H_0"] / 100
        ) * scaling_catalog["rcom_zobs"]

        rcom_max = kwargs.get("rcom_max", None)
        grid_size = kwargs.get("grid_size", None)
        grid_type = kwargs.get("grid_type", "sphere")
        kind = kwargs.get("kind", "ngc")

        scaled_mesh = DensMesh.init_from_catalog(
            scaling_catalog,
            rcom_max,
            grid_size,
            grid_type,
            kind,
            **kwargs,
        )

    return scaled_mesh


class BaseLikelihood(abc.ABC):

    _default_likelihood_properties = {
        "use_jit": False,
        "use_gradient": False,
    }

    def __init__(
        self,
        parameter_names=None,
        likelihood_properties={},
    ):
        self.parameter_names = parameter_names

        self.likelihood_properties = {
            **self._default_likelihood_properties,
            **likelihood_properties,
        }
        self.prior = self.initialize_prior()

        self.likelihood_call, self.likelihood_grad = self._init_likelihood()

    def __call__(self, parameter_values):
        """Evaluate likelihood at parameter values.

        Args:
            parameter_values (array-like): Parameter vector aligned with `parameter_names`.

        Returns:
            float: Likelihood value, sign controlled by `negative_log_likelihood`.
        """
        return self.likelihood_call(parameter_values)

    @abc.abstractmethod
    def _init_likelihood(self, *args):
        """Initialize likelihood and optional gradient.

        Returns:
            tuple[Callable, Callable|None]: `(likelihood_call, likelihood_grad)`.
        """
        likelihood_fun = None
        likelihood_grad = None
        return likelihood_fun, likelihood_grad

    def initialize_prior(
        self,
    ):
        """Build prior function from likelihood properties.

        Returns:
            Callable: Prior function mapping parameter dict to log-prior.

        Raises:
            ValueError: If an unsupported prior type is requested.
        """
        if "prior" not in self.likelihood_properties.keys():
            return lambda x: 0
        else:
            prior_dict = self.likelihood_properties["prior"]
            priors = []
            for parameter_name, prior_properties in prior_dict.items():
                prior = return_prior(
                    parameter_name,
                    prior_properties,
                )
                priors.append(prior)

            prior_function = partial(prior_sum, priors)
            return prior_function


class GaussianFieldComparisonLikelihood(BaseLikelihood):

    def __init__(
        self,
        basefield=None,
        scaling_catalog=None,
        parameter_names=None,
        covariance_basefield=None,
        likelihood_properties={},
    ):

        super(GaussianFieldComparisonLikelihood, self).__init__(
            parameter_names=parameter_names,
            likelihood_properties=likelihood_properties,
        )
        self.basefield = basefield
        self.scaling_catalog = scaling_catalog
        self.covariance_basefield = covariance_basefield

    def _init_likelihood(self):
        """Build callable likelihood and optional gradient for Gaussian model.

        Returns:
            tuple[Callable, Callable|None]: `(likelihood_call, likelihood_grad)`.
        """

        use_jit = self.likelihood_properties["use_jit"]

        if jax_installed & use_jit:
            prior = jit(self.prior)
        else:
            prior = self.prior

        def likelihood_evaluation(
            parameter_values,
        ):
            """Evaluate likelihood for given parameters.

            Args:
                parameter_values (array-like): Parameter vector aligned to names.

            Returns:
                float: Likelihood value (sign depends on `negative_log_likelihood`).
            """
            parameter_values_dict = dict(zip(self.parameter_names, parameter_values))

            common_grid_x = self.basefield.data["x"]
            common_grid_y = self.basefield.data["y"]
            common_grid_z = self.basefield.data["z"]

            fixed_grid = jnp.stack(
                jnp.meshgrid(
                    common_grid_x, common_grid_y, common_grid_z, indexing="ij"
                ),
                axis=-1,
            )

            scaled_field = express_field_on_grid(
                self.scaling_catalog,
                parameter_values_dict,
                fixed_grid,
                method=self.likelihood_properties["scaling_method"],
                **self.likelihood_properties["scaling_kwargs"],
            )

            likelihood = jnp.sum(
                (scaled_field["density"] - self.basefield["density"]) ** 2
                / (
                    self.scaled_field["density_error"] ** 2
                    + self.covariance_basefield["density_error"] ** 2
                )
            )

            likelihood_value = likelihood + prior(parameter_values_dict)

            if self.likelihood_properties["negative_log_likelihood"]:
                likelihood_value *= -1
            return likelihood_value

        if jax_installed:
            likelihood_grad = grad(likelihood_evaluation)
            if use_jit:
                likelihood_evaluation = jit(likelihood_evaluation)
                likelihood_grad = jit(likelihood_grad)
        else:
            likelihood_grad = None
        return likelihood_evaluation, likelihood_grad
