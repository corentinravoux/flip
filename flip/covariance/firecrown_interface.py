"""Experimental firecrown interface for flip likelihoods.

``FlipLikelihood`` wraps a flip likelihood as a ``firecrown.likelihood.Likelihood``,
so that it can be sampled through the firecrown connectors (Cobaya, CosmoSIS,
NumCosmo). firecrown's ``GaussFamily`` does not fit flip, whose data vector and
covariance both depend on the parameters.

The flip likelihood is built by the user and given to ``FlipLikelihood``, either
directly or as a function ``ccl_cosmology -> flip likelihood`` called again when
the cosmology changes. When the sampler is run from Python, the module is the
firecrown factory (``likelihood_source = flip.covariance.firecrown_interface``):
``build_likelihood`` returns the flip likelihood registered with
``set_flip_likelihood``. From the command line (``cosmosis``, ``cobaya-run``),
the factory is a user file whose ``build_likelihood`` builds the flip likelihood,
for instance from a covariance saved with ``CovMatrix.write``.

Requires firecrown (Python >= 3.12). Not imported by ``flip.covariance``.
"""

import numpy as np

from flip.utils import create_log

try:
    from firecrown.likelihood import Likelihood
    from firecrown.modeling_tools import CCLFactory, ModelingTools
    from firecrown.updatable import register_new_updatable_parameter

    firecrown_installed = True
except ImportError:
    firecrown_installed = False
if not firecrown_installed:
    Likelihood = object

log = create_log()

_FLIP_LIKELIHOODS = {}


def _check_flip_likelihood(flip_likelihood):
    if flip_likelihood.likelihood_properties["negative_log_likelihood"]:
        raise ValueError(
            "The flip likelihood must return the log-likelihood: "
            "set negative_log_likelihood=False."
        )
    if "prior" in flip_likelihood.likelihood_properties:
        log.add(
            "The flip likelihood has a prior: it adds to the sampler prior.",
            level="warning",
        )


class FlipLikelihood(Likelihood):
    """firecrown likelihood delegating to a flip likelihood.

    Attributes:
        flip_likelihood (BaseLikelihood): Current flip likelihood, returning the
            log-likelihood (``negative_log_likelihood=False``).
        parameter_names (list[str]): flip parameter order.
    """

    def __init__(
        self,
        flip_likelihood,
        parameter_names=None,
        fixed_parameters=None,
        parameter_prefix=None,
    ):
        """Register the flip parameters as firecrown parameters (default NaN).

        Args:
            flip_likelihood (BaseLikelihood|Callable): flip likelihood, or a
                function ``ccl_cosmology -> flip likelihood``, called again when
                the cosmology changes.
            parameter_names (list[str], optional): flip parameter order;
                required with a function, else taken from the likelihood.
            fixed_parameters (dict, optional): Parameters kept fixed.
            parameter_prefix (str, optional): Prefix of the sampler names,
                ``{prefix}_{name}``.
        """
        super().__init__(parameter_prefix=parameter_prefix)
        if hasattr(flip_likelihood, "parameter_names"):
            _check_flip_likelihood(flip_likelihood)
            self.build_flip_likelihood = None
            self.flip_likelihood = flip_likelihood
            parameter_names = flip_likelihood.parameter_names
        elif parameter_names is None:
            raise ValueError("parameter_names is needed with a build function")
        else:
            self.build_flip_likelihood = flip_likelihood
            self.flip_likelihood = None
        self.parameter_names = list(parameter_names)
        self._ccl_cosmology = None

        fixed_parameters = fixed_parameters or {}
        for name in self.parameter_names:
            value = fixed_parameters.get(name)
            value = None if value is None else float(value)
            setattr(
                self,
                name,
                register_new_updatable_parameter(
                    value, default_value=np.nan if value is None else value
                ),
            )

    def read(self, sacc_data):
        """Do nothing: the data are held by the flip ``DataVector``."""

    def compute_loglike(self, tools):
        """Evaluate the flip log-likelihood, rebuilding it if the cosmology changed.

        Args:
            tools (ModelingTools): Prepared firecrown modeling tools.

        Returns:
            float: Log-likelihood, -inf if flip returns a non-finite value.
        """
        ccl_cosmology = tools.get_ccl_cosmology()
        # firecrown creates a new cosmology at each prepare(); pyccl compares
        # cosmologies by their parameters.
        if self.build_flip_likelihood is not None and (
            ccl_cosmology != self._ccl_cosmology
        ):
            if self._ccl_cosmology is not None:
                log.add("Cosmology changed: rebuilding the flip likelihood")
            self.flip_likelihood = self.build_flip_likelihood(ccl_cosmology)
            _check_flip_likelihood(self.flip_likelihood)
            self._ccl_cosmology = ccl_cosmology

        parameter_values = [getattr(self, name) for name in self.parameter_names]
        loglike = float(self.flip_likelihood(np.array(parameter_values)))
        return loglike if np.isfinite(loglike) else -np.inf


def set_flip_likelihood(flip_likelihood, name="flip"):
    """Register a flip likelihood for `build_likelihood` (same Python process).

    Args:
        flip_likelihood (BaseLikelihood): flip likelihood returning the
            log-likelihood (``negative_log_likelihood=False``).
        name (str): Build parameter ``flip_likelihood`` selecting it.
    """
    _FLIP_LIKELIHOODS[name] = flip_likelihood


def build_likelihood(build_parameters):
    """firecrown factory: the registered flip likelihood wrapped by `FlipLikelihood`.

    Build parameters:
        flip_likelihood (str): Name given to `set_flip_likelihood`, default ``flip``.
        prefix (str): Sampler parameter prefix, default ``flip``.
        ccl_creation_mode, ccl_transfer_function, mass_split (str): firecrown
            ``CCLFactory`` options, defaults ``default``, ``boltzmann_camb``,
            ``normal``.

    Args:
        build_parameters (firecrown.likelihood.NamedParameters): Parameters of
            the connector configuration.

    Returns:
        tuple[FlipLikelihood, ModelingTools]: Likelihood and modeling tools.
    """
    options = build_parameters.data
    name = options.get("flip_likelihood", "flip")
    if name not in _FLIP_LIKELIHOODS:
        raise KeyError(
            f"No flip likelihood registered as {name!r}: call set_flip_likelihood "
            "in the same Python process, or use your own factory file."
        )
    ccl_factory = CCLFactory(
        creation_mode=options.get("ccl_creation_mode", "default"),
        pure_ccl_transfer_function=options.get(
            "ccl_transfer_function", "boltzmann_camb"
        ),
        mass_split=options.get("mass_split", "normal"),
    )
    likelihood = FlipLikelihood(
        _FLIP_LIKELIHOODS[name], parameter_prefix=options.get("prefix", "flip")
    )
    return likelihood, ModelingTools(ccl_factory=ccl_factory)
