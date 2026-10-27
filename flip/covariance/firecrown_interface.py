"""Experimental firecrown interface for flip likelihoods.

Wraps a flip ``BaseLikelihood`` (e.g. ``MultivariateGaussianLikelihood``) as a
``firecrown.likelihood.Likelihood`` so that it can be sampled through the
firecrown connectors (Cobaya, CosmoSIS, NumCosmo).

flip cannot reuse firecrown's ``GaussFamily``: there the covariance is fixed and
the mean is parameter dependent, while flip has a zero mean, a parameter
dependent data vector and a parameter dependent covariance. The wrapper thus
subclasses ``Likelihood`` directly and delegates the evaluation to flip.

Two modes:

- all flip parameters are firecrown sampler parameters (default);
- ``fs8_redshift`` is given: ``fs8`` is no longer sampled but computed from the
  firecrown cosmology, ``fs8 = f(z) sigma_8(z)``, and returned as a derived
  parameter. The covariance shape stays the one used to build the ``CovMatrix``.

Requires firecrown (Python >= 3.12). This module is not imported by
``flip.covariance``.
"""

import numpy as np

from flip.utils import create_log

try:
    from firecrown.likelihood import Likelihood
    from firecrown.modeling_tools import ModelingTools
    from firecrown.updatable import (
        DerivedParameter,
        DerivedParameterCollection,
        register_new_updatable_parameter,
    )
except ImportError as error:
    raise ImportError(
        "firecrown is needed for flip.covariance.firecrown_interface, "
        "see https://github.com/LSSTDESC/firecrown"
    ) from error

log = create_log()

_derived_section = "flip"


def compute_fs8_ccl(ccl_cosmo, redshift):
    r"""Return $f(z)\,\sigma_8(z)$ from a pyccl cosmology.

    Args:
        ccl_cosmo (pyccl.Cosmology): Cosmology prepared by firecrown.
        redshift (float): Effective redshift.

    Returns:
        float: $f\sigma_8$ at `redshift`.
    """
    scale_factor = 1 / (1 + redshift)
    return float(
        ccl_cosmo.growth_rate(scale_factor)
        * ccl_cosmo.sigma8()
        * ccl_cosmo.growth_factor(scale_factor)
    )


class FlipLikelihood(Likelihood):
    """firecrown likelihood delegating to a flip likelihood.

    Attributes:
        flip_likelihood (BaseLikelihood): Wrapped flip likelihood. It must return
            the log-likelihood (``negative_log_likelihood=False``) and carry no
            prior: priors belong to the sampler.
        parameter_names (list[str]): flip parameter order.
        fs8_redshift (float|None): If set, ``fs8`` is derived from the cosmology.
    """

    def __init__(
        self,
        flip_likelihood,
        parameter_defaults,
        fixed_parameters=None,
        fs8_redshift=None,
        parameter_prefix=None,
    ):
        """Register the flip parameters as firecrown parameters.

        Args:
            flip_likelihood (BaseLikelihood): flip likelihood to wrap.
            parameter_defaults (dict): Default value of each sampled parameter.
            fixed_parameters (dict, optional): Parameters kept fixed
                (firecrown ``InternalParameter``).
            fs8_redshift (float, optional): Effective redshift at which ``fs8``
                is computed from the firecrown cosmology.
            parameter_prefix (str, optional): Prefix of the sampler names,
                ``{prefix}_{name}``.
        """
        super().__init__(parameter_prefix=parameter_prefix)
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
        fixed_parameters = {} if fixed_parameters is None else fixed_parameters

        self.flip_likelihood = flip_likelihood
        self.parameter_names = list(flip_likelihood.parameter_names)
        self.fs8_redshift = fs8_redshift
        self.derived_fs8 = None
        if fs8_redshift is not None and "fs8" not in self.parameter_names:
            raise ValueError("fs8_redshift is set but fs8 is not a flip parameter.")

        for name in self.parameter_names:
            if name == "fs8" and fs8_redshift is not None:
                continue
            if name in fixed_parameters:
                parameter = register_new_updatable_parameter(
                    float(fixed_parameters[name]),
                    default_value=float(fixed_parameters[name]),
                )
            else:
                parameter = register_new_updatable_parameter(
                    default_value=float(parameter_defaults[name])
                )
            setattr(self, name, parameter)

    def read(self, sacc_data):
        """Do nothing: the data are held by the flip ``DataVector``.

        Args:
            sacc_data (sacc.Sacc|None): Unused.
        """

    def _reset(self):
        """Forget the derived ``fs8``."""
        self.derived_fs8 = None

    def get_parameter_values(self, tools):
        """Return the flip parameter vector for the current firecrown state.

        Args:
            tools (ModelingTools): Prepared firecrown modeling tools.

        Returns:
            numpy.ndarray: Values ordered as ``parameter_names``.
        """
        if self.fs8_redshift is not None:
            self.derived_fs8 = compute_fs8_ccl(
                tools.get_ccl_cosmology(), self.fs8_redshift
            )
        return np.array(
            [
                (
                    self.derived_fs8
                    if name == "fs8" and self.fs8_redshift is not None
                    else getattr(self, name)
                )
                for name in self.parameter_names
            ]
        )

    def compute_loglike(self, tools):
        """Evaluate the flip log-likelihood.

        Args:
            tools (ModelingTools): Prepared firecrown modeling tools.

        Returns:
            float: Log-likelihood, -inf if flip returns a non-finite value.
        """
        loglike = float(self.flip_likelihood(self.get_parameter_values(tools)))
        if not np.isfinite(loglike):
            return -np.inf
        return loglike

    def _get_derived_parameters(self):
        """Return ``fs8`` as derived parameter when computed from cosmology.

        The section is the parameter prefix (``"flip"`` without prefix), so the
        connectors name it ``{prefix}__fs8``.

        Returns:
            DerivedParameterCollection: Derived parameters.
        """
        if self.derived_fs8 is None:
            return DerivedParameterCollection([])
        section = self.parameter_prefix or _derived_section
        return DerivedParameterCollection(
            [DerivedParameter(section, "fs8", self.derived_fs8)]
        )


def create_firecrown_likelihood(
    flip_likelihood,
    parameter_defaults,
    fixed_parameters=None,
    fs8_redshift=None,
    parameter_prefix=None,
    modeling_tools=None,
):
    """Build the ``(likelihood, tools)`` pair a firecrown factory returns.

    Use it inside a ``build_likelihood(build_parameters)`` function given to a
    firecrown connector.

    Args:
        flip_likelihood (BaseLikelihood): flip likelihood to wrap.
        parameter_defaults (dict): Default value of each sampled parameter.
        fixed_parameters (dict, optional): Parameters kept fixed.
        fs8_redshift (float, optional): Effective redshift of the derived ``fs8``.
        parameter_prefix (str, optional): Prefix of the sampler names.
        modeling_tools (ModelingTools, optional): Defaults to ``ModelingTools()``.

    Returns:
        tuple[FlipLikelihood, ModelingTools]: Likelihood and modeling tools.
    """
    likelihood = FlipLikelihood(
        flip_likelihood,
        parameter_defaults,
        fixed_parameters=fixed_parameters,
        fs8_redshift=fs8_redshift,
        parameter_prefix=parameter_prefix,
    )
    if modeling_tools is None:
        modeling_tools = ModelingTools()
    return likelihood, modeling_tools
