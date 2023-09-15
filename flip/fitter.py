import numpy as np
import emcee
import iminuit
import flip.likelihood as flik
from flip.covariance.covariance import CovMatrix
from flip.utils import create_log

log = create_log()


class BaseFitter(object):
    def __init__(
        self,
        covariance=None,
        data=None,
    ):
        self.covariance = covariance
        self.data = data

    @classmethod
    def init_from_cov(
        cls,
        covariance,
        data,
        parameter_dict,
        likelihood_type=None,
    ):
        log.add("Method to override, no initialization will be done here")
        return None

    @classmethod
    def init_from_file(
        cls,
        model_name,
        model_type,
        filename,
        data,
        parameter_dict,
        likelihood_type="multivariate_gaussian",
    ):
        covariance = CovMatrix.init_from_file(model_name, model_type, filename)

        fitter = cls.init_from_cov(
            covariance,
            data,
            parameter_dict,
            likelihood_type=likelihood_type,
        )
        return fitter

    def get_likelihood(
        self,
        parameter_dict,
        likelihood_type="multivariate_gaussian",
    ):
        if "density" in self.data.keys():
            density = self.data["density"]
            density_err = self.data["density_err"]
        else:
            density = None
            density_err = None
        if "velocity" in self.data.keys():
            velocity = self.data["velocity"]
            velocity_err = self.data["velocity_err"]
        else:
            velocity = None
            velocity_err = None

        parameter_names = [parameters for parameters in parameter_dict]

        likelihood_class = self.select_likelihood(likelihood_type)

        likelihood = likelihood_class.init_from_covariance(
            self.covariance,
            parameter_names,
            density=density,
            density_err=density_err,
            velocity=velocity,
            velocity_err=velocity_err,
        )

        return likelihood

    def select_likelihood(likelihood_type):
        if likelihood_type == "multivariate_gaussian":
            likelihood_class = flik.MultivariateGaussianLikelihood
        return likelihood_class


class FitMinuit(BaseFitter):
    """Class to maximize likelihood with Minuit."""

    def __init__(
        self,
        covariance=None,
        data=None,
        minuit=None,
    ):
        super(FitMinuit, self).__init__(
            covariance=covariance,
            data=data,
        )
        self.minuit = minuit

    @classmethod
    def init_from_cov(
        cls,
        covariance,
        data,
        parameter_dict,
        likelihood_type="multivariate_gaussian",
    ):
        minuit_fitter = cls(
            covariance=covariance,
            data=data,
        )
        likelihood = minuit_fitter.get_likelihood(
            parameter_dict,
            likelihood_type=likelihood_type,
        )

        minuit_fitter.minuit = iminuit.Minuit(
            likelihood, name=likelihood.parameter_names
        )

        minuit_fitter.setup_minuit(parameter_dict)

        return minuit_fitter

    def setup_minuit(self, parameter_dict):
        self.minuit.errordef = 1.0
        for parameters in parameter_dict:
            d = parameter_dict[parameters]
            self.minuit.errors[parameters] = d["error"] if "error" in d else d["value"]
            self.minuit.errors[parameters] = (
                0 if "fixed" in d and d["fixed"] else self.minuit.errors[parameters]
            )
            self.minuit.fixed[parameters] = d["fixed"] if "fixed" in d else False
            limit_low = d["limit_low"] if "limit_low" in d else None
            limit_up = d["limit_up"] if "limit_up" in d else None
            self.minuit.limits[parameters] = (limit_low, limit_up)

    def run(
        self,
        migrad=True,
        hesse=False,
        minos=False,
    ):
        if migrad:
            log.add(self.minuit.migrad())
        if hesse:
            log.add(self.minuit.hesse())
        if minos:
            log.add(self.minuit.minos())


class FitMcmc:
    """Class to create and run a MCMC sampler with emcee package."""

    def __init__(
        self,
        covariance=None,
        data=None,
        sampler=None,
    ):
        super(FitMinuit, self).__init__(
            covariance=covariance,
            data=data,
        )
        self.sampler = sampler

    @classmethod
    def init_from_cov(
        cls,
        covariance,
        data,
        parameter_dict,
        likelihood_type="multivariate_gaussian",
    ):
        mcmc_fitter = cls(
            covariance=covariance,
            data=data,
        )
        likelihood = mcmc_fitter.get_likelihood(
            parameter_dict,
            likelihood_type=likelihood_type,
        )

        # CR - need to add the sampler here.

        return mcmc_fitter
