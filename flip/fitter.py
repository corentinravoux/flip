import abc
import multiprocessing as mp
import os
from contextlib import nullcontext

import emcee
import iminuit
import numpy as np

try:
    from jax import grad as jax_grad

    jax_installed = True
except ImportError:
    jax_installed = False
    pass


from flip.utils import create_log

log = create_log()

import flip.likelihood as flik
from flip.covariance.covariance import CovMatrix


class BaseFitter(abc.ABC):
    def __init__(
        self,
        covariance=None,
        data=None,
    ):
        """
        The __init__ function is called when the class is instantiated.
        It sets up the instance of the class, and defines all of its attributes.
        The __init__ function should always accept at least one argument, self, which refers to the instance of the object being created.

        Args:
            self: Represent the instance of the class
            covariance: Set the covariance matrix of the model
            data: Store the data that will be used to train the model

        Returns:
            The object itself

        """
        self.covariance = covariance
        self.data = data

    @abc.abstractmethod
    def init_from_covariance(
        cls,
    ):
        """
        The init_from_covariance function is a class method that initializes the
            fitter from the covariance matrix. It is here an abstract method that
            needs to be override

        Args:
            cls: Pass a class object into a method
        """
        return

    @abc.abstractmethod
    def init_from_file(
        cls,
    ):
        """
        The init_from_covariance function is a class method that initializes the
            fitter from the a file containing covariance matrix. It is here an
            abstract method that needs to be override

        Args:
            cls: Pass a class object into a method
        """
        return

    def get_likelihood(
        self,
        parameter_dict,
        likelihood_type="multivariate_gaussian",
        likelihood_properties=None,
        **kwargs,
    ):
        """
        The get_likelihood function is used to create a likelihood object from the covariance matrix.
        The function takes in a dictionary of parameters, and returns an instance of the likelihood class.


        Args:
            self: Bind the method to a class
            parameter_dict: Pass the parameters to be used in the likelihood function
            likelihood_type: Select the likelihood class
            : Select the likelihood function

        Returns:
            A likelihood object

        """

        parameter_names = [parameters for parameters in parameter_dict]

        likelihood_class = BaseFitter.select_likelihood(likelihood_type)

        likelihood = likelihood_class.init_from_covariance(
            self.covariance,
            self.data,
            parameter_names,
            likelihood_properties=likelihood_properties,
            **kwargs,
        )

        return likelihood

    @staticmethod
    def select_likelihood(likelihood_type):
        """
        The select_likelihood function takes in a string, likelihood_type, and returns the corresponding class.

        Args:
            likelihood_type: Determine which likelihood function to use

        Returns:
            The likelihood class

        """
        if likelihood_type == "multivariate_gaussian":
            likelihood_class = flik.MultivariateGaussianLikelihood
        elif likelihood_type == "multivariate_gaussian_interp1d":
            likelihood_class = flik.MultivariateGaussianLikelihoodInterpolate1D
        elif likelihood_type == "multivariate_gaussian_interp2d":
            likelihood_class = flik.MultivariateGaussianLikelihoodInterpolate2D
        return likelihood_class


class FitMinuit(BaseFitter):
    """Class to maximize likelihood with Minuit."""

    def __init__(
        self,
        covariance=None,
        data=None,
        likelihood=None,
        minuit=None,
    ):
        """
        The __init__ function is called when the class is instantiated.
        It sets up the instance of the class, and defines all of its attributes.
        The self argument refers to the instance of this object that has been created.

        Args:
            self: Represent the instance of the class
            covariance: Pass the covariance matrix to the fit
            data: Pass the data to be fitted
            likelihood: Pass the likelihood function to the fit
            minuit: Pass a minuit object to the fitminuit class
            : Set the minuit object

        Returns:
            The object that is being created

        """
        super(FitMinuit, self).__init__(
            covariance=covariance,
            data=data,
        )
        self.likelihood = likelihood
        self.minuit = minuit

    @classmethod
    def init_from_covariance(
        cls,
        covariance,
        data,
        parameter_dict,
        likelihood_type="multivariate_gaussian",
        likelihood_properties={},
        **kwargs,
    ):
        """
        The init_from_covariance function is a class method that initializes the MinuitFitter object.
        It takes in the covariance matrix, data, parameter dictionary and likelihood type as arguments.
        The minuit_fitter object is initialized with the covariance matrix and data. The likelihood function
        is then calculated using get_likelihood() which returns an instance of LikelihoodFunction(). This
        instance is assigned to minuit_fitter's attribute 'likelihood'. The parameter values are extracted from
        the parameter dictionary and stored in a list called 'parameter_values'. A Minuit object called 'minuit'
        is

        Args:
            cls: Create a new instance of the class
            covariance: Initialize the covariance matrix of the likelihood
            data: Pass the data to the likelihood function
            parameter_dict: Pass the parameters to be fitted
            likelihood_type: Specify the type of likelihood function to be used
            : Set the covariance matrix of the data

        Returns:
            A minuit_fitter object

        """
        minuit_fitter = cls(
            covariance=covariance,
            data=data,
        )
        likelihood = minuit_fitter.get_likelihood(
            parameter_dict,
            likelihood_type=likelihood_type,
            likelihood_properties={
                **likelihood_properties,
                "negative_log_likelihood": True,
            },
            **kwargs,
        )
        minuit_fitter.likelihood = likelihood
        parameter_values = [
            parameter_dict[parameters]["value"] for parameters in parameter_dict
        ]

        if jax_installed & likelihood.likelihood_properties["use_gradient"]:
            grad = jax_grad(likelihood)
        else:
            grad = None
        minuit_fitter.minuit = iminuit.Minuit(
            likelihood,
            parameter_values,
            grad=grad,
            name=likelihood.parameter_names,
        )

        minuit_fitter.setup_minuit(parameter_dict)

        return minuit_fitter

    @classmethod
    def init_from_file(
        cls,
        model_name,
        model_kind,
        filename,
        data,
        parameter_dict,
        likelihood_type="multivariate_gaussian",
        likelihood_properties=None,
    ):
        """
        The init_from_file function is a class method that initializes the fitter object from a covariance matrix.

        Args:
            cls: Pass the class object to the function
            model_name: Specify the name of the model
            model_kind: Specify the type of model
            filename: Load the covariance matrix from a file
            data: Initialize the fitter's data attribute
            parameter_dict: Pass in the parameters that are used to
            likelihood_type: Specify the type of likelihood function to use
            : Specify the type of likelihood

        Returns:
            A fitter object

        """
        covariance = CovMatrix.init_from_file(model_name, model_kind, filename)

        return cls.init_from_covariance(
            covariance,
            data,
            parameter_dict,
            likelihood_type=likelihood_type,
            likelihood_properties={
                **likelihood_properties,
                "negative_log_likelihood": True,
            },
        )

    def setup_minuit(self, parameter_dict):
        """
        The setup_minuit function is used to set up the minuit object.
        It takes a dictionary of parameters as input and sets the errors, fixed values, and limits for each parameter.
        The error is set to be equal to the value if no error is specified in the dictionary. If a parameter has been fixed then its error will be zero.

        Args:
            self: Refer to the object itself
            parameter_dict: Set the initial values of the parameters

        Returns:
            A minuit object

        """
        self.minuit.errordef = iminuit.Minuit.LIKELIHOOD
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

    def run(self, migrad=True, hesse=False, minos=False, n_iter=1):
        """
        The run function is the main function of the class. It takes in a number of
        arguments, and then runs them through Minuit. The arguments are:

        Args:
            self: Bind the method to the object
            migrad: Run the migrad algorithm
            hesse: Run the hesse function
            minos: Run the minos function, which is a
            : Set the number of iterations for migrad

        Returns:
            A dictionary with the results of the minimization

        """
        if migrad:
            for i in range(n_iter):
                if n_iter != 1:
                    log.add(f"Iteration {i+1}/{n_iter}\n")
                log.add(self.minuit.migrad())
        if hesse:
            log.add(self.minuit.hesse())
        if minos:
            try:
                log.add(self.minuit.minos())
            except:
                pass

        return self.minuit.values.to_dict()


class FitMCMC(BaseFitter):
    """Class to create and run a MCMC sampler with emcee package."""

    def __init__(
        self,
        covariance=None,
        data=None,
        sampler_name="emcee",
    ):
        """
        The __init__ function is called when the class is instantiated.
        It sets up the instance of the class, and defines all of its attributes.
        The __init__ function should always accept at least one argument, self,
        which refers to the instance of the object being created.

        Args:
            self: Represent the instance of the object itself
            covariance: Set the covariance matrix of the fit
            data: Pass the data to the likelihood function
            likelihood: Define the likelihood function
            sampler: Pass the sampler object to the fit
            : Define the sampler that will be used in the fit

        Returns:
            The object itself, so the return value is self

        """
        super().__init__(
            covariance=covariance,
            data=data,
        )
        self.sampler_name = sampler_name
        self.sampler = None

    @classmethod
    def init_from_covariance(
        cls,
        covariance,
        data,
        parameter_dict,
        likelihood_type="multivariate_gaussian",
        likelihood_properties={},
        sampler_name="emcee",
        nwalkers=1,
        backend_file=None,
        **kwargs,
    ):
        """
        The init_from_covariance function is a class method that initializes the MCMC fitter from a covariance matrix.

        Args:
            cls: Create a new instance of the class
            covariance: Set the covariance matrix of the multivariate gaussian
            data: Calculate the likelihood
            parameter_dict: Pass in the parameters of the model
            likelihood_type: Specify the type of likelihood function to use
            : Set the covariance matrix

        Returns:
            A mcmc_fitter object

        """

        mcmc_fitter = cls(
            covariance=covariance,
            data=data,
            sampler_name=sampler_name,
        )
        likelihood = mcmc_fitter.get_likelihood(
            parameter_dict,
            likelihood_type=likelihood_type,
            likelihood_properties={
                **likelihood_properties,
                "negative_log_likelihood": False,
            },
            **kwargs,
        )
        p0 = None
        if backend_file is None:
            p0 = np.stack(
                [p["randfun"](size=nwalkers) for p in parameter_dict.values()]
            ).T
        else:
            if not (os.path.exists(backend_file)):
                p0 = np.stack(
                    [p["randfun"](size=nwalkers) for p in parameter_dict.values()]
                ).T
        mcmc_fitter.set_sampler(likelihood, p0=p0, backend_file=backend_file)

        return mcmc_fitter

    @classmethod
    def init_from_file(
        cls,
    ):
        """
        The init_from_covariance function is a class method that initializes the
            fitter from the a file containing covariance matrix. It is here an
            abstract method that needs to be override

        Args:
            cls: Pass a class object into a method
        """

        raise NotImplementedError

    def set_sampler(self, likelihood, p0=None, **kwargs):
        if self.sampler_name == "emcee":
            self.sampler = EMCEESampler(likelihood, p0=p0, **kwargs)
        else:
            raise ValueError("Only emcee is available now")


class Sampler(abc.ABC):

    def __init__(self, likelihood, p0=None):
        self.likelihood = likelihood
        self._p0 = None
        if p0 is not None:
            self.p0 = p0

    @abc.abstractmethod
    def run_chains(self, nsteps):
        return

    @property
    def ndim(self):
        return len(self.likelihood.parameter_names)

    @property
    def p0(self):
        return self._p0

    @p0.setter
    def p0(self, value):
        if value.shape[1] != self.ndim:
            raise ValueError(
                f"p0.shape[1] is equal to ndim={self.ndim}, currently {value.shape[1]}"
            )
        self._p0 = value
        self.nwalkers = value.shape[0]


class EMCEESampler(Sampler):
    def __init__(self, likelihood, p0=None, backend_file=None):
        super().__init__(likelihood, p0=p0)

        self.backend = None
        if backend_file is not None:
            backend_file_exists = os.path.exists(backend_file)
            self.backend = emcee.backends.HDFBackend(backend_file)
            if backend_file_exists:
                log.add(
                    "File already exist"
                    "Initial size: {0}".format(self.backend.iteration)
                )
                if self.backend.iteration == 0:
                    log.add("Backend file is empty, please delete it and relaunch")
                self._p0 = None
                self.nwalkers = self.backend.shape[0]
            else:
                log.add("No file initialize, will create a new one")
            self.backend_file_exists = backend_file_exists
        else:
            self.backend_file_exists = False

    def run_chains(self, nsteps, number_worker=1, progress=False):
        with mp.Pool(number_worker) if number_worker != 1 else nullcontext() as pool:
            sampler = emcee.EnsembleSampler(
                self.nwalkers,
                self.ndim,
                self.likelihood,
                pool=pool,
                backend=self.backend,
            )
            sampler.run_mcmc(
                self.p0,
                nsteps,
                progress=progress,
            )
        return sampler

    def run_chains_untilconv(
        self,
        number_worker=1,
        maxstep=100,
        tau_conv=0.01,
        progress=False,
    ):
        """Run chains until reaching auto correlation convergence criteria."""
        old_tau = np.inf
        with mp.Pool(number_worker) if number_worker != 1 else nullcontext() as pool:
            sampler = emcee.EnsembleSampler(
                self.nwalkers,
                self.ndim,
                self.likelihood,
                pool=pool,
                backend=self.backend,
            )
            if not self.backend_file_exists:
                for _ in sampler.sample(self.p0, iterations=maxstep, progress=progress):
                    if sampler.iteration % 500 == 0:
                        # Compute tau
                        tau = sampler.get_autocorr_time(tol=0)
                        # Check convergence
                        converged = np.all(tau * 100 < sampler.iteration)
                        converged &= np.all(np.abs(old_tau - tau) / tau < tau_conv)
                        if converged:
                            break
                        old_tau = tau
            else:
                # If the file already exists run to max step.
                sampler.run_mcmc(
                    None,
                    maxstep - self.backend.iteration,
                    progress=progress,
                )

        return sampler
