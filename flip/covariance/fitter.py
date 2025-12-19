import abc
import multiprocessing as mp
import os
from contextlib import nullcontext

import emcee
import iminuit
import numpy as np

import flip.covariance.likelihood as flik
from flip.covariance.covariance import CovMatrix
from flip.utils import create_log

log = create_log()


class BaseFitter(abc.ABC):
    """Abstract interface for fitters.

    Provides common wiring between covariance, data, and likelihood construction,
    and defines the contract for initialization from covariance or files.

    Attributes:
        covariance (CovMatrix): Covariance model to use for fits.
        data (object): Data provider passed to likelihoods.
    """

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
        """Initialize fitter from covariance.

        Returns:
            BaseFitter: Implementations must return an initialized fitter.
        """
        return

    @abc.abstractmethod
    def init_from_file(
        cls,
    ):
        """Initialize fitter from a covariance file.

        Returns:
            BaseFitter: Implementations must return an initialized fitter.
        """
        return

    def get_likelihood(
        self,
        parameter_dict,
        likelihood_type="multivariate_gaussian",
        likelihood_properties=None,
        **kwargs,
    ):
        """Construct a likelihood from the fitter's covariance and data.

        Args:
            parameter_dict (dict): Parameters with keys as names and values/priors.
            likelihood_type (str): Likelihood class key; see `select_likelihood`.
            likelihood_properties (dict, optional): Options overriding defaults.
            **kwargs: Extra args forwarded to likelihood constructors.

        Returns:
            BaseLikelihood: Initialized likelihood instance.
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
        """Map a likelihood type key to its class.

        Args:
            likelihood_type (str): One of `multivariate_gaussian`,
                `multivariate_gaussian_interp1d`, `multivariate_gaussian_interp2d`.

        Returns:
            type: Likelihood class.
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
        """Initialize Minuit fitter.

        Args:
            covariance (CovMatrix, optional): Covariance model for the fit.
            data (object, optional): Data provider passed to likelihoods.
            likelihood (BaseLikelihood, optional): Prebuilt likelihood.
            minuit (iminuit.Minuit, optional): Preconfigured Minuit instance.
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
        """Build a Minuit fitter from covariance and data.

        Args:
            covariance (CovMatrix): Covariance model.
            data (object): Data provider.
            parameter_dict (dict): Parameter specs including values, errors, limits.
            likelihood_type (str): Likelihood variant key.
            likelihood_properties (dict): Options (e.g., use_jit, use_gradient).
            **kwargs: Extra args forwarded to likelihood construction.

        Returns:
            FitMinuit: Configured fitter with `iminuit.Minuit` ready.
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

        if (likelihood.likelihood_grad is not None) & likelihood.likelihood_properties[
            "use_gradient"
        ]:
            log.add("Using jax gradient")
            grad = likelihood.likelihood_grad
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
        """Initialize a Minuit fitter from a covariance file.

        Detects supported formats by extension (`.pickle`, `.npz`), loads a
        `CovMatrix`, and delegates to `init_from_covariance`.

        Args:
            model_name (str): Model name (unused here; kept for API parity).
            model_kind (str): Model kind (unused here).
            filename (str): Path with or without extension.
            data (object): Data provider.
            parameter_dict (dict): Parameter specs.
            likelihood_type (str): Likelihood variant key.
            likelihood_properties (dict): Likelihood options.

        Returns:
            FitMinuit: Configured fitter with `iminuit.Minuit` ready.
        """
        # Detect supported formats by extension
        if filename.endswith(".pickle"):
            fname = filename[:-7]
            file_format = "pickle"
        elif filename.endswith(".npz"):
            fname = filename[:-4]
            file_format = "npz"
        else:
            # Assume base name without extension, default to pickle
            fname = filename
            file_format = "pickle"

        covariance = CovMatrix.init_from_file(fname, file_format)

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
        """Configure Minuit parameter errors, limits, and fixed flags.

        Args:
            parameter_dict (dict): Parameter config with keys `value`, optional
                `error`, `fixed`, `limit_low`, `limit_up`.
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
        """Run Minuit optimization and return fitted parameter values.

        Args:
            migrad (bool): Run MIGRAD algorithm.
            hesse (bool): Compute HESSE errors.
            minos (bool): Compute MINOS intervals (may be slow).
            n_iter (int): Number of MIGRAD iterations to perform.

        Returns:
            dict: Fitted parameter values.
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
            except RuntimeError:
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
        """Initialize MCMC fitter.

        Args:
            covariance (CovMatrix, optional): Covariance model.
            data (object, optional): Data provider.
            sampler_name (str): Sampler backend name (only `emcee` supported).
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
        """Build an MCMC fitter from covariance and data.

        Args:
            covariance (CovMatrix): Covariance model.
            data (object): Data provider.
            parameter_dict (dict): Parameter specs including random initialization.
            likelihood_type (str): Likelihood variant key.
            likelihood_properties (dict): Options; sets negative_log_likelihood=False.
            sampler_name (str): Sampler backend (`emcee`).
            nwalkers (int): Number of walkers.
            backend_file (str, optional): HDF backend path for resume/checkpoint.
            **kwargs: Extra args forwarded to likelihood.

        Returns:
            FitMCMC: Configured fitter with sampler set.
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
        """Not implemented for MCMC from file.

        Raises:
            NotImplementedError: Always.
        """

        raise NotImplementedError

    def set_sampler(self, likelihood, p0=None, **kwargs):
        """Create sampler backend from likelihood and initial positions.

        Args:
            likelihood (Callable): Log-probability callable.
            p0 (numpy.ndarray, optional): Initial walker positions `(nwalkers, ndim)`.
            **kwargs: Backend-specific options (e.g., `backend_file`).
        """
        if self.sampler_name == "emcee":
            self.sampler = EMCEESampler(likelihood, p0=p0, **kwargs)
        else:
            raise ValueError("Only emcee is available now")


class Sampler(abc.ABC):
    """Abstract sampler interface wrapping different MCMC engines."""

    def __init__(self, likelihood, p0=None):
        """Initialize sampler.

        Args:
            likelihood (Callable): Log-probability function.
            p0 (numpy.ndarray, optional): Initial positions `(nwalkers, ndim)`.
        """
        self.likelihood = likelihood
        self._p0 = None
        if p0 is not None:
            self.p0 = p0

    @abc.abstractmethod
    def run_chains(self, nsteps):
        """Run sampler chains for a fixed number of steps.

        Args:
            nsteps (int): Number of steps per walker.

        Returns:
            Any: Backend-specific sampler object.
        """
        return

    @property
    def ndim(self):
        """Return dimensionality of parameter space.

        Returns:
            int: Number of parameters.
        """
        return len(self.likelihood.parameter_names)

    @property
    def p0(self):
        """Initial positions of walkers.

        Returns:
            numpy.ndarray: Array of shape `(nwalkers, ndim)`.
        """
        return self._p0

    @p0.setter
    def p0(self, value):
        """Set initial positions ensuring shape consistency.

        Args:
            value (numpy.ndarray): Initial positions `(nwalkers, ndim)`.

        Raises:
            ValueError: If `ndim` mismatch.
        """
        if value.shape[1] != self.ndim:
            raise ValueError(
                f"p0.shape[1] is equal to ndim={self.ndim}, currently {value.shape[1]}"
            )
        self._p0 = value
        self.nwalkers = value.shape[0]


class EMCEESampler(Sampler):
    def __init__(self, likelihood, p0=None, backend_file=None):
        super().__init__(likelihood, p0=p0)
        """Create an emcee sampler with optional HDF backend.

        Args:
            likelihood (Callable): Log-probability function.
            p0 (numpy.ndarray, optional): Initial positions.
            backend_file (str, optional): HDF backend filename to resume/checkpoint.
        """
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
        """Run emcee chains for a fixed number of steps.

        Args:
            nsteps (int): Number of steps to run.
            number_worker (int): Parallel workers via multiprocessing.
            progress (bool): Show progress bar.

        Returns:
            emcee.EnsembleSampler: The sampler instance.
        """
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
        """Run chains until reaching autocorrelation convergence criteria.

        Uses emcee's `get_autocorr_time` to check stabilization of autocorrelation
        time and sufficient chain length.

        Args:
            number_worker (int): Parallel workers.
            maxstep (int): Maximum steps if not converged earlier.
            tau_conv (float): Relative change threshold for convergence.
            progress (bool): Show progress bar.

        Returns:
            emcee.EnsembleSampler: The sampler instance.
        """
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
