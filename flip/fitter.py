import emcee
import iminuit
import numpy as np

import flip.likelihood as flik
from flip.covariance.covariance import CovMatrix
from flip.utils import create_log

log = create_log()


class BaseFitter(object):
    def __init__(
        self,
        covariance=None,
        data=None,
        likelihood=None,
    ):
        """
        The __init__ function is called when the class is instantiated.
        It sets up the instance of the class, and defines all of its attributes.
        The __init__ function should always accept at least one argument, self, which refers to the instance of the object being created.

        Args:
            self: Represent the instance of the class
            covariance: Set the covariance matrix of the model
            data: Store the data that will be used to train the model
            likelihood: Specify the likelihood function
            : Initialize the covariance matrix

        Returns:
            The object itself

        """
        self.covariance = covariance
        self.data = data
        self.likelihood = likelihood

    @classmethod
    def init_from_covariance(
        cls,
        covariance,
        data,
        parameter_dict,
        likelihood_type=None,
    ):
        """
        The init_from_covariance function is used to initialize the parameters of a likelihood class.
        It takes in a covariance matrix, data, and parameter dictionary as inputs. The covariance matrix
        is used to calculate the inverse of it (the precision matrix) which is then passed into the
        likelihood function along with the data and parameter dictionary. This allows for an initial guess
        of what values should be passed into each parameter when running MCMC or Minuit.

        Args:
            cls: Call the class that is being used
            covariance: Initialize the covariance matrix
            data: Calculate the number of parameters in the model
            parameter_dict: Pass the parameter dictionary to the
            likelihood_type: Determine the type of likelihood to use
            : Initialize the covariance matrix

        """
        log.add("Method to override, no initialization is done in this super class")
        raise RuntimeError("Ghost override method")

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
        """
        The init_from_file function is a class method that initializes the fitter object from a covariance matrix.

        Args:
            cls: Pass the class object to the function
            model_name: Specify the name of the model
            model_type: Specify the type of model
            filename: Load the covariance matrix from a file
            data: Initialize the fitter's data attribute
            parameter_dict: Pass in the parameters that are used to
            likelihood_type: Specify the type of likelihood function to use
            : Specify the type of likelihood

        Returns:
            A fitter object

        """
        covariance = CovMatrix.init_from_file(model_name, model_type, filename)

        fitter = cls.init_from_covariance(
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

        likelihood_class = BaseFitter.select_likelihood(likelihood_type)

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
        """
        The select_likelihood function takes in a string, likelihood_type, and returns the corresponding class.

        Args:
            likelihood_type: Determine which likelihood function to use

        Returns:
            The likelihood class

        """
        if likelihood_type == "multivariate_gaussian":
            likelihood_class = flik.MultivariateGaussianLikelihood
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
            likelihood=likelihood,
        )
        self.minuit = minuit

    @classmethod
    def init_from_covariance(
        cls,
        covariance,
        data,
        parameter_dict,
        likelihood_type="multivariate_gaussian",
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
        )
        minuit_fitter.likelihood = likelihood
        parameter_values = [
            parameter_dict[parameters]["value"] for parameters in parameter_dict
        ]

        minuit_fitter.minuit = iminuit.Minuit(
            likelihood,
            parameter_values,
            name=likelihood.parameter_names,
        )

        minuit_fitter.setup_minuit(parameter_dict)

        return minuit_fitter

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
        likelihood=None,
        sampler=None,
    ):
        """
        The __init__ function is called when the class is instantiated.
        It sets up the instance of the class, and defines all of its attributes.
        The __init__ function should always accept at least one argument, self, which refers to the instance of the object being created.

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
        super(FitMinuit, self).__init__(
            covariance=covariance,
            data=data,
            likelihood=likelihood,
        )
        self.sampler = sampler

    @classmethod
    def init_from_covariance(
        cls,
        covariance,
        data,
        parameter_dict,
        likelihood_type="multivariate_gaussian",
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
        )
        likelihood = mcmc_fitter.get_likelihood(
            parameter_dict,
            likelihood_type=likelihood_type,
        )
        mcmc_fitter.likelihood = likelihood

        # CR - need to add the sampler here.

        return mcmc_fitter
