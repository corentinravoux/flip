import numpy as np
import emcee
import iminuit
import flip.likelihood as flik
from flip.covariance.covariance import CovMatrix


class FitMinuit:
    """Class to maximize likelihood with Minuit. 
    """    
    def __init__(data, cov):
        self.data = data
        self.cov = cov

        if self.cov.model_type == "joint":
            self.likelihood = flik.JointLikelihood()
        elif self.cov.model_type == "velocity":
            self.likelihood = flik.VelocityLikelihood()
        elif self.cov.model_type == "density":
            self.likelihood = flik.VelocityLikelihood()

    
    @classmethod
    def init_from_cov(cls, data, cov):
        return cls(data, cov)
    
    @classmethod
    def init_from_file(cls, data, model_name, filename):
        cov = CovMatrix.init_from_file(model_name, model_type, filename)
        return cls(data, cov)
    



