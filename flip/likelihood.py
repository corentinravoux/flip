import numpy as np
from scipy.linalg import cho_factor, cho_solve
import iminuit


# CR - first start at building an oriented object framework for likelihood + sampling
# --> interface between the model and the fitting method (Add minuit + MCMC)

# CR - We need a generalized way to define the likelihood based on:
# - A type of likelihood: Gaussian multivariate
# - The model: link with flip.covariance.covariance


def parameters_lai22(params, covariance, variant=None):
    if variant == "densitynoRSD":
        b = params[0]
    elif variant == "density":
        b = params[0]
        f = params[1]
        sig_g = params[2]
    else:
        b = params[0]
        f = params[1]
        sig_g = params[2]
    return (covariance,)


def log_likelihood(self):
    cholesky = cho_factor(self.covariance)
    logdet = 2 * np.sum(np.log(np.diag(cholesky[0])))
    chi2 = np.dot(self.vector, cho_solve(cholesky, self.vector))
    return -0.5 * (self.vector * np.log(2 * np.pi) + logdet + chi2)


def init_fit_from_cov(
    vector,
    vector_err,
    covariance,
    parameters_name,
    params_init,
    params_fixed,
    params_lim,
):
    def neg_log_like(params_values):
        covariance_sum = np.sum(
            [covariance[i] * params_values[i] for i, _ in enumerate(covariance)], axis=0
        )
        covariance_sum += np.diag(vector_err**2)
        log_like = log_likelihood(vector, covariance_sum)
        return -log_like

    m = iminuit.Minuit(neg_log_like, params_init, name=parameters_name)

    m.errordef = iminuit.Minuit.LIKELIHOOD
    m.limits = params_lim
    m.fixed = params_fixed
    return m


class BaseLikelihood(object):
    def __init__(self, likelihood=None):
        self.likelihood = likelihood


class MultivariateGaussianLikelihood(BaseLikelihood):
    def __init__(self, likelihood=None, covariance=None, vector=None):
        super(BaseLikelihood, self).__init__(likelihood=likelihood)
        self.covariance = covariance
        self.vector = vector

    def compute_likelihood(self):
        cholesky = cho_factor(self.covariance)
        logdet = 2 * np.sum(np.log(np.diag(cholesky[0])))
        chi2 = np.dot(self.vector, cho_solve(cholesky, self.vector))
        return -0.5 * (self.vector * np.log(2 * np.pi) + logdet + chi2)


class VelocityLikelihood(MultivariateGaussianLikelihood):
    def __init__(self, likelihood=None, model_name=None, covariance=None, vector=None):
        self.vector = vector
        self.covariance = covariance
        self.model_name = model_name
        pass


class DensityLikelihood(MultivariateGaussianLikelihood):
    def __init__(self, function=None, rsd=None):
        self.function = function
        self.rsd = rsd
        pass

    def get_log_like_density(
        self, fitpar, density, density_err, cov_gg_b2, cov_gg_bf, cov_gg_f2, m_index_gg
    ):
        fs8 = fitpar[0]
        bs8 = fitpar[1]
        sig_g = fitpar[2]

        # Build cov matrix
        cov_gg_b2_all = np.sum(
            np.array([cov_gg_b2[i] * sig_g**m for i, m in enumerate(m_index_gg)]),
            axis=0,
        )
        cov_gg_bf_all = np.sum(
            np.array([cov_gg_bf[i] * sig_g**m for i, m in enumerate(m_index_gg)]),
            axis=0,
        )
        cov_gg_f2_all = np.sum(
            np.array([cov_gg_f2[i] * sig_g**m for i, m in enumerate(m_index_gg)]),
            axis=0,
        )
        cov_matrix = (
            bs8**2 * cov_gg_b2_all
            + bs8 * fs8 * cov_gg_bf_all
            + fs8**2 * cov_gg_f2_all
        )

        # Add shot noise
        diag_add = density_err**2
        cov_matrix += np.diag(diag_add)

        log_like = self.log_likelihood(density, cov_matrix)
        return log_like

    def get_log_like_density_no_rsd(
        self, fitpar, density, density_err, cov_gg_b2, m_index_gg
    ):
        bs8 = fitpar[0]
        sig_g = fitpar[1]

        # Build cov matrix
        cov_gg_b2_all = np.sum(
            np.array([cov_gg_b2[i] * sig_g**m for i, m in enumerate(m_index_gg)]),
            axis=0,
        )
        cov_matrix = bs8**2 * cov_gg_b2_all

        # Add shot noise
        diag_add = density_err**2
        cov_matrix += np.diag(diag_add)

        log_like = self.log_likelihood(density, cov_matrix)
        return log_like

    @classmethod
    def init_from_matrix(cls):
        return cls.init_from_property_files()


class JointLikelihood(MultivariateGaussianLikelihood):
    def __init__(self, likelihood=None, model_name=None, covariance=None, vector=None):
        self.vector = vector
        self.covariance = covariance
        self.model_name = model_name
        pass
