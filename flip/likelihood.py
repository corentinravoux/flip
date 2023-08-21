import numpy as np
import jax
import jax.numpy as jnp
import jax.scipy as jsc


# CR - first start at building an oriented object framework for likelihood + sampling
# --> interface between the model and the fitting method (Add minuit + MCMC)


class BaseLikelihood(object):
    def __init__(self, likelihood=None):
        self.likelihood = likelihood


class MultivariateGaussianLikelihood(BaseLikelihood):
    def __init__(self, likelihood=None, covariance=None, vector=None):
        super(BaseLikelihood, self).__init__(likelihood=likelihood)
        self.covariance = covariance
        self.vector = vector

    @jax.jit
    def compute_likelihood(self):
        cholesky = jsc.linalg.cho_factor(self.covariance)
        logdet = 2 * jnp.sum(jnp.log(jnp.diag(cholesky[0])))
        chi2 = jnp.dot(self.vector, jsc.linalg.cho_solve(cholesky, self.vector))
        return -0.5 * (self.vector * jnp.log(2 * np.pi) + logdet + chi2)


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

    @jax.jit
    def get_log_like_density(
        fitpar, density, density_err, cov_gg_b2, cov_gg_bf, cov_gg_f2, m_index_gg
    ):
        fs8 = fitpar[0]
        bs8 = fitpar[1]
        sig_g = fitpar[2]

        # Build cov matrix
        cov_gg_b2_all = jnp.sum(
            jnp.array([cov_gg_b2[i] * sig_g**m for i, m in enumerate(m_index_gg)]),
            axis=0,
        )
        cov_gg_bf_all = jnp.sum(
            jnp.array([cov_gg_bf[i] * sig_g**m for i, m in enumerate(m_index_gg)]),
            axis=0,
        )
        cov_gg_f2_all = jnp.sum(
            jnp.array([cov_gg_f2[i] * sig_g**m for i, m in enumerate(m_index_gg)]),
            axis=0,
        )
        cov_matrix = (
            bs8**2 * cov_gg_b2_all
            + bs8 * fs8 * cov_gg_bf_all
            + fs8**2 * cov_gg_f2_all
        )

        # Add shot noise
        diag_add = density_err**2
        cov_matrix += jnp.diag(diag_add)

        log_like = log_likelihood(density, cov_matrix)
        return log_like

    @jax.jit
    def get_log_like_density_no_rsd(
        fitpar, density, density_err, cov_gg_b2, m_index_gg
    ):
        bs8 = fitpar[0]
        sig_g = fitpar[1]

        # Build cov matrix
        cov_gg_b2_all = jnp.sum(
            jnp.array([cov_gg_b2[i] * sig_g**m for i, m in enumerate(m_index_gg)]),
            axis=0,
        )
        cov_matrix = bs8**2 * cov_gg_b2_all

        # Add shot noise
        diag_add = density_err**2
        cov_matrix += jnp.diag(diag_add)

        log_like = log_likelihood(density, cov_matrix)
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

    # CR - include cross terms or not
