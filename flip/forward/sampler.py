import jax
import tensorflow_probability.substrates.jax as tfp
import tensorflow_probability.substrates.jax.mcmc as mcmc
from tensorflow_probability.substrates.jax.mcmc import NoUTurnSampler as NUTS

from .likelihood import VelocityGaussianGridComparisonLikelihood

# CR - Implement nuts


class BaseSampler(object):
    def __init__(self, likelihood):
        self.likelihood = likelihood


class NutsSampler(BaseSampler):
    def __init__(
        self,
        likelihood,
        sampling_parameters,
    ):

        super().__init__(likelihood=likelihood)
        self.likelihood = likelihood
        self.sampling_parameters = sampling_parameters

    @classmethod
    def init_from_simulator(
        cls,
        simulator,
        velocity_data_vector,
        coordinates_velocity,
        parameter_names,
        sampling_parameters,
    ):
        likelihood = VelocityGaussianGridComparisonLikelihood(
            simulator=simulator,
            velocity_data_vector=velocity_data_vector,
            coordinates_velocity=coordinates_velocity,
            parameter_names=parameter_names,
        )
        return cls(likelihood=likelihood, sampling_parameters=sampling_parameters)

    def set_parameters(self, parameters):
        # CR - fix the parameter to vary and the one to keep fixed
        self._parameters = parameters

    def target_logprob_fn(self, *paramv):
        """ """
        param = dict(zip(self.sampling_parameters, paramv))

        # data to compare to
        data_candles = {
            "mag": self._data_mag,
            "redshift": self._data_redshift,
            "mag_err": self._data_mag_err,
            "redshift_err": self._data_redshift_err,
        }

        # box structure for computation
        box_struct = {
            "pk0": self.pk,
            "d2v": self.d2v,
            "dist_mpch_vec": self.dist_mpch_vec,
            "r1d": self.r1d,
            "nbins": self.nbins,
            "kmaxindex": self.kmaxindex,
            "deltak_sampling": self.deltak_sampling,
            # density sampling
            "dist_mpch_los": self.dist_mpch_los,
            "targets_voxel_dir": self.targets_voxel_dir,
        }

        # compute log-probability
        loglikelihood = self.likelihood
        return loglikelihood

    def run_nuts(
        self,
        initial_guess,
        sample_key=jax.random.split(jax.random.PRNGKey(0))[1],
        num_burnin_steps=1000,
        step_size=0.1,
        mc_steps=2000,
        **kwargs,
    ):
        """"""

        print(f"num_burnin_steps = {num_burnin_steps}")
        print(f"mc_steps = {mc_steps}")

        # NUTS
        kernel = NUTS(self.target_logprob_fn, step_size=step_size)

        num_adaptation_steps = num_burnin_steps * 0.8
        kernel = tfp.mcmc.DualAveragingStepSizeAdaptation(
            inner_kernel=kernel,
            num_adaptation_steps=int(num_adaptation_steps),
            step_size_setter_fn=lambda pkr, new_step_size: pkr._replace(
                step_size=new_step_size
            ),
            step_size_getter_fn=lambda pkr: pkr.step_size,
            log_accept_prob_getter_fn=lambda pkr: pkr.log_accept_ratio,
        )

        out = mcmc.sample_chain(
            mc_steps,
            current_state=initial_guess,
            kernel=kernel,
            num_burnin_steps=num_burnin_steps,
            seed=sample_key,
        )
        return out

    @property
    def data(self):
        """data"""
        return self._data

    @property
    def sampling_parameters(self):
        """parameters to sample"""
        return self._sampling_parameters

    @property
    def parameters(self):
        """parameters used in the sampling"""
        return self._parameters
