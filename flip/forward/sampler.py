import jax
import tensorflow_probability.substrates.jax as tfp
import tensorflow_probability.substrates.jax.mcmc as mcmc
from tensorflow_probability.substrates.jax.mcmc import NoUTurnSampler as NUTS

from . import likelihood as forward_model_likelihood


class BaseSampler(object):
    def __init__(self, likelihood):
        self.likelihood = likelihood

    @staticmethod
    def select_likelihood(likelihood_type):
        if likelihood_type == "candle_grid_gaussian":
            likelihood_class = forward_model_likelihood.CandleGridGaussianLikelihood
        return likelihood_class


class NutsSampler(BaseSampler):
    def __init__(
        self,
        likelihood,
        parameter_dict,
    ):

        super().__init__(likelihood=likelihood)
        self.parameter_dict = parameter_dict
        self.init_parameters()

    @classmethod
    def init_from_simulator(
        cls,
        likelihood_type,
        simulator,
        velocity_data_vector,
        coordinates_velocity,
        parameter_dict,
    ):

        likelihood_class = BaseSampler.select_likelihood(likelihood_type)
        likelihood = likelihood_class(
            simulator=simulator,
            velocity_data_vector=velocity_data_vector,
            coordinates_velocity=coordinates_velocity,
            parameter_dict=parameter_dict,
        )
        return cls(likelihood=likelihood, parameter_dict=parameter_dict)

    def init_parameters(self):

        parameters_fixed = {
            key: value for key, value in self.parameter_dict.items() if value["fixed"]
        }
        parameters_to_sample = {
            key: value
            for key, value in self.parameter_dict.items()
            if not value["fixed"]
        }

        self._parameters = parameters_fixed
        self._sampling_parameters = list(parameters_to_sample.keys())

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
        kernel = NUTS(self.likelihood, step_size=step_size)

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
