import jax
import tensorflow_probability.substrates.jax as tfp
import tensorflow_probability.substrates.jax.mcmc as mcmc
from tensorflow_probability.substrates.jax.mcmc import NoUTurnSampler as NUTS

from flip.forward import likelihood as forward_model_likelihood

# CR - forward modeling is not like covariance.
# The likelihood is too specific to be stored at the sampler level
# Try to make the simulator general enough. If not, put the likelihood in the simulator
# And rename it models.


class BaseSampler(object):
    def __init__(
        self,
        likelihood=None,
        parameter_dict=None,
    ):
        self.likelihood = likelihood
        self.parameter_dict = parameter_dict
        self.init_parameters()

    @staticmethod
    def select_likelihood(likelihood_type):
        if likelihood_type == "candle_grid_gaussian":
            likelihood_class = forward_model_likelihood.CandleGridGaussianLikelihood
        return likelihood_class

    def init_parameters(self):

        parameters_fixed = [
            key for key, value in self.parameter_dict.items() if value["fixed"]
        ]
        parameters_sample = [
            key for key, value in self.parameter_dict.items() if not value["fixed"]
        ]

        self._parameters_fixed = parameters_fixed
        self._parameters_sample = parameters_sample

    @property
    def parameters_sample(self):
        """parameters to sample"""
        return self._parameters_sample

    @property
    def parameters_fixed(self):
        """parameters fixed"""
        return self._parameters_fixed


class NutsSampler(BaseSampler):
    def __init__(
        self,
        likelihood=None,
        parameter_dict=None,
    ):

        super().__init__(likelihood=likelihood, parameter_dict=parameter_dict)

    @classmethod
    def init_from_simulator(
        cls,
        likelihood_type,
        simulator,
        velocity_data_vector,
        coordinates_velocity,
        parameter_dict,
    ):

        parameter_names = list(parameter_dict.keys())

        likelihood_class = BaseSampler.select_likelihood(likelihood_type)
        likelihood = likelihood_class(
            simulator=simulator,
            velocity_data_vector=velocity_data_vector,
            coordinates_velocity=coordinates_velocity,
            parameter_names=parameter_names,
        )
        return cls(likelihood=likelihood, parameter_dict=parameter_dict)

    def likelihood_call(self, *param):
        parameters_sample = dict(zip(self._parameters_sample, param))

        parameters = [
            (
                self.parameter_dict[key]["value"]
                if key in self._parameters_fixed
                else parameters_sample[key]
            )
            for key in self.likelihood.parameter_names
        ]
        return self.likelihood(parameters)

    def run(
        self,
        sample_key=jax.random.split(jax.random.PRNGKey(0))[1],
        num_burnin_steps=1000,
        mc_steps=2000,
        **kwargs,
    ):

        print(f"num_burnin_steps = {num_burnin_steps}")
        print(f"mc_steps = {mc_steps}")

        initial_guess = [
            self.parameter_dict[key]["value"] for key in self._parameters_sample
        ]
        step_size = [
            self.parameter_dict[key]["step_size"] for key in self._parameters_sample
        ]

        # NUTS
        kernel = NUTS(self.likelihood_call, step_size=step_size)
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

        sampler_states = mcmc.sample_chain(
            mc_steps,
            current_state=initial_guess,
            kernel=kernel,
            num_burnin_steps=num_burnin_steps,
            seed=sample_key,
        )

        return sampler_states
