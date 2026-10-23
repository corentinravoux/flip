import multiprocessing

import jax
import jax.numpy as jnp

from flip.forward import likelihood as forward_model_likelihood
from flip.utils import create_log

log = create_log()

try:
    import tensorflow_probability.substrates.jax as tfp
    from tensorflow_probability.substrates.jax import mcmc
    from tensorflow_probability.substrates.jax.mcmc import NoUTurnSampler as NUTS

    tfp_installed = True
# AttributeError: tfp releases that do not match the installed jax fail at import
except (ImportError, AttributeError):
    tfp_installed = False
    if multiprocessing.parent_process() is None:
        log.add(
            "Install a tensorflow-probability compatible with jax to use the NutsSampler",
            level="warning",
        )

try:
    import blackjax

    blackjax_installed = True
except ImportError:
    blackjax_installed = False
    if multiprocessing.parent_process() is None:
        log.add(
            "Install blackjax to use the BlackjaxNutsSampler",
            level="warning",
        )

# CR - forward modeling is not like covariance.
# The likelihood is too specific to be stored at the sampler level
# Try to make the simulator general enough. If not, put the likelihood in the simulator
# And rename it models.


class BaseSampler:
    def __init__(
        self,
        likelihood=None,
        parameter_dict=None,
    ):
        self.likelihood = likelihood
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
        likelihood_properties=None,
    ):

        parameter_names = list(parameter_dict.keys())

        likelihood_class = BaseSampler.select_likelihood(likelihood_type)
        likelihood = likelihood_class(
            simulator=simulator,
            velocity_data_vector=velocity_data_vector,
            coordinates_velocity=coordinates_velocity,
            parameter_names=parameter_names,
            likelihood_properties=likelihood_properties,
        )
        return cls(
            likelihood=likelihood,
            parameter_dict=parameter_dict,
        )

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

        super().__init__(
            likelihood=likelihood,
            parameter_dict=parameter_dict,
        )

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
        sample_key=None,
        num_burnin_steps=1000,
        mc_steps=2000,
        **kwargs,
    ):

        if sample_key is None:
            sample_key = jax.random.split(jax.random.PRNGKey(0))[1]

        print(f"num_burnin_steps = {num_burnin_steps}")
        print(f"mc_steps = {mc_steps}")

        initial_guess = [
            self.parameter_dict[key]["value"] for key in self._parameters_sample
        ]
        step_size = [
            self.parameter_dict[key]["step_size"] for key in self._parameters_sample
        ]

        # NUTS
        kernel = NUTS(
            self.likelihood_call,
            step_size=step_size,
            # max_tree_depth=6,
        )
        num_adaptation_steps = num_burnin_steps * 0.8
        kernel = tfp.mcmc.DualAveragingStepSizeAdaptation(
            inner_kernel=kernel,
            num_adaptation_steps=int(num_adaptation_steps),
            step_size_setter_fn=lambda pkr, new_step_size: pkr._replace(
                step_size=new_step_size
            ),
            step_size_getter_fn=lambda pkr: pkr.step_size,
            log_accept_prob_getter_fn=lambda pkr: pkr.log_accept_ratio,
            # target_accept_prob=0.65,
        )

        sampler_states = mcmc.sample_chain(
            mc_steps,
            current_state=initial_guess,
            kernel=kernel,
            num_burnin_steps=num_burnin_steps,
            seed=sample_key,
        )

        return sampler_states


class BlackjaxNutsSampler(BaseSampler):
    """NUTS from blackjax. The burn-in runs Stan's window adaptation, which tunes
    the step size and a diagonal mass matrix: no hand-set step sizes are needed,
    the "step_size" entries of parameter_dict are ignored."""

    def likelihood_call(self, position):
        parameters = [
            position[key] if key in position else self.parameter_dict[key]["value"]
            for key in self.likelihood.parameter_names
        ]
        return self.likelihood(parameters)

    def run(
        self,
        sample_key=None,
        num_burnin_steps=1000,
        mc_steps=2000,
        target_acceptance_rate=0.8,
        **kwargs,
    ):
        if sample_key is None:
            sample_key = jax.random.PRNGKey(0)
        key_warmup, key_sample = jax.random.split(sample_key)

        initial_position = {
            key: jnp.asarray(self.parameter_dict[key]["value"])
            for key in self._parameters_sample
        }

        warmup = blackjax.window_adaptation(
            blackjax.nuts,
            self.likelihood_call,
            target_acceptance_rate=target_acceptance_rate,
            **kwargs,
        )
        (state, tuned_parameters), _ = warmup.run(
            key_warmup, initial_position, num_steps=num_burnin_steps
        )
        nuts_step = blackjax.nuts(self.likelihood_call, **tuned_parameters).step

        def one_step(state, key):
            state, info = nuts_step(key, state)
            return state, (
                state.position,
                info.acceptance_rate,
                info.num_integration_steps,
                info.is_divergent,
            )

        _, (positions, acceptance_rate, num_integration_steps, is_divergent) = (
            jax.lax.scan(one_step, state, jax.random.split(key_sample, mc_steps))
        )

        return {
            "positions": positions,
            "acceptance_rate": acceptance_rate,
            "num_integration_steps": num_integration_steps,
            "is_divergent": is_divergent,
            "tuned_parameters": tuned_parameters,
        }

    def return_sampled_chains_from_states(
        self,
        sampler_states,
        burnin_steps=0,
    ):

        chains_parameters_sample = {
            self._parameters_sample[i]: sampler_states[0][i][burnin_steps:]
            for i in range(len(self._parameters_sample))
        }

        return chains_parameters_sample

    def return_all_chains_from_states(
        self,
        sampler_states,
        burnin_steps=0,
    ):

        chains_parameters_sample = self.return_sampled_chains_from_states(
            sampler_states,
            burnin_steps=burnin_steps,
        )

        size_chain = len(sampler_states[0][0][burnin_steps:])

        chains_parameters_fixed = {
            self._parameters_fixed[i]: jnp.array(
                [
                    self.parameter_dict[self._parameters_fixed[i]]["value"]
                    for _ in range(size_chain)
                ]
            )
            for i in range(len(self._parameters_fixed))
        }

        chains_parameters = chains_parameters_sample | chains_parameters_fixed

        return chains_parameters

    def return_average_chains_from_states(
        self,
        sampler_states,
        burnin_steps=0,
    ):

        chains_parameters = self.return_all_chains_from_states(
            sampler_states,
            burnin_steps=burnin_steps,
        )

        average_chains_parameters = {
            key: jnp.mean(value, axis=0) for key, value in chains_parameters.items()
        }

        return average_chains_parameters
