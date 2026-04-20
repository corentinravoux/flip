import jax
import tensorflow_probability.substrates.jax as tfp
import tensorflow_probability.substrates.jax.mcmc as mcmc
from tensorflow_probability.substrates.jax.mcmc import NoUTurnSampler as NUTS

from flip.forward import likelihood as forward_model_likelihood


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
        sampler = cls(likelihood=likelihood, parameter_dict=parameter_dict)

        # If field modes are fixed, precompute base fields to avoid
        # expensive FFTs at every sampling step.
        sampler._precompute_if_modes_fixed()

        return sampler

    def _precompute_if_modes_fixed(self):
        """Precompute FFT-based base fields when modes are not sampled.

        When only scalar parameters (f, s8, b, sigma_v, h, etc.) are
        being sampled, the delta_fourier modes don't change between
        steps. Precomputing the real-space fields eliminates 4 FFTs
        (1 density + 3 velocity components) per likelihood evaluation.
        """
        modes_fixed = (
            "delta_modes_real" not in self._parameters_sample
            and "delta_modes_imag" not in self._parameters_sample
        )
        if modes_fixed and hasattr(self.likelihood, "simulator"):
            fixed_params = {
                key: value["value"] for key, value in self.parameter_dict.items()
            }
            self.likelihood.simulator.precompute_base_fields(fixed_params)

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

        import jax.numpy as jnp

        # Ensure initial values are JAX arrays with consistent dtype
        # to prevent JIT recompilation from type mismatches.
        initial_guess = [
            jnp.asarray(self.parameter_dict[key]["value"], dtype=jnp.float32)
            for key in self._parameters_sample
        ]
        step_size = [
            jnp.asarray(self.parameter_dict[key]["step_size"], dtype=jnp.float32)
            for key in self._parameters_sample
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

        # Only trace accept ratio to minimize per-step overhead.
        sampler_states = mcmc.sample_chain(
            mc_steps,
            current_state=initial_guess,
            kernel=kernel,
            num_burnin_steps=num_burnin_steps,
            seed=sample_key,
        )

        return sampler_states
