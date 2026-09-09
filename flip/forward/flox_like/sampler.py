import jax
import jax.numpy as jnp
import numpy as np
from flip.forward.flox_like import probabilities


def spherical_to_cartesian(r, ra, dec):
    """Converts dist-radec [r,ra,dec] to cartersian [x, y, z] coordinates
    (in degrees).

    Parameters
    ----------
    vec: array
        r, ra, dec

    Returns
    -------
    array
        [x, y, z]
    """
    theta = jnp.pi / 2 - dec * jnp.pi / 180
    phi = ra * jnp.pi / 180

    x = r * jnp.sin(theta) * jnp.cos(phi)
    y = r * jnp.sin(theta) * jnp.sin(phi)
    z = r * jnp.cos(theta)

    return jnp.stack([x, y, z]).T


def get_unity_3dcoords(ra, dec):
    """"""
    return spherical_to_cartesian(jnp.ones_like(ra), ra, dec)


class Sampler(object):

    def __init__(self, simulator, cosmo, data, sampling_parameters):

        self.simulator = simulator
        self.cosmo = cosmo
        self._data = data
        self._sampling_parameters = sampling_parameters

    def set_parameters(
        self, kmax=0.1, deltak_true=None, dist_mpch_true=None, use_density=True
    ):
        """ """

        self.kmaxindex = np.where(self.simulator.wavenumber_norm_squared <= kmax**2)
        # place holder which mode may be fitted
        self.deltak_sampling = jnp.zeros(
            shape=(
                self.simulator.number_bins,
                self.simulator.number_bins,
                self.simulator.number_bins // 2 + 1,
            ),
            dtype="complex64",
        )

        if deltak_true is not None:
            modes_real = deltak_true[self.kmaxindex].real
            modes_imag = deltak_true[self.kmaxindex].imag
        else:
            modes_real = None
            modes_imag = None

        self.r1d = (
            jnp.linspace(0, self.simulator.box_size, self.simulator.number_bins)
            - self.simulator.box_size / 2
        )
        self.dist_mpch_vec = get_unity_3dcoords(
            self.data["ra"].values, self.data["dec"].values
        )

        self.targets_voxel_dir = self.simulator.get_voxels_in_direction(
            self.data["ra"].values,
            self.data["dec"].values,
            dist_range=[0, 200],
            physical_unit=True,
            unique=False,
        )
        self.dist_mpch_los = self.simulator._dist_mpch[
            self.targets_voxel_dir[:, 0],
            self.targets_voxel_dir[:, 1],
            self.targets_voxel_dir[:, 2],
        ]

        # Data
        self._data_mag = jnp.asarray(self.data["mag"].values, dtype="float32")
        self._data_mag_err = jnp.asarray(self.data["mag_err"], dtype="float32")

        self._data_redshift = jnp.asarray(self.data["redshift"].values, dtype="float32")
        self._data_redshift_err = jnp.asarray(
            self.data["redshift_err"].values, dtype="float32"
        )

        parameters = {
            "modes_real": modes_real,
            "modes_imag": modes_imag,
            "dist_mpch": dist_mpch_true,
            "h": self.cosmo.h,
            "use_density": use_density,
        }

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
            "number_bins": self.number_bins,
            "kmaxindex": self.kmaxindex,
            "deltak_sampling": self.deltak_sampling,
            # density sampling
            "dist_mpch_los": self.dist_mpch_los,
            "targets_voxel_dir": self.targets_voxel_dir,
        }

        # compute log-probability
        loglikelihood = probabilities.get_logprob(
            data_candles=data_candles,
            # cosmology H0
            box_struct=box_struct,
            #
            **(self.parameters | param),
        )
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
        import tensorflow_probability.substrates.jax.mcmc as mcmc
        from tensorflow_probability.substrates.jax.mcmc import NoUTurnSampler as NUTS

        print(f"num_burnin_steps = {num_burnin_steps}")
        print(f"mc_steps = {mc_steps}")

        # NUTS
        kernel = NUTS(self.target_logprob_fn, step_size=step_size)

        num_adaptation_steps = num_burnin_steps * 0.8
        kernel = mcmc.DualAveragingStepSizeAdaptation(
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
