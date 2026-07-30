import abc
import copy
import importlib

import numpy as np

try:
    from flip.covariance import CovMatrix
except ImportError:
    CovMatrix = None
    pass


from flip.data_vector import mesh
from flip.utils import create_log

from .._config import __use_jax__
from . import vector_utils

if __use_jax__:
    try:
        import jax.numpy as jnp
        from jax import jit, random
        from jax.experimental.sparse import BCOO

        jax_installed = True

    except ImportError:
        import numpy as jnp

        jax_installed = False
else:

    import numpy as jnp
    from numpy import random

    jax_installed = False

log = create_log()


class DataVector(abc.ABC):
    """Abstract base for data vectors used in fits.

    Provides common storage, key validation, optional JAX acceleration,
    covariance-aware masking, and covariance construction helpers.

    Attributes:
        _free_par (list[str]): Model parameters this vector depends on.
        _kind (str): One of "velocity", "density" or "cross".
    """

    _kind = ""  # 'velocity', 'density' or 'cross'
    _needed_keys = []
    _free_par = []
    _number_dimension_observation_covariance = 0
    _parameters_observation_covariance = []

    @property
    def conditional_free_par(self):
        """Conditional extra parameters required by this vector.

        Returns:
            list[str]: Parameter names required depending on data content.
        """
        return []

    @property
    def free_par(self):
        """All free parameters for this vector.

        Returns:
            list[str]: Base plus conditional parameters.
        """
        return self._free_par + self.conditional_free_par

    @property
    def kind(self):
        """Return the data vector type.

        Returns:
            str: "velocity", "density" or "cross".
        """
        return self._kind

    @property
    def conditional_needed_keys(self):
        """Keys conditionally required in input `data`.

        Returns:
            list[str]: Extra keys required depending on configuration.
        """
        return []

    @property
    def needed_keys(self):
        """All required keys for this data vector.

        Returns:
            list[str]: Static plus conditional keys.
        """
        return self._needed_keys + self.conditional_needed_keys

    @property
    def data(self):
        """Access the underlying data dictionary.

        Returns:
            dict: Data fields as arrays.
        """
        return self._data

    @abc.abstractmethod
    def give_data_and_variance(self, **kwargs):
        """Return data vector and its variance/covariance.

        Returns:
            tuple: (data_array, variance_or_cov).
        """
        pass

    def __init__(self, data, covariance_observation=None, **kwargs):
        """Initialize data vector with data and optional observation covariance.

        Args:
            data (dict): Mapping of required fields to arrays.
            covariance_observation (ndarray|None): Observation covariance matrix or None.
            **kwargs: Extra configuration for subclasses.
        """
        self._covariance_observation = covariance_observation
        self._check_keys(data)
        self._number_datapoints = len(data[self.needed_keys[0]])
        self.check_covariance_observation()
        self._data = copy.copy(data)
        self._kwargs = kwargs

        for k in self._data:
            self._data[k] = jnp.array(self._data[k])

        if jax_installed:
            self.give_data_and_variance_jit = jit(self.give_data_and_variance)

    def check_covariance_observation(self):
        """Validate the shape of a user-supplied observation covariance.

        Ensures ``covariance_observation`` is ``(D*N, D*N)`` where ``N`` is the
        number of data points and ``D`` the per-point dimension declared by the
        subclass (``_number_dimension_observation_covariance``). Does nothing when
        no observation covariance was provided.

        Raises:
            ValueError: If the covariance shape does not match ``D*N x D*N``.
        """
        if self._covariance_observation is not None:
            if self._covariance_observation.shape != (
                self._number_dimension_observation_covariance * self._number_datapoints,
                self._number_dimension_observation_covariance * self._number_datapoints,
            ):
                raise ValueError(
                    f"Observation covariance matrix should be {self._number_dimension_observation_covariance}N "
                    f"x {self._number_dimension_observation_covariance}N"
                )
            log.add(
                f"Loading observation covariance matrix, "
                f"expecting {self._parameters_observation_covariance} parameters."
            )

    def _check_keys(self, data):
        """Validate that `data` contains all required keys.

        Raises:
            ValueError: When a required key is missing.
        """
        for k in self.needed_keys:
            if k not in data:
                raise ValueError(f"{k} field is needed in data")

    def get_masked_data_and_cov(self, bool_mask):
        """Return masked data and corresponding masked observation covariance.

        Args:
            bool_mask (array-like): Boolean mask aligned with first data key length.

        Returns:
            tuple: (new_data_dict, new_cov) with covariance masked or None.

        Raises:
            ValueError: If mask length mismatches data length.
        """
        if len(bool_mask) != len(self.data[self.needed_keys[0]]):
            raise ValueError("Boolean mask does not align with data")
        new_data = {k: v[bool_mask] for k, v in self._data.items()}

        new_cov = None
        if self._covariance_observation is not None:
            new_cov = self._covariance_observation[np.ix_(bool_mask, bool_mask)]
        return new_data, new_cov

    def compute_covariance(self, model, power_spectrum_dict, **kwargs):
        """Build a `CovMatrix` for this vector and model.

        Args:
            model (str): Covariance model module under `flip.covariance`.
            power_spectrum_dict (dict): Power spectra inputs for model.
            **kwargs: Model-specific options.

        Returns:
            CovMatrix: Initialized covariance matrix object.
        """

        if CovMatrix is None:
            raise ImportError(
                "flip.covariance module is not loaded."
                " Cannot compute covariance without it."
                " Try 'import flip' to see the module needed for covariance."
            )
        coordinate_keys = importlib.import_module(
            f"flip.covariance.analytical.{model}"
        )._coordinate_keys

        coords = np.vstack([self.data[k] for k in coordinate_keys])

        return CovMatrix.init_from_flip(
            model,
            self._kind,
            power_spectrum_dict,
            **{f"coordinates_{self._kind}": coords},
            **kwargs,
        )


class Dens(DataVector):
    """Density-contrast data vector.

    Wraps a measured density field ``density`` with its per-point error
    ``density_error`` (or a full observation covariance). Used to build the
    density-density block of the covariance and the density likelihood.

    Required keys:
        ``density``, ``density_error``.
    """

    _kind = "density"
    _needed_keys = ["density", "density_error"]
    _free_par = []
    _number_dimension_observation_covariance = 1
    _parameters_observation_covariance = ["density"]

    def give_data_and_variance(self, *args):
        """Return density data and diagonal variance from `density_error`.

        Returns:
            tuple: (density, density_error^2).
        """

        if self._covariance_observation is not None:
            return self._data["density"], self._covariance_observation
        return self._data["density"], self._data["density_error"] ** 2

    def __init__(self, data, covariance_observation=None):
        """Initialize the density vector.

        Args:
            data (dict): Must contain ``density`` and ``density_error`` plus the
                coordinate keys required by the chosen covariance model.
            covariance_observation (ndarray|None): Optional full density
                covariance replacing the diagonal ``density_error**2``.
        """
        super().__init__(data, covariance_observation=covariance_observation)


class DirectVel(DataVector):
    """Peculiar-velocity data vector from directly measured velocities.

    Wraps line-of-sight peculiar velocities ``velocity`` (km/s) with their error
    ``velocity_error`` (or a full observation covariance). If the data contain a
    ``host_group_id`` column, velocities sharing a host are averaged through a
    host matrix (see :func:`flip.data_vector.vector_utils.compute_host_matrix`),
    which is important for e.g. multiple SNe Ia in the same galaxy.

    Required keys:
        ``velocity`` (and ``velocity_error`` when no observation covariance is
        provided).
    """

    _kind = "velocity"
    _needed_keys = ["velocity"]
    _free_par = []
    _number_dimension_observation_covariance = 1
    _parameters_observation_covariance = ["velocity"]

    @property
    def conditional_needed_keys(self):
        """Add ``velocity_error`` to the required keys when no covariance is set.

        Returns:
            list[str]: ``["velocity_error"]`` if no observation covariance was
            supplied, otherwise an empty list.
        """
        cond_keys = []
        if self._covariance_observation is None:
            cond_keys += ["velocity_error"]
        return cond_keys

    def give_data_and_variance(self, *args):
        """Return velocities and their variance.

        Returns:
            tuple: ``(velocity, velocity_error**2)`` for the diagonal case, or
            ``(velocity, covariance_observation)`` when a full covariance is set.
        """
        if self._covariance_observation is not None:
            return self._data["velocity"], self._covariance_observation
        return self._data["velocity"], self._data["velocity_error"] ** 2

    def __init__(self, data, covariance_observation=None):
        """Initialize the velocity vector, optionally grouping shared hosts.

        Args:
            data (dict): Must contain ``velocity`` (and ``velocity_error`` when no
                observation covariance is given) plus the coordinate keys required
                by the covariance model. An optional ``host_group_id`` triggers
                host averaging of velocities sharing the same host.
            covariance_observation (ndarray|None): Optional full velocity
                covariance replacing the diagonal ``velocity_error**2``.
        """
        super().__init__(data, covariance_observation=covariance_observation)

        if "host_group_id" in self._data:
            # Copy full length velocities and velocity errors
            self._data["velocity_full"] = copy.copy(self._data["velocity"])

            # Init host matrix
            self._host_matrix, self._data_to_group_mapping = (
                vector_utils.compute_host_matrix(self._data["host_group_id"])
            )
            self._data = vector_utils.format_data_multiple_host(
                self._data, self._host_matrix
            )

            if jax_installed:
                self._host_matrix = BCOO.from_scipy_sparse(self._host_matrix)

            if self._covariance_observation is None:
                self._data["velocity_error_full"] = copy.copy(
                    self._data["velocity_error"]
                )
                velocity_variance = self._data["velocity_error"] ** 2
            else:
                velocity_variance = self._covariance_observation

            self._data["velocity"], velocity_variance = (
                vector_utils.get_grouped_data_variance(
                    self._host_matrix, self._data["velocity"], velocity_variance
                )
            )

            if self._covariance_observation is None:
                self._data["velocity_error"] = jnp.sqrt(velocity_variance)
            else:
                self._covariance_observation = velocity_variance


class DensMesh(Dens):
    """Density vector defined on a regular mesh built from a sky catalog.

    Same data/variance behaviour as :class:`Dens`, but adds
    :meth:`init_from_catalog` to grid an object catalog into a density field on a
    Cartesian mesh via :func:`flip.data_vector.mesh.grid_data_density`.
    """

    _kind = "density"
    _needed_keys = ["density", "density_error"]
    _free_par = []
    _number_dimension_observation_covariance = 1
    _parameters_observation_covariance = ["density"]

    def give_data_and_variance(self, *args):
        """Return density data and diagonal variance from `density_error`.

        Returns:
            tuple: (density, density_error^2).
        """

        if self._covariance_observation is not None:
            return self._data["density"], self._covariance_observation
        return self._data["density"], self._data["density_error"] ** 2

    def __init__(self, data, covariance_observation=None):
        """Initialize from a pre-gridded density dictionary (see :class:`Dens`)."""
        super().__init__(data, covariance_observation=covariance_observation)

    @classmethod
    def init_from_catalog(
        cls,
        data_position_sky,
        rcom_max,
        grid_size,
        grid_type,
        kind,
        **kwargs,
    ):
        """Build a :class:`DensMesh` by gridding a sky catalog onto a mesh.

        Args:
            data_position_sky (dict): Sky positions (and comoving distances) of
                the objects to grid.
            rcom_max (float): Comoving half-size of the box (Mpc/h).
            grid_size (float): Cell size of the mesh (Mpc/h).
            grid_type (str): Mesh geometry passed to
                :func:`flip.data_vector.mesh.grid_data_density`.
            kind (str): Density estimator / normalization convention.
            **kwargs: Extra options forwarded to the mesh gridder.

        Returns:
            DensMesh: Vector wrapping the gridded density and its error.
        """
        grid = mesh.grid_data_density(
            data_position_sky,
            rcom_max,
            grid_size,
            grid_type,
            kind,
            **kwargs,
        )

        return cls(grid)


class DirectVelMesh(DirectVel):
    """Velocity vector defined on a regular mesh built from a sky catalog.

    Same data/variance behaviour as :class:`DirectVel`, but adds
    :meth:`init_from_catalog` to grid a velocity catalog into a momentum/velocity
    field on a Cartesian mesh via
    :func:`flip.data_vector.mesh.grid_data_velocity`.
    """

    _kind = "velocity"
    _needed_keys = ["velocity", "velocity_error"]
    _free_par = []
    _number_dimension_observation_covariance = 1
    _parameters_observation_covariance = ["velocity"]

    def give_data_and_variance(self, *args):
        """Return velocity data and diagonal variance from `velocity_error`.

        Returns:
            tuple: (velocity, velocity_error^2).
        """

        if self._covariance_observation is not None:
            return self._data["velocity"], self._covariance_observation
        return self._data["velocity"], self._data["velocity_error"] ** 2

    def __init__(self, data, covariance_observation=None):
        """Initialize from a pre-gridded velocity dictionary (see :class:`DirectVel`)."""
        super().__init__(data, covariance_observation=covariance_observation)

    @classmethod
    def init_from_catalog(
        cls,
        data_position_sky,
        data,
        rcom_max,
        grid_size,
        grid_type,
        kind,
        **kwargs,
    ):
        """Build a :class:`DirectVelMesh` by gridding a velocity catalog.

        Args:
            data_position_sky (dict): Sky positions (and comoving distances) of
                the objects to grid.
            data (dict): Must contain ``velocity`` and ``velocity_error`` for the
                objects, used to grid the velocity field and its variance.
            rcom_max (float): Comoving half-size of the box (Mpc/h).
            grid_size (float): Cell size of the mesh (Mpc/h).
            grid_type (str): Mesh geometry passed to
                :func:`flip.data_vector.mesh.grid_data_velocity`.
            kind (str): Velocity estimator / normalization convention.
            **kwargs: Extra options forwarded to the mesh gridder.

        Returns:
            DirectVelMesh: Vector wrapping the gridded velocity and its error.
        """
        grid_velocity = mesh.grid_data_velocity(
            data_position_sky,
            rcom_max,
            grid_size,
            grid_type,
            kind,
            data["velocity_error"] ** 2,
            velocity=data["velocity"],
            **kwargs,
        )

        return cls(grid_velocity)


class VelFromHDres(DataVector):
    """Peculiar velocities derived from Hubble-diagram residuals.

    Converts distance-modulus residuals ``dmu`` (:math:`\\Delta\\mu`) into
    peculiar velocities using a redshift-dependent estimator
    :math:`\\hat{v} = J(z)\\,(\\Delta\\mu - M_0)`, where the coefficient
    :math:`J(z)` is selected by ``velocity_estimator`` (see
    :func:`flip.data_vector.vector_utils.redshift_dependence_velocity` and the
    "Velocity estimators" documentation page). ``M_0`` is a free zero-point
    parameter of the fit.

    Required keys:
        ``dmu``, ``zobs`` (and ``dmu_error`` when no observation covariance is
        provided).
    """

    _kind = "velocity"
    _needed_keys = ["dmu", "zobs"]
    _free_par = ["M_0"]
    _number_dimension_observation_covariance = 1
    _parameters_observation_covariance = ["dmu"]

    @property
    def conditional_needed_keys(self):
        """Add ``dmu_error`` to the required keys when no covariance is set.

        Returns:
            list[str]: Base keys plus ``["dmu_error"]`` when no observation
            covariance was supplied.
        """
        cond_keys = []
        if self._covariance_observation is None:
            cond_keys += ["dmu_error"]
        return self._needed_keys + cond_keys

    def give_data_and_variance(self, parameter_values_dict):
        """Convert HD residuals into velocities and propagate the variance.

        Args:
            parameter_values_dict (dict): Fit parameters; must include ``M_0``
                and any parameters required by the chosen velocity estimator
                (e.g. ``q0``, ``j0`` for the Hubble high-order estimator).

        Returns:
            tuple: ``(velocity, velocity_variance)`` where ``velocity_variance``
            is diagonal (:math:`(J\\,\\sigma_{\\Delta\\mu})^2`) or the full
            :math:`J\\,C_{\\Delta\\mu}\\,J^T` when an observation covariance is set.
        """
        distance_modulus_difference_to_velocity = (
            vector_utils.redshift_dependence_velocity(
                self._data, self.velocity_estimator, **parameter_values_dict
            )
        )
        velocity = (
            distance_modulus_difference_to_velocity * self._data["dmu"]
            - distance_modulus_difference_to_velocity * parameter_values_dict["M_0"]
        )

        if self._covariance_observation is None:
            velocity_variance = (
                distance_modulus_difference_to_velocity * self._data["dmu_error"]
            ) ** 2
        else:
            velocity_variance = (
                distance_modulus_difference_to_velocity[:, None]
                * self._covariance_observation
                * distance_modulus_difference_to_velocity[None, :]
            )

        return velocity, velocity_variance

    def __init__(
        self, data, covariance_observation=None, velocity_estimator="full", **kwargs
    ):
        """Initialize the HD-residual velocity vector.

        Args:
            data (dict): Must contain ``dmu`` and ``zobs`` (and ``dmu_error`` when
                no observation covariance is given), plus any fields required by
                the estimator (e.g. ``hubble_norm``, ``rcom_zobs`` for ``"full"``).
            covariance_observation (ndarray|None): Optional full covariance of the
                distance-modulus residuals.
            velocity_estimator (str): Estimator name selecting :math:`J(z)`; one of
                ``"full"``, ``"watkins"``, ``"lowz"``, ``"hubble highorder"``.
            **kwargs: Extra options forwarded to the base class.
        """
        self.velocity_estimator = velocity_estimator

        super().__init__(data, covariance_observation=covariance_observation)


class VelFromIntrinsicScatter(DataVector):
    """Velocity contribution from the intrinsic magnitude scatter.

    Models the peculiar-velocity noise induced by the intrinsic magnitude
    scatter :math:`\\sigma_M` of standard candles: a random distance modulus of
    dispersion ``sigma_M`` mapped to velocity through :math:`J(z)`. The mean is a
    random realization; the variance is :math:`J(z)^2\\,\\sigma_M^2`. Useful as an
    additive velocity term whose amplitude ``sigma_M`` is fitted.

    Required keys:
        ``zobs``.
    """

    _kind = "velocity"
    _needed_keys = ["zobs"]
    _free_par = ["sigma_M"]
    _number_dimension_observation_covariance = 0
    _parameters_observation_covariance = []

    def give_data_and_variance(self, parameter_values_dict):
        """Return the scatter-induced velocity realization and its variance.

        Args:
            parameter_values_dict (dict): Must include ``sigma_M`` and any
                parameters required by the chosen velocity estimator.

        Returns:
            tuple: ``(velocity, velocity_variance)`` with variance
            :math:`J(z)^2\\,\\sigma_M^2`.
        """
        distance_modulus_difference_to_velocity = (
            vector_utils.redshift_dependence_velocity(
                self._data, self.velocity_estimator, **parameter_values_dict
            )
        )
        if jax_installed:
            key = random.PRNGKey(0)
            distance_modulus = parameter_values_dict["sigma_M"] * random.normal(
                key, (len(self._data["zobs"]),)
            )
        else:
            distance_modulus = random.normal(
                loc=0.0,
                scale=parameter_values_dict["sigma_M"],
                size=len(self._data["zobs"]),
            )

        variance = parameter_values_dict["sigma_M"] ** 2

        return (
            distance_modulus_difference_to_velocity * distance_modulus,
            distance_modulus_difference_to_velocity**2 * variance,
        )

    def __init__(self, data, velocity_estimator="full"):
        """Initialize the intrinsic-scatter velocity vector.

        Args:
            data (dict): Must contain ``zobs`` plus any fields required by the
                estimator.
            velocity_estimator (str): Estimator name selecting :math:`J(z)`.
        """
        super().__init__(data)
        self.velocity_estimator = velocity_estimator


class DensVel(DataVector):
    """Joint density x velocity (cross) data vector.

    Concatenates a density vector and a velocity vector so a single fit can use
    the density-density, velocity-velocity and density-velocity cross covariance
    blocks. Observation covariance on the velocity side combined with density is
    not yet supported.
    """

    _kind = "cross"

    @property
    def needed_keys(self):
        """Union of the required keys of the density and velocity sub-vectors."""
        return self.densities.needed_keys + self.velocities.needed_keys

    @property
    def free_par(self):
        """Concatenated free parameters of the density and velocity sub-vectors."""
        return self.densities.free_par + self.velocities.free_par

    def give_data_and_variance(self, *args):
        """Stack density and velocity data and their (diagonal) variances.

        Returns:
            tuple: ``(data, variance)`` with density entries first, then velocity.
        """
        data_density, density_variance = self.densities.give_data_and_variance(*args)
        data_velocity, velocity_variance = self.velocities.give_data_and_variance(*args)
        data = jnp.hstack((data_density, data_velocity))
        variance = jnp.hstack((density_variance, velocity_variance))
        return data, variance

    def __init__(self, density_vector, velocity_vector):
        """Initialize from an existing density and velocity vector.

        Args:
            density_vector (DataVector): A density-kind vector (e.g. :class:`Dens`).
            velocity_vector (DataVector): A velocity-kind vector (e.g.
                :class:`DirectVel`).

        Raises:
            NotImplementedError: If the velocity vector carries a full observation
                covariance (not yet supported in the cross case).
        """
        self.densities = density_vector
        self.velocities = velocity_vector

        if self.velocities._covariance_observation is not None:
            raise NotImplementedError(
                "Velocity with observed covariance + density not implemented yet"
            )

        if jax_installed:
            self.give_data_and_variance_jit = jit(self.give_data_and_variance)

    def compute_covariance(self, model, power_spectrum_dict, **kwargs):
        """Build the full density+velocity :class:`CovMatrix` for this pair.

        Extracts ``(ra, dec, rcom_zobs)`` coordinates for both the density and
        velocity sub-vectors and initializes the ``"full"`` covariance (all three
        blocks) for the requested model.

        Args:
            model (str): Covariance model name under ``flip.covariance.analytical``.
            power_spectrum_dict (dict): Power spectra inputs for the model.
            **kwargs: Model-specific options.

        Returns:
            CovMatrix: Full cross covariance object.

        Raises:
            ImportError: If ``flip.covariance`` could not be imported.
        """
        if CovMatrix is None:
            raise ImportError(
                "flip.covariance module is not loaded."
                " Cannot compute covariance without it."
                " Try 'import flip' to see the module needed for covariance."
            )

        coords_dens = np.vstack(
            (
                self.densities.data["ra"],
                self.densities.data["dec"],
                self.densities.data["rcom_zobs"],
            )
        )

        coords_vel = np.vstack(
            (
                self.velocities.data["ra"],
                self.velocities.data["dec"],
                self.velocities.data["rcom_zobs"],
            )
        )
        return CovMatrix.init_from_flip(
            model,
            "full",
            power_spectrum_dict,
            coordinates_density=coords_dens,
            coordinates_velocity=coords_vel,
            **kwargs,
        )
