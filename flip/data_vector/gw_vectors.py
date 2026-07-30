"""Gravitational-wave density vectors on a mesh.

Density data vectors built from gravitational-wave host localizations, where
each event contributes a probability kernel over sky position and distance
rather than a single point. The kernels are gridded onto a Cartesian mesh via
:mod:`flip.data_vector.mesh`.
"""

from . import mesh
from .basic import Dens


class GWDensMesh(Dens):
    """Density vector from gravitational-wave localization kernels on a mesh.

    Same data/variance behaviour as :class:`flip.data_vector.basic.Dens`, but
    provides constructors that grid GW host localization kernels into a density
    field: either by convolving multivariate kernels
    (:meth:`init_from_multivariate_kernel`) or by Monte-Carlo sampling the
    kernels (:meth:`init_from_kernel_sampling`).
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
        """Initialize from a pre-gridded density dictionary (see :class:`~flip.data_vector.basic.Dens`)."""
        super().__init__(data, covariance_observation=covariance_observation)

    @classmethod
    def init_from_multivariate_kernel(
        cls,
        data_position_sky,
        data_position_sky_kernel_properties,
        rcom_max,
        grid_size,
        grid_type,
        kind,
        **kwargs,
    ):
        """Build a :class:`GWDensMesh` by gridding multivariate localization kernels.

        Args:
            data_position_sky (dict): Central sky positions of the GW events.
            data_position_sky_kernel_properties (dict): Kernel parameters (widths,
                correlations) describing each event's localization uncertainty.
            rcom_max (float): Comoving half-size of the box (Mpc/h).
            grid_size (float): Cell size of the mesh (Mpc/h).
            grid_type (str): Mesh geometry passed to the gridder.
            kind (str): Density estimator / normalization convention.
            **kwargs: Extra options forwarded to
                :func:`flip.data_vector.mesh.grid_data_density_multivariate_kernel`.

        Returns:
            GWDensMesh: Vector wrapping the gridded density and its error.
        """
        grid = mesh.grid_data_density_multivariate_kernel(
            data_position_sky,
            data_position_sky_kernel_properties,
            rcom_max,
            grid_size,
            grid_type,
            kind,
            **kwargs,
        )
        return cls(grid)

    @classmethod
    def init_from_kernel_sampling(
        cls,
        data_position_sky_kernel,
        rcom_max,
        grid_size,
        grid_type,
        kind,
        **kwargs,
    ):
        """Build a :class:`GWDensMesh` by Monte-Carlo sampling localization kernels.

        Args:
            data_position_sky_kernel (dict): Per-event samples drawn from each GW
                localization kernel (sky position and comoving distance).
            rcom_max (float): Comoving half-size of the box (Mpc/h).
            grid_size (float): Cell size of the mesh (Mpc/h).
            grid_type (str): Mesh geometry passed to the gridder.
            kind (str): Density estimator / normalization convention.
            **kwargs: Extra options forwarded to
                :func:`flip.data_vector.mesh.grid_data_density_kernel_sampling`.

        Returns:
            GWDensMesh: Vector wrapping the gridded density and its error.
        """
        grid = mesh.grid_data_density_kernel_sampling(
            data_position_sky_kernel,
            rcom_max,
            grid_size,
            grid_type,
            kind,
            **kwargs,
        )
        return cls(grid)
