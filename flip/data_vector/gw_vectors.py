from . import mesh
from .basic import Dens


class GWDensMesh(Dens):
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

        grid = mesh.grid_data_density_kernel_sampling(
            data_position_sky_kernel,
            rcom_max,
            grid_size,
            grid_type,
            kind,
            **kwargs,
        )
        return cls(grid)
