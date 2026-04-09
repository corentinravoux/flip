import abc


class BaseSimulator(abc.ABC):
    def __init__(self):
        pass

    @abc.abstractmethod
    def give_density_velocity_fields(self, **kwargs):
        """Return density and velocity fields.

        Returns:
            tuple: (density_field, velocity_field).
        """
        pass

    @abc.abstractmethod
    def give_density_velocity_fields_from_delta_fourier(
        self,
        delta_fourier,
        parameter_values_dict,
    ):
        """Return density and velocity fields from delta fourier.

        Returns:
            tuple: (density_field, velocity_field).
        """
        pass


def return_simulator(model_name, **kwargs):
    """Factory function to return a simulator instance based on the specified model.

    Args:
        model (str): Name of the model to use for the simulator. Supported values are:
            - 'flox': Uses the FourierBox-based simulator from flip.forward.flox.box.
            - 'gaussian': Uses a Gaussian random field simulator from flip.forward.flox.gaussian_box.
        **kwargs: Additional keyword arguments to pass to the simulator constructor."""

    # CR - the flox simulator might be removed.
    if model_name == "flox":
        from flip.forward.flox.box import CosmoBox

        return CosmoBox(**kwargs)

    elif model_name == "gaussian":
        from flip.forward.simulation.gaussian.field_model import GaussianRandomFieldBox

        return GaussianRandomFieldBox(**kwargs)

    else:
        raise ValueError(
            f"Unsupported model '{model_name}'. Supported models are 'flox' and 'gaussian'."
        )
