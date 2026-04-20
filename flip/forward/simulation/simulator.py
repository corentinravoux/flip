import abc

# CR - improve the base simulator and the loading of the several ones.


class BaseSimulator(abc.ABC):
    def __init__(self):
        pass

    @abc.abstractmethod
    def sample_density_velocity_fields(
        self,
        **kwargs,
    ):
        pass

    @abc.abstractmethod
    def sample_density_velocity_fields_from_modes(
        self,
        **kwargs,
    ):
        pass


def return_simulator(model_name, **kwargs):

    # CR - the flox simulator will be removed.
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
