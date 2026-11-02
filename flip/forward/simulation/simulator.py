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

    if model_name == "gaussian":
        from flip.forward.simulation.gaussian.field_model import GaussianRandomFieldBox

        return GaussianRandomFieldBox(**kwargs)

    else:
        raise ValueError(
            f"Unsupported model '{model_name}'. Supported model is 'gaussian'."
        )
