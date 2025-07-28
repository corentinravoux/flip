"""Init file of the flip.covariance package."""
import os
import importlib

__all__ = []
__flip_covariance_models_path__ = os.path.dirname(__file__)

# List available models
__available_models__ = []
for directory in os.listdir(__flip_covariance_models_path__):
    model_directory_path = os.path.join(__flip_covariance_models_path__, directory)
    model_init_file = os.path.join(model_directory_path, '__init__.py')

    if os.path.isdir(model_directory_path) and os.path.isfile(model_init_file):
        __available_models__.append(f"{directory}")

# Perform relative import of the models from the covariance package
for model in __available_models__:
    importlib.import_module(f".{model}", __package__)
    __all__.append(model)   

from .covariance import CovMatrix

