"""Init file of the flip.data package."""

def load_velocity_data():
    import pandas as pd

    from flip import __flip_dir_path__

    return pd.read_parquet(__flip_dir_path__ + "/data/velocity_data.parquet")


def load_density_data():
    import pandas as pd

    from flip import __flip_dir_path__

    return pd.read_parquet(__flip_dir_path__ + "/data/density_data.parquet")


def load_power_spectrum(kind="mm"):
    import numpy as np

    from flip import __flip_dir_path__

    return np.loadtxt(__flip_dir_path__ + f"/data/power_spectrum_{kind}.txt")
