import numpy as np


def K_vv_0_0(theta, phi):
    def func(k):
        return (1 / 3) * np.cos(theta) / k**2

    return func


def K_vv_0_2(theta, phi):
    def func(k):
        return (1 / 6) * (3 * np.cos(2 * phi) + np.cos(theta)) / k**2

    return func
