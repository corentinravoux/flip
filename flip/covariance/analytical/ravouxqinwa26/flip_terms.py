import numpy
import scipy


def set_backend(module):
    global np, erf
    if module == "numpy":
        np = numpy
        erf = scipy.special.erf
    elif module == "mpmath":
        import mpmath

        np = mpmath.mp
        erf = mpmath.erf


set_backend("numpy")


def M_gg_0_0_0():
    def func(k):
        return 1

    return func


def N_gg_0_0_0(theta, phi):
    return 1


def M_gg_1_0_0():
    def func(k):
        return 2

    return func


def N_gg_1_0_0(theta, phi):
    return 1


def M_gg_2_0_0():
    def func(k):
        return 2

    return func


def N_gg_2_0_0(theta, phi):
    return 1


def M_gg_3_0_0():
    def func(k):
        return 2

    return func


def N_gg_3_0_0(theta, phi):
    return 1


def M_gg_4_0_0():
    def func(k):
        return 6 * k**2

    return func


def N_gg_4_0_0(theta, phi):
    return 1


def M_gg_5_0_0():
    def func(k):
        return 1 / 2

    return func


def N_gg_5_0_0(theta, phi):
    return 1


def M_gg_6_0_0():
    def func(k):
        return 1 / 2

    return func


def N_gg_6_0_0(theta, phi):
    return 1


def M_gg_7_0_0():
    def func(k):
        return 1

    return func


def N_gg_7_0_0(theta, phi):
    return 1


def M_gg_8_0_0():
    def func(k):
        return 2

    return func


def N_gg_8_0_0(theta, phi):
    return 1


def M_gg_9_0_0():
    def func(k):
        return 2 / 3

    return func


def N_gg_9_0_0(theta, phi):
    return 1


def M_gg_9_2_0():
    def func(k):
        return 2 / 15

    return func


def N_gg_9_2_0(theta, phi):
    return (15 / 4) * np.cos(2 * phi + theta) + 5 / 4


def M_gg_9_2_1():
    def func(k):
        return 2 / 15

    return func


def N_gg_9_2_1(theta, phi):
    return (15 / 4) * np.cos(2 * phi - theta) + 5 / 4


def M_gg_10_0_0():
    def func(k):
        return 4 / 3

    return func


def N_gg_10_0_0(theta, phi):
    return 1


def M_gg_10_2_0():
    def func(k):
        return 4 / 15

    return func


def N_gg_10_2_0(theta, phi):
    return (15 / 4) * np.cos(2 * phi + theta) + 5 / 4


def M_gg_10_2_1():
    def func(k):
        return 4 / 15

    return func


def N_gg_10_2_1(theta, phi):
    return (15 / 4) * np.cos(2 * phi - theta) + 5 / 4


def M_gg_11_0_0():
    def func(k):
        return 4 / 3

    return func


def N_gg_11_0_0(theta, phi):
    return 1


def M_gg_11_2_0():
    def func(k):
        return 4 / 15

    return func


def N_gg_11_2_0(theta, phi):
    return (15 / 4) * np.cos(2 * phi + theta) + 5 / 4


def M_gg_11_2_1():
    def func(k):
        return 4 / 15

    return func


def N_gg_11_2_1(theta, phi):
    return (15 / 4) * np.cos(2 * phi - theta) + 5 / 4


def M_gg_12_0_0():
    def func(k):
        return 4 * k**2

    return func


def N_gg_12_0_0(theta, phi):
    return 1


def M_gg_12_2_0():
    def func(k):
        return (4 / 5) * k**2

    return func


def N_gg_12_2_0(theta, phi):
    return (15 / 4) * np.cos(2 * phi + theta) + 5 / 4


def M_gg_12_2_1():
    def func(k):
        return (4 / 5) * k**2

    return func


def N_gg_12_2_1(theta, phi):
    return (15 / 4) * np.cos(2 * phi - theta) + 5 / 4


def M_gg_13_0_0():
    def func(k):
        return 4 * k**2

    return func


def N_gg_13_0_0(theta, phi):
    return 1


def M_gg_13_2_0():
    def func(k):
        return (4 / 5) * k**2

    return func


def N_gg_13_2_0(theta, phi):
    return (15 / 4) * np.cos(2 * phi + theta) + 5 / 4


def M_gg_13_2_1():
    def func(k):
        return (4 / 5) * k**2

    return func


def N_gg_13_2_1(theta, phi):
    return (15 / 4) * np.cos(2 * phi - theta) + 5 / 4


def M_gg_14_0_0():
    def func(k):
        return -2 / 3

    return func


def N_gg_14_0_0(theta, phi):
    return 1


def M_gg_14_2_0():
    def func(k):
        return -2 / 15

    return func


def N_gg_14_2_0(theta, phi):
    return (15 / 4) * np.cos(2 * phi + theta) + 5 / 4


def M_gg_14_2_1():
    def func(k):
        return -2 / 15

    return func


def N_gg_14_2_1(theta, phi):
    return (15 / 4) * np.cos(2 * phi - theta) + 5 / 4


def M_gg_15_0_0():
    def func(k):
        return -2 / 3

    return func


def N_gg_15_0_0(theta, phi):
    return 1


def M_gg_15_2_0():
    def func(k):
        return -2 / 15

    return func


def N_gg_15_2_0(theta, phi):
    return (15 / 4) * np.cos(2 * phi + theta) + 5 / 4


def M_gg_15_2_1():
    def func(k):
        return -2 / 15

    return func


def N_gg_15_2_1(theta, phi):
    return (15 / 4) * np.cos(2 * phi - theta) + 5 / 4


def M_gg_16_0_0():
    def func(k):
        return -2 / 3

    return func


def N_gg_16_0_0(theta, phi):
    return 1


def M_gg_16_2_0():
    def func(k):
        return -2 / 15

    return func


def N_gg_16_2_0(theta, phi):
    return (15 / 4) * np.cos(2 * phi + theta) + 5 / 4


def M_gg_16_2_1():
    def func(k):
        return -2 / 15

    return func


def N_gg_16_2_1(theta, phi):
    return (15 / 4) * np.cos(2 * phi - theta) + 5 / 4


def M_gg_17_0_0():
    def func(k):
        return -2 / 3

    return func


def N_gg_17_0_0(theta, phi):
    return 1


def M_gg_17_2_0():
    def func(k):
        return -2 / 15

    return func


def N_gg_17_2_0(theta, phi):
    return (15 / 4) * np.cos(2 * phi + theta) + 5 / 4


def M_gg_17_2_1():
    def func(k):
        return -2 / 15

    return func


def N_gg_17_2_1(theta, phi):
    return (15 / 4) * np.cos(2 * phi - theta) + 5 / 4


def M_gg_18_0_0():
    def func(k):
        return -2 / 3

    return func


def N_gg_18_0_0(theta, phi):
    return 1


def M_gg_18_2_0():
    def func(k):
        return -2 / 15

    return func


def N_gg_18_2_0(theta, phi):
    return (15 / 4) * np.cos(2 * phi + theta) + 5 / 4


def M_gg_18_2_1():
    def func(k):
        return -2 / 15

    return func


def N_gg_18_2_1(theta, phi):
    return (15 / 4) * np.cos(2 * phi - theta) + 5 / 4


def M_gg_19_0_0():
    def func(k):
        return 1 / 3

    return func


def N_gg_19_0_0(theta, phi):
    return 1


def M_gg_19_2_0():
    def func(k):
        return 1 / 15

    return func


def N_gg_19_2_0(theta, phi):
    return (15 / 4) * np.cos(2 * phi + theta) + 5 / 4


def M_gg_19_2_1():
    def func(k):
        return 1 / 15

    return func


def N_gg_19_2_1(theta, phi):
    return (15 / 4) * np.cos(2 * phi - theta) + 5 / 4


def M_gg_20_0_0():
    def func(k):
        return 1 / 5

    return func


def N_gg_20_0_0(theta, phi):
    return 1


def M_gg_20_2_0():
    def func(k):
        return 2 / 35

    return func


def N_gg_20_2_0(theta, phi):
    return (15 / 4) * np.cos(2 * phi + theta) + 5 / 4


def M_gg_20_2_1():
    def func(k):
        return 2 / 35

    return func


def N_gg_20_2_1(theta, phi):
    return (15 / 4) * np.cos(2 * phi - theta) + 5 / 4


def M_gg_20_4_0():
    def func(k):
        return 4 / 315

    return func


def N_gg_20_4_0(theta, phi):
    return (
        (45 / 16) * np.cos(2 * phi + theta)
        + (315 / 64) * np.cos(4 * phi + 2 * theta)
        + 81 / 64
    )


def M_gg_20_4_1():
    def func(k):
        return 4 / 315

    return func


def N_gg_20_4_1(theta, phi):
    return (
        (45 / 16) * np.cos(2 * phi - theta)
        + (315 / 64) * np.cos(4 * phi - 2 * theta)
        + 81 / 64
    )


def M_gg_21_0_0():
    def func(k):
        return (2 / 3) * k**2

    return func


def N_gg_21_0_0(theta, phi):
    return 1


def M_gg_21_2_0():
    def func(k):
        return (2 / 15) * k**2

    return func


def N_gg_21_2_0(theta, phi):
    return (15 / 4) * np.cos(2 * phi + theta) + 5 / 4


def M_gg_21_2_1():
    def func(k):
        return (2 / 15) * k**2

    return func


def N_gg_21_2_1(theta, phi):
    return (15 / 4) * np.cos(2 * phi - theta) + 5 / 4


def M_gg_22_0_0():
    def func(k):
        return (2 / 5) * k**2

    return func


def N_gg_22_0_0(theta, phi):
    return 1


def M_gg_22_2_0():
    def func(k):
        return (4 / 35) * k**2

    return func


def N_gg_22_2_0(theta, phi):
    return (15 / 4) * np.cos(2 * phi + theta) + 5 / 4


def M_gg_22_2_1():
    def func(k):
        return (4 / 35) * k**2

    return func


def N_gg_22_2_1(theta, phi):
    return (15 / 4) * np.cos(2 * phi - theta) + 5 / 4


def M_gg_22_4_0():
    def func(k):
        return (8 / 315) * k**2

    return func


def N_gg_22_4_0(theta, phi):
    return (
        (45 / 16) * np.cos(2 * phi + theta)
        + (315 / 64) * np.cos(4 * phi + 2 * theta)
        + 81 / 64
    )


def M_gg_22_4_1():
    def func(k):
        return (8 / 315) * k**2

    return func


def N_gg_22_4_1(theta, phi):
    return (
        (45 / 16) * np.cos(2 * phi - theta)
        + (315 / 64) * np.cos(4 * phi - 2 * theta)
        + 81 / 64
    )


def M_gg_23_0_0():
    def func(k):
        return -1 / 3 * k**2

    return func


def N_gg_23_0_0(theta, phi):
    return 1


def M_gg_23_2_0():
    def func(k):
        return -1 / 15 * k**2

    return func


def N_gg_23_2_0(theta, phi):
    return (15 / 4) * np.cos(2 * phi + theta) + 5 / 4


def M_gg_23_2_1():
    def func(k):
        return -1 / 15 * k**2

    return func


def N_gg_23_2_1(theta, phi):
    return (15 / 4) * np.cos(2 * phi - theta) + 5 / 4


def M_gg_24_0_0():
    def func(k):
        return -2 / 3 * k**2

    return func


def N_gg_24_0_0(theta, phi):
    return 1


def M_gg_24_2_0():
    def func(k):
        return -2 / 15 * k**2

    return func


def N_gg_24_2_0(theta, phi):
    return (15 / 4) * np.cos(2 * phi + theta) + 5 / 4


def M_gg_24_2_1():
    def func(k):
        return -2 / 15 * k**2

    return func


def N_gg_24_2_1(theta, phi):
    return (15 / 4) * np.cos(2 * phi - theta) + 5 / 4


def M_gg_25_0_0():
    def func(k):
        return -2 / 3 * k**2

    return func


def N_gg_25_0_0(theta, phi):
    return 1


def M_gg_25_2_0():
    def func(k):
        return -2 / 15 * k**2

    return func


def N_gg_25_2_0(theta, phi):
    return (15 / 4) * np.cos(2 * phi + theta) + 5 / 4


def M_gg_25_2_1():
    def func(k):
        return -2 / 15 * k**2

    return func


def N_gg_25_2_1(theta, phi):
    return (15 / 4) * np.cos(2 * phi - theta) + 5 / 4


def M_gg_26_0_0():
    def func(k):
        return -2 / 3 * k**2

    return func


def N_gg_26_0_0(theta, phi):
    return 1


def M_gg_26_2_0():
    def func(k):
        return -2 / 15 * k**2

    return func


def N_gg_26_2_0(theta, phi):
    return (15 / 4) * np.cos(2 * phi + theta) + 5 / 4


def M_gg_26_2_1():
    def func(k):
        return -2 / 15 * k**2

    return func


def N_gg_26_2_1(theta, phi):
    return (15 / 4) * np.cos(2 * phi - theta) + 5 / 4


def M_gg_27_0_0():
    def func(k):
        return -2 * k**4

    return func


def N_gg_27_0_0(theta, phi):
    return 1


def M_gg_27_2_0():
    def func(k):
        return -2 / 5 * k**4

    return func


def N_gg_27_2_0(theta, phi):
    return (15 / 4) * np.cos(2 * phi + theta) + 5 / 4


def M_gg_27_2_1():
    def func(k):
        return -2 / 5 * k**4

    return func


def N_gg_27_2_1(theta, phi):
    return (15 / 4) * np.cos(2 * phi - theta) + 5 / 4


def M_gg_28_0_0():
    def func(k):
        return -1 / 6 * k**2

    return func


def N_gg_28_0_0(theta, phi):
    return 1


def M_gg_28_2_0():
    def func(k):
        return -1 / 30 * k**2

    return func


def N_gg_28_2_0(theta, phi):
    return (15 / 4) * np.cos(2 * phi + theta) + 5 / 4


def M_gg_28_2_1():
    def func(k):
        return -1 / 30 * k**2

    return func


def N_gg_28_2_1(theta, phi):
    return (15 / 4) * np.cos(2 * phi - theta) + 5 / 4


def M_gg_29_0_0():
    def func(k):
        return -1 / 6 * k**2

    return func


def N_gg_29_0_0(theta, phi):
    return 1


def M_gg_29_2_0():
    def func(k):
        return -1 / 30 * k**2

    return func


def N_gg_29_2_0(theta, phi):
    return (15 / 4) * np.cos(2 * phi + theta) + 5 / 4


def M_gg_29_2_1():
    def func(k):
        return -1 / 30 * k**2

    return func


def N_gg_29_2_1(theta, phi):
    return (15 / 4) * np.cos(2 * phi - theta) + 5 / 4


def M_gg_30_0_0():
    def func(k):
        return -1 / 3 * k**2

    return func


def N_gg_30_0_0(theta, phi):
    return 1


def M_gg_30_2_0():
    def func(k):
        return -1 / 15 * k**2

    return func


def N_gg_30_2_0(theta, phi):
    return (15 / 4) * np.cos(2 * phi + theta) + 5 / 4


def M_gg_30_2_1():
    def func(k):
        return -1 / 15 * k**2

    return func


def N_gg_30_2_1(theta, phi):
    return (15 / 4) * np.cos(2 * phi - theta) + 5 / 4


def M_gg_31_0_0():
    def func(k):
        return -2 / 3 * k**2

    return func


def N_gg_31_0_0(theta, phi):
    return 1


def M_gg_31_2_0():
    def func(k):
        return -2 / 15 * k**2

    return func


def N_gg_31_2_0(theta, phi):
    return (15 / 4) * np.cos(2 * phi + theta) + 5 / 4


def M_gg_31_2_1():
    def func(k):
        return -2 / 15 * k**2

    return func


def N_gg_31_2_1(theta, phi):
    return (15 / 4) * np.cos(2 * phi - theta) + 5 / 4


def M_gg_32_0_0():
    def func(k):
        return 1 / 3

    return func


def N_gg_32_0_0(theta, phi):
    return 1


def M_gg_32_2_0():
    def func(k):
        return 1 / 15

    return func


def N_gg_32_2_0(theta, phi):
    return (15 / 4) * np.cos(2 * phi + theta) + 5 / 4


def M_gg_32_2_1():
    def func(k):
        return 1 / 15

    return func


def N_gg_32_2_1(theta, phi):
    return (15 / 4) * np.cos(2 * phi - theta) + 5 / 4


def M_gg_33_0_0():
    def func(k):
        return 1 / 3

    return func


def N_gg_33_0_0(theta, phi):
    return 1


def M_gg_33_2_0():
    def func(k):
        return 1 / 15

    return func


def N_gg_33_2_0(theta, phi):
    return (15 / 4) * np.cos(2 * phi + theta) + 5 / 4


def M_gg_33_2_1():
    def func(k):
        return 1 / 15

    return func


def N_gg_33_2_1(theta, phi):
    return (15 / 4) * np.cos(2 * phi - theta) + 5 / 4


def M_gg_34_0_0():
    def func(k):
        return 1 / 5

    return func


def N_gg_34_0_0(theta, phi):
    return 1


def M_gg_34_2_0():
    def func(k):
        return 2 / 35

    return func


def N_gg_34_2_0(theta, phi):
    return (15 / 4) * np.cos(2 * phi + theta) + 5 / 4


def M_gg_34_2_1():
    def func(k):
        return 2 / 35

    return func


def N_gg_34_2_1(theta, phi):
    return (15 / 4) * np.cos(2 * phi - theta) + 5 / 4


def M_gg_34_4_0():
    def func(k):
        return 4 / 315

    return func


def N_gg_34_4_0(theta, phi):
    return (
        (45 / 16) * np.cos(2 * phi + theta)
        + (315 / 64) * np.cos(4 * phi + 2 * theta)
        + 81 / 64
    )


def M_gg_34_4_1():
    def func(k):
        return 4 / 315

    return func


def N_gg_34_4_1(theta, phi):
    return (
        (45 / 16) * np.cos(2 * phi - theta)
        + (315 / 64) * np.cos(4 * phi - 2 * theta)
        + 81 / 64
    )


def M_gg_35_0_0():
    def func(k):
        return 1 / 5

    return func


def N_gg_35_0_0(theta, phi):
    return 1


def M_gg_35_2_0():
    def func(k):
        return 2 / 35

    return func


def N_gg_35_2_0(theta, phi):
    return (15 / 4) * np.cos(2 * phi + theta) + 5 / 4


def M_gg_35_2_1():
    def func(k):
        return 2 / 35

    return func


def N_gg_35_2_1(theta, phi):
    return (15 / 4) * np.cos(2 * phi - theta) + 5 / 4


def M_gg_35_4_0():
    def func(k):
        return 4 / 315

    return func


def N_gg_35_4_0(theta, phi):
    return (
        (45 / 16) * np.cos(2 * phi + theta)
        + (315 / 64) * np.cos(4 * phi + 2 * theta)
        + 81 / 64
    )


def M_gg_35_4_1():
    def func(k):
        return 4 / 315

    return func


def N_gg_35_4_1(theta, phi):
    return (
        (45 / 16) * np.cos(2 * phi - theta)
        + (315 / 64) * np.cos(4 * phi - 2 * theta)
        + 81 / 64
    )


def M_gg_36_0_0():
    def func(k):
        return -1 / 5 * k**2

    return func


def N_gg_36_0_0(theta, phi):
    return 1


def M_gg_36_2_0():
    def func(k):
        return -2 / 35 * k**2

    return func


def N_gg_36_2_0(theta, phi):
    return (15 / 4) * np.cos(2 * phi + theta) + 5 / 4


def M_gg_36_2_1():
    def func(k):
        return -2 / 35 * k**2

    return func


def N_gg_36_2_1(theta, phi):
    return (15 / 4) * np.cos(2 * phi - theta) + 5 / 4


def M_gg_36_4_0():
    def func(k):
        return -4 / 315 * k**2

    return func


def N_gg_36_4_0(theta, phi):
    return (
        (45 / 16) * np.cos(2 * phi + theta)
        + (315 / 64) * np.cos(4 * phi + 2 * theta)
        + 81 / 64
    )


def M_gg_36_4_1():
    def func(k):
        return -4 / 315 * k**2

    return func


def N_gg_36_4_1(theta, phi):
    return (
        (45 / 16) * np.cos(2 * phi - theta)
        + (315 / 64) * np.cos(4 * phi - 2 * theta)
        + 81 / 64
    )


def M_gg_37_0_0():
    def func(k):
        return -2 / 5 * k**2

    return func


def N_gg_37_0_0(theta, phi):
    return 1


def M_gg_37_2_0():
    def func(k):
        return -4 / 35 * k**2

    return func


def N_gg_37_2_0(theta, phi):
    return (15 / 4) * np.cos(2 * phi + theta) + 5 / 4


def M_gg_37_2_1():
    def func(k):
        return -4 / 35 * k**2

    return func


def N_gg_37_2_1(theta, phi):
    return (15 / 4) * np.cos(2 * phi - theta) + 5 / 4


def M_gg_37_4_0():
    def func(k):
        return -8 / 315 * k**2

    return func


def N_gg_37_4_0(theta, phi):
    return (
        (45 / 16) * np.cos(2 * phi + theta)
        + (315 / 64) * np.cos(4 * phi + 2 * theta)
        + 81 / 64
    )


def M_gg_37_4_1():
    def func(k):
        return -8 / 315 * k**2

    return func


def N_gg_37_4_1(theta, phi):
    return (
        (45 / 16) * np.cos(2 * phi - theta)
        + (315 / 64) * np.cos(4 * phi - 2 * theta)
        + 81 / 64
    )


def M_gg_38_0_0():
    def func(k):
        return -2 / 5 * k**2

    return func


def N_gg_38_0_0(theta, phi):
    return 1


def M_gg_38_2_0():
    def func(k):
        return -4 / 35 * k**2

    return func


def N_gg_38_2_0(theta, phi):
    return (15 / 4) * np.cos(2 * phi + theta) + 5 / 4


def M_gg_38_2_1():
    def func(k):
        return -4 / 35 * k**2

    return func


def N_gg_38_2_1(theta, phi):
    return (15 / 4) * np.cos(2 * phi - theta) + 5 / 4


def M_gg_38_4_0():
    def func(k):
        return -8 / 315 * k**2

    return func


def N_gg_38_4_0(theta, phi):
    return (
        (45 / 16) * np.cos(2 * phi + theta)
        + (315 / 64) * np.cos(4 * phi + 2 * theta)
        + 81 / 64
    )


def M_gg_38_4_1():
    def func(k):
        return -8 / 315 * k**2

    return func


def N_gg_38_4_1(theta, phi):
    return (
        (45 / 16) * np.cos(2 * phi - theta)
        + (315 / 64) * np.cos(4 * phi - 2 * theta)
        + 81 / 64
    )


def M_gg_39_0_0():
    def func(k):
        return -6 / 5 * k**4

    return func


def N_gg_39_0_0(theta, phi):
    return 1


def M_gg_39_2_0():
    def func(k):
        return -12 / 35 * k**4

    return func


def N_gg_39_2_0(theta, phi):
    return (15 / 4) * np.cos(2 * phi + theta) + 5 / 4


def M_gg_39_2_1():
    def func(k):
        return -12 / 35 * k**4

    return func


def N_gg_39_2_1(theta, phi):
    return (15 / 4) * np.cos(2 * phi - theta) + 5 / 4


def M_gg_39_4_0():
    def func(k):
        return -8 / 105 * k**4

    return func


def N_gg_39_4_0(theta, phi):
    return (
        (45 / 16) * np.cos(2 * phi + theta)
        + (315 / 64) * np.cos(4 * phi + 2 * theta)
        + 81 / 64
    )


def M_gg_39_4_1():
    def func(k):
        return -8 / 105 * k**4

    return func


def N_gg_39_4_1(theta, phi):
    return (
        (45 / 16) * np.cos(2 * phi - theta)
        + (315 / 64) * np.cos(4 * phi - 2 * theta)
        + 81 / 64
    )


def M_gg_40_0_0():
    def func(k):
        return -6 / 5 * k**4

    return func


def N_gg_40_0_0(theta, phi):
    return 1


def M_gg_40_2_0():
    def func(k):
        return -12 / 35 * k**4

    return func


def N_gg_40_2_0(theta, phi):
    return (15 / 4) * np.cos(2 * phi + theta) + 5 / 4


def M_gg_40_2_1():
    def func(k):
        return -12 / 35 * k**4

    return func


def N_gg_40_2_1(theta, phi):
    return (15 / 4) * np.cos(2 * phi - theta) + 5 / 4


def M_gg_40_4_0():
    def func(k):
        return -8 / 105 * k**4

    return func


def N_gg_40_4_0(theta, phi):
    return (
        (45 / 16) * np.cos(2 * phi + theta)
        + (315 / 64) * np.cos(4 * phi + 2 * theta)
        + 81 / 64
    )


def M_gg_40_4_1():
    def func(k):
        return -8 / 105 * k**4

    return func


def N_gg_40_4_1(theta, phi):
    return (
        (45 / 16) * np.cos(2 * phi - theta)
        + (315 / 64) * np.cos(4 * phi - 2 * theta)
        + 81 / 64
    )


def M_gg_41_0_0():
    def func(k):
        return (1 / 5) * k**2

    return func


def N_gg_41_0_0(theta, phi):
    return 1


def M_gg_41_2_0():
    def func(k):
        return (2 / 35) * k**2

    return func


def N_gg_41_2_0(theta, phi):
    return (15 / 4) * np.cos(2 * phi + theta) + 5 / 4


def M_gg_41_2_1():
    def func(k):
        return (2 / 35) * k**2

    return func


def N_gg_41_2_1(theta, phi):
    return (15 / 4) * np.cos(2 * phi - theta) + 5 / 4


def M_gg_41_4_0():
    def func(k):
        return (4 / 315) * k**2

    return func


def N_gg_41_4_0(theta, phi):
    return (
        (45 / 16) * np.cos(2 * phi + theta)
        + (315 / 64) * np.cos(4 * phi + 2 * theta)
        + 81 / 64
    )


def M_gg_41_4_1():
    def func(k):
        return (4 / 315) * k**2

    return func


def N_gg_41_4_1(theta, phi):
    return (
        (45 / 16) * np.cos(2 * phi - theta)
        + (315 / 64) * np.cos(4 * phi - 2 * theta)
        + 81 / 64
    )


def M_gg_42_0_0():
    def func(k):
        return (1 / 5) * k**2

    return func


def N_gg_42_0_0(theta, phi):
    return 1


def M_gg_42_2_0():
    def func(k):
        return (2 / 35) * k**2

    return func


def N_gg_42_2_0(theta, phi):
    return (15 / 4) * np.cos(2 * phi + theta) + 5 / 4


def M_gg_42_2_1():
    def func(k):
        return (2 / 35) * k**2

    return func


def N_gg_42_2_1(theta, phi):
    return (15 / 4) * np.cos(2 * phi - theta) + 5 / 4


def M_gg_42_4_0():
    def func(k):
        return (4 / 315) * k**2

    return func


def N_gg_42_4_0(theta, phi):
    return (
        (45 / 16) * np.cos(2 * phi + theta)
        + (315 / 64) * np.cos(4 * phi + 2 * theta)
        + 81 / 64
    )


def M_gg_42_4_1():
    def func(k):
        return (4 / 315) * k**2

    return func


def N_gg_42_4_1(theta, phi):
    return (
        (45 / 16) * np.cos(2 * phi - theta)
        + (315 / 64) * np.cos(4 * phi - 2 * theta)
        + 81 / 64
    )


def M_gg_43_0_0():
    def func(k):
        return (1 / 5) * k**2

    return func


def N_gg_43_0_0(theta, phi):
    return 1


def M_gg_43_2_0():
    def func(k):
        return (2 / 35) * k**2

    return func


def N_gg_43_2_0(theta, phi):
    return (15 / 4) * np.cos(2 * phi + theta) + 5 / 4


def M_gg_43_2_1():
    def func(k):
        return (2 / 35) * k**2

    return func


def N_gg_43_2_1(theta, phi):
    return (15 / 4) * np.cos(2 * phi - theta) + 5 / 4


def M_gg_43_4_0():
    def func(k):
        return (4 / 315) * k**2

    return func


def N_gg_43_4_0(theta, phi):
    return (
        (45 / 16) * np.cos(2 * phi + theta)
        + (315 / 64) * np.cos(4 * phi + 2 * theta)
        + 81 / 64
    )


def M_gg_43_4_1():
    def func(k):
        return (4 / 315) * k**2

    return func


def N_gg_43_4_1(theta, phi):
    return (
        (45 / 16) * np.cos(2 * phi - theta)
        + (315 / 64) * np.cos(4 * phi - 2 * theta)
        + 81 / 64
    )


def M_gg_44_0_0():
    def func(k):
        return (1 / 5) * k**2

    return func


def N_gg_44_0_0(theta, phi):
    return 1


def M_gg_44_2_0():
    def func(k):
        return (2 / 35) * k**2

    return func


def N_gg_44_2_0(theta, phi):
    return (15 / 4) * np.cos(2 * phi + theta) + 5 / 4


def M_gg_44_2_1():
    def func(k):
        return (2 / 35) * k**2

    return func


def N_gg_44_2_1(theta, phi):
    return (15 / 4) * np.cos(2 * phi - theta) + 5 / 4


def M_gg_44_4_0():
    def func(k):
        return (4 / 315) * k**2

    return func


def N_gg_44_4_0(theta, phi):
    return (
        (45 / 16) * np.cos(2 * phi + theta)
        + (315 / 64) * np.cos(4 * phi + 2 * theta)
        + 81 / 64
    )


def M_gg_44_4_1():
    def func(k):
        return (4 / 315) * k**2

    return func


def N_gg_44_4_1(theta, phi):
    return (
        (45 / 16) * np.cos(2 * phi - theta)
        + (315 / 64) * np.cos(4 * phi - 2 * theta)
        + 81 / 64
    )


def M_gg_45_0_0():
    def func(k):
        return (1 / 5) * k**2

    return func


def N_gg_45_0_0(theta, phi):
    return 1


def M_gg_45_2_0():
    def func(k):
        return (2 / 35) * k**2

    return func


def N_gg_45_2_0(theta, phi):
    return (15 / 4) * np.cos(2 * phi + theta) + 5 / 4


def M_gg_45_2_1():
    def func(k):
        return (2 / 35) * k**2

    return func


def N_gg_45_2_1(theta, phi):
    return (15 / 4) * np.cos(2 * phi - theta) + 5 / 4


def M_gg_45_4_0():
    def func(k):
        return (4 / 315) * k**2

    return func


def N_gg_45_4_0(theta, phi):
    return (
        (45 / 16) * np.cos(2 * phi + theta)
        + (315 / 64) * np.cos(4 * phi + 2 * theta)
        + 81 / 64
    )


def M_gg_45_4_1():
    def func(k):
        return (4 / 315) * k**2

    return func


def N_gg_45_4_1(theta, phi):
    return (
        (45 / 16) * np.cos(2 * phi - theta)
        + (315 / 64) * np.cos(4 * phi - 2 * theta)
        + 81 / 64
    )


def M_gg_46_0_0():
    def func(k):
        return -1 / 10 * k**2

    return func


def N_gg_46_0_0(theta, phi):
    return 1


def M_gg_46_2_0():
    def func(k):
        return -1 / 35 * k**2

    return func


def N_gg_46_2_0(theta, phi):
    return (15 / 4) * np.cos(2 * phi + theta) + 5 / 4


def M_gg_46_2_1():
    def func(k):
        return -1 / 35 * k**2

    return func


def N_gg_46_2_1(theta, phi):
    return (15 / 4) * np.cos(2 * phi - theta) + 5 / 4


def M_gg_46_4_0():
    def func(k):
        return -2 / 315 * k**2

    return func


def N_gg_46_4_0(theta, phi):
    return (
        (45 / 16) * np.cos(2 * phi + theta)
        + (315 / 64) * np.cos(4 * phi + 2 * theta)
        + 81 / 64
    )


def M_gg_46_4_1():
    def func(k):
        return -2 / 315 * k**2

    return func


def N_gg_46_4_1(theta, phi):
    return (
        (45 / 16) * np.cos(2 * phi - theta)
        + (315 / 64) * np.cos(4 * phi - 2 * theta)
        + 81 / 64
    )


def M_gg_47_0_0():
    def func(k):
        return -1 / 14 * k**2

    return func


def N_gg_47_0_0(theta, phi):
    return 1


def M_gg_47_2_0():
    def func(k):
        return -1 / 42 * k**2

    return func


def N_gg_47_2_0(theta, phi):
    return (15 / 4) * np.cos(2 * phi + theta) + 5 / 4


def M_gg_47_2_1():
    def func(k):
        return -1 / 42 * k**2

    return func


def N_gg_47_2_1(theta, phi):
    return (15 / 4) * np.cos(2 * phi - theta) + 5 / 4


def M_gg_47_4_0():
    def func(k):
        return -2 / 231 * k**2

    return func


def N_gg_47_4_0(theta, phi):
    return (
        (45 / 16) * np.cos(2 * phi + theta)
        + (315 / 64) * np.cos(4 * phi + 2 * theta)
        + 81 / 64
    )


def M_gg_47_4_1():
    def func(k):
        return -2 / 231 * k**2

    return func


def N_gg_47_4_1(theta, phi):
    return (
        (45 / 16) * np.cos(2 * phi - theta)
        + (315 / 64) * np.cos(4 * phi - 2 * theta)
        + 81 / 64
    )


def M_gg_47_6_0():
    def func(k):
        return -4 / 3003 * k**2

    return func


def N_gg_47_6_0(theta, phi):
    return (
        (1365 / 512) * np.cos(2 * phi + theta)
        + (819 / 256) * np.cos(4 * phi + 2 * theta)
        + (3003 / 512) * np.cos(6 * phi + 3 * theta)
        + 325 / 256
    )


def M_gg_47_6_1():
    def func(k):
        return -4 / 3003 * k**2

    return func


def N_gg_47_6_1(theta, phi):
    return (
        (1365 / 512) * np.cos(2 * phi - theta)
        + (819 / 256) * np.cos(4 * phi - 2 * theta)
        + (3003 / 512) * np.cos(6 * phi - 3 * theta)
        + 325 / 256
    )


def M_gg_48_0_0():
    def func(k):
        return -1 / 5 * k**4

    return func


def N_gg_48_0_0(theta, phi):
    return 1


def M_gg_48_2_0():
    def func(k):
        return -2 / 35 * k**4

    return func


def N_gg_48_2_0(theta, phi):
    return (15 / 4) * np.cos(2 * phi + theta) + 5 / 4


def M_gg_48_2_1():
    def func(k):
        return -2 / 35 * k**4

    return func


def N_gg_48_2_1(theta, phi):
    return (15 / 4) * np.cos(2 * phi - theta) + 5 / 4


def M_gg_48_4_0():
    def func(k):
        return -4 / 315 * k**4

    return func


def N_gg_48_4_0(theta, phi):
    return (
        (45 / 16) * np.cos(2 * phi + theta)
        + (315 / 64) * np.cos(4 * phi + 2 * theta)
        + 81 / 64
    )


def M_gg_48_4_1():
    def func(k):
        return -4 / 315 * k**4

    return func


def N_gg_48_4_1(theta, phi):
    return (
        (45 / 16) * np.cos(2 * phi - theta)
        + (315 / 64) * np.cos(4 * phi - 2 * theta)
        + 81 / 64
    )


def M_gg_49_0_0():
    def func(k):
        return -1 / 7 * k**4

    return func


def N_gg_49_0_0(theta, phi):
    return 1


def M_gg_49_2_0():
    def func(k):
        return -1 / 21 * k**4

    return func


def N_gg_49_2_0(theta, phi):
    return (15 / 4) * np.cos(2 * phi + theta) + 5 / 4


def M_gg_49_2_1():
    def func(k):
        return -1 / 21 * k**4

    return func


def N_gg_49_2_1(theta, phi):
    return (15 / 4) * np.cos(2 * phi - theta) + 5 / 4


def M_gg_49_4_0():
    def func(k):
        return -4 / 231 * k**4

    return func


def N_gg_49_4_0(theta, phi):
    return (
        (45 / 16) * np.cos(2 * phi + theta)
        + (315 / 64) * np.cos(4 * phi + 2 * theta)
        + 81 / 64
    )


def M_gg_49_4_1():
    def func(k):
        return -4 / 231 * k**4

    return func


def N_gg_49_4_1(theta, phi):
    return (
        (45 / 16) * np.cos(2 * phi - theta)
        + (315 / 64) * np.cos(4 * phi - 2 * theta)
        + 81 / 64
    )


def M_gg_49_6_0():
    def func(k):
        return -8 / 3003 * k**4

    return func


def N_gg_49_6_0(theta, phi):
    return (
        (1365 / 512) * np.cos(2 * phi + theta)
        + (819 / 256) * np.cos(4 * phi + 2 * theta)
        + (3003 / 512) * np.cos(6 * phi + 3 * theta)
        + 325 / 256
    )


def M_gg_49_6_1():
    def func(k):
        return -8 / 3003 * k**4

    return func


def N_gg_49_6_1(theta, phi):
    return (
        (1365 / 512) * np.cos(2 * phi - theta)
        + (819 / 256) * np.cos(4 * phi - 2 * theta)
        + (3003 / 512) * np.cos(6 * phi - 3 * theta)
        + 325 / 256
    )


def M_gg_50_0_0():
    def func(k):
        return (1 / 20) * k**4

    return func


def N_gg_50_0_0(theta, phi):
    return 1


def M_gg_50_2_0():
    def func(k):
        return (1 / 70) * k**4

    return func


def N_gg_50_2_0(theta, phi):
    return (15 / 4) * np.cos(2 * phi + theta) + 5 / 4


def M_gg_50_2_1():
    def func(k):
        return (1 / 70) * k**4

    return func


def N_gg_50_2_1(theta, phi):
    return (15 / 4) * np.cos(2 * phi - theta) + 5 / 4


def M_gg_50_4_0():
    def func(k):
        return (1 / 315) * k**4

    return func


def N_gg_50_4_0(theta, phi):
    return (
        (45 / 16) * np.cos(2 * phi + theta)
        + (315 / 64) * np.cos(4 * phi + 2 * theta)
        + 81 / 64
    )


def M_gg_50_4_1():
    def func(k):
        return (1 / 315) * k**4

    return func


def N_gg_50_4_1(theta, phi):
    return (
        (45 / 16) * np.cos(2 * phi - theta)
        + (315 / 64) * np.cos(4 * phi - 2 * theta)
        + 81 / 64
    )


def M_gg_51_0_0():
    def func(k):
        return (1 / 10) * k**4

    return func


def N_gg_51_0_0(theta, phi):
    return 1


def M_gg_51_2_0():
    def func(k):
        return (1 / 35) * k**4

    return func


def N_gg_51_2_0(theta, phi):
    return (15 / 4) * np.cos(2 * phi + theta) + 5 / 4


def M_gg_51_2_1():
    def func(k):
        return (1 / 35) * k**4

    return func


def N_gg_51_2_1(theta, phi):
    return (15 / 4) * np.cos(2 * phi - theta) + 5 / 4


def M_gg_51_4_0():
    def func(k):
        return (2 / 315) * k**4

    return func


def N_gg_51_4_0(theta, phi):
    return (
        (45 / 16) * np.cos(2 * phi + theta)
        + (315 / 64) * np.cos(4 * phi + 2 * theta)
        + 81 / 64
    )


def M_gg_51_4_1():
    def func(k):
        return (2 / 315) * k**4

    return func


def N_gg_51_4_1(theta, phi):
    return (
        (45 / 16) * np.cos(2 * phi - theta)
        + (315 / 64) * np.cos(4 * phi - 2 * theta)
        + 81 / 64
    )


def M_gg_52_0_0():
    def func(k):
        return (1 / 10) * k**4

    return func


def N_gg_52_0_0(theta, phi):
    return 1


def M_gg_52_2_0():
    def func(k):
        return (1 / 35) * k**4

    return func


def N_gg_52_2_0(theta, phi):
    return (15 / 4) * np.cos(2 * phi + theta) + 5 / 4


def M_gg_52_2_1():
    def func(k):
        return (1 / 35) * k**4

    return func


def N_gg_52_2_1(theta, phi):
    return (15 / 4) * np.cos(2 * phi - theta) + 5 / 4


def M_gg_52_4_0():
    def func(k):
        return (2 / 315) * k**4

    return func


def N_gg_52_4_0(theta, phi):
    return (
        (45 / 16) * np.cos(2 * phi + theta)
        + (315 / 64) * np.cos(4 * phi + 2 * theta)
        + 81 / 64
    )


def M_gg_52_4_1():
    def func(k):
        return (2 / 315) * k**4

    return func


def N_gg_52_4_1(theta, phi):
    return (
        (45 / 16) * np.cos(2 * phi - theta)
        + (315 / 64) * np.cos(4 * phi - 2 * theta)
        + 81 / 64
    )


def M_gg_53_0_0():
    def func(k):
        return (1 / 10) * k**4

    return func


def N_gg_53_0_0(theta, phi):
    return 1


def M_gg_53_2_0():
    def func(k):
        return (1 / 35) * k**4

    return func


def N_gg_53_2_0(theta, phi):
    return (15 / 4) * np.cos(2 * phi + theta) + 5 / 4


def M_gg_53_2_1():
    def func(k):
        return (1 / 35) * k**4

    return func


def N_gg_53_2_1(theta, phi):
    return (15 / 4) * np.cos(2 * phi - theta) + 5 / 4


def M_gg_53_4_0():
    def func(k):
        return (2 / 315) * k**4

    return func


def N_gg_53_4_0(theta, phi):
    return (
        (45 / 16) * np.cos(2 * phi + theta)
        + (315 / 64) * np.cos(4 * phi + 2 * theta)
        + 81 / 64
    )


def M_gg_53_4_1():
    def func(k):
        return (2 / 315) * k**4

    return func


def N_gg_53_4_1(theta, phi):
    return (
        (45 / 16) * np.cos(2 * phi - theta)
        + (315 / 64) * np.cos(4 * phi - 2 * theta)
        + 81 / 64
    )


def M_gg_54_0_0():
    def func(k):
        return (3 / 10) * k**6

    return func


def N_gg_54_0_0(theta, phi):
    return 1


def M_gg_54_2_0():
    def func(k):
        return (3 / 35) * k**6

    return func


def N_gg_54_2_0(theta, phi):
    return (15 / 4) * np.cos(2 * phi + theta) + 5 / 4


def M_gg_54_2_1():
    def func(k):
        return (3 / 35) * k**6

    return func


def N_gg_54_2_1(theta, phi):
    return (15 / 4) * np.cos(2 * phi - theta) + 5 / 4


def M_gg_54_4_0():
    def func(k):
        return (2 / 105) * k**6

    return func


def N_gg_54_4_0(theta, phi):
    return (
        (45 / 16) * np.cos(2 * phi + theta)
        + (315 / 64) * np.cos(4 * phi + 2 * theta)
        + 81 / 64
    )


def M_gg_54_4_1():
    def func(k):
        return (2 / 105) * k**6

    return func


def N_gg_54_4_1(theta, phi):
    return (
        (45 / 16) * np.cos(2 * phi - theta)
        + (315 / 64) * np.cos(4 * phi - 2 * theta)
        + 81 / 64
    )


def M_gg_55_0_0():
    def func(k):
        return (1 / 40) * k**4

    return func


def N_gg_55_0_0(theta, phi):
    return 1


def M_gg_55_2_0():
    def func(k):
        return (1 / 140) * k**4

    return func


def N_gg_55_2_0(theta, phi):
    return (15 / 4) * np.cos(2 * phi + theta) + 5 / 4


def M_gg_55_2_1():
    def func(k):
        return (1 / 140) * k**4

    return func


def N_gg_55_2_1(theta, phi):
    return (15 / 4) * np.cos(2 * phi - theta) + 5 / 4


def M_gg_55_4_0():
    def func(k):
        return (1 / 630) * k**4

    return func


def N_gg_55_4_0(theta, phi):
    return (
        (45 / 16) * np.cos(2 * phi + theta)
        + (315 / 64) * np.cos(4 * phi + 2 * theta)
        + 81 / 64
    )


def M_gg_55_4_1():
    def func(k):
        return (1 / 630) * k**4

    return func


def N_gg_55_4_1(theta, phi):
    return (
        (45 / 16) * np.cos(2 * phi - theta)
        + (315 / 64) * np.cos(4 * phi - 2 * theta)
        + 81 / 64
    )


def M_gg_56_0_0():
    def func(k):
        return (1 / 40) * k**4

    return func


def N_gg_56_0_0(theta, phi):
    return 1


def M_gg_56_2_0():
    def func(k):
        return (1 / 140) * k**4

    return func


def N_gg_56_2_0(theta, phi):
    return (15 / 4) * np.cos(2 * phi + theta) + 5 / 4


def M_gg_56_2_1():
    def func(k):
        return (1 / 140) * k**4

    return func


def N_gg_56_2_1(theta, phi):
    return (15 / 4) * np.cos(2 * phi - theta) + 5 / 4


def M_gg_56_4_0():
    def func(k):
        return (1 / 630) * k**4

    return func


def N_gg_56_4_0(theta, phi):
    return (
        (45 / 16) * np.cos(2 * phi + theta)
        + (315 / 64) * np.cos(4 * phi + 2 * theta)
        + 81 / 64
    )


def M_gg_56_4_1():
    def func(k):
        return (1 / 630) * k**4

    return func


def N_gg_56_4_1(theta, phi):
    return (
        (45 / 16) * np.cos(2 * phi - theta)
        + (315 / 64) * np.cos(4 * phi - 2 * theta)
        + 81 / 64
    )


def M_gg_57_0_0():
    def func(k):
        return (1 / 20) * k**4

    return func


def N_gg_57_0_0(theta, phi):
    return 1


def M_gg_57_2_0():
    def func(k):
        return (1 / 70) * k**4

    return func


def N_gg_57_2_0(theta, phi):
    return (15 / 4) * np.cos(2 * phi + theta) + 5 / 4


def M_gg_57_2_1():
    def func(k):
        return (1 / 70) * k**4

    return func


def N_gg_57_2_1(theta, phi):
    return (15 / 4) * np.cos(2 * phi - theta) + 5 / 4


def M_gg_57_4_0():
    def func(k):
        return (1 / 315) * k**4

    return func


def N_gg_57_4_0(theta, phi):
    return (
        (45 / 16) * np.cos(2 * phi + theta)
        + (315 / 64) * np.cos(4 * phi + 2 * theta)
        + 81 / 64
    )


def M_gg_57_4_1():
    def func(k):
        return (1 / 315) * k**4

    return func


def N_gg_57_4_1(theta, phi):
    return (
        (45 / 16) * np.cos(2 * phi - theta)
        + (315 / 64) * np.cos(4 * phi - 2 * theta)
        + 81 / 64
    )


def M_gg_58_0_0():
    def func(k):
        return (1 / 10) * k**4

    return func


def N_gg_58_0_0(theta, phi):
    return 1


def M_gg_58_2_0():
    def func(k):
        return (1 / 35) * k**4

    return func


def N_gg_58_2_0(theta, phi):
    return (15 / 4) * np.cos(2 * phi + theta) + 5 / 4


def M_gg_58_2_1():
    def func(k):
        return (1 / 35) * k**4

    return func


def N_gg_58_2_1(theta, phi):
    return (15 / 4) * np.cos(2 * phi - theta) + 5 / 4


def M_gg_58_4_0():
    def func(k):
        return (2 / 315) * k**4

    return func


def N_gg_58_4_0(theta, phi):
    return (
        (45 / 16) * np.cos(2 * phi + theta)
        + (315 / 64) * np.cos(4 * phi + 2 * theta)
        + 81 / 64
    )


def M_gg_58_4_1():
    def func(k):
        return (2 / 315) * k**4

    return func


def N_gg_58_4_1(theta, phi):
    return (
        (45 / 16) * np.cos(2 * phi - theta)
        + (315 / 64) * np.cos(4 * phi - 2 * theta)
        + 81 / 64
    )


def M_gg_59_0_0():
    def func(k):
        return (1 / 20) * k**4

    return func


def N_gg_59_0_0(theta, phi):
    return 1


def M_gg_59_2_0():
    def func(k):
        return (1 / 70) * k**4

    return func


def N_gg_59_2_0(theta, phi):
    return (15 / 4) * np.cos(2 * phi + theta) + 5 / 4


def M_gg_59_2_1():
    def func(k):
        return (1 / 70) * k**4

    return func


def N_gg_59_2_1(theta, phi):
    return (15 / 4) * np.cos(2 * phi - theta) + 5 / 4


def M_gg_59_4_0():
    def func(k):
        return (1 / 315) * k**4

    return func


def N_gg_59_4_0(theta, phi):
    return (
        (45 / 16) * np.cos(2 * phi + theta)
        + (315 / 64) * np.cos(4 * phi + 2 * theta)
        + 81 / 64
    )


def M_gg_59_4_1():
    def func(k):
        return (1 / 315) * k**4

    return func


def N_gg_59_4_1(theta, phi):
    return (
        (45 / 16) * np.cos(2 * phi - theta)
        + (315 / 64) * np.cos(4 * phi - 2 * theta)
        + 81 / 64
    )


def M_gg_60_0_0():
    def func(k):
        return (1 / 10) * k**4

    return func


def N_gg_60_0_0(theta, phi):
    return 1


def M_gg_60_2_0():
    def func(k):
        return (1 / 35) * k**4

    return func


def N_gg_60_2_0(theta, phi):
    return (15 / 4) * np.cos(2 * phi + theta) + 5 / 4


def M_gg_60_2_1():
    def func(k):
        return (1 / 35) * k**4

    return func


def N_gg_60_2_1(theta, phi):
    return (15 / 4) * np.cos(2 * phi - theta) + 5 / 4


def M_gg_60_4_0():
    def func(k):
        return (2 / 315) * k**4

    return func


def N_gg_60_4_0(theta, phi):
    return (
        (45 / 16) * np.cos(2 * phi + theta)
        + (315 / 64) * np.cos(4 * phi + 2 * theta)
        + 81 / 64
    )


def M_gg_60_4_1():
    def func(k):
        return (2 / 315) * k**4

    return func


def N_gg_60_4_1(theta, phi):
    return (
        (45 / 16) * np.cos(2 * phi - theta)
        + (315 / 64) * np.cos(4 * phi - 2 * theta)
        + 81 / 64
    )


def M_gg_61_0_0():
    def func(k):
        return (1 / 10) * k**4

    return func


def N_gg_61_0_0(theta, phi):
    return 1


def M_gg_61_2_0():
    def func(k):
        return (1 / 35) * k**4

    return func


def N_gg_61_2_0(theta, phi):
    return (15 / 4) * np.cos(2 * phi + theta) + 5 / 4


def M_gg_61_2_1():
    def func(k):
        return (1 / 35) * k**4

    return func


def N_gg_61_2_1(theta, phi):
    return (15 / 4) * np.cos(2 * phi - theta) + 5 / 4


def M_gg_61_4_0():
    def func(k):
        return (2 / 315) * k**4

    return func


def N_gg_61_4_0(theta, phi):
    return (
        (45 / 16) * np.cos(2 * phi + theta)
        + (315 / 64) * np.cos(4 * phi + 2 * theta)
        + 81 / 64
    )


def M_gg_61_4_1():
    def func(k):
        return (2 / 315) * k**4

    return func


def N_gg_61_4_1(theta, phi):
    return (
        (45 / 16) * np.cos(2 * phi - theta)
        + (315 / 64) * np.cos(4 * phi - 2 * theta)
        + 81 / 64
    )


def M_gg_62_0_0():
    def func(k):
        return (1 / 10) * k**4

    return func


def N_gg_62_0_0(theta, phi):
    return 1


def M_gg_62_2_0():
    def func(k):
        return (1 / 35) * k**4

    return func


def N_gg_62_2_0(theta, phi):
    return (15 / 4) * np.cos(2 * phi + theta) + 5 / 4


def M_gg_62_2_1():
    def func(k):
        return (1 / 35) * k**4

    return func


def N_gg_62_2_1(theta, phi):
    return (15 / 4) * np.cos(2 * phi - theta) + 5 / 4


def M_gg_62_4_0():
    def func(k):
        return (2 / 315) * k**4

    return func


def N_gg_62_4_0(theta, phi):
    return (
        (45 / 16) * np.cos(2 * phi + theta)
        + (315 / 64) * np.cos(4 * phi + 2 * theta)
        + 81 / 64
    )


def M_gg_62_4_1():
    def func(k):
        return (2 / 315) * k**4

    return func


def N_gg_62_4_1(theta, phi):
    return (
        (45 / 16) * np.cos(2 * phi - theta)
        + (315 / 64) * np.cos(4 * phi - 2 * theta)
        + 81 / 64
    )


def M_gg_63_0_0():
    def func(k):
        return (3 / 10) * k**6

    return func


def N_gg_63_0_0(theta, phi):
    return 1


def M_gg_63_2_0():
    def func(k):
        return (3 / 35) * k**6

    return func


def N_gg_63_2_0(theta, phi):
    return (15 / 4) * np.cos(2 * phi + theta) + 5 / 4


def M_gg_63_2_1():
    def func(k):
        return (3 / 35) * k**6

    return func


def N_gg_63_2_1(theta, phi):
    return (15 / 4) * np.cos(2 * phi - theta) + 5 / 4


def M_gg_63_4_0():
    def func(k):
        return (2 / 105) * k**6

    return func


def N_gg_63_4_0(theta, phi):
    return (
        (45 / 16) * np.cos(2 * phi + theta)
        + (315 / 64) * np.cos(4 * phi + 2 * theta)
        + 81 / 64
    )


def M_gg_63_4_1():
    def func(k):
        return (2 / 105) * k**6

    return func


def N_gg_63_4_1(theta, phi):
    return (
        (45 / 16) * np.cos(2 * phi - theta)
        + (315 / 64) * np.cos(4 * phi - 2 * theta)
        + 81 / 64
    )


def M_gg_64_0_0():
    def func(k):
        return (1 / 40) * k**4

    return func


def N_gg_64_0_0(theta, phi):
    return 1


def M_gg_64_2_0():
    def func(k):
        return (1 / 140) * k**4

    return func


def N_gg_64_2_0(theta, phi):
    return (15 / 4) * np.cos(2 * phi + theta) + 5 / 4


def M_gg_64_2_1():
    def func(k):
        return (1 / 140) * k**4

    return func


def N_gg_64_2_1(theta, phi):
    return (15 / 4) * np.cos(2 * phi - theta) + 5 / 4


def M_gg_64_4_0():
    def func(k):
        return (1 / 630) * k**4

    return func


def N_gg_64_4_0(theta, phi):
    return (
        (45 / 16) * np.cos(2 * phi + theta)
        + (315 / 64) * np.cos(4 * phi + 2 * theta)
        + 81 / 64
    )


def M_gg_64_4_1():
    def func(k):
        return (1 / 630) * k**4

    return func


def N_gg_64_4_1(theta, phi):
    return (
        (45 / 16) * np.cos(2 * phi - theta)
        + (315 / 64) * np.cos(4 * phi - 2 * theta)
        + 81 / 64
    )


def M_gg_65_0_0():
    def func(k):
        return (1 / 40) * k**4

    return func


def N_gg_65_0_0(theta, phi):
    return 1


def M_gg_65_2_0():
    def func(k):
        return (1 / 140) * k**4

    return func


def N_gg_65_2_0(theta, phi):
    return (15 / 4) * np.cos(2 * phi + theta) + 5 / 4


def M_gg_65_2_1():
    def func(k):
        return (1 / 140) * k**4

    return func


def N_gg_65_2_1(theta, phi):
    return (15 / 4) * np.cos(2 * phi - theta) + 5 / 4


def M_gg_65_4_0():
    def func(k):
        return (1 / 630) * k**4

    return func


def N_gg_65_4_0(theta, phi):
    return (
        (45 / 16) * np.cos(2 * phi + theta)
        + (315 / 64) * np.cos(4 * phi + 2 * theta)
        + 81 / 64
    )


def M_gg_65_4_1():
    def func(k):
        return (1 / 630) * k**4

    return func


def N_gg_65_4_1(theta, phi):
    return (
        (45 / 16) * np.cos(2 * phi - theta)
        + (315 / 64) * np.cos(4 * phi - 2 * theta)
        + 81 / 64
    )


def M_gg_66_0_0():
    def func(k):
        return (1 / 20) * k**4

    return func


def N_gg_66_0_0(theta, phi):
    return 1


def M_gg_66_2_0():
    def func(k):
        return (1 / 70) * k**4

    return func


def N_gg_66_2_0(theta, phi):
    return (15 / 4) * np.cos(2 * phi + theta) + 5 / 4


def M_gg_66_2_1():
    def func(k):
        return (1 / 70) * k**4

    return func


def N_gg_66_2_1(theta, phi):
    return (15 / 4) * np.cos(2 * phi - theta) + 5 / 4


def M_gg_66_4_0():
    def func(k):
        return (1 / 315) * k**4

    return func


def N_gg_66_4_0(theta, phi):
    return (
        (45 / 16) * np.cos(2 * phi + theta)
        + (315 / 64) * np.cos(4 * phi + 2 * theta)
        + 81 / 64
    )


def M_gg_66_4_1():
    def func(k):
        return (1 / 315) * k**4

    return func


def N_gg_66_4_1(theta, phi):
    return (
        (45 / 16) * np.cos(2 * phi - theta)
        + (315 / 64) * np.cos(4 * phi - 2 * theta)
        + 81 / 64
    )


def M_gg_67_0_0():
    def func(k):
        return (1 / 10) * k**4

    return func


def N_gg_67_0_0(theta, phi):
    return 1


def M_gg_67_2_0():
    def func(k):
        return (1 / 35) * k**4

    return func


def N_gg_67_2_0(theta, phi):
    return (15 / 4) * np.cos(2 * phi + theta) + 5 / 4


def M_gg_67_2_1():
    def func(k):
        return (1 / 35) * k**4

    return func


def N_gg_67_2_1(theta, phi):
    return (15 / 4) * np.cos(2 * phi - theta) + 5 / 4


def M_gg_67_4_0():
    def func(k):
        return (2 / 315) * k**4

    return func


def N_gg_67_4_0(theta, phi):
    return (
        (45 / 16) * np.cos(2 * phi + theta)
        + (315 / 64) * np.cos(4 * phi + 2 * theta)
        + 81 / 64
    )


def M_gg_67_4_1():
    def func(k):
        return (2 / 315) * k**4

    return func


def N_gg_67_4_1(theta, phi):
    return (
        (45 / 16) * np.cos(2 * phi - theta)
        + (315 / 64) * np.cos(4 * phi - 2 * theta)
        + 81 / 64
    )


def M_gg_68_0_0():
    def func(k):
        return 1 / 9

    return func


def N_gg_68_0_0(theta, phi):
    return 1


def M_gg_68_0_1():
    def func(k):
        return 4 / 225

    return func


def N_gg_68_0_1(theta, phi):
    return (15 / 4) * np.cos(2 * theta) + 5 / 4


def M_gg_68_2_0():
    def func(k):
        return 2 / 45

    return func


def N_gg_68_2_0(theta, phi):
    return (15 / 4) * np.cos(2 * phi + theta) + 5 / 4


def M_gg_68_2_1():
    def func(k):
        return 2 / 45

    return func


def N_gg_68_2_1(theta, phi):
    return (15 / 4) * np.cos(2 * phi - theta) + 5 / 4


def M_gg_68_2_2():
    def func(k):
        return 4 / 225

    return func


def N_gg_68_2_2(theta, phi):
    return (
        (75 / 28) * np.cos(2 * theta)
        + (75 / 28) * np.cos(2 * phi - theta)
        + (75 / 28) * np.cos(2 * phi + theta)
        - 25 / 28
    )


def M_gg_68_4_0():
    def func(k):
        return 4 / 225

    return func


def N_gg_68_4_0(theta, phi):
    return (
        (225 / 32) * np.cos(4 * phi)
        + (135 / 224) * np.cos(2 * theta)
        + (225 / 112) * np.cos(2 * phi - theta)
        + (225 / 112) * np.cos(2 * phi + theta)
        + 135 / 112
    )


def M_gg_69_0_0():
    def func(k):
        return 2 / 9

    return func


def N_gg_69_0_0(theta, phi):
    return 1


def M_gg_69_0_1():
    def func(k):
        return 8 / 225

    return func


def N_gg_69_0_1(theta, phi):
    return (15 / 4) * np.cos(2 * theta) + 5 / 4


def M_gg_69_2_0():
    def func(k):
        return 4 / 45

    return func


def N_gg_69_2_0(theta, phi):
    return (15 / 4) * np.cos(2 * phi + theta) + 5 / 4


def M_gg_69_2_1():
    def func(k):
        return 4 / 45

    return func


def N_gg_69_2_1(theta, phi):
    return (15 / 4) * np.cos(2 * phi - theta) + 5 / 4


def M_gg_69_2_2():
    def func(k):
        return 8 / 225

    return func


def N_gg_69_2_2(theta, phi):
    return (
        (75 / 28) * np.cos(2 * theta)
        + (75 / 28) * np.cos(2 * phi - theta)
        + (75 / 28) * np.cos(2 * phi + theta)
        - 25 / 28
    )


def M_gg_69_4_0():
    def func(k):
        return 8 / 225

    return func


def N_gg_69_4_0(theta, phi):
    return (
        (225 / 32) * np.cos(4 * phi)
        + (135 / 224) * np.cos(2 * theta)
        + (225 / 112) * np.cos(2 * phi - theta)
        + (225 / 112) * np.cos(2 * phi + theta)
        + 135 / 112
    )


def M_gg_70_0_0():
    def func(k):
        return 4 / 9

    return func


def N_gg_70_0_0(theta, phi):
    return 1


def M_gg_70_0_1():
    def func(k):
        return 16 / 225

    return func


def N_gg_70_0_1(theta, phi):
    return (15 / 4) * np.cos(2 * theta) + 5 / 4


def M_gg_70_2_0():
    def func(k):
        return 8 / 45

    return func


def N_gg_70_2_0(theta, phi):
    return (15 / 4) * np.cos(2 * phi + theta) + 5 / 4


def M_gg_70_2_1():
    def func(k):
        return 8 / 45

    return func


def N_gg_70_2_1(theta, phi):
    return (15 / 4) * np.cos(2 * phi - theta) + 5 / 4


def M_gg_70_2_2():
    def func(k):
        return 16 / 225

    return func


def N_gg_70_2_2(theta, phi):
    return (
        (75 / 28) * np.cos(2 * theta)
        + (75 / 28) * np.cos(2 * phi - theta)
        + (75 / 28) * np.cos(2 * phi + theta)
        - 25 / 28
    )


def M_gg_70_4_0():
    def func(k):
        return 16 / 225

    return func


def N_gg_70_4_0(theta, phi):
    return (
        (225 / 32) * np.cos(4 * phi)
        + (135 / 224) * np.cos(2 * theta)
        + (225 / 112) * np.cos(2 * phi - theta)
        + (225 / 112) * np.cos(2 * phi + theta)
        + 135 / 112
    )


def M_gg_71_0_0():
    def func(k):
        return 1 / 9

    return func


def N_gg_71_0_0(theta, phi):
    return (3 / 2) * np.cos(2 * theta) + 3 / 2


def M_gg_71_2_0():
    def func(k):
        return 1 / 9

    return func


def N_gg_71_2_0(theta, phi):
    return (
        (3 / 4) * np.cos(2 * theta)
        + (9 / 4) * np.cos(2 * phi - theta)
        + (9 / 4) * np.cos(2 * phi + theta)
        + 3 / 4
    )


def M_gg_72_0_0():
    def func(k):
        return 1 / 9

    return func


def N_gg_72_0_0(theta, phi):
    return 1


def M_gg_72_0_1():
    def func(k):
        return 4 / 225

    return func


def N_gg_72_0_1(theta, phi):
    return (15 / 4) * np.cos(2 * theta) + 5 / 4


def M_gg_72_2_0():
    def func(k):
        return 2 / 45

    return func


def N_gg_72_2_0(theta, phi):
    return (15 / 4) * np.cos(2 * phi + theta) + 5 / 4


def M_gg_72_2_1():
    def func(k):
        return 2 / 45

    return func


def N_gg_72_2_1(theta, phi):
    return (15 / 4) * np.cos(2 * phi - theta) + 5 / 4


def M_gg_72_2_2():
    def func(k):
        return 4 / 225

    return func


def N_gg_72_2_2(theta, phi):
    return (
        (75 / 28) * np.cos(2 * theta)
        + (75 / 28) * np.cos(2 * phi - theta)
        + (75 / 28) * np.cos(2 * phi + theta)
        - 25 / 28
    )


def M_gg_72_4_0():
    def func(k):
        return 4 / 225

    return func


def N_gg_72_4_0(theta, phi):
    return (
        (225 / 32) * np.cos(4 * phi)
        + (135 / 224) * np.cos(2 * theta)
        + (225 / 112) * np.cos(2 * phi - theta)
        + (225 / 112) * np.cos(2 * phi + theta)
        + 135 / 112
    )


def M_gg_73_0_0():
    def func(k):
        return (4 / 3) * k**2

    return func


def N_gg_73_0_0(theta, phi):
    return 1


def M_gg_73_0_1():
    def func(k):
        return (16 / 75) * k**2

    return func


def N_gg_73_0_1(theta, phi):
    return (15 / 4) * np.cos(2 * theta) + 5 / 4


def M_gg_73_2_0():
    def func(k):
        return (8 / 15) * k**2

    return func


def N_gg_73_2_0(theta, phi):
    return (15 / 4) * np.cos(2 * phi + theta) + 5 / 4


def M_gg_73_2_1():
    def func(k):
        return (8 / 15) * k**2

    return func


def N_gg_73_2_1(theta, phi):
    return (15 / 4) * np.cos(2 * phi - theta) + 5 / 4


def M_gg_73_2_2():
    def func(k):
        return (16 / 75) * k**2

    return func


def N_gg_73_2_2(theta, phi):
    return (
        (75 / 28) * np.cos(2 * theta)
        + (75 / 28) * np.cos(2 * phi - theta)
        + (75 / 28) * np.cos(2 * phi + theta)
        - 25 / 28
    )


def M_gg_73_4_0():
    def func(k):
        return (16 / 75) * k**2

    return func


def N_gg_73_4_0(theta, phi):
    return (
        (225 / 32) * np.cos(4 * phi)
        + (135 / 224) * np.cos(2 * theta)
        + (225 / 112) * np.cos(2 * phi - theta)
        + (225 / 112) * np.cos(2 * phi + theta)
        + 135 / 112
    )


def M_gg_74_0_0():
    def func(k):
        return (2 / 3) * k**2

    return func


def N_gg_74_0_0(theta, phi):
    return 1


def M_gg_74_0_1():
    def func(k):
        return (8 / 75) * k**2

    return func


def N_gg_74_0_1(theta, phi):
    return (15 / 4) * np.cos(2 * theta) + 5 / 4


def M_gg_74_2_0():
    def func(k):
        return (4 / 15) * k**2

    return func


def N_gg_74_2_0(theta, phi):
    return (15 / 4) * np.cos(2 * phi + theta) + 5 / 4


def M_gg_74_2_1():
    def func(k):
        return (4 / 15) * k**2

    return func


def N_gg_74_2_1(theta, phi):
    return (15 / 4) * np.cos(2 * phi - theta) + 5 / 4


def M_gg_74_2_2():
    def func(k):
        return (8 / 75) * k**2

    return func


def N_gg_74_2_2(theta, phi):
    return (
        (75 / 28) * np.cos(2 * theta)
        + (75 / 28) * np.cos(2 * phi - theta)
        + (75 / 28) * np.cos(2 * phi + theta)
        - 25 / 28
    )


def M_gg_74_4_0():
    def func(k):
        return (8 / 75) * k**2

    return func


def N_gg_74_4_0(theta, phi):
    return (
        (225 / 32) * np.cos(4 * phi)
        + (135 / 224) * np.cos(2 * theta)
        + (225 / 112) * np.cos(2 * phi - theta)
        + (225 / 112) * np.cos(2 * phi + theta)
        + 135 / 112
    )


def M_gg_75_0_0():
    def func(k):
        return 1 / 9

    return func


def N_gg_75_0_0(theta, phi):
    return 1


def M_gg_75_0_1():
    def func(k):
        return 4 / 225

    return func


def N_gg_75_0_1(theta, phi):
    return (15 / 4) * np.cos(2 * theta) + 5 / 4


def M_gg_75_2_0():
    def func(k):
        return 2 / 45

    return func


def N_gg_75_2_0(theta, phi):
    return (15 / 4) * np.cos(2 * phi + theta) + 5 / 4


def M_gg_75_2_1():
    def func(k):
        return 2 / 45

    return func


def N_gg_75_2_1(theta, phi):
    return (15 / 4) * np.cos(2 * phi - theta) + 5 / 4


def M_gg_75_2_2():
    def func(k):
        return 4 / 225

    return func


def N_gg_75_2_2(theta, phi):
    return (
        (75 / 28) * np.cos(2 * theta)
        + (75 / 28) * np.cos(2 * phi - theta)
        + (75 / 28) * np.cos(2 * phi + theta)
        - 25 / 28
    )


def M_gg_75_4_0():
    def func(k):
        return 4 / 225

    return func


def N_gg_75_4_0(theta, phi):
    return (
        (225 / 32) * np.cos(4 * phi)
        + (135 / 224) * np.cos(2 * theta)
        + (225 / 112) * np.cos(2 * phi - theta)
        + (225 / 112) * np.cos(2 * phi + theta)
        + 135 / 112
    )


def M_gg_76_0_0():
    def func(k):
        return 1 / 15

    return func


def N_gg_76_0_0(theta, phi):
    return 1


def M_gg_76_0_1():
    def func(k):
        return 8 / 525

    return func


def N_gg_76_0_1(theta, phi):
    return (15 / 4) * np.cos(2 * theta) + 5 / 4


def M_gg_76_2_0():
    def func(k):
        return 17 / 525

    return func


def N_gg_76_2_0(theta, phi):
    return (15 / 4) * np.cos(2 * phi + theta) + 5 / 4


def M_gg_76_2_1():
    def func(k):
        return 17 / 525

    return func


def N_gg_76_2_1(theta, phi):
    return (15 / 4) * np.cos(2 * phi - theta) + 5 / 4


def M_gg_76_2_2():
    def func(k):
        return 8 / 525

    return func


def N_gg_76_2_2(theta, phi):
    return (
        (75 / 28) * np.cos(2 * theta)
        + (75 / 28) * np.cos(2 * phi - theta)
        + (75 / 28) * np.cos(2 * phi + theta)
        - 25 / 28
    )


def M_gg_76_2_3():
    def func(k):
        return 8 / 4725

    return func


def N_gg_76_2_3(theta, phi):
    return (
        (225 / 112) * np.cos(2 * theta)
        + (135 / 224) * np.cos(2 * phi - theta)
        + (225 / 112) * np.cos(2 * phi + theta)
        + (225 / 32) * np.cos(2 * phi + 3 * theta)
        + 135 / 112
    )


def M_gg_76_2_4():
    def func(k):
        return 8 / 4725

    return func


def N_gg_76_2_4(theta, phi):
    return (
        (225 / 112) * np.cos(2 * theta)
        + (225 / 32) * np.cos(2 * phi - 3 * theta)
        + (225 / 112) * np.cos(2 * phi - theta)
        + (135 / 224) * np.cos(2 * phi + theta)
        + 135 / 112
    )


def M_gg_76_4_0():
    def func(k):
        return 4 / 945

    return func


def N_gg_76_4_0(theta, phi):
    return (
        (45 / 16) * np.cos(2 * phi + theta)
        + (315 / 64) * np.cos(4 * phi + 2 * theta)
        + 81 / 64
    )


def M_gg_76_4_1():
    def func(k):
        return 8 / 525

    return func


def N_gg_76_4_1(theta, phi):
    return (
        (225 / 32) * np.cos(4 * phi)
        + (135 / 224) * np.cos(2 * theta)
        + (225 / 112) * np.cos(2 * phi - theta)
        + (225 / 112) * np.cos(2 * phi + theta)
        + 135 / 112
    )


def M_gg_76_4_2():
    def func(k):
        return 8 / 4725

    return func


def N_gg_76_4_2(theta, phi):
    return (
        (675 / 352) * np.cos(4 * phi)
        + (6075 / 2464) * np.cos(2 * theta)
        + (6075 / 2464) * np.cos(2 * phi - theta)
        - 225 / 308 * np.cos(2 * phi + theta)
        + (675 / 352) * np.cos(2 * phi + 3 * theta)
        + (1575 / 352) * np.cos(4 * phi + 2 * theta)
        - 2025 / 2464
    )


def M_gg_76_4_3():
    def func(k):
        return 4 / 945

    return func


def N_gg_76_4_3(theta, phi):
    return (
        (45 / 16) * np.cos(2 * phi - theta)
        + (315 / 64) * np.cos(4 * phi - 2 * theta)
        + 81 / 64
    )


def M_gg_76_4_4():
    def func(k):
        return 8 / 4725

    return func


def N_gg_76_4_4(theta, phi):
    return (
        (675 / 352) * np.cos(4 * phi)
        + (6075 / 2464) * np.cos(2 * theta)
        + (675 / 352) * np.cos(2 * phi - 3 * theta)
        - 225 / 308 * np.cos(2 * phi - theta)
        + (6075 / 2464) * np.cos(2 * phi + theta)
        + (1575 / 352) * np.cos(4 * phi - 2 * theta)
        - 2025 / 2464
    )


def M_gg_76_6_0():
    def func(k):
        return 8 / 4725

    return func


def N_gg_76_6_0(theta, phi):
    return (
        (4725 / 1408) * np.cos(4 * phi)
        + (1125 / 1408) * np.cos(2 * theta)
        + (4725 / 2816) * np.cos(2 * phi - theta)
        + (1575 / 704) * np.cos(2 * phi + theta)
        + (1575 / 5632) * np.cos(2 * phi + 3 * theta)
        + (4725 / 2816) * np.cos(4 * phi + 2 * theta)
        + (4725 / 512) * np.cos(6 * phi + theta)
        + 3375 / 2816
    )


def M_gg_76_6_1():
    def func(k):
        return 8 / 4725

    return func


def N_gg_76_6_1(theta, phi):
    return (
        (4725 / 1408) * np.cos(4 * phi)
        + (1125 / 1408) * np.cos(2 * theta)
        + (1575 / 5632) * np.cos(2 * phi - 3 * theta)
        + (1575 / 704) * np.cos(2 * phi - theta)
        + (4725 / 2816) * np.cos(2 * phi + theta)
        + (4725 / 2816) * np.cos(4 * phi - 2 * theta)
        + (4725 / 512) * np.cos(6 * phi - theta)
        + 3375 / 2816
    )


def M_gg_77_0_0():
    def func(k):
        return -1 / 9

    return func


def N_gg_77_0_0(theta, phi):
    return 1


def M_gg_77_0_1():
    def func(k):
        return -4 / 225

    return func


def N_gg_77_0_1(theta, phi):
    return (15 / 4) * np.cos(2 * theta) + 5 / 4


def M_gg_77_2_0():
    def func(k):
        return -2 / 45

    return func


def N_gg_77_2_0(theta, phi):
    return (15 / 4) * np.cos(2 * phi + theta) + 5 / 4


def M_gg_77_2_1():
    def func(k):
        return -2 / 45

    return func


def N_gg_77_2_1(theta, phi):
    return (15 / 4) * np.cos(2 * phi - theta) + 5 / 4


def M_gg_77_2_2():
    def func(k):
        return -4 / 225

    return func


def N_gg_77_2_2(theta, phi):
    return (
        (75 / 28) * np.cos(2 * theta)
        + (75 / 28) * np.cos(2 * phi - theta)
        + (75 / 28) * np.cos(2 * phi + theta)
        - 25 / 28
    )


def M_gg_77_4_0():
    def func(k):
        return -4 / 225

    return func


def N_gg_77_4_0(theta, phi):
    return (
        (225 / 32) * np.cos(4 * phi)
        + (135 / 224) * np.cos(2 * theta)
        + (225 / 112) * np.cos(2 * phi - theta)
        + (225 / 112) * np.cos(2 * phi + theta)
        + 135 / 112
    )


def M_gg_78_0_0():
    def func(k):
        return -1 / 15

    return func


def N_gg_78_0_0(theta, phi):
    return 1


def M_gg_78_0_1():
    def func(k):
        return -8 / 525

    return func


def N_gg_78_0_1(theta, phi):
    return (15 / 4) * np.cos(2 * theta) + 5 / 4


def M_gg_78_2_0():
    def func(k):
        return -17 / 525

    return func


def N_gg_78_2_0(theta, phi):
    return (15 / 4) * np.cos(2 * phi + theta) + 5 / 4


def M_gg_78_2_1():
    def func(k):
        return -17 / 525

    return func


def N_gg_78_2_1(theta, phi):
    return (15 / 4) * np.cos(2 * phi - theta) + 5 / 4


def M_gg_78_2_2():
    def func(k):
        return -8 / 525

    return func


def N_gg_78_2_2(theta, phi):
    return (
        (75 / 28) * np.cos(2 * theta)
        + (75 / 28) * np.cos(2 * phi - theta)
        + (75 / 28) * np.cos(2 * phi + theta)
        - 25 / 28
    )


def M_gg_78_2_3():
    def func(k):
        return -8 / 4725

    return func


def N_gg_78_2_3(theta, phi):
    return (
        (225 / 112) * np.cos(2 * theta)
        + (135 / 224) * np.cos(2 * phi - theta)
        + (225 / 112) * np.cos(2 * phi + theta)
        + (225 / 32) * np.cos(2 * phi + 3 * theta)
        + 135 / 112
    )


def M_gg_78_2_4():
    def func(k):
        return -8 / 4725

    return func


def N_gg_78_2_4(theta, phi):
    return (
        (225 / 112) * np.cos(2 * theta)
        + (225 / 32) * np.cos(2 * phi - 3 * theta)
        + (225 / 112) * np.cos(2 * phi - theta)
        + (135 / 224) * np.cos(2 * phi + theta)
        + 135 / 112
    )


def M_gg_78_4_0():
    def func(k):
        return -4 / 945

    return func


def N_gg_78_4_0(theta, phi):
    return (
        (45 / 16) * np.cos(2 * phi + theta)
        + (315 / 64) * np.cos(4 * phi + 2 * theta)
        + 81 / 64
    )


def M_gg_78_4_1():
    def func(k):
        return -8 / 525

    return func


def N_gg_78_4_1(theta, phi):
    return (
        (225 / 32) * np.cos(4 * phi)
        + (135 / 224) * np.cos(2 * theta)
        + (225 / 112) * np.cos(2 * phi - theta)
        + (225 / 112) * np.cos(2 * phi + theta)
        + 135 / 112
    )


def M_gg_78_4_2():
    def func(k):
        return -8 / 4725

    return func


def N_gg_78_4_2(theta, phi):
    return (
        (675 / 352) * np.cos(4 * phi)
        + (6075 / 2464) * np.cos(2 * theta)
        + (6075 / 2464) * np.cos(2 * phi - theta)
        - 225 / 308 * np.cos(2 * phi + theta)
        + (675 / 352) * np.cos(2 * phi + 3 * theta)
        + (1575 / 352) * np.cos(4 * phi + 2 * theta)
        - 2025 / 2464
    )


def M_gg_78_4_3():
    def func(k):
        return -4 / 945

    return func


def N_gg_78_4_3(theta, phi):
    return (
        (45 / 16) * np.cos(2 * phi - theta)
        + (315 / 64) * np.cos(4 * phi - 2 * theta)
        + 81 / 64
    )


def M_gg_78_4_4():
    def func(k):
        return -8 / 4725

    return func


def N_gg_78_4_4(theta, phi):
    return (
        (675 / 352) * np.cos(4 * phi)
        + (6075 / 2464) * np.cos(2 * theta)
        + (675 / 352) * np.cos(2 * phi - 3 * theta)
        - 225 / 308 * np.cos(2 * phi - theta)
        + (6075 / 2464) * np.cos(2 * phi + theta)
        + (1575 / 352) * np.cos(4 * phi - 2 * theta)
        - 2025 / 2464
    )


def M_gg_78_6_0():
    def func(k):
        return -8 / 4725

    return func


def N_gg_78_6_0(theta, phi):
    return (
        (4725 / 1408) * np.cos(4 * phi)
        + (1125 / 1408) * np.cos(2 * theta)
        + (4725 / 2816) * np.cos(2 * phi - theta)
        + (1575 / 704) * np.cos(2 * phi + theta)
        + (1575 / 5632) * np.cos(2 * phi + 3 * theta)
        + (4725 / 2816) * np.cos(4 * phi + 2 * theta)
        + (4725 / 512) * np.cos(6 * phi + theta)
        + 3375 / 2816
    )


def M_gg_78_6_1():
    def func(k):
        return -8 / 4725

    return func


def N_gg_78_6_1(theta, phi):
    return (
        (4725 / 1408) * np.cos(4 * phi)
        + (1125 / 1408) * np.cos(2 * theta)
        + (1575 / 5632) * np.cos(2 * phi - 3 * theta)
        + (1575 / 704) * np.cos(2 * phi - theta)
        + (4725 / 2816) * np.cos(2 * phi + theta)
        + (4725 / 2816) * np.cos(4 * phi - 2 * theta)
        + (4725 / 512) * np.cos(6 * phi - theta)
        + 3375 / 2816
    )


def M_gg_79_0_0():
    def func(k):
        return (2 / 9) * k**2

    return func


def N_gg_79_0_0(theta, phi):
    return 1


def M_gg_79_0_1():
    def func(k):
        return (8 / 225) * k**2

    return func


def N_gg_79_0_1(theta, phi):
    return (15 / 4) * np.cos(2 * theta) + 5 / 4


def M_gg_79_2_0():
    def func(k):
        return (4 / 45) * k**2

    return func


def N_gg_79_2_0(theta, phi):
    return (15 / 4) * np.cos(2 * phi + theta) + 5 / 4


def M_gg_79_2_1():
    def func(k):
        return (4 / 45) * k**2

    return func


def N_gg_79_2_1(theta, phi):
    return (15 / 4) * np.cos(2 * phi - theta) + 5 / 4


def M_gg_79_2_2():
    def func(k):
        return (8 / 225) * k**2

    return func


def N_gg_79_2_2(theta, phi):
    return (
        (75 / 28) * np.cos(2 * theta)
        + (75 / 28) * np.cos(2 * phi - theta)
        + (75 / 28) * np.cos(2 * phi + theta)
        - 25 / 28
    )


def M_gg_79_4_0():
    def func(k):
        return (8 / 225) * k**2

    return func


def N_gg_79_4_0(theta, phi):
    return (
        (225 / 32) * np.cos(4 * phi)
        + (135 / 224) * np.cos(2 * theta)
        + (225 / 112) * np.cos(2 * phi - theta)
        + (225 / 112) * np.cos(2 * phi + theta)
        + 135 / 112
    )


def M_gg_80_0_0():
    def func(k):
        return (2 / 15) * k**2

    return func


def N_gg_80_0_0(theta, phi):
    return 1


def M_gg_80_0_1():
    def func(k):
        return (16 / 525) * k**2

    return func


def N_gg_80_0_1(theta, phi):
    return (15 / 4) * np.cos(2 * theta) + 5 / 4


def M_gg_80_2_0():
    def func(k):
        return (34 / 525) * k**2

    return func


def N_gg_80_2_0(theta, phi):
    return (15 / 4) * np.cos(2 * phi + theta) + 5 / 4


def M_gg_80_2_1():
    def func(k):
        return (34 / 525) * k**2

    return func


def N_gg_80_2_1(theta, phi):
    return (15 / 4) * np.cos(2 * phi - theta) + 5 / 4


def M_gg_80_2_2():
    def func(k):
        return (16 / 525) * k**2

    return func


def N_gg_80_2_2(theta, phi):
    return (
        (75 / 28) * np.cos(2 * theta)
        + (75 / 28) * np.cos(2 * phi - theta)
        + (75 / 28) * np.cos(2 * phi + theta)
        - 25 / 28
    )


def M_gg_80_2_3():
    def func(k):
        return (16 / 4725) * k**2

    return func


def N_gg_80_2_3(theta, phi):
    return (
        (225 / 112) * np.cos(2 * theta)
        + (135 / 224) * np.cos(2 * phi - theta)
        + (225 / 112) * np.cos(2 * phi + theta)
        + (225 / 32) * np.cos(2 * phi + 3 * theta)
        + 135 / 112
    )


def M_gg_80_2_4():
    def func(k):
        return (16 / 4725) * k**2

    return func


def N_gg_80_2_4(theta, phi):
    return (
        (225 / 112) * np.cos(2 * theta)
        + (225 / 32) * np.cos(2 * phi - 3 * theta)
        + (225 / 112) * np.cos(2 * phi - theta)
        + (135 / 224) * np.cos(2 * phi + theta)
        + 135 / 112
    )


def M_gg_80_4_0():
    def func(k):
        return (8 / 945) * k**2

    return func


def N_gg_80_4_0(theta, phi):
    return (
        (45 / 16) * np.cos(2 * phi + theta)
        + (315 / 64) * np.cos(4 * phi + 2 * theta)
        + 81 / 64
    )


def M_gg_80_4_1():
    def func(k):
        return (16 / 525) * k**2

    return func


def N_gg_80_4_1(theta, phi):
    return (
        (225 / 32) * np.cos(4 * phi)
        + (135 / 224) * np.cos(2 * theta)
        + (225 / 112) * np.cos(2 * phi - theta)
        + (225 / 112) * np.cos(2 * phi + theta)
        + 135 / 112
    )


def M_gg_80_4_2():
    def func(k):
        return (16 / 4725) * k**2

    return func


def N_gg_80_4_2(theta, phi):
    return (
        (675 / 352) * np.cos(4 * phi)
        + (6075 / 2464) * np.cos(2 * theta)
        + (6075 / 2464) * np.cos(2 * phi - theta)
        - 225 / 308 * np.cos(2 * phi + theta)
        + (675 / 352) * np.cos(2 * phi + 3 * theta)
        + (1575 / 352) * np.cos(4 * phi + 2 * theta)
        - 2025 / 2464
    )


def M_gg_80_4_3():
    def func(k):
        return (8 / 945) * k**2

    return func


def N_gg_80_4_3(theta, phi):
    return (
        (45 / 16) * np.cos(2 * phi - theta)
        + (315 / 64) * np.cos(4 * phi - 2 * theta)
        + 81 / 64
    )


def M_gg_80_4_4():
    def func(k):
        return (16 / 4725) * k**2

    return func


def N_gg_80_4_4(theta, phi):
    return (
        (675 / 352) * np.cos(4 * phi)
        + (6075 / 2464) * np.cos(2 * theta)
        + (675 / 352) * np.cos(2 * phi - 3 * theta)
        - 225 / 308 * np.cos(2 * phi - theta)
        + (6075 / 2464) * np.cos(2 * phi + theta)
        + (1575 / 352) * np.cos(4 * phi - 2 * theta)
        - 2025 / 2464
    )


def M_gg_80_6_0():
    def func(k):
        return (16 / 4725) * k**2

    return func


def N_gg_80_6_0(theta, phi):
    return (
        (4725 / 1408) * np.cos(4 * phi)
        + (1125 / 1408) * np.cos(2 * theta)
        + (4725 / 2816) * np.cos(2 * phi - theta)
        + (1575 / 704) * np.cos(2 * phi + theta)
        + (1575 / 5632) * np.cos(2 * phi + 3 * theta)
        + (4725 / 2816) * np.cos(4 * phi + 2 * theta)
        + (4725 / 512) * np.cos(6 * phi + theta)
        + 3375 / 2816
    )


def M_gg_80_6_1():
    def func(k):
        return (16 / 4725) * k**2

    return func


def N_gg_80_6_1(theta, phi):
    return (
        (4725 / 1408) * np.cos(4 * phi)
        + (1125 / 1408) * np.cos(2 * theta)
        + (1575 / 5632) * np.cos(2 * phi - 3 * theta)
        + (1575 / 704) * np.cos(2 * phi - theta)
        + (4725 / 2816) * np.cos(2 * phi + theta)
        + (4725 / 2816) * np.cos(4 * phi - 2 * theta)
        + (4725 / 512) * np.cos(6 * phi - theta)
        + 3375 / 2816
    )


def M_gg_81_0_0():
    def func(k):
        return -1 / 9 * k**2

    return func


def N_gg_81_0_0(theta, phi):
    return 1


def M_gg_81_0_1():
    def func(k):
        return -4 / 225 * k**2

    return func


def N_gg_81_0_1(theta, phi):
    return (15 / 4) * np.cos(2 * theta) + 5 / 4


def M_gg_81_2_0():
    def func(k):
        return -2 / 45 * k**2

    return func


def N_gg_81_2_0(theta, phi):
    return (15 / 4) * np.cos(2 * phi + theta) + 5 / 4


def M_gg_81_2_1():
    def func(k):
        return -2 / 45 * k**2

    return func


def N_gg_81_2_1(theta, phi):
    return (15 / 4) * np.cos(2 * phi - theta) + 5 / 4


def M_gg_81_2_2():
    def func(k):
        return -4 / 225 * k**2

    return func


def N_gg_81_2_2(theta, phi):
    return (
        (75 / 28) * np.cos(2 * theta)
        + (75 / 28) * np.cos(2 * phi - theta)
        + (75 / 28) * np.cos(2 * phi + theta)
        - 25 / 28
    )


def M_gg_81_4_0():
    def func(k):
        return -4 / 225 * k**2

    return func


def N_gg_81_4_0(theta, phi):
    return (
        (225 / 32) * np.cos(4 * phi)
        + (135 / 224) * np.cos(2 * theta)
        + (225 / 112) * np.cos(2 * phi - theta)
        + (225 / 112) * np.cos(2 * phi + theta)
        + 135 / 112
    )


def M_gg_82_0_0():
    def func(k):
        return -2 / 9 * k**2

    return func


def N_gg_82_0_0(theta, phi):
    return 1


def M_gg_82_0_1():
    def func(k):
        return -8 / 225 * k**2

    return func


def N_gg_82_0_1(theta, phi):
    return (15 / 4) * np.cos(2 * theta) + 5 / 4


def M_gg_82_2_0():
    def func(k):
        return -4 / 45 * k**2

    return func


def N_gg_82_2_0(theta, phi):
    return (15 / 4) * np.cos(2 * phi + theta) + 5 / 4


def M_gg_82_2_1():
    def func(k):
        return -4 / 45 * k**2

    return func


def N_gg_82_2_1(theta, phi):
    return (15 / 4) * np.cos(2 * phi - theta) + 5 / 4


def M_gg_82_2_2():
    def func(k):
        return -8 / 225 * k**2

    return func


def N_gg_82_2_2(theta, phi):
    return (
        (75 / 28) * np.cos(2 * theta)
        + (75 / 28) * np.cos(2 * phi - theta)
        + (75 / 28) * np.cos(2 * phi + theta)
        - 25 / 28
    )


def M_gg_82_4_0():
    def func(k):
        return -8 / 225 * k**2

    return func


def N_gg_82_4_0(theta, phi):
    return (
        (225 / 32) * np.cos(4 * phi)
        + (135 / 224) * np.cos(2 * theta)
        + (225 / 112) * np.cos(2 * phi - theta)
        + (225 / 112) * np.cos(2 * phi + theta)
        + 135 / 112
    )


def M_gg_83_0_0():
    def func(k):
        return -2 / 9 * k**2

    return func


def N_gg_83_0_0(theta, phi):
    return 1


def M_gg_83_0_1():
    def func(k):
        return -8 / 225 * k**2

    return func


def N_gg_83_0_1(theta, phi):
    return (15 / 4) * np.cos(2 * theta) + 5 / 4


def M_gg_83_2_0():
    def func(k):
        return -4 / 45 * k**2

    return func


def N_gg_83_2_0(theta, phi):
    return (15 / 4) * np.cos(2 * phi + theta) + 5 / 4


def M_gg_83_2_1():
    def func(k):
        return -4 / 45 * k**2

    return func


def N_gg_83_2_1(theta, phi):
    return (15 / 4) * np.cos(2 * phi - theta) + 5 / 4


def M_gg_83_2_2():
    def func(k):
        return -8 / 225 * k**2

    return func


def N_gg_83_2_2(theta, phi):
    return (
        (75 / 28) * np.cos(2 * theta)
        + (75 / 28) * np.cos(2 * phi - theta)
        + (75 / 28) * np.cos(2 * phi + theta)
        - 25 / 28
    )


def M_gg_83_4_0():
    def func(k):
        return -8 / 225 * k**2

    return func


def N_gg_83_4_0(theta, phi):
    return (
        (225 / 32) * np.cos(4 * phi)
        + (135 / 224) * np.cos(2 * theta)
        + (225 / 112) * np.cos(2 * phi - theta)
        + (225 / 112) * np.cos(2 * phi + theta)
        + 135 / 112
    )


def M_gg_84_0_0():
    def func(k):
        return -2 / 3 * k**4

    return func


def N_gg_84_0_0(theta, phi):
    return 1


def M_gg_84_0_1():
    def func(k):
        return -8 / 75 * k**4

    return func


def N_gg_84_0_1(theta, phi):
    return (15 / 4) * np.cos(2 * theta) + 5 / 4


def M_gg_84_2_0():
    def func(k):
        return -4 / 15 * k**4

    return func


def N_gg_84_2_0(theta, phi):
    return (15 / 4) * np.cos(2 * phi + theta) + 5 / 4


def M_gg_84_2_1():
    def func(k):
        return -4 / 15 * k**4

    return func


def N_gg_84_2_1(theta, phi):
    return (15 / 4) * np.cos(2 * phi - theta) + 5 / 4


def M_gg_84_2_2():
    def func(k):
        return -8 / 75 * k**4

    return func


def N_gg_84_2_2(theta, phi):
    return (
        (75 / 28) * np.cos(2 * theta)
        + (75 / 28) * np.cos(2 * phi - theta)
        + (75 / 28) * np.cos(2 * phi + theta)
        - 25 / 28
    )


def M_gg_84_4_0():
    def func(k):
        return -8 / 75 * k**4

    return func


def N_gg_84_4_0(theta, phi):
    return (
        (225 / 32) * np.cos(4 * phi)
        + (135 / 224) * np.cos(2 * theta)
        + (225 / 112) * np.cos(2 * phi - theta)
        + (225 / 112) * np.cos(2 * phi + theta)
        + 135 / 112
    )


def M_gg_85_0_0():
    def func(k):
        return -2 / 3 * k**4

    return func


def N_gg_85_0_0(theta, phi):
    return 1


def M_gg_85_0_1():
    def func(k):
        return -8 / 75 * k**4

    return func


def N_gg_85_0_1(theta, phi):
    return (15 / 4) * np.cos(2 * theta) + 5 / 4


def M_gg_85_2_0():
    def func(k):
        return -4 / 15 * k**4

    return func


def N_gg_85_2_0(theta, phi):
    return (15 / 4) * np.cos(2 * phi + theta) + 5 / 4


def M_gg_85_2_1():
    def func(k):
        return -4 / 15 * k**4

    return func


def N_gg_85_2_1(theta, phi):
    return (15 / 4) * np.cos(2 * phi - theta) + 5 / 4


def M_gg_85_2_2():
    def func(k):
        return -8 / 75 * k**4

    return func


def N_gg_85_2_2(theta, phi):
    return (
        (75 / 28) * np.cos(2 * theta)
        + (75 / 28) * np.cos(2 * phi - theta)
        + (75 / 28) * np.cos(2 * phi + theta)
        - 25 / 28
    )


def M_gg_85_4_0():
    def func(k):
        return -8 / 75 * k**4

    return func


def N_gg_85_4_0(theta, phi):
    return (
        (225 / 32) * np.cos(4 * phi)
        + (135 / 224) * np.cos(2 * theta)
        + (225 / 112) * np.cos(2 * phi - theta)
        + (225 / 112) * np.cos(2 * phi + theta)
        + 135 / 112
    )


def M_gg_86_0_0():
    def func(k):
        return (1 / 9) * k**2

    return func


def N_gg_86_0_0(theta, phi):
    return 1


def M_gg_86_0_1():
    def func(k):
        return (4 / 225) * k**2

    return func


def N_gg_86_0_1(theta, phi):
    return (15 / 4) * np.cos(2 * theta) + 5 / 4


def M_gg_86_2_0():
    def func(k):
        return (2 / 45) * k**2

    return func


def N_gg_86_2_0(theta, phi):
    return (15 / 4) * np.cos(2 * phi + theta) + 5 / 4


def M_gg_86_2_1():
    def func(k):
        return (2 / 45) * k**2

    return func


def N_gg_86_2_1(theta, phi):
    return (15 / 4) * np.cos(2 * phi - theta) + 5 / 4


def M_gg_86_2_2():
    def func(k):
        return (4 / 225) * k**2

    return func


def N_gg_86_2_2(theta, phi):
    return (
        (75 / 28) * np.cos(2 * theta)
        + (75 / 28) * np.cos(2 * phi - theta)
        + (75 / 28) * np.cos(2 * phi + theta)
        - 25 / 28
    )


def M_gg_86_4_0():
    def func(k):
        return (4 / 225) * k**2

    return func


def N_gg_86_4_0(theta, phi):
    return (
        (225 / 32) * np.cos(4 * phi)
        + (135 / 224) * np.cos(2 * theta)
        + (225 / 112) * np.cos(2 * phi - theta)
        + (225 / 112) * np.cos(2 * phi + theta)
        + 135 / 112
    )


def M_gg_87_0_0():
    def func(k):
        return (1 / 9) * k**2

    return func


def N_gg_87_0_0(theta, phi):
    return 1


def M_gg_87_0_1():
    def func(k):
        return (4 / 225) * k**2

    return func


def N_gg_87_0_1(theta, phi):
    return (15 / 4) * np.cos(2 * theta) + 5 / 4


def M_gg_87_2_0():
    def func(k):
        return (2 / 45) * k**2

    return func


def N_gg_87_2_0(theta, phi):
    return (15 / 4) * np.cos(2 * phi + theta) + 5 / 4


def M_gg_87_2_1():
    def func(k):
        return (2 / 45) * k**2

    return func


def N_gg_87_2_1(theta, phi):
    return (15 / 4) * np.cos(2 * phi - theta) + 5 / 4


def M_gg_87_2_2():
    def func(k):
        return (4 / 225) * k**2

    return func


def N_gg_87_2_2(theta, phi):
    return (
        (75 / 28) * np.cos(2 * theta)
        + (75 / 28) * np.cos(2 * phi - theta)
        + (75 / 28) * np.cos(2 * phi + theta)
        - 25 / 28
    )


def M_gg_87_4_0():
    def func(k):
        return (4 / 225) * k**2

    return func


def N_gg_87_4_0(theta, phi):
    return (
        (225 / 32) * np.cos(4 * phi)
        + (135 / 224) * np.cos(2 * theta)
        + (225 / 112) * np.cos(2 * phi - theta)
        + (225 / 112) * np.cos(2 * phi + theta)
        + 135 / 112
    )


def M_gg_88_0_0():
    def func(k):
        return (1 / 9) * k**2

    return func


def N_gg_88_0_0(theta, phi):
    return 1


def M_gg_88_0_1():
    def func(k):
        return (4 / 225) * k**2

    return func


def N_gg_88_0_1(theta, phi):
    return (15 / 4) * np.cos(2 * theta) + 5 / 4


def M_gg_88_2_0():
    def func(k):
        return (2 / 45) * k**2

    return func


def N_gg_88_2_0(theta, phi):
    return (15 / 4) * np.cos(2 * phi + theta) + 5 / 4


def M_gg_88_2_1():
    def func(k):
        return (2 / 45) * k**2

    return func


def N_gg_88_2_1(theta, phi):
    return (15 / 4) * np.cos(2 * phi - theta) + 5 / 4


def M_gg_88_2_2():
    def func(k):
        return (4 / 225) * k**2

    return func


def N_gg_88_2_2(theta, phi):
    return (
        (75 / 28) * np.cos(2 * theta)
        + (75 / 28) * np.cos(2 * phi - theta)
        + (75 / 28) * np.cos(2 * phi + theta)
        - 25 / 28
    )


def M_gg_88_4_0():
    def func(k):
        return (4 / 225) * k**2

    return func


def N_gg_88_4_0(theta, phi):
    return (
        (225 / 32) * np.cos(4 * phi)
        + (135 / 224) * np.cos(2 * theta)
        + (225 / 112) * np.cos(2 * phi - theta)
        + (225 / 112) * np.cos(2 * phi + theta)
        + 135 / 112
    )


def M_gg_89_0_0():
    def func(k):
        return (1 / 9) * k**2

    return func


def N_gg_89_0_0(theta, phi):
    return 1


def M_gg_89_0_1():
    def func(k):
        return (4 / 225) * k**2

    return func


def N_gg_89_0_1(theta, phi):
    return (15 / 4) * np.cos(2 * theta) + 5 / 4


def M_gg_89_2_0():
    def func(k):
        return (2 / 45) * k**2

    return func


def N_gg_89_2_0(theta, phi):
    return (15 / 4) * np.cos(2 * phi + theta) + 5 / 4


def M_gg_89_2_1():
    def func(k):
        return (2 / 45) * k**2

    return func


def N_gg_89_2_1(theta, phi):
    return (15 / 4) * np.cos(2 * phi - theta) + 5 / 4


def M_gg_89_2_2():
    def func(k):
        return (4 / 225) * k**2

    return func


def N_gg_89_2_2(theta, phi):
    return (
        (75 / 28) * np.cos(2 * theta)
        + (75 / 28) * np.cos(2 * phi - theta)
        + (75 / 28) * np.cos(2 * phi + theta)
        - 25 / 28
    )


def M_gg_89_4_0():
    def func(k):
        return (4 / 225) * k**2

    return func


def N_gg_89_4_0(theta, phi):
    return (
        (225 / 32) * np.cos(4 * phi)
        + (135 / 224) * np.cos(2 * theta)
        + (225 / 112) * np.cos(2 * phi - theta)
        + (225 / 112) * np.cos(2 * phi + theta)
        + 135 / 112
    )


def M_gg_90_0_0():
    def func(k):
        return (1 / 9) * k**2

    return func


def N_gg_90_0_0(theta, phi):
    return 1


def M_gg_90_0_1():
    def func(k):
        return (4 / 225) * k**2

    return func


def N_gg_90_0_1(theta, phi):
    return (15 / 4) * np.cos(2 * theta) + 5 / 4


def M_gg_90_2_0():
    def func(k):
        return (2 / 45) * k**2

    return func


def N_gg_90_2_0(theta, phi):
    return (15 / 4) * np.cos(2 * phi + theta) + 5 / 4


def M_gg_90_2_1():
    def func(k):
        return (2 / 45) * k**2

    return func


def N_gg_90_2_1(theta, phi):
    return (15 / 4) * np.cos(2 * phi - theta) + 5 / 4


def M_gg_90_2_2():
    def func(k):
        return (4 / 225) * k**2

    return func


def N_gg_90_2_2(theta, phi):
    return (
        (75 / 28) * np.cos(2 * theta)
        + (75 / 28) * np.cos(2 * phi - theta)
        + (75 / 28) * np.cos(2 * phi + theta)
        - 25 / 28
    )


def M_gg_90_4_0():
    def func(k):
        return (4 / 225) * k**2

    return func


def N_gg_90_4_0(theta, phi):
    return (
        (225 / 32) * np.cos(4 * phi)
        + (135 / 224) * np.cos(2 * theta)
        + (225 / 112) * np.cos(2 * phi - theta)
        + (225 / 112) * np.cos(2 * phi + theta)
        + 135 / 112
    )


def M_gg_91_0_0():
    def func(k):
        return (2 / 9) * k**2

    return func


def N_gg_91_0_0(theta, phi):
    return 1


def M_gg_91_0_1():
    def func(k):
        return (8 / 225) * k**2

    return func


def N_gg_91_0_1(theta, phi):
    return (15 / 4) * np.cos(2 * theta) + 5 / 4


def M_gg_91_2_0():
    def func(k):
        return (4 / 45) * k**2

    return func


def N_gg_91_2_0(theta, phi):
    return (15 / 4) * np.cos(2 * phi + theta) + 5 / 4


def M_gg_91_2_1():
    def func(k):
        return (4 / 45) * k**2

    return func


def N_gg_91_2_1(theta, phi):
    return (15 / 4) * np.cos(2 * phi - theta) + 5 / 4


def M_gg_91_2_2():
    def func(k):
        return (8 / 225) * k**2

    return func


def N_gg_91_2_2(theta, phi):
    return (
        (75 / 28) * np.cos(2 * theta)
        + (75 / 28) * np.cos(2 * phi - theta)
        + (75 / 28) * np.cos(2 * phi + theta)
        - 25 / 28
    )


def M_gg_91_4_0():
    def func(k):
        return (8 / 225) * k**2

    return func


def N_gg_91_4_0(theta, phi):
    return (
        (225 / 32) * np.cos(4 * phi)
        + (135 / 224) * np.cos(2 * theta)
        + (225 / 112) * np.cos(2 * phi - theta)
        + (225 / 112) * np.cos(2 * phi + theta)
        + 135 / 112
    )


def M_gg_92_0_0():
    def func(k):
        return (2 / 3) * k**4

    return func


def N_gg_92_0_0(theta, phi):
    return 1


def M_gg_92_0_1():
    def func(k):
        return (8 / 75) * k**4

    return func


def N_gg_92_0_1(theta, phi):
    return (15 / 4) * np.cos(2 * theta) + 5 / 4


def M_gg_92_2_0():
    def func(k):
        return (4 / 15) * k**4

    return func


def N_gg_92_2_0(theta, phi):
    return (15 / 4) * np.cos(2 * phi + theta) + 5 / 4


def M_gg_92_2_1():
    def func(k):
        return (4 / 15) * k**4

    return func


def N_gg_92_2_1(theta, phi):
    return (15 / 4) * np.cos(2 * phi - theta) + 5 / 4


def M_gg_92_2_2():
    def func(k):
        return (8 / 75) * k**4

    return func


def N_gg_92_2_2(theta, phi):
    return (
        (75 / 28) * np.cos(2 * theta)
        + (75 / 28) * np.cos(2 * phi - theta)
        + (75 / 28) * np.cos(2 * phi + theta)
        - 25 / 28
    )


def M_gg_92_4_0():
    def func(k):
        return (8 / 75) * k**4

    return func


def N_gg_92_4_0(theta, phi):
    return (
        (225 / 32) * np.cos(4 * phi)
        + (135 / 224) * np.cos(2 * theta)
        + (225 / 112) * np.cos(2 * phi - theta)
        + (225 / 112) * np.cos(2 * phi + theta)
        + 135 / 112
    )


def M_gg_93_0_0():
    def func(k):
        return 1 / 9

    return func


def N_gg_93_0_0(theta, phi):
    return -2


def M_gg_93_0_1():
    def func(k):
        return 4 / 225

    return func


def N_gg_93_0_1(theta, phi):
    return -15 / 2 * np.cos(2 * theta) - 5 / 2


def M_gg_93_2_0():
    def func(k):
        return 2 / 45

    return func


def N_gg_93_2_0(theta, phi):
    return -15 / 2 * np.cos(2 * phi + theta) - 5 / 2


def M_gg_93_2_1():
    def func(k):
        return 2 / 45

    return func


def N_gg_93_2_1(theta, phi):
    return -15 / 2 * np.cos(2 * phi - theta) - 5 / 2


def M_gg_93_2_2():
    def func(k):
        return 4 / 225

    return func


def N_gg_93_2_2(theta, phi):
    return (
        -75 / 14 * np.cos(2 * theta)
        - 75 / 14 * np.cos(2 * phi - theta)
        - 75 / 14 * np.cos(2 * phi + theta)
        + 25 / 14
    )


def M_gg_93_4_0():
    def func(k):
        return 4 / 225

    return func


def N_gg_93_4_0(theta, phi):
    return (
        -225 / 16 * np.cos(4 * phi)
        - 135 / 112 * np.cos(2 * theta)
        - 225 / 56 * np.cos(2 * phi - theta)
        - 225 / 56 * np.cos(2 * phi + theta)
        - 135 / 56
    )


def M_gg_94_0_0():
    def func(k):
        return 1 / 15

    return func


def N_gg_94_0_0(theta, phi):
    return 3 * np.cos(2 * theta) + 3


def M_gg_94_2_0():
    def func(k):
        return 1 / 15

    return func


def N_gg_94_2_0(theta, phi):
    return (
        (3 / 2) * np.cos(2 * theta)
        + (9 / 2) * np.cos(2 * phi - theta)
        + (9 / 2) * np.cos(2 * phi + theta)
        + 3 / 2
    )


def M_gg_94_2_1():
    def func(k):
        return 1 / 105

    return func


def N_gg_94_2_1(theta, phi):
    return (
        (9 / 4) * np.cos(2 * theta)
        + (9 / 8) * np.cos(2 * phi - theta)
        + (27 / 4) * np.cos(2 * phi + theta)
        + (45 / 8) * np.cos(2 * phi + 3 * theta)
        + 9 / 4
    )


def M_gg_94_2_2():
    def func(k):
        return 1 / 105

    return func


def N_gg_94_2_2(theta, phi):
    return (
        (9 / 4) * np.cos(2 * theta)
        + (45 / 8) * np.cos(2 * phi - 3 * theta)
        + (27 / 4) * np.cos(2 * phi - theta)
        + (9 / 8) * np.cos(2 * phi + theta)
        + 9 / 4
    )


def M_gg_94_4_0():
    def func(k):
        return 1 / 105

    return func


def N_gg_94_4_0(theta, phi):
    return (
        (105 / 16) * np.cos(4 * phi)
        + (27 / 16) * np.cos(2 * theta)
        + (45 / 16) * np.cos(2 * phi - theta)
        + (15 / 4) * np.cos(2 * phi + theta)
        + (15 / 16) * np.cos(2 * phi + 3 * theta)
        + (105 / 16) * np.cos(4 * phi + 2 * theta)
        + 27 / 16
    )


def M_gg_94_4_1():
    def func(k):
        return 1 / 105

    return func


def N_gg_94_4_1(theta, phi):
    return (
        (105 / 16) * np.cos(4 * phi)
        + (27 / 16) * np.cos(2 * theta)
        + (15 / 16) * np.cos(2 * phi - 3 * theta)
        + (15 / 4) * np.cos(2 * phi - theta)
        + (45 / 16) * np.cos(2 * phi + theta)
        + (105 / 16) * np.cos(4 * phi - 2 * theta)
        + 27 / 16
    )


def M_gg_95_0_0():
    def func(k):
        return -1 / 15 * k**2

    return func


def N_gg_95_0_0(theta, phi):
    return 1


def M_gg_95_0_1():
    def func(k):
        return -8 / 525 * k**2

    return func


def N_gg_95_0_1(theta, phi):
    return (15 / 4) * np.cos(2 * theta) + 5 / 4


def M_gg_95_2_0():
    def func(k):
        return -17 / 525 * k**2

    return func


def N_gg_95_2_0(theta, phi):
    return (15 / 4) * np.cos(2 * phi + theta) + 5 / 4


def M_gg_95_2_1():
    def func(k):
        return -17 / 525 * k**2

    return func


def N_gg_95_2_1(theta, phi):
    return (15 / 4) * np.cos(2 * phi - theta) + 5 / 4


def M_gg_95_2_2():
    def func(k):
        return -8 / 525 * k**2

    return func


def N_gg_95_2_2(theta, phi):
    return (
        (75 / 28) * np.cos(2 * theta)
        + (75 / 28) * np.cos(2 * phi - theta)
        + (75 / 28) * np.cos(2 * phi + theta)
        - 25 / 28
    )


def M_gg_95_2_3():
    def func(k):
        return -8 / 4725 * k**2

    return func


def N_gg_95_2_3(theta, phi):
    return (
        (225 / 112) * np.cos(2 * theta)
        + (135 / 224) * np.cos(2 * phi - theta)
        + (225 / 112) * np.cos(2 * phi + theta)
        + (225 / 32) * np.cos(2 * phi + 3 * theta)
        + 135 / 112
    )


def M_gg_95_2_4():
    def func(k):
        return -8 / 4725 * k**2

    return func


def N_gg_95_2_4(theta, phi):
    return (
        (225 / 112) * np.cos(2 * theta)
        + (225 / 32) * np.cos(2 * phi - 3 * theta)
        + (225 / 112) * np.cos(2 * phi - theta)
        + (135 / 224) * np.cos(2 * phi + theta)
        + 135 / 112
    )


def M_gg_95_4_0():
    def func(k):
        return -4 / 945 * k**2

    return func


def N_gg_95_4_0(theta, phi):
    return (
        (45 / 16) * np.cos(2 * phi + theta)
        + (315 / 64) * np.cos(4 * phi + 2 * theta)
        + 81 / 64
    )


def M_gg_95_4_1():
    def func(k):
        return -8 / 525 * k**2

    return func


def N_gg_95_4_1(theta, phi):
    return (
        (225 / 32) * np.cos(4 * phi)
        + (135 / 224) * np.cos(2 * theta)
        + (225 / 112) * np.cos(2 * phi - theta)
        + (225 / 112) * np.cos(2 * phi + theta)
        + 135 / 112
    )


def M_gg_95_4_2():
    def func(k):
        return -8 / 4725 * k**2

    return func


def N_gg_95_4_2(theta, phi):
    return (
        (675 / 352) * np.cos(4 * phi)
        + (6075 / 2464) * np.cos(2 * theta)
        + (6075 / 2464) * np.cos(2 * phi - theta)
        - 225 / 308 * np.cos(2 * phi + theta)
        + (675 / 352) * np.cos(2 * phi + 3 * theta)
        + (1575 / 352) * np.cos(4 * phi + 2 * theta)
        - 2025 / 2464
    )


def M_gg_95_4_3():
    def func(k):
        return -4 / 945 * k**2

    return func


def N_gg_95_4_3(theta, phi):
    return (
        (45 / 16) * np.cos(2 * phi - theta)
        + (315 / 64) * np.cos(4 * phi - 2 * theta)
        + 81 / 64
    )


def M_gg_95_4_4():
    def func(k):
        return -8 / 4725 * k**2

    return func


def N_gg_95_4_4(theta, phi):
    return (
        (675 / 352) * np.cos(4 * phi)
        + (6075 / 2464) * np.cos(2 * theta)
        + (675 / 352) * np.cos(2 * phi - 3 * theta)
        - 225 / 308 * np.cos(2 * phi - theta)
        + (6075 / 2464) * np.cos(2 * phi + theta)
        + (1575 / 352) * np.cos(4 * phi - 2 * theta)
        - 2025 / 2464
    )


def M_gg_95_6_0():
    def func(k):
        return -8 / 4725 * k**2

    return func


def N_gg_95_6_0(theta, phi):
    return (
        (4725 / 1408) * np.cos(4 * phi)
        + (1125 / 1408) * np.cos(2 * theta)
        + (4725 / 2816) * np.cos(2 * phi - theta)
        + (1575 / 704) * np.cos(2 * phi + theta)
        + (1575 / 5632) * np.cos(2 * phi + 3 * theta)
        + (4725 / 2816) * np.cos(4 * phi + 2 * theta)
        + (4725 / 512) * np.cos(6 * phi + theta)
        + 3375 / 2816
    )


def M_gg_95_6_1():
    def func(k):
        return -8 / 4725 * k**2

    return func


def N_gg_95_6_1(theta, phi):
    return (
        (4725 / 1408) * np.cos(4 * phi)
        + (1125 / 1408) * np.cos(2 * theta)
        + (1575 / 5632) * np.cos(2 * phi - 3 * theta)
        + (1575 / 704) * np.cos(2 * phi - theta)
        + (4725 / 2816) * np.cos(2 * phi + theta)
        + (4725 / 2816) * np.cos(4 * phi - 2 * theta)
        + (4725 / 512) * np.cos(6 * phi - theta)
        + 3375 / 2816
    )


def M_gg_96_0_0():
    def func(k):
        return -2 / 15 * k**2

    return func


def N_gg_96_0_0(theta, phi):
    return 1


def M_gg_96_0_1():
    def func(k):
        return -16 / 525 * k**2

    return func


def N_gg_96_0_1(theta, phi):
    return (15 / 4) * np.cos(2 * theta) + 5 / 4


def M_gg_96_2_0():
    def func(k):
        return -34 / 525 * k**2

    return func


def N_gg_96_2_0(theta, phi):
    return (15 / 4) * np.cos(2 * phi + theta) + 5 / 4


def M_gg_96_2_1():
    def func(k):
        return -34 / 525 * k**2

    return func


def N_gg_96_2_1(theta, phi):
    return (15 / 4) * np.cos(2 * phi - theta) + 5 / 4


def M_gg_96_2_2():
    def func(k):
        return -16 / 525 * k**2

    return func


def N_gg_96_2_2(theta, phi):
    return (
        (75 / 28) * np.cos(2 * theta)
        + (75 / 28) * np.cos(2 * phi - theta)
        + (75 / 28) * np.cos(2 * phi + theta)
        - 25 / 28
    )


def M_gg_96_2_3():
    def func(k):
        return -16 / 4725 * k**2

    return func


def N_gg_96_2_3(theta, phi):
    return (
        (225 / 112) * np.cos(2 * theta)
        + (135 / 224) * np.cos(2 * phi - theta)
        + (225 / 112) * np.cos(2 * phi + theta)
        + (225 / 32) * np.cos(2 * phi + 3 * theta)
        + 135 / 112
    )


def M_gg_96_2_4():
    def func(k):
        return -16 / 4725 * k**2

    return func


def N_gg_96_2_4(theta, phi):
    return (
        (225 / 112) * np.cos(2 * theta)
        + (225 / 32) * np.cos(2 * phi - 3 * theta)
        + (225 / 112) * np.cos(2 * phi - theta)
        + (135 / 224) * np.cos(2 * phi + theta)
        + 135 / 112
    )


def M_gg_96_4_0():
    def func(k):
        return -8 / 945 * k**2

    return func


def N_gg_96_4_0(theta, phi):
    return (
        (45 / 16) * np.cos(2 * phi + theta)
        + (315 / 64) * np.cos(4 * phi + 2 * theta)
        + 81 / 64
    )


def M_gg_96_4_1():
    def func(k):
        return -16 / 525 * k**2

    return func


def N_gg_96_4_1(theta, phi):
    return (
        (225 / 32) * np.cos(4 * phi)
        + (135 / 224) * np.cos(2 * theta)
        + (225 / 112) * np.cos(2 * phi - theta)
        + (225 / 112) * np.cos(2 * phi + theta)
        + 135 / 112
    )


def M_gg_96_4_2():
    def func(k):
        return -16 / 4725 * k**2

    return func


def N_gg_96_4_2(theta, phi):
    return (
        (675 / 352) * np.cos(4 * phi)
        + (6075 / 2464) * np.cos(2 * theta)
        + (6075 / 2464) * np.cos(2 * phi - theta)
        - 225 / 308 * np.cos(2 * phi + theta)
        + (675 / 352) * np.cos(2 * phi + 3 * theta)
        + (1575 / 352) * np.cos(4 * phi + 2 * theta)
        - 2025 / 2464
    )


def M_gg_96_4_3():
    def func(k):
        return -8 / 945 * k**2

    return func


def N_gg_96_4_3(theta, phi):
    return (
        (45 / 16) * np.cos(2 * phi - theta)
        + (315 / 64) * np.cos(4 * phi - 2 * theta)
        + 81 / 64
    )


def M_gg_96_4_4():
    def func(k):
        return -16 / 4725 * k**2

    return func


def N_gg_96_4_4(theta, phi):
    return (
        (675 / 352) * np.cos(4 * phi)
        + (6075 / 2464) * np.cos(2 * theta)
        + (675 / 352) * np.cos(2 * phi - 3 * theta)
        - 225 / 308 * np.cos(2 * phi - theta)
        + (6075 / 2464) * np.cos(2 * phi + theta)
        + (1575 / 352) * np.cos(4 * phi - 2 * theta)
        - 2025 / 2464
    )


def M_gg_96_6_0():
    def func(k):
        return -16 / 4725 * k**2

    return func


def N_gg_96_6_0(theta, phi):
    return (
        (4725 / 1408) * np.cos(4 * phi)
        + (1125 / 1408) * np.cos(2 * theta)
        + (4725 / 2816) * np.cos(2 * phi - theta)
        + (1575 / 704) * np.cos(2 * phi + theta)
        + (1575 / 5632) * np.cos(2 * phi + 3 * theta)
        + (4725 / 2816) * np.cos(4 * phi + 2 * theta)
        + (4725 / 512) * np.cos(6 * phi + theta)
        + 3375 / 2816
    )


def M_gg_96_6_1():
    def func(k):
        return -16 / 4725 * k**2

    return func


def N_gg_96_6_1(theta, phi):
    return (
        (4725 / 1408) * np.cos(4 * phi)
        + (1125 / 1408) * np.cos(2 * theta)
        + (1575 / 5632) * np.cos(2 * phi - 3 * theta)
        + (1575 / 704) * np.cos(2 * phi - theta)
        + (4725 / 2816) * np.cos(2 * phi + theta)
        + (4725 / 2816) * np.cos(4 * phi - 2 * theta)
        + (4725 / 512) * np.cos(6 * phi - theta)
        + 3375 / 2816
    )


def M_gg_97_0_0():
    def func(k):
        return -4 / 15 * k**2

    return func


def N_gg_97_0_0(theta, phi):
    return 1


def M_gg_97_0_1():
    def func(k):
        return -32 / 525 * k**2

    return func


def N_gg_97_0_1(theta, phi):
    return (15 / 4) * np.cos(2 * theta) + 5 / 4


def M_gg_97_2_0():
    def func(k):
        return -68 / 525 * k**2

    return func


def N_gg_97_2_0(theta, phi):
    return (15 / 4) * np.cos(2 * phi + theta) + 5 / 4


def M_gg_97_2_1():
    def func(k):
        return -68 / 525 * k**2

    return func


def N_gg_97_2_1(theta, phi):
    return (15 / 4) * np.cos(2 * phi - theta) + 5 / 4


def M_gg_97_2_2():
    def func(k):
        return -32 / 525 * k**2

    return func


def N_gg_97_2_2(theta, phi):
    return (
        (75 / 28) * np.cos(2 * theta)
        + (75 / 28) * np.cos(2 * phi - theta)
        + (75 / 28) * np.cos(2 * phi + theta)
        - 25 / 28
    )


def M_gg_97_2_3():
    def func(k):
        return -32 / 4725 * k**2

    return func


def N_gg_97_2_3(theta, phi):
    return (
        (225 / 112) * np.cos(2 * theta)
        + (135 / 224) * np.cos(2 * phi - theta)
        + (225 / 112) * np.cos(2 * phi + theta)
        + (225 / 32) * np.cos(2 * phi + 3 * theta)
        + 135 / 112
    )


def M_gg_97_2_4():
    def func(k):
        return -32 / 4725 * k**2

    return func


def N_gg_97_2_4(theta, phi):
    return (
        (225 / 112) * np.cos(2 * theta)
        + (225 / 32) * np.cos(2 * phi - 3 * theta)
        + (225 / 112) * np.cos(2 * phi - theta)
        + (135 / 224) * np.cos(2 * phi + theta)
        + 135 / 112
    )


def M_gg_97_4_0():
    def func(k):
        return -16 / 945 * k**2

    return func


def N_gg_97_4_0(theta, phi):
    return (
        (45 / 16) * np.cos(2 * phi + theta)
        + (315 / 64) * np.cos(4 * phi + 2 * theta)
        + 81 / 64
    )


def M_gg_97_4_1():
    def func(k):
        return -32 / 525 * k**2

    return func


def N_gg_97_4_1(theta, phi):
    return (
        (225 / 32) * np.cos(4 * phi)
        + (135 / 224) * np.cos(2 * theta)
        + (225 / 112) * np.cos(2 * phi - theta)
        + (225 / 112) * np.cos(2 * phi + theta)
        + 135 / 112
    )


def M_gg_97_4_2():
    def func(k):
        return -32 / 4725 * k**2

    return func


def N_gg_97_4_2(theta, phi):
    return (
        (675 / 352) * np.cos(4 * phi)
        + (6075 / 2464) * np.cos(2 * theta)
        + (6075 / 2464) * np.cos(2 * phi - theta)
        - 225 / 308 * np.cos(2 * phi + theta)
        + (675 / 352) * np.cos(2 * phi + 3 * theta)
        + (1575 / 352) * np.cos(4 * phi + 2 * theta)
        - 2025 / 2464
    )


def M_gg_97_4_3():
    def func(k):
        return -16 / 945 * k**2

    return func


def N_gg_97_4_3(theta, phi):
    return (
        (45 / 16) * np.cos(2 * phi - theta)
        + (315 / 64) * np.cos(4 * phi - 2 * theta)
        + 81 / 64
    )


def M_gg_97_4_4():
    def func(k):
        return -32 / 4725 * k**2

    return func


def N_gg_97_4_4(theta, phi):
    return (
        (675 / 352) * np.cos(4 * phi)
        + (6075 / 2464) * np.cos(2 * theta)
        + (675 / 352) * np.cos(2 * phi - 3 * theta)
        - 225 / 308 * np.cos(2 * phi - theta)
        + (6075 / 2464) * np.cos(2 * phi + theta)
        + (1575 / 352) * np.cos(4 * phi - 2 * theta)
        - 2025 / 2464
    )


def M_gg_97_6_0():
    def func(k):
        return -32 / 4725 * k**2

    return func


def N_gg_97_6_0(theta, phi):
    return (
        (4725 / 1408) * np.cos(4 * phi)
        + (1125 / 1408) * np.cos(2 * theta)
        + (4725 / 2816) * np.cos(2 * phi - theta)
        + (1575 / 704) * np.cos(2 * phi + theta)
        + (1575 / 5632) * np.cos(2 * phi + 3 * theta)
        + (4725 / 2816) * np.cos(4 * phi + 2 * theta)
        + (4725 / 512) * np.cos(6 * phi + theta)
        + 3375 / 2816
    )


def M_gg_97_6_1():
    def func(k):
        return -32 / 4725 * k**2

    return func


def N_gg_97_6_1(theta, phi):
    return (
        (4725 / 1408) * np.cos(4 * phi)
        + (1125 / 1408) * np.cos(2 * theta)
        + (1575 / 5632) * np.cos(2 * phi - 3 * theta)
        + (1575 / 704) * np.cos(2 * phi - theta)
        + (4725 / 2816) * np.cos(2 * phi + theta)
        + (4725 / 2816) * np.cos(4 * phi - 2 * theta)
        + (4725 / 512) * np.cos(6 * phi - theta)
        + 3375 / 2816
    )


def M_gg_98_0_0():
    def func(k):
        return -2 / 5 * k**4

    return func


def N_gg_98_0_0(theta, phi):
    return 1


def M_gg_98_0_1():
    def func(k):
        return -16 / 175 * k**4

    return func


def N_gg_98_0_1(theta, phi):
    return (15 / 4) * np.cos(2 * theta) + 5 / 4


def M_gg_98_2_0():
    def func(k):
        return -34 / 175 * k**4

    return func


def N_gg_98_2_0(theta, phi):
    return (15 / 4) * np.cos(2 * phi + theta) + 5 / 4


def M_gg_98_2_1():
    def func(k):
        return -34 / 175 * k**4

    return func


def N_gg_98_2_1(theta, phi):
    return (15 / 4) * np.cos(2 * phi - theta) + 5 / 4


def M_gg_98_2_2():
    def func(k):
        return -16 / 175 * k**4

    return func


def N_gg_98_2_2(theta, phi):
    return (
        (75 / 28) * np.cos(2 * theta)
        + (75 / 28) * np.cos(2 * phi - theta)
        + (75 / 28) * np.cos(2 * phi + theta)
        - 25 / 28
    )


def M_gg_98_2_3():
    def func(k):
        return -16 / 1575 * k**4

    return func


def N_gg_98_2_3(theta, phi):
    return (
        (225 / 112) * np.cos(2 * theta)
        + (135 / 224) * np.cos(2 * phi - theta)
        + (225 / 112) * np.cos(2 * phi + theta)
        + (225 / 32) * np.cos(2 * phi + 3 * theta)
        + 135 / 112
    )


def M_gg_98_2_4():
    def func(k):
        return -16 / 1575 * k**4

    return func


def N_gg_98_2_4(theta, phi):
    return (
        (225 / 112) * np.cos(2 * theta)
        + (225 / 32) * np.cos(2 * phi - 3 * theta)
        + (225 / 112) * np.cos(2 * phi - theta)
        + (135 / 224) * np.cos(2 * phi + theta)
        + 135 / 112
    )


def M_gg_98_4_0():
    def func(k):
        return -8 / 315 * k**4

    return func


def N_gg_98_4_0(theta, phi):
    return (
        (45 / 16) * np.cos(2 * phi + theta)
        + (315 / 64) * np.cos(4 * phi + 2 * theta)
        + 81 / 64
    )


def M_gg_98_4_1():
    def func(k):
        return -16 / 175 * k**4

    return func


def N_gg_98_4_1(theta, phi):
    return (
        (225 / 32) * np.cos(4 * phi)
        + (135 / 224) * np.cos(2 * theta)
        + (225 / 112) * np.cos(2 * phi - theta)
        + (225 / 112) * np.cos(2 * phi + theta)
        + 135 / 112
    )


def M_gg_98_4_2():
    def func(k):
        return -16 / 1575 * k**4

    return func


def N_gg_98_4_2(theta, phi):
    return (
        (675 / 352) * np.cos(4 * phi)
        + (6075 / 2464) * np.cos(2 * theta)
        + (6075 / 2464) * np.cos(2 * phi - theta)
        - 225 / 308 * np.cos(2 * phi + theta)
        + (675 / 352) * np.cos(2 * phi + 3 * theta)
        + (1575 / 352) * np.cos(4 * phi + 2 * theta)
        - 2025 / 2464
    )


def M_gg_98_4_3():
    def func(k):
        return -8 / 315 * k**4

    return func


def N_gg_98_4_3(theta, phi):
    return (
        (45 / 16) * np.cos(2 * phi - theta)
        + (315 / 64) * np.cos(4 * phi - 2 * theta)
        + 81 / 64
    )


def M_gg_98_4_4():
    def func(k):
        return -16 / 1575 * k**4

    return func


def N_gg_98_4_4(theta, phi):
    return (
        (675 / 352) * np.cos(4 * phi)
        + (6075 / 2464) * np.cos(2 * theta)
        + (675 / 352) * np.cos(2 * phi - 3 * theta)
        - 225 / 308 * np.cos(2 * phi - theta)
        + (6075 / 2464) * np.cos(2 * phi + theta)
        + (1575 / 352) * np.cos(4 * phi - 2 * theta)
        - 2025 / 2464
    )


def M_gg_98_6_0():
    def func(k):
        return -16 / 1575 * k**4

    return func


def N_gg_98_6_0(theta, phi):
    return (
        (4725 / 1408) * np.cos(4 * phi)
        + (1125 / 1408) * np.cos(2 * theta)
        + (4725 / 2816) * np.cos(2 * phi - theta)
        + (1575 / 704) * np.cos(2 * phi + theta)
        + (1575 / 5632) * np.cos(2 * phi + 3 * theta)
        + (4725 / 2816) * np.cos(4 * phi + 2 * theta)
        + (4725 / 512) * np.cos(6 * phi + theta)
        + 3375 / 2816
    )


def M_gg_98_6_1():
    def func(k):
        return -16 / 1575 * k**4

    return func


def N_gg_98_6_1(theta, phi):
    return (
        (4725 / 1408) * np.cos(4 * phi)
        + (1125 / 1408) * np.cos(2 * theta)
        + (1575 / 5632) * np.cos(2 * phi - 3 * theta)
        + (1575 / 704) * np.cos(2 * phi - theta)
        + (4725 / 2816) * np.cos(2 * phi + theta)
        + (4725 / 2816) * np.cos(4 * phi - 2 * theta)
        + (4725 / 512) * np.cos(6 * phi - theta)
        + 3375 / 2816
    )


def M_gg_99_0_0():
    def func(k):
        return -4 / 5 * k**4

    return func


def N_gg_99_0_0(theta, phi):
    return 1


def M_gg_99_0_1():
    def func(k):
        return -32 / 175 * k**4

    return func


def N_gg_99_0_1(theta, phi):
    return (15 / 4) * np.cos(2 * theta) + 5 / 4


def M_gg_99_2_0():
    def func(k):
        return -68 / 175 * k**4

    return func


def N_gg_99_2_0(theta, phi):
    return (15 / 4) * np.cos(2 * phi + theta) + 5 / 4


def M_gg_99_2_1():
    def func(k):
        return -68 / 175 * k**4

    return func


def N_gg_99_2_1(theta, phi):
    return (15 / 4) * np.cos(2 * phi - theta) + 5 / 4


def M_gg_99_2_2():
    def func(k):
        return -32 / 175 * k**4

    return func


def N_gg_99_2_2(theta, phi):
    return (
        (75 / 28) * np.cos(2 * theta)
        + (75 / 28) * np.cos(2 * phi - theta)
        + (75 / 28) * np.cos(2 * phi + theta)
        - 25 / 28
    )


def M_gg_99_2_3():
    def func(k):
        return -32 / 1575 * k**4

    return func


def N_gg_99_2_3(theta, phi):
    return (
        (225 / 112) * np.cos(2 * theta)
        + (135 / 224) * np.cos(2 * phi - theta)
        + (225 / 112) * np.cos(2 * phi + theta)
        + (225 / 32) * np.cos(2 * phi + 3 * theta)
        + 135 / 112
    )


def M_gg_99_2_4():
    def func(k):
        return -32 / 1575 * k**4

    return func


def N_gg_99_2_4(theta, phi):
    return (
        (225 / 112) * np.cos(2 * theta)
        + (225 / 32) * np.cos(2 * phi - 3 * theta)
        + (225 / 112) * np.cos(2 * phi - theta)
        + (135 / 224) * np.cos(2 * phi + theta)
        + 135 / 112
    )


def M_gg_99_4_0():
    def func(k):
        return -16 / 315 * k**4

    return func


def N_gg_99_4_0(theta, phi):
    return (
        (45 / 16) * np.cos(2 * phi + theta)
        + (315 / 64) * np.cos(4 * phi + 2 * theta)
        + 81 / 64
    )


def M_gg_99_4_1():
    def func(k):
        return -32 / 175 * k**4

    return func


def N_gg_99_4_1(theta, phi):
    return (
        (225 / 32) * np.cos(4 * phi)
        + (135 / 224) * np.cos(2 * theta)
        + (225 / 112) * np.cos(2 * phi - theta)
        + (225 / 112) * np.cos(2 * phi + theta)
        + 135 / 112
    )


def M_gg_99_4_2():
    def func(k):
        return -32 / 1575 * k**4

    return func


def N_gg_99_4_2(theta, phi):
    return (
        (675 / 352) * np.cos(4 * phi)
        + (6075 / 2464) * np.cos(2 * theta)
        + (6075 / 2464) * np.cos(2 * phi - theta)
        - 225 / 308 * np.cos(2 * phi + theta)
        + (675 / 352) * np.cos(2 * phi + 3 * theta)
        + (1575 / 352) * np.cos(4 * phi + 2 * theta)
        - 2025 / 2464
    )


def M_gg_99_4_3():
    def func(k):
        return -16 / 315 * k**4

    return func


def N_gg_99_4_3(theta, phi):
    return (
        (45 / 16) * np.cos(2 * phi - theta)
        + (315 / 64) * np.cos(4 * phi - 2 * theta)
        + 81 / 64
    )


def M_gg_99_4_4():
    def func(k):
        return -32 / 1575 * k**4

    return func


def N_gg_99_4_4(theta, phi):
    return (
        (675 / 352) * np.cos(4 * phi)
        + (6075 / 2464) * np.cos(2 * theta)
        + (675 / 352) * np.cos(2 * phi - 3 * theta)
        - 225 / 308 * np.cos(2 * phi - theta)
        + (6075 / 2464) * np.cos(2 * phi + theta)
        + (1575 / 352) * np.cos(4 * phi - 2 * theta)
        - 2025 / 2464
    )


def M_gg_99_6_0():
    def func(k):
        return -32 / 1575 * k**4

    return func


def N_gg_99_6_0(theta, phi):
    return (
        (4725 / 1408) * np.cos(4 * phi)
        + (1125 / 1408) * np.cos(2 * theta)
        + (4725 / 2816) * np.cos(2 * phi - theta)
        + (1575 / 704) * np.cos(2 * phi + theta)
        + (1575 / 5632) * np.cos(2 * phi + 3 * theta)
        + (4725 / 2816) * np.cos(4 * phi + 2 * theta)
        + (4725 / 512) * np.cos(6 * phi + theta)
        + 3375 / 2816
    )


def M_gg_99_6_1():
    def func(k):
        return -32 / 1575 * k**4

    return func


def N_gg_99_6_1(theta, phi):
    return (
        (4725 / 1408) * np.cos(4 * phi)
        + (1125 / 1408) * np.cos(2 * theta)
        + (1575 / 5632) * np.cos(2 * phi - 3 * theta)
        + (1575 / 704) * np.cos(2 * phi - theta)
        + (4725 / 2816) * np.cos(2 * phi + theta)
        + (4725 / 2816) * np.cos(4 * phi - 2 * theta)
        + (4725 / 512) * np.cos(6 * phi - theta)
        + 3375 / 2816
    )


def M_gg_100_0_0():
    def func(k):
        return -1 / 15 * k**2

    return func


def N_gg_100_0_0(theta, phi):
    return 1


def M_gg_100_0_1():
    def func(k):
        return -8 / 525 * k**2

    return func


def N_gg_100_0_1(theta, phi):
    return (15 / 4) * np.cos(2 * theta) + 5 / 4


def M_gg_100_2_0():
    def func(k):
        return -17 / 525 * k**2

    return func


def N_gg_100_2_0(theta, phi):
    return (15 / 4) * np.cos(2 * phi + theta) + 5 / 4


def M_gg_100_2_1():
    def func(k):
        return -17 / 525 * k**2

    return func


def N_gg_100_2_1(theta, phi):
    return (15 / 4) * np.cos(2 * phi - theta) + 5 / 4


def M_gg_100_2_2():
    def func(k):
        return -8 / 525 * k**2

    return func


def N_gg_100_2_2(theta, phi):
    return (
        (75 / 28) * np.cos(2 * theta)
        + (75 / 28) * np.cos(2 * phi - theta)
        + (75 / 28) * np.cos(2 * phi + theta)
        - 25 / 28
    )


def M_gg_100_2_3():
    def func(k):
        return -8 / 4725 * k**2

    return func


def N_gg_100_2_3(theta, phi):
    return (
        (225 / 112) * np.cos(2 * theta)
        + (135 / 224) * np.cos(2 * phi - theta)
        + (225 / 112) * np.cos(2 * phi + theta)
        + (225 / 32) * np.cos(2 * phi + 3 * theta)
        + 135 / 112
    )


def M_gg_100_2_4():
    def func(k):
        return -8 / 4725 * k**2

    return func


def N_gg_100_2_4(theta, phi):
    return (
        (225 / 112) * np.cos(2 * theta)
        + (225 / 32) * np.cos(2 * phi - 3 * theta)
        + (225 / 112) * np.cos(2 * phi - theta)
        + (135 / 224) * np.cos(2 * phi + theta)
        + 135 / 112
    )


def M_gg_100_4_0():
    def func(k):
        return -4 / 945 * k**2

    return func


def N_gg_100_4_0(theta, phi):
    return (
        (45 / 16) * np.cos(2 * phi + theta)
        + (315 / 64) * np.cos(4 * phi + 2 * theta)
        + 81 / 64
    )


def M_gg_100_4_1():
    def func(k):
        return -8 / 525 * k**2

    return func


def N_gg_100_4_1(theta, phi):
    return (
        (225 / 32) * np.cos(4 * phi)
        + (135 / 224) * np.cos(2 * theta)
        + (225 / 112) * np.cos(2 * phi - theta)
        + (225 / 112) * np.cos(2 * phi + theta)
        + 135 / 112
    )


def M_gg_100_4_2():
    def func(k):
        return -8 / 4725 * k**2

    return func


def N_gg_100_4_2(theta, phi):
    return (
        (675 / 352) * np.cos(4 * phi)
        + (6075 / 2464) * np.cos(2 * theta)
        + (6075 / 2464) * np.cos(2 * phi - theta)
        - 225 / 308 * np.cos(2 * phi + theta)
        + (675 / 352) * np.cos(2 * phi + 3 * theta)
        + (1575 / 352) * np.cos(4 * phi + 2 * theta)
        - 2025 / 2464
    )


def M_gg_100_4_3():
    def func(k):
        return -4 / 945 * k**2

    return func


def N_gg_100_4_3(theta, phi):
    return (
        (45 / 16) * np.cos(2 * phi - theta)
        + (315 / 64) * np.cos(4 * phi - 2 * theta)
        + 81 / 64
    )


def M_gg_100_4_4():
    def func(k):
        return -8 / 4725 * k**2

    return func


def N_gg_100_4_4(theta, phi):
    return (
        (675 / 352) * np.cos(4 * phi)
        + (6075 / 2464) * np.cos(2 * theta)
        + (675 / 352) * np.cos(2 * phi - 3 * theta)
        - 225 / 308 * np.cos(2 * phi - theta)
        + (6075 / 2464) * np.cos(2 * phi + theta)
        + (1575 / 352) * np.cos(4 * phi - 2 * theta)
        - 2025 / 2464
    )


def M_gg_100_6_0():
    def func(k):
        return -8 / 4725 * k**2

    return func


def N_gg_100_6_0(theta, phi):
    return (
        (4725 / 1408) * np.cos(4 * phi)
        + (1125 / 1408) * np.cos(2 * theta)
        + (4725 / 2816) * np.cos(2 * phi - theta)
        + (1575 / 704) * np.cos(2 * phi + theta)
        + (1575 / 5632) * np.cos(2 * phi + 3 * theta)
        + (4725 / 2816) * np.cos(4 * phi + 2 * theta)
        + (4725 / 512) * np.cos(6 * phi + theta)
        + 3375 / 2816
    )


def M_gg_100_6_1():
    def func(k):
        return -8 / 4725 * k**2

    return func


def N_gg_100_6_1(theta, phi):
    return (
        (4725 / 1408) * np.cos(4 * phi)
        + (1125 / 1408) * np.cos(2 * theta)
        + (1575 / 5632) * np.cos(2 * phi - 3 * theta)
        + (1575 / 704) * np.cos(2 * phi - theta)
        + (4725 / 2816) * np.cos(2 * phi + theta)
        + (4725 / 2816) * np.cos(4 * phi - 2 * theta)
        + (4725 / 512) * np.cos(6 * phi - theta)
        + 3375 / 2816
    )


def M_gg_101_0_0():
    def func(k):
        return -1 / 15 * k**2

    return func


def N_gg_101_0_0(theta, phi):
    return (3 / 2) * np.cos(2 * theta) + 3 / 2


def M_gg_101_2_0():
    def func(k):
        return -1 / 15 * k**2

    return func


def N_gg_101_2_0(theta, phi):
    return (
        (3 / 4) * np.cos(2 * theta)
        + (9 / 4) * np.cos(2 * phi - theta)
        + (9 / 4) * np.cos(2 * phi + theta)
        + 3 / 4
    )


def M_gg_101_2_1():
    def func(k):
        return -1 / 105 * k**2

    return func


def N_gg_101_2_1(theta, phi):
    return (
        (9 / 8) * np.cos(2 * theta)
        + (9 / 16) * np.cos(2 * phi - theta)
        + (27 / 8) * np.cos(2 * phi + theta)
        + (45 / 16) * np.cos(2 * phi + 3 * theta)
        + 9 / 8
    )


def M_gg_101_2_2():
    def func(k):
        return -1 / 105 * k**2

    return func


def N_gg_101_2_2(theta, phi):
    return (
        (9 / 8) * np.cos(2 * theta)
        + (45 / 16) * np.cos(2 * phi - 3 * theta)
        + (27 / 8) * np.cos(2 * phi - theta)
        + (9 / 16) * np.cos(2 * phi + theta)
        + 9 / 8
    )


def M_gg_101_4_0():
    def func(k):
        return -1 / 105 * k**2

    return func


def N_gg_101_4_0(theta, phi):
    return (
        (105 / 32) * np.cos(4 * phi)
        + (27 / 32) * np.cos(2 * theta)
        + (45 / 32) * np.cos(2 * phi - theta)
        + (15 / 8) * np.cos(2 * phi + theta)
        + (15 / 32) * np.cos(2 * phi + 3 * theta)
        + (105 / 32) * np.cos(4 * phi + 2 * theta)
        + 27 / 32
    )


def M_gg_101_4_1():
    def func(k):
        return -1 / 105 * k**2

    return func


def N_gg_101_4_1(theta, phi):
    return (
        (105 / 32) * np.cos(4 * phi)
        + (27 / 32) * np.cos(2 * theta)
        + (15 / 32) * np.cos(2 * phi - 3 * theta)
        + (15 / 8) * np.cos(2 * phi - theta)
        + (45 / 32) * np.cos(2 * phi + theta)
        + (105 / 32) * np.cos(4 * phi - 2 * theta)
        + 27 / 32
    )


def M_gg_102_0_0():
    def func(k):
        return 1 / 36

    return func


def N_gg_102_0_0(theta, phi):
    return (1 / 12) * np.cos(2 * theta) + 1 / 6


def M_gg_102_0_1():
    def func(k):
        return 1 / 225

    return func


def N_gg_102_0_1(theta, phi):
    return (35 / 48) * np.cos(2 * theta) + (5 / 32) * np.cos(4 * theta) + 35 / 96


def M_gg_102_2_0():
    def func(k):
        return 1 / 90

    return func


def N_gg_102_2_0(theta, phi):
    return (
        (5 / 48) * np.cos(2 * theta)
        + (5 / 32) * np.cos(2 * phi - theta)
        + (5 / 8) * np.cos(2 * phi + theta)
        + (5 / 32) * np.cos(2 * phi + 3 * theta)
        + 5 / 24
    )


def M_gg_102_2_1():
    def func(k):
        return 1 / 90

    return func


def N_gg_102_2_1(theta, phi):
    return (
        (5 / 48) * np.cos(2 * theta)
        + (5 / 32) * np.cos(2 * phi - 3 * theta)
        + (5 / 8) * np.cos(2 * phi - theta)
        + (5 / 32) * np.cos(2 * phi + theta)
        + 5 / 24
    )


def M_gg_102_2_2():
    def func(k):
        return 1 / 225

    return func


def N_gg_102_2_2(theta, phi):
    return (
        (125 / 336) * np.cos(2 * theta)
        + (25 / 224) * np.cos(4 * theta)
        + (25 / 224) * np.cos(2 * phi - 3 * theta)
        + (125 / 224) * np.cos(2 * phi - theta)
        + (125 / 224) * np.cos(2 * phi + theta)
        + (25 / 224) * np.cos(2 * phi + 3 * theta)
        - 25 / 672
    )


def M_gg_102_4_0():
    def func(k):
        return 1 / 225

    return func


def N_gg_102_4_0(theta, phi):
    return (
        (75 / 64) * np.cos(4 * phi)
        + (45 / 224) * np.cos(2 * theta)
        + (45 / 1792) * np.cos(4 * theta)
        + (75 / 896) * np.cos(2 * phi - 3 * theta)
        + (375 / 896) * np.cos(2 * phi - theta)
        + (375 / 896) * np.cos(2 * phi + theta)
        + (75 / 896) * np.cos(2 * phi + 3 * theta)
        + (75 / 256) * np.cos(4 * phi - 2 * theta)
        + (75 / 256) * np.cos(4 * phi + 2 * theta)
        + 405 / 1792
    )


def M_gg_103_0_0():
    def func(k):
        return 1 / 100

    return func


def N_gg_103_0_0(theta, phi):
    return (3 / 4) * np.cos(2 * theta) + 3 / 4


def M_gg_103_0_1():
    def func(k):
        return 1 / 1225

    return func


def N_gg_103_0_1(theta, phi):
    return (7 / 4) * np.cos(2 * theta) + (35 / 32) * np.cos(4 * theta) + 21 / 32


def M_gg_103_2_0():
    def func(k):
        return 1 / 100

    return func


def N_gg_103_2_0(theta, phi):
    return (
        (3 / 8) * np.cos(2 * theta)
        + (9 / 8) * np.cos(2 * phi - theta)
        + (9 / 8) * np.cos(2 * phi + theta)
        + 3 / 8
    )


def M_gg_103_2_1():
    def func(k):
        return 1 / 350

    return func


def N_gg_103_2_1(theta, phi):
    return (
        (9 / 16) * np.cos(2 * theta)
        + (9 / 32) * np.cos(2 * phi - theta)
        + (27 / 16) * np.cos(2 * phi + theta)
        + (45 / 32) * np.cos(2 * phi + 3 * theta)
        + 9 / 16
    )


def M_gg_103_2_2():
    def func(k):
        return 1 / 350

    return func


def N_gg_103_2_2(theta, phi):
    return (
        (9 / 16) * np.cos(2 * theta)
        + (45 / 32) * np.cos(2 * phi - 3 * theta)
        + (27 / 16) * np.cos(2 * phi - theta)
        + (9 / 32) * np.cos(2 * phi + theta)
        + 9 / 16
    )


def M_gg_103_2_3():
    def func(k):
        return 1 / 1225

    return func


def N_gg_103_2_3(theta, phi):
    return (
        (7 / 12) * np.cos(2 * theta)
        + (175 / 192) * np.cos(4 * theta)
        + (35 / 64) * np.cos(2 * phi - 3 * theta)
        + (77 / 64) * np.cos(2 * phi - theta)
        + (77 / 64) * np.cos(2 * phi + theta)
        + (35 / 64) * np.cos(2 * phi + 3 * theta)
        - 21 / 64
    )


def M_gg_103_4_0():
    def func(k):
        return 1 / 350

    return func


def N_gg_103_4_0(theta, phi):
    return (
        (105 / 64) * np.cos(4 * phi)
        + (27 / 64) * np.cos(2 * theta)
        + (45 / 64) * np.cos(2 * phi - theta)
        + (15 / 16) * np.cos(2 * phi + theta)
        + (15 / 64) * np.cos(2 * phi + 3 * theta)
        + (105 / 64) * np.cos(4 * phi + 2 * theta)
        + 27 / 64
    )


def M_gg_103_4_1():
    def func(k):
        return 1 / 350

    return func


def N_gg_103_4_1(theta, phi):
    return (
        (105 / 64) * np.cos(4 * phi)
        + (27 / 64) * np.cos(2 * theta)
        + (15 / 64) * np.cos(2 * phi - 3 * theta)
        + (15 / 16) * np.cos(2 * phi - theta)
        + (45 / 64) * np.cos(2 * phi + theta)
        + (105 / 64) * np.cos(4 * phi - 2 * theta)
        + 27 / 64
    )


def M_gg_103_4_2():
    def func(k):
        return 1 / 1225

    return func


def N_gg_103_4_2(theta, phi):
    return (
        (2205 / 1408) * np.cos(4 * phi)
        + (567 / 1408) * np.cos(2 * theta)
        + (945 / 2816) * np.cos(4 * theta)
        + (945 / 1408) * np.cos(2 * phi - 3 * theta)
        + (315 / 1408) * np.cos(2 * phi - theta)
        + (315 / 1408) * np.cos(2 * phi + theta)
        + (945 / 1408) * np.cos(2 * phi + 3 * theta)
        + (2205 / 2816) * np.cos(4 * phi - 2 * theta)
        + (2205 / 2816) * np.cos(4 * phi + 2 * theta)
        + 189 / 2816
    )


def M_gg_103_6_0():
    def func(k):
        return 1 / 1225

    return func


def N_gg_103_6_0(theta, phi):
    return (
        (3675 / 2816) * np.cos(4 * phi)
        + (4375 / 8448) * np.cos(2 * theta)
        + (875 / 16896) * np.cos(4 * theta)
        + (1225 / 5632) * np.cos(2 * phi - 3 * theta)
        + (1225 / 1408) * np.cos(2 * phi - theta)
        + (1225 / 1408) * np.cos(2 * phi + theta)
        + (1225 / 5632) * np.cos(2 * phi + 3 * theta)
        + (3675 / 5632) * np.cos(4 * phi - 2 * theta)
        + (3675 / 5632) * np.cos(4 * phi + 2 * theta)
        + (1225 / 512) * np.cos(6 * phi - theta)
        + (1225 / 512) * np.cos(6 * phi + theta)
        + 2625 / 5632
    )


def M_gg_104_0_0():
    def func(k):
        return 1 / 400

    return func


def N_gg_104_0_0(theta, phi):
    return 1


def M_gg_104_0_1():
    def func(k):
        return 1 / 1225

    return func


def N_gg_104_0_1(theta, phi):
    return (15 / 4) * np.cos(2 * theta) + 5 / 4


def M_gg_104_0_2():
    def func(k):
        return 4 / 99225

    return func


def N_gg_104_0_2(theta, phi):
    return (45 / 16) * np.cos(2 * theta) + (315 / 64) * np.cos(4 * theta) + 81 / 64


def M_gg_104_2_0():
    def func(k):
        return 1 / 700

    return func


def N_gg_104_2_0(theta, phi):
    return (15 / 4) * np.cos(2 * phi + theta) + 5 / 4


def M_gg_104_2_1():
    def func(k):
        return 1 / 700

    return func


def N_gg_104_2_1(theta, phi):
    return (15 / 4) * np.cos(2 * phi - theta) + 5 / 4


def M_gg_104_2_2():
    def func(k):
        return 1 / 1225

    return func


def N_gg_104_2_2(theta, phi):
    return (
        (75 / 28) * np.cos(2 * theta)
        + (75 / 28) * np.cos(2 * phi - theta)
        + (75 / 28) * np.cos(2 * phi + theta)
        - 25 / 28
    )


def M_gg_104_2_3():
    def func(k):
        return 2 / 11025

    return func


def N_gg_104_2_3(theta, phi):
    return (
        (225 / 112) * np.cos(2 * theta)
        + (135 / 224) * np.cos(2 * phi - theta)
        + (225 / 112) * np.cos(2 * phi + theta)
        + (225 / 32) * np.cos(2 * phi + 3 * theta)
        + 135 / 112
    )


def M_gg_104_2_4():
    def func(k):
        return 2 / 11025

    return func


def N_gg_104_2_4(theta, phi):
    return (
        (225 / 112) * np.cos(2 * theta)
        + (225 / 32) * np.cos(2 * phi - 3 * theta)
        + (225 / 112) * np.cos(2 * phi - theta)
        + (135 / 224) * np.cos(2 * phi + theta)
        + 135 / 112
    )


def M_gg_104_2_5():
    def func(k):
        return 4 / 99225

    return func


def N_gg_104_2_5(theta, phi):
    return (
        -225 / 308 * np.cos(2 * theta)
        + (1575 / 352) * np.cos(4 * theta)
        + (675 / 352) * np.cos(2 * phi - 3 * theta)
        + (6075 / 2464) * np.cos(2 * phi - theta)
        + (6075 / 2464) * np.cos(2 * phi + theta)
        + (675 / 352) * np.cos(2 * phi + 3 * theta)
        - 2025 / 2464
    )


def M_gg_104_4_0():
    def func(k):
        return 1 / 3150

    return func


def N_gg_104_4_0(theta, phi):
    return (
        (45 / 16) * np.cos(2 * phi + theta)
        + (315 / 64) * np.cos(4 * phi + 2 * theta)
        + 81 / 64
    )


def M_gg_104_4_1():
    def func(k):
        return 1 / 1225

    return func


def N_gg_104_4_1(theta, phi):
    return (
        (225 / 32) * np.cos(4 * phi)
        + (135 / 224) * np.cos(2 * theta)
        + (225 / 112) * np.cos(2 * phi - theta)
        + (225 / 112) * np.cos(2 * phi + theta)
        + 135 / 112
    )


def M_gg_104_4_2():
    def func(k):
        return 2 / 11025

    return func


def N_gg_104_4_2(theta, phi):
    return (
        (675 / 352) * np.cos(4 * phi)
        + (6075 / 2464) * np.cos(2 * theta)
        + (6075 / 2464) * np.cos(2 * phi - theta)
        - 225 / 308 * np.cos(2 * phi + theta)
        + (675 / 352) * np.cos(2 * phi + 3 * theta)
        + (1575 / 352) * np.cos(4 * phi + 2 * theta)
        - 2025 / 2464
    )


def M_gg_104_4_3():
    def func(k):
        return 1 / 3150

    return func


def N_gg_104_4_3(theta, phi):
    return (
        (45 / 16) * np.cos(2 * phi - theta)
        + (315 / 64) * np.cos(4 * phi - 2 * theta)
        + 81 / 64
    )


def M_gg_104_4_4():
    def func(k):
        return 2 / 11025

    return func


def N_gg_104_4_4(theta, phi):
    return (
        (675 / 352) * np.cos(4 * phi)
        + (6075 / 2464) * np.cos(2 * theta)
        + (675 / 352) * np.cos(2 * phi - 3 * theta)
        - 225 / 308 * np.cos(2 * phi - theta)
        + (6075 / 2464) * np.cos(2 * phi + theta)
        + (1575 / 352) * np.cos(4 * phi - 2 * theta)
        - 2025 / 2464
    )


def M_gg_104_4_5():
    def func(k):
        return 4 / 99225

    return func


def N_gg_104_4_5(theta, phi):
    return (
        (54675 / 18304) * np.cos(4 * phi)
        - 10935 / 11648 * np.cos(2 * theta)
        + (76545 / 36608) * np.cos(4 * theta)
        + (54675 / 18304) * np.cos(2 * phi - 3 * theta)
        - 10935 / 11648 * np.cos(2 * phi - theta)
        - 10935 / 11648 * np.cos(2 * phi + theta)
        + (54675 / 18304) * np.cos(2 * phi + 3 * theta)
        + (76545 / 36608) * np.cos(4 * phi - 2 * theta)
        + (76545 / 36608) * np.cos(4 * phi + 2 * theta)
        + 177147 / 256256
    )


def M_gg_104_6_0():
    def func(k):
        return 2 / 11025

    return func


def N_gg_104_6_0(theta, phi):
    return (
        (4725 / 1408) * np.cos(4 * phi)
        + (1125 / 1408) * np.cos(2 * theta)
        + (4725 / 2816) * np.cos(2 * phi - theta)
        + (1575 / 704) * np.cos(2 * phi + theta)
        + (1575 / 5632) * np.cos(2 * phi + 3 * theta)
        + (4725 / 2816) * np.cos(4 * phi + 2 * theta)
        + (4725 / 512) * np.cos(6 * phi + theta)
        + 3375 / 2816
    )


def M_gg_104_6_1():
    def func(k):
        return 2 / 11025

    return func


def N_gg_104_6_1(theta, phi):
    return (
        (4725 / 1408) * np.cos(4 * phi)
        + (1125 / 1408) * np.cos(2 * theta)
        + (1575 / 5632) * np.cos(2 * phi - 3 * theta)
        + (1575 / 704) * np.cos(2 * phi - theta)
        + (4725 / 2816) * np.cos(2 * phi + theta)
        + (4725 / 2816) * np.cos(4 * phi - 2 * theta)
        + (4725 / 512) * np.cos(6 * phi - theta)
        + 3375 / 2816
    )


def M_gg_104_6_2():
    def func(k):
        return 4 / 99225

    return func


def N_gg_104_6_2(theta, phi):
    return (
        -2835 / 1408 * np.cos(4 * phi)
        + (225 / 128) * np.cos(2 * theta)
        + (1575 / 2816) * np.cos(4 * theta)
        + (4725 / 2816) * np.cos(2 * phi - 3 * theta)
        + (4725 / 2816) * np.cos(2 * phi + 3 * theta)
        + (8505 / 2816) * np.cos(4 * phi - 2 * theta)
        + (8505 / 2816) * np.cos(4 * phi + 2 * theta)
        + (945 / 256) * np.cos(6 * phi - theta)
        + (945 / 256) * np.cos(6 * phi + theta)
        - 2025 / 2816
    )


def M_gg_104_8_0():
    def func(k):
        return 4 / 99225

    return func


def N_gg_104_8_0(theta, phi):
    return (
        (19845 / 6656) * np.cos(4 * phi)
        + (99225 / 8192) * np.cos(8 * phi)
        + (77175 / 73216) * np.cos(2 * theta)
        + (77175 / 1171456) * np.cos(4 * theta)
        + (99225 / 292864) * np.cos(2 * phi - 3 * theta)
        + (297675 / 146432) * np.cos(2 * phi - theta)
        + (297675 / 146432) * np.cos(2 * phi + theta)
        + (99225 / 292864) * np.cos(2 * phi + 3 * theta)
        + (59535 / 53248) * np.cos(4 * phi - 2 * theta)
        + (59535 / 53248) * np.cos(4 * phi + 2 * theta)
        + (6615 / 2048) * np.cos(6 * phi - theta)
        + (6615 / 2048) * np.cos(6 * phi + theta)
        + 694575 / 585728
    )


def M_gg_105_0_0():
    def func(k):
        return (1 / 12) * k**4

    return func


def N_gg_105_0_0(theta, phi):
    return 1


def M_gg_105_0_1():
    def func(k):
        return (1 / 75) * k**4

    return func


def N_gg_105_0_1(theta, phi):
    return (15 / 4) * np.cos(2 * theta) + 5 / 4


def M_gg_105_2_0():
    def func(k):
        return (1 / 30) * k**4

    return func


def N_gg_105_2_0(theta, phi):
    return (15 / 4) * np.cos(2 * phi + theta) + 5 / 4


def M_gg_105_2_1():
    def func(k):
        return (1 / 30) * k**4

    return func


def N_gg_105_2_1(theta, phi):
    return (15 / 4) * np.cos(2 * phi - theta) + 5 / 4


def M_gg_105_2_2():
    def func(k):
        return (1 / 75) * k**4

    return func


def N_gg_105_2_2(theta, phi):
    return (
        (75 / 28) * np.cos(2 * theta)
        + (75 / 28) * np.cos(2 * phi - theta)
        + (75 / 28) * np.cos(2 * phi + theta)
        - 25 / 28
    )


def M_gg_105_4_0():
    def func(k):
        return (1 / 75) * k**4

    return func


def N_gg_105_4_0(theta, phi):
    return (
        (225 / 32) * np.cos(4 * phi)
        + (135 / 224) * np.cos(2 * theta)
        + (225 / 112) * np.cos(2 * phi - theta)
        + (225 / 112) * np.cos(2 * phi + theta)
        + 135 / 112
    )


def M_gg_106_0_0():
    def func(k):
        return (1 / 6) * k**4

    return func


def N_gg_106_0_0(theta, phi):
    return 1


def M_gg_106_0_1():
    def func(k):
        return (2 / 75) * k**4

    return func


def N_gg_106_0_1(theta, phi):
    return (15 / 4) * np.cos(2 * theta) + 5 / 4


def M_gg_106_2_0():
    def func(k):
        return (1 / 15) * k**4

    return func


def N_gg_106_2_0(theta, phi):
    return (15 / 4) * np.cos(2 * phi + theta) + 5 / 4


def M_gg_106_2_1():
    def func(k):
        return (1 / 15) * k**4

    return func


def N_gg_106_2_1(theta, phi):
    return (15 / 4) * np.cos(2 * phi - theta) + 5 / 4


def M_gg_106_2_2():
    def func(k):
        return (2 / 75) * k**4

    return func


def N_gg_106_2_2(theta, phi):
    return (
        (75 / 28) * np.cos(2 * theta)
        + (75 / 28) * np.cos(2 * phi - theta)
        + (75 / 28) * np.cos(2 * phi + theta)
        - 25 / 28
    )


def M_gg_106_4_0():
    def func(k):
        return (2 / 75) * k**4

    return func


def N_gg_106_4_0(theta, phi):
    return (
        (225 / 32) * np.cos(4 * phi)
        + (135 / 224) * np.cos(2 * theta)
        + (225 / 112) * np.cos(2 * phi - theta)
        + (225 / 112) * np.cos(2 * phi + theta)
        + 135 / 112
    )


def M_gg_107_0_0():
    def func(k):
        return (1 / 6) * k**4

    return func


def N_gg_107_0_0(theta, phi):
    return 1


def M_gg_107_0_1():
    def func(k):
        return (2 / 75) * k**4

    return func


def N_gg_107_0_1(theta, phi):
    return (15 / 4) * np.cos(2 * theta) + 5 / 4


def M_gg_107_2_0():
    def func(k):
        return (1 / 15) * k**4

    return func


def N_gg_107_2_0(theta, phi):
    return (15 / 4) * np.cos(2 * phi + theta) + 5 / 4


def M_gg_107_2_1():
    def func(k):
        return (1 / 15) * k**4

    return func


def N_gg_107_2_1(theta, phi):
    return (15 / 4) * np.cos(2 * phi - theta) + 5 / 4


def M_gg_107_2_2():
    def func(k):
        return (2 / 75) * k**4

    return func


def N_gg_107_2_2(theta, phi):
    return (
        (75 / 28) * np.cos(2 * theta)
        + (75 / 28) * np.cos(2 * phi - theta)
        + (75 / 28) * np.cos(2 * phi + theta)
        - 25 / 28
    )


def M_gg_107_4_0():
    def func(k):
        return (2 / 75) * k**4

    return func


def N_gg_107_4_0(theta, phi):
    return (
        (225 / 32) * np.cos(4 * phi)
        + (135 / 224) * np.cos(2 * theta)
        + (225 / 112) * np.cos(2 * phi - theta)
        + (225 / 112) * np.cos(2 * phi + theta)
        + 135 / 112
    )


def M_gg_108_0_0():
    def func(k):
        return (1 / 6) * k**4

    return func


def N_gg_108_0_0(theta, phi):
    return 1


def M_gg_108_0_1():
    def func(k):
        return (2 / 75) * k**4

    return func


def N_gg_108_0_1(theta, phi):
    return (15 / 4) * np.cos(2 * theta) + 5 / 4


def M_gg_108_2_0():
    def func(k):
        return (1 / 15) * k**4

    return func


def N_gg_108_2_0(theta, phi):
    return (15 / 4) * np.cos(2 * phi + theta) + 5 / 4


def M_gg_108_2_1():
    def func(k):
        return (1 / 15) * k**4

    return func


def N_gg_108_2_1(theta, phi):
    return (15 / 4) * np.cos(2 * phi - theta) + 5 / 4


def M_gg_108_2_2():
    def func(k):
        return (2 / 75) * k**4

    return func


def N_gg_108_2_2(theta, phi):
    return (
        (75 / 28) * np.cos(2 * theta)
        + (75 / 28) * np.cos(2 * phi - theta)
        + (75 / 28) * np.cos(2 * phi + theta)
        - 25 / 28
    )


def M_gg_108_4_0():
    def func(k):
        return (2 / 75) * k**4

    return func


def N_gg_108_4_0(theta, phi):
    return (
        (225 / 32) * np.cos(4 * phi)
        + (135 / 224) * np.cos(2 * theta)
        + (225 / 112) * np.cos(2 * phi - theta)
        + (225 / 112) * np.cos(2 * phi + theta)
        + 135 / 112
    )


def M_gg_109_0_0():
    def func(k):
        return (1 / 2) * k**6

    return func


def N_gg_109_0_0(theta, phi):
    return 1


def M_gg_109_0_1():
    def func(k):
        return (2 / 25) * k**6

    return func


def N_gg_109_0_1(theta, phi):
    return (15 / 4) * np.cos(2 * theta) + 5 / 4


def M_gg_109_2_0():
    def func(k):
        return (1 / 5) * k**6

    return func


def N_gg_109_2_0(theta, phi):
    return (15 / 4) * np.cos(2 * phi + theta) + 5 / 4


def M_gg_109_2_1():
    def func(k):
        return (1 / 5) * k**6

    return func


def N_gg_109_2_1(theta, phi):
    return (15 / 4) * np.cos(2 * phi - theta) + 5 / 4


def M_gg_109_2_2():
    def func(k):
        return (2 / 25) * k**6

    return func


def N_gg_109_2_2(theta, phi):
    return (
        (75 / 28) * np.cos(2 * theta)
        + (75 / 28) * np.cos(2 * phi - theta)
        + (75 / 28) * np.cos(2 * phi + theta)
        - 25 / 28
    )


def M_gg_109_4_0():
    def func(k):
        return (2 / 25) * k**6

    return func


def N_gg_109_4_0(theta, phi):
    return (
        (225 / 32) * np.cos(4 * phi)
        + (135 / 224) * np.cos(2 * theta)
        + (225 / 112) * np.cos(2 * phi - theta)
        + (225 / 112) * np.cos(2 * phi + theta)
        + 135 / 112
    )


def M_gg_110_0_0():
    def func(k):
        return (1 / 24) * k**4

    return func


def N_gg_110_0_0(theta, phi):
    return 1


def M_gg_110_0_1():
    def func(k):
        return (1 / 150) * k**4

    return func


def N_gg_110_0_1(theta, phi):
    return (15 / 4) * np.cos(2 * theta) + 5 / 4


def M_gg_110_2_0():
    def func(k):
        return (1 / 60) * k**4

    return func


def N_gg_110_2_0(theta, phi):
    return (15 / 4) * np.cos(2 * phi + theta) + 5 / 4


def M_gg_110_2_1():
    def func(k):
        return (1 / 60) * k**4

    return func


def N_gg_110_2_1(theta, phi):
    return (15 / 4) * np.cos(2 * phi - theta) + 5 / 4


def M_gg_110_2_2():
    def func(k):
        return (1 / 150) * k**4

    return func


def N_gg_110_2_2(theta, phi):
    return (
        (75 / 28) * np.cos(2 * theta)
        + (75 / 28) * np.cos(2 * phi - theta)
        + (75 / 28) * np.cos(2 * phi + theta)
        - 25 / 28
    )


def M_gg_110_4_0():
    def func(k):
        return (1 / 150) * k**4

    return func


def N_gg_110_4_0(theta, phi):
    return (
        (225 / 32) * np.cos(4 * phi)
        + (135 / 224) * np.cos(2 * theta)
        + (225 / 112) * np.cos(2 * phi - theta)
        + (225 / 112) * np.cos(2 * phi + theta)
        + 135 / 112
    )


def M_gg_111_0_0():
    def func(k):
        return (1 / 24) * k**4

    return func


def N_gg_111_0_0(theta, phi):
    return 1


def M_gg_111_0_1():
    def func(k):
        return (1 / 150) * k**4

    return func


def N_gg_111_0_1(theta, phi):
    return (15 / 4) * np.cos(2 * theta) + 5 / 4


def M_gg_111_2_0():
    def func(k):
        return (1 / 60) * k**4

    return func


def N_gg_111_2_0(theta, phi):
    return (15 / 4) * np.cos(2 * phi + theta) + 5 / 4


def M_gg_111_2_1():
    def func(k):
        return (1 / 60) * k**4

    return func


def N_gg_111_2_1(theta, phi):
    return (15 / 4) * np.cos(2 * phi - theta) + 5 / 4


def M_gg_111_2_2():
    def func(k):
        return (1 / 150) * k**4

    return func


def N_gg_111_2_2(theta, phi):
    return (
        (75 / 28) * np.cos(2 * theta)
        + (75 / 28) * np.cos(2 * phi - theta)
        + (75 / 28) * np.cos(2 * phi + theta)
        - 25 / 28
    )


def M_gg_111_4_0():
    def func(k):
        return (1 / 150) * k**4

    return func


def N_gg_111_4_0(theta, phi):
    return (
        (225 / 32) * np.cos(4 * phi)
        + (135 / 224) * np.cos(2 * theta)
        + (225 / 112) * np.cos(2 * phi - theta)
        + (225 / 112) * np.cos(2 * phi + theta)
        + 135 / 112
    )


def M_gg_112_0_0():
    def func(k):
        return (1 / 12) * k**4

    return func


def N_gg_112_0_0(theta, phi):
    return 1


def M_gg_112_0_1():
    def func(k):
        return (1 / 75) * k**4

    return func


def N_gg_112_0_1(theta, phi):
    return (15 / 4) * np.cos(2 * theta) + 5 / 4


def M_gg_112_2_0():
    def func(k):
        return (1 / 30) * k**4

    return func


def N_gg_112_2_0(theta, phi):
    return (15 / 4) * np.cos(2 * phi + theta) + 5 / 4


def M_gg_112_2_1():
    def func(k):
        return (1 / 30) * k**4

    return func


def N_gg_112_2_1(theta, phi):
    return (15 / 4) * np.cos(2 * phi - theta) + 5 / 4


def M_gg_112_2_2():
    def func(k):
        return (1 / 75) * k**4

    return func


def N_gg_112_2_2(theta, phi):
    return (
        (75 / 28) * np.cos(2 * theta)
        + (75 / 28) * np.cos(2 * phi - theta)
        + (75 / 28) * np.cos(2 * phi + theta)
        - 25 / 28
    )


def M_gg_112_4_0():
    def func(k):
        return (1 / 75) * k**4

    return func


def N_gg_112_4_0(theta, phi):
    return (
        (225 / 32) * np.cos(4 * phi)
        + (135 / 224) * np.cos(2 * theta)
        + (225 / 112) * np.cos(2 * phi - theta)
        + (225 / 112) * np.cos(2 * phi + theta)
        + 135 / 112
    )


def M_gg_113_0_0():
    def func(k):
        return (1 / 6) * k**4

    return func


def N_gg_113_0_0(theta, phi):
    return 1


def M_gg_113_0_1():
    def func(k):
        return (2 / 75) * k**4

    return func


def N_gg_113_0_1(theta, phi):
    return (15 / 4) * np.cos(2 * theta) + 5 / 4


def M_gg_113_2_0():
    def func(k):
        return (1 / 15) * k**4

    return func


def N_gg_113_2_0(theta, phi):
    return (15 / 4) * np.cos(2 * phi + theta) + 5 / 4


def M_gg_113_2_1():
    def func(k):
        return (1 / 15) * k**4

    return func


def N_gg_113_2_1(theta, phi):
    return (15 / 4) * np.cos(2 * phi - theta) + 5 / 4


def M_gg_113_2_2():
    def func(k):
        return (2 / 75) * k**4

    return func


def N_gg_113_2_2(theta, phi):
    return (
        (75 / 28) * np.cos(2 * theta)
        + (75 / 28) * np.cos(2 * phi - theta)
        + (75 / 28) * np.cos(2 * phi + theta)
        - 25 / 28
    )


def M_gg_113_4_0():
    def func(k):
        return (2 / 75) * k**4

    return func


def N_gg_113_4_0(theta, phi):
    return (
        (225 / 32) * np.cos(4 * phi)
        + (135 / 224) * np.cos(2 * theta)
        + (225 / 112) * np.cos(2 * phi - theta)
        + (225 / 112) * np.cos(2 * phi + theta)
        + 135 / 112
    )


def M_gg_114_0_0():
    def func(k):
        return -1 / 18 * k**2

    return func


def N_gg_114_0_0(theta, phi):
    return 1


def M_gg_114_0_1():
    def func(k):
        return -2 / 225 * k**2

    return func


def N_gg_114_0_1(theta, phi):
    return (15 / 4) * np.cos(2 * theta) + 5 / 4


def M_gg_114_2_0():
    def func(k):
        return -1 / 45 * k**2

    return func


def N_gg_114_2_0(theta, phi):
    return (15 / 4) * np.cos(2 * phi + theta) + 5 / 4


def M_gg_114_2_1():
    def func(k):
        return -1 / 45 * k**2

    return func


def N_gg_114_2_1(theta, phi):
    return (15 / 4) * np.cos(2 * phi - theta) + 5 / 4


def M_gg_114_2_2():
    def func(k):
        return -2 / 225 * k**2

    return func


def N_gg_114_2_2(theta, phi):
    return (
        (75 / 28) * np.cos(2 * theta)
        + (75 / 28) * np.cos(2 * phi - theta)
        + (75 / 28) * np.cos(2 * phi + theta)
        - 25 / 28
    )


def M_gg_114_4_0():
    def func(k):
        return -2 / 225 * k**2

    return func


def N_gg_114_4_0(theta, phi):
    return (
        (225 / 32) * np.cos(4 * phi)
        + (135 / 224) * np.cos(2 * theta)
        + (225 / 112) * np.cos(2 * phi - theta)
        + (225 / 112) * np.cos(2 * phi + theta)
        + 135 / 112
    )


def M_gg_115_0_0():
    def func(k):
        return -1 / 30 * k**2

    return func


def N_gg_115_0_0(theta, phi):
    return 1


def M_gg_115_0_1():
    def func(k):
        return -4 / 525 * k**2

    return func


def N_gg_115_0_1(theta, phi):
    return (15 / 4) * np.cos(2 * theta) + 5 / 4


def M_gg_115_2_0():
    def func(k):
        return -17 / 1050 * k**2

    return func


def N_gg_115_2_0(theta, phi):
    return (15 / 4) * np.cos(2 * phi + theta) + 5 / 4


def M_gg_115_2_1():
    def func(k):
        return -17 / 1050 * k**2

    return func


def N_gg_115_2_1(theta, phi):
    return (15 / 4) * np.cos(2 * phi - theta) + 5 / 4


def M_gg_115_2_2():
    def func(k):
        return -4 / 525 * k**2

    return func


def N_gg_115_2_2(theta, phi):
    return (
        (75 / 28) * np.cos(2 * theta)
        + (75 / 28) * np.cos(2 * phi - theta)
        + (75 / 28) * np.cos(2 * phi + theta)
        - 25 / 28
    )


def M_gg_115_2_3():
    def func(k):
        return -4 / 4725 * k**2

    return func


def N_gg_115_2_3(theta, phi):
    return (
        (225 / 112) * np.cos(2 * theta)
        + (135 / 224) * np.cos(2 * phi - theta)
        + (225 / 112) * np.cos(2 * phi + theta)
        + (225 / 32) * np.cos(2 * phi + 3 * theta)
        + 135 / 112
    )


def M_gg_115_2_4():
    def func(k):
        return -4 / 4725 * k**2

    return func


def N_gg_115_2_4(theta, phi):
    return (
        (225 / 112) * np.cos(2 * theta)
        + (225 / 32) * np.cos(2 * phi - 3 * theta)
        + (225 / 112) * np.cos(2 * phi - theta)
        + (135 / 224) * np.cos(2 * phi + theta)
        + 135 / 112
    )


def M_gg_115_4_0():
    def func(k):
        return -2 / 945 * k**2

    return func


def N_gg_115_4_0(theta, phi):
    return (
        (45 / 16) * np.cos(2 * phi + theta)
        + (315 / 64) * np.cos(4 * phi + 2 * theta)
        + 81 / 64
    )


def M_gg_115_4_1():
    def func(k):
        return -4 / 525 * k**2

    return func


def N_gg_115_4_1(theta, phi):
    return (
        (225 / 32) * np.cos(4 * phi)
        + (135 / 224) * np.cos(2 * theta)
        + (225 / 112) * np.cos(2 * phi - theta)
        + (225 / 112) * np.cos(2 * phi + theta)
        + 135 / 112
    )


def M_gg_115_4_2():
    def func(k):
        return -4 / 4725 * k**2

    return func


def N_gg_115_4_2(theta, phi):
    return (
        (675 / 352) * np.cos(4 * phi)
        + (6075 / 2464) * np.cos(2 * theta)
        + (6075 / 2464) * np.cos(2 * phi - theta)
        - 225 / 308 * np.cos(2 * phi + theta)
        + (675 / 352) * np.cos(2 * phi + 3 * theta)
        + (1575 / 352) * np.cos(4 * phi + 2 * theta)
        - 2025 / 2464
    )


def M_gg_115_4_3():
    def func(k):
        return -2 / 945 * k**2

    return func


def N_gg_115_4_3(theta, phi):
    return (
        (45 / 16) * np.cos(2 * phi - theta)
        + (315 / 64) * np.cos(4 * phi - 2 * theta)
        + 81 / 64
    )


def M_gg_115_4_4():
    def func(k):
        return -4 / 4725 * k**2

    return func


def N_gg_115_4_4(theta, phi):
    return (
        (675 / 352) * np.cos(4 * phi)
        + (6075 / 2464) * np.cos(2 * theta)
        + (675 / 352) * np.cos(2 * phi - 3 * theta)
        - 225 / 308 * np.cos(2 * phi - theta)
        + (6075 / 2464) * np.cos(2 * phi + theta)
        + (1575 / 352) * np.cos(4 * phi - 2 * theta)
        - 2025 / 2464
    )


def M_gg_115_6_0():
    def func(k):
        return -4 / 4725 * k**2

    return func


def N_gg_115_6_0(theta, phi):
    return (
        (4725 / 1408) * np.cos(4 * phi)
        + (1125 / 1408) * np.cos(2 * theta)
        + (4725 / 2816) * np.cos(2 * phi - theta)
        + (1575 / 704) * np.cos(2 * phi + theta)
        + (1575 / 5632) * np.cos(2 * phi + 3 * theta)
        + (4725 / 2816) * np.cos(4 * phi + 2 * theta)
        + (4725 / 512) * np.cos(6 * phi + theta)
        + 3375 / 2816
    )


def M_gg_115_6_1():
    def func(k):
        return -4 / 4725 * k**2

    return func


def N_gg_115_6_1(theta, phi):
    return (
        (4725 / 1408) * np.cos(4 * phi)
        + (1125 / 1408) * np.cos(2 * theta)
        + (1575 / 5632) * np.cos(2 * phi - 3 * theta)
        + (1575 / 704) * np.cos(2 * phi - theta)
        + (4725 / 2816) * np.cos(2 * phi + theta)
        + (4725 / 2816) * np.cos(4 * phi - 2 * theta)
        + (4725 / 512) * np.cos(6 * phi - theta)
        + 3375 / 2816
    )


def M_gg_116_0_0():
    def func(k):
        return -1 / 9 * k**4

    return func


def N_gg_116_0_0(theta, phi):
    return 1


def M_gg_116_0_1():
    def func(k):
        return -4 / 225 * k**4

    return func


def N_gg_116_0_1(theta, phi):
    return (15 / 4) * np.cos(2 * theta) + 5 / 4


def M_gg_116_2_0():
    def func(k):
        return -2 / 45 * k**4

    return func


def N_gg_116_2_0(theta, phi):
    return (15 / 4) * np.cos(2 * phi + theta) + 5 / 4


def M_gg_116_2_1():
    def func(k):
        return -2 / 45 * k**4

    return func


def N_gg_116_2_1(theta, phi):
    return (15 / 4) * np.cos(2 * phi - theta) + 5 / 4


def M_gg_116_2_2():
    def func(k):
        return -4 / 225 * k**4

    return func


def N_gg_116_2_2(theta, phi):
    return (
        (75 / 28) * np.cos(2 * theta)
        + (75 / 28) * np.cos(2 * phi - theta)
        + (75 / 28) * np.cos(2 * phi + theta)
        - 25 / 28
    )


def M_gg_116_4_0():
    def func(k):
        return -4 / 225 * k**4

    return func


def N_gg_116_4_0(theta, phi):
    return (
        (225 / 32) * np.cos(4 * phi)
        + (135 / 224) * np.cos(2 * theta)
        + (225 / 112) * np.cos(2 * phi - theta)
        + (225 / 112) * np.cos(2 * phi + theta)
        + 135 / 112
    )


def M_gg_117_0_0():
    def func(k):
        return -1 / 15 * k**4

    return func


def N_gg_117_0_0(theta, phi):
    return 1


def M_gg_117_0_1():
    def func(k):
        return -8 / 525 * k**4

    return func


def N_gg_117_0_1(theta, phi):
    return (15 / 4) * np.cos(2 * theta) + 5 / 4


def M_gg_117_2_0():
    def func(k):
        return -17 / 525 * k**4

    return func


def N_gg_117_2_0(theta, phi):
    return (15 / 4) * np.cos(2 * phi + theta) + 5 / 4


def M_gg_117_2_1():
    def func(k):
        return -17 / 525 * k**4

    return func


def N_gg_117_2_1(theta, phi):
    return (15 / 4) * np.cos(2 * phi - theta) + 5 / 4


def M_gg_117_2_2():
    def func(k):
        return -8 / 525 * k**4

    return func


def N_gg_117_2_2(theta, phi):
    return (
        (75 / 28) * np.cos(2 * theta)
        + (75 / 28) * np.cos(2 * phi - theta)
        + (75 / 28) * np.cos(2 * phi + theta)
        - 25 / 28
    )


def M_gg_117_2_3():
    def func(k):
        return -8 / 4725 * k**4

    return func


def N_gg_117_2_3(theta, phi):
    return (
        (225 / 112) * np.cos(2 * theta)
        + (135 / 224) * np.cos(2 * phi - theta)
        + (225 / 112) * np.cos(2 * phi + theta)
        + (225 / 32) * np.cos(2 * phi + 3 * theta)
        + 135 / 112
    )


def M_gg_117_2_4():
    def func(k):
        return -8 / 4725 * k**4

    return func


def N_gg_117_2_4(theta, phi):
    return (
        (225 / 112) * np.cos(2 * theta)
        + (225 / 32) * np.cos(2 * phi - 3 * theta)
        + (225 / 112) * np.cos(2 * phi - theta)
        + (135 / 224) * np.cos(2 * phi + theta)
        + 135 / 112
    )


def M_gg_117_4_0():
    def func(k):
        return -4 / 945 * k**4

    return func


def N_gg_117_4_0(theta, phi):
    return (
        (45 / 16) * np.cos(2 * phi + theta)
        + (315 / 64) * np.cos(4 * phi + 2 * theta)
        + 81 / 64
    )


def M_gg_117_4_1():
    def func(k):
        return -8 / 525 * k**4

    return func


def N_gg_117_4_1(theta, phi):
    return (
        (225 / 32) * np.cos(4 * phi)
        + (135 / 224) * np.cos(2 * theta)
        + (225 / 112) * np.cos(2 * phi - theta)
        + (225 / 112) * np.cos(2 * phi + theta)
        + 135 / 112
    )


def M_gg_117_4_2():
    def func(k):
        return -8 / 4725 * k**4

    return func


def N_gg_117_4_2(theta, phi):
    return (
        (675 / 352) * np.cos(4 * phi)
        + (6075 / 2464) * np.cos(2 * theta)
        + (6075 / 2464) * np.cos(2 * phi - theta)
        - 225 / 308 * np.cos(2 * phi + theta)
        + (675 / 352) * np.cos(2 * phi + 3 * theta)
        + (1575 / 352) * np.cos(4 * phi + 2 * theta)
        - 2025 / 2464
    )


def M_gg_117_4_3():
    def func(k):
        return -4 / 945 * k**4

    return func


def N_gg_117_4_3(theta, phi):
    return (
        (45 / 16) * np.cos(2 * phi - theta)
        + (315 / 64) * np.cos(4 * phi - 2 * theta)
        + 81 / 64
    )


def M_gg_117_4_4():
    def func(k):
        return -8 / 4725 * k**4

    return func


def N_gg_117_4_4(theta, phi):
    return (
        (675 / 352) * np.cos(4 * phi)
        + (6075 / 2464) * np.cos(2 * theta)
        + (675 / 352) * np.cos(2 * phi - 3 * theta)
        - 225 / 308 * np.cos(2 * phi - theta)
        + (6075 / 2464) * np.cos(2 * phi + theta)
        + (1575 / 352) * np.cos(4 * phi - 2 * theta)
        - 2025 / 2464
    )


def M_gg_117_6_0():
    def func(k):
        return -8 / 4725 * k**4

    return func


def N_gg_117_6_0(theta, phi):
    return (
        (4725 / 1408) * np.cos(4 * phi)
        + (1125 / 1408) * np.cos(2 * theta)
        + (4725 / 2816) * np.cos(2 * phi - theta)
        + (1575 / 704) * np.cos(2 * phi + theta)
        + (1575 / 5632) * np.cos(2 * phi + 3 * theta)
        + (4725 / 2816) * np.cos(4 * phi + 2 * theta)
        + (4725 / 512) * np.cos(6 * phi + theta)
        + 3375 / 2816
    )


def M_gg_117_6_1():
    def func(k):
        return -8 / 4725 * k**4

    return func


def N_gg_117_6_1(theta, phi):
    return (
        (4725 / 1408) * np.cos(4 * phi)
        + (1125 / 1408) * np.cos(2 * theta)
        + (1575 / 5632) * np.cos(2 * phi - 3 * theta)
        + (1575 / 704) * np.cos(2 * phi - theta)
        + (4725 / 2816) * np.cos(2 * phi + theta)
        + (4725 / 2816) * np.cos(4 * phi - 2 * theta)
        + (4725 / 512) * np.cos(6 * phi - theta)
        + 3375 / 2816
    )


def M_gg_118_0_0():
    def func(k):
        return -1 / 36 * k**2

    return func


def N_gg_118_0_0(theta, phi):
    return 1


def M_gg_118_0_1():
    def func(k):
        return -1 / 225 * k**2

    return func


def N_gg_118_0_1(theta, phi):
    return (15 / 4) * np.cos(2 * theta) + 5 / 4


def M_gg_118_2_0():
    def func(k):
        return -1 / 90 * k**2

    return func


def N_gg_118_2_0(theta, phi):
    return (15 / 4) * np.cos(2 * phi + theta) + 5 / 4


def M_gg_118_2_1():
    def func(k):
        return -1 / 90 * k**2

    return func


def N_gg_118_2_1(theta, phi):
    return (15 / 4) * np.cos(2 * phi - theta) + 5 / 4


def M_gg_118_2_2():
    def func(k):
        return -1 / 225 * k**2

    return func


def N_gg_118_2_2(theta, phi):
    return (
        (75 / 28) * np.cos(2 * theta)
        + (75 / 28) * np.cos(2 * phi - theta)
        + (75 / 28) * np.cos(2 * phi + theta)
        - 25 / 28
    )


def M_gg_118_4_0():
    def func(k):
        return -1 / 225 * k**2

    return func


def N_gg_118_4_0(theta, phi):
    return (
        (225 / 32) * np.cos(4 * phi)
        + (135 / 224) * np.cos(2 * theta)
        + (225 / 112) * np.cos(2 * phi - theta)
        + (225 / 112) * np.cos(2 * phi + theta)
        + 135 / 112
    )


def M_gg_119_0_0():
    def func(k):
        return -1 / 36 * k**2

    return func


def N_gg_119_0_0(theta, phi):
    return 1


def M_gg_119_0_1():
    def func(k):
        return -1 / 225 * k**2

    return func


def N_gg_119_0_1(theta, phi):
    return (15 / 4) * np.cos(2 * theta) + 5 / 4


def M_gg_119_2_0():
    def func(k):
        return -1 / 90 * k**2

    return func


def N_gg_119_2_0(theta, phi):
    return (15 / 4) * np.cos(2 * phi + theta) + 5 / 4


def M_gg_119_2_1():
    def func(k):
        return -1 / 90 * k**2

    return func


def N_gg_119_2_1(theta, phi):
    return (15 / 4) * np.cos(2 * phi - theta) + 5 / 4


def M_gg_119_2_2():
    def func(k):
        return -1 / 225 * k**2

    return func


def N_gg_119_2_2(theta, phi):
    return (
        (75 / 28) * np.cos(2 * theta)
        + (75 / 28) * np.cos(2 * phi - theta)
        + (75 / 28) * np.cos(2 * phi + theta)
        - 25 / 28
    )


def M_gg_119_4_0():
    def func(k):
        return -1 / 225 * k**2

    return func


def N_gg_119_4_0(theta, phi):
    return (
        (225 / 32) * np.cos(4 * phi)
        + (135 / 224) * np.cos(2 * theta)
        + (225 / 112) * np.cos(2 * phi - theta)
        + (225 / 112) * np.cos(2 * phi + theta)
        + 135 / 112
    )


def M_gg_120_0_0():
    def func(k):
        return -1 / 60 * k**2

    return func


def N_gg_120_0_0(theta, phi):
    return 1


def M_gg_120_0_1():
    def func(k):
        return -2 / 525 * k**2

    return func


def N_gg_120_0_1(theta, phi):
    return (15 / 4) * np.cos(2 * theta) + 5 / 4


def M_gg_120_2_0():
    def func(k):
        return -17 / 2100 * k**2

    return func


def N_gg_120_2_0(theta, phi):
    return (15 / 4) * np.cos(2 * phi + theta) + 5 / 4


def M_gg_120_2_1():
    def func(k):
        return -17 / 2100 * k**2

    return func


def N_gg_120_2_1(theta, phi):
    return (15 / 4) * np.cos(2 * phi - theta) + 5 / 4


def M_gg_120_2_2():
    def func(k):
        return -2 / 525 * k**2

    return func


def N_gg_120_2_2(theta, phi):
    return (
        (75 / 28) * np.cos(2 * theta)
        + (75 / 28) * np.cos(2 * phi - theta)
        + (75 / 28) * np.cos(2 * phi + theta)
        - 25 / 28
    )


def M_gg_120_2_3():
    def func(k):
        return -2 / 4725 * k**2

    return func


def N_gg_120_2_3(theta, phi):
    return (
        (225 / 112) * np.cos(2 * theta)
        + (135 / 224) * np.cos(2 * phi - theta)
        + (225 / 112) * np.cos(2 * phi + theta)
        + (225 / 32) * np.cos(2 * phi + 3 * theta)
        + 135 / 112
    )


def M_gg_120_2_4():
    def func(k):
        return -2 / 4725 * k**2

    return func


def N_gg_120_2_4(theta, phi):
    return (
        (225 / 112) * np.cos(2 * theta)
        + (225 / 32) * np.cos(2 * phi - 3 * theta)
        + (225 / 112) * np.cos(2 * phi - theta)
        + (135 / 224) * np.cos(2 * phi + theta)
        + 135 / 112
    )


def M_gg_120_4_0():
    def func(k):
        return -1 / 945 * k**2

    return func


def N_gg_120_4_0(theta, phi):
    return (
        (45 / 16) * np.cos(2 * phi + theta)
        + (315 / 64) * np.cos(4 * phi + 2 * theta)
        + 81 / 64
    )


def M_gg_120_4_1():
    def func(k):
        return -2 / 525 * k**2

    return func


def N_gg_120_4_1(theta, phi):
    return (
        (225 / 32) * np.cos(4 * phi)
        + (135 / 224) * np.cos(2 * theta)
        + (225 / 112) * np.cos(2 * phi - theta)
        + (225 / 112) * np.cos(2 * phi + theta)
        + 135 / 112
    )


def M_gg_120_4_2():
    def func(k):
        return -2 / 4725 * k**2

    return func


def N_gg_120_4_2(theta, phi):
    return (
        (675 / 352) * np.cos(4 * phi)
        + (6075 / 2464) * np.cos(2 * theta)
        + (6075 / 2464) * np.cos(2 * phi - theta)
        - 225 / 308 * np.cos(2 * phi + theta)
        + (675 / 352) * np.cos(2 * phi + 3 * theta)
        + (1575 / 352) * np.cos(4 * phi + 2 * theta)
        - 2025 / 2464
    )


def M_gg_120_4_3():
    def func(k):
        return -1 / 945 * k**2

    return func


def N_gg_120_4_3(theta, phi):
    return (
        (45 / 16) * np.cos(2 * phi - theta)
        + (315 / 64) * np.cos(4 * phi - 2 * theta)
        + 81 / 64
    )


def M_gg_120_4_4():
    def func(k):
        return -2 / 4725 * k**2

    return func


def N_gg_120_4_4(theta, phi):
    return (
        (675 / 352) * np.cos(4 * phi)
        + (6075 / 2464) * np.cos(2 * theta)
        + (675 / 352) * np.cos(2 * phi - 3 * theta)
        - 225 / 308 * np.cos(2 * phi - theta)
        + (6075 / 2464) * np.cos(2 * phi + theta)
        + (1575 / 352) * np.cos(4 * phi - 2 * theta)
        - 2025 / 2464
    )


def M_gg_120_6_0():
    def func(k):
        return -2 / 4725 * k**2

    return func


def N_gg_120_6_0(theta, phi):
    return (
        (4725 / 1408) * np.cos(4 * phi)
        + (1125 / 1408) * np.cos(2 * theta)
        + (4725 / 2816) * np.cos(2 * phi - theta)
        + (1575 / 704) * np.cos(2 * phi + theta)
        + (1575 / 5632) * np.cos(2 * phi + 3 * theta)
        + (4725 / 2816) * np.cos(4 * phi + 2 * theta)
        + (4725 / 512) * np.cos(6 * phi + theta)
        + 3375 / 2816
    )


def M_gg_120_6_1():
    def func(k):
        return -2 / 4725 * k**2

    return func


def N_gg_120_6_1(theta, phi):
    return (
        (4725 / 1408) * np.cos(4 * phi)
        + (1125 / 1408) * np.cos(2 * theta)
        + (1575 / 5632) * np.cos(2 * phi - 3 * theta)
        + (1575 / 704) * np.cos(2 * phi - theta)
        + (4725 / 2816) * np.cos(2 * phi + theta)
        + (4725 / 2816) * np.cos(4 * phi - 2 * theta)
        + (4725 / 512) * np.cos(6 * phi - theta)
        + 3375 / 2816
    )


def M_gg_121_0_0():
    def func(k):
        return -1 / 60 * k**2

    return func


def N_gg_121_0_0(theta, phi):
    return 1


def M_gg_121_0_1():
    def func(k):
        return -2 / 525 * k**2

    return func


def N_gg_121_0_1(theta, phi):
    return (15 / 4) * np.cos(2 * theta) + 5 / 4


def M_gg_121_2_0():
    def func(k):
        return -17 / 2100 * k**2

    return func


def N_gg_121_2_0(theta, phi):
    return (15 / 4) * np.cos(2 * phi + theta) + 5 / 4


def M_gg_121_2_1():
    def func(k):
        return -17 / 2100 * k**2

    return func


def N_gg_121_2_1(theta, phi):
    return (15 / 4) * np.cos(2 * phi - theta) + 5 / 4


def M_gg_121_2_2():
    def func(k):
        return -2 / 525 * k**2

    return func


def N_gg_121_2_2(theta, phi):
    return (
        (75 / 28) * np.cos(2 * theta)
        + (75 / 28) * np.cos(2 * phi - theta)
        + (75 / 28) * np.cos(2 * phi + theta)
        - 25 / 28
    )


def M_gg_121_2_3():
    def func(k):
        return -2 / 4725 * k**2

    return func


def N_gg_121_2_3(theta, phi):
    return (
        (225 / 112) * np.cos(2 * theta)
        + (135 / 224) * np.cos(2 * phi - theta)
        + (225 / 112) * np.cos(2 * phi + theta)
        + (225 / 32) * np.cos(2 * phi + 3 * theta)
        + 135 / 112
    )


def M_gg_121_2_4():
    def func(k):
        return -2 / 4725 * k**2

    return func


def N_gg_121_2_4(theta, phi):
    return (
        (225 / 112) * np.cos(2 * theta)
        + (225 / 32) * np.cos(2 * phi - 3 * theta)
        + (225 / 112) * np.cos(2 * phi - theta)
        + (135 / 224) * np.cos(2 * phi + theta)
        + 135 / 112
    )


def M_gg_121_4_0():
    def func(k):
        return -1 / 945 * k**2

    return func


def N_gg_121_4_0(theta, phi):
    return (
        (45 / 16) * np.cos(2 * phi + theta)
        + (315 / 64) * np.cos(4 * phi + 2 * theta)
        + 81 / 64
    )


def M_gg_121_4_1():
    def func(k):
        return -2 / 525 * k**2

    return func


def N_gg_121_4_1(theta, phi):
    return (
        (225 / 32) * np.cos(4 * phi)
        + (135 / 224) * np.cos(2 * theta)
        + (225 / 112) * np.cos(2 * phi - theta)
        + (225 / 112) * np.cos(2 * phi + theta)
        + 135 / 112
    )


def M_gg_121_4_2():
    def func(k):
        return -2 / 4725 * k**2

    return func


def N_gg_121_4_2(theta, phi):
    return (
        (675 / 352) * np.cos(4 * phi)
        + (6075 / 2464) * np.cos(2 * theta)
        + (6075 / 2464) * np.cos(2 * phi - theta)
        - 225 / 308 * np.cos(2 * phi + theta)
        + (675 / 352) * np.cos(2 * phi + 3 * theta)
        + (1575 / 352) * np.cos(4 * phi + 2 * theta)
        - 2025 / 2464
    )


def M_gg_121_4_3():
    def func(k):
        return -1 / 945 * k**2

    return func


def N_gg_121_4_3(theta, phi):
    return (
        (45 / 16) * np.cos(2 * phi - theta)
        + (315 / 64) * np.cos(4 * phi - 2 * theta)
        + 81 / 64
    )


def M_gg_121_4_4():
    def func(k):
        return -2 / 4725 * k**2

    return func


def N_gg_121_4_4(theta, phi):
    return (
        (675 / 352) * np.cos(4 * phi)
        + (6075 / 2464) * np.cos(2 * theta)
        + (675 / 352) * np.cos(2 * phi - 3 * theta)
        - 225 / 308 * np.cos(2 * phi - theta)
        + (6075 / 2464) * np.cos(2 * phi + theta)
        + (1575 / 352) * np.cos(4 * phi - 2 * theta)
        - 2025 / 2464
    )


def M_gg_121_6_0():
    def func(k):
        return -2 / 4725 * k**2

    return func


def N_gg_121_6_0(theta, phi):
    return (
        (4725 / 1408) * np.cos(4 * phi)
        + (1125 / 1408) * np.cos(2 * theta)
        + (4725 / 2816) * np.cos(2 * phi - theta)
        + (1575 / 704) * np.cos(2 * phi + theta)
        + (1575 / 5632) * np.cos(2 * phi + 3 * theta)
        + (4725 / 2816) * np.cos(4 * phi + 2 * theta)
        + (4725 / 512) * np.cos(6 * phi + theta)
        + 3375 / 2816
    )


def M_gg_121_6_1():
    def func(k):
        return -2 / 4725 * k**2

    return func


def N_gg_121_6_1(theta, phi):
    return (
        (4725 / 1408) * np.cos(4 * phi)
        + (1125 / 1408) * np.cos(2 * theta)
        + (1575 / 5632) * np.cos(2 * phi - 3 * theta)
        + (1575 / 704) * np.cos(2 * phi - theta)
        + (4725 / 2816) * np.cos(2 * phi + theta)
        + (4725 / 2816) * np.cos(4 * phi - 2 * theta)
        + (4725 / 512) * np.cos(6 * phi - theta)
        + 3375 / 2816
    )


def M_gg_122_0_0():
    def func(k):
        return -77 / 480

    return func


def N_gg_122_0_0(theta, phi):
    return 1


def M_gg_122_0_1():
    def func(k):
        return -11 / 300

    return func


def N_gg_122_0_1(theta, phi):
    return (15 / 4) * np.cos(2 * theta) + 5 / 4


def M_gg_122_2_0():
    def func(k):
        return -187 / 2400

    return func


def N_gg_122_2_0(theta, phi):
    return (15 / 4) * np.cos(2 * phi + theta) + 5 / 4


def M_gg_122_2_1():
    def func(k):
        return -187 / 2400

    return func


def N_gg_122_2_1(theta, phi):
    return (15 / 4) * np.cos(2 * phi - theta) + 5 / 4


def M_gg_122_2_2():
    def func(k):
        return -11 / 300

    return func


def N_gg_122_2_2(theta, phi):
    return (
        (75 / 28) * np.cos(2 * theta)
        + (75 / 28) * np.cos(2 * phi - theta)
        + (75 / 28) * np.cos(2 * phi + theta)
        - 25 / 28
    )


def M_gg_122_2_3():
    def func(k):
        return -11 / 2700

    return func


def N_gg_122_2_3(theta, phi):
    return (
        (225 / 112) * np.cos(2 * theta)
        + (135 / 224) * np.cos(2 * phi - theta)
        + (225 / 112) * np.cos(2 * phi + theta)
        + (225 / 32) * np.cos(2 * phi + 3 * theta)
        + 135 / 112
    )


def M_gg_122_2_4():
    def func(k):
        return -11 / 2700

    return func


def N_gg_122_2_4(theta, phi):
    return (
        (225 / 112) * np.cos(2 * theta)
        + (225 / 32) * np.cos(2 * phi - 3 * theta)
        + (225 / 112) * np.cos(2 * phi - theta)
        + (135 / 224) * np.cos(2 * phi + theta)
        + 135 / 112
    )


def M_gg_122_4_0():
    def func(k):
        return -11 / 1080

    return func


def N_gg_122_4_0(theta, phi):
    return (
        (45 / 16) * np.cos(2 * phi + theta)
        + (315 / 64) * np.cos(4 * phi + 2 * theta)
        + 81 / 64
    )


def M_gg_122_4_1():
    def func(k):
        return -11 / 300

    return func


def N_gg_122_4_1(theta, phi):
    return (
        (225 / 32) * np.cos(4 * phi)
        + (135 / 224) * np.cos(2 * theta)
        + (225 / 112) * np.cos(2 * phi - theta)
        + (225 / 112) * np.cos(2 * phi + theta)
        + 135 / 112
    )


def M_gg_122_4_2():
    def func(k):
        return -11 / 2700

    return func


def N_gg_122_4_2(theta, phi):
    return (
        (675 / 352) * np.cos(4 * phi)
        + (6075 / 2464) * np.cos(2 * theta)
        + (6075 / 2464) * np.cos(2 * phi - theta)
        - 225 / 308 * np.cos(2 * phi + theta)
        + (675 / 352) * np.cos(2 * phi + 3 * theta)
        + (1575 / 352) * np.cos(4 * phi + 2 * theta)
        - 2025 / 2464
    )


def M_gg_122_4_3():
    def func(k):
        return -11 / 1080

    return func


def N_gg_122_4_3(theta, phi):
    return (
        (45 / 16) * np.cos(2 * phi - theta)
        + (315 / 64) * np.cos(4 * phi - 2 * theta)
        + 81 / 64
    )


def M_gg_122_4_4():
    def func(k):
        return -11 / 2700

    return func


def N_gg_122_4_4(theta, phi):
    return (
        (675 / 352) * np.cos(4 * phi)
        + (6075 / 2464) * np.cos(2 * theta)
        + (675 / 352) * np.cos(2 * phi - 3 * theta)
        - 225 / 308 * np.cos(2 * phi - theta)
        + (6075 / 2464) * np.cos(2 * phi + theta)
        + (1575 / 352) * np.cos(4 * phi - 2 * theta)
        - 2025 / 2464
    )


def M_gg_122_6_0():
    def func(k):
        return -11 / 2700

    return func


def N_gg_122_6_0(theta, phi):
    return (
        (4725 / 1408) * np.cos(4 * phi)
        + (1125 / 1408) * np.cos(2 * theta)
        + (4725 / 2816) * np.cos(2 * phi - theta)
        + (1575 / 704) * np.cos(2 * phi + theta)
        + (1575 / 5632) * np.cos(2 * phi + 3 * theta)
        + (4725 / 2816) * np.cos(4 * phi + 2 * theta)
        + (4725 / 512) * np.cos(6 * phi + theta)
        + 3375 / 2816
    )


def M_gg_122_6_1():
    def func(k):
        return -11 / 2700

    return func


def N_gg_122_6_1(theta, phi):
    return (
        (4725 / 1408) * np.cos(4 * phi)
        + (1125 / 1408) * np.cos(2 * theta)
        + (1575 / 5632) * np.cos(2 * phi - 3 * theta)
        + (1575 / 704) * np.cos(2 * phi - theta)
        + (4725 / 2816) * np.cos(2 * phi + theta)
        + (4725 / 2816) * np.cos(4 * phi - 2 * theta)
        + (4725 / 512) * np.cos(6 * phi - theta)
        + 3375 / 2816
    )


def M_gg_123_0_0():
    def func(k):
        return -77 / 1600

    return func


def N_gg_123_0_0(theta, phi):
    return -3 * np.cos(2 * theta) - 3


def M_gg_123_0_1():
    def func(k):
        return -11 / 2800

    return func


def N_gg_123_0_1(theta, phi):
    return -7 * np.cos(2 * theta) - 35 / 8 * np.cos(4 * theta) - 21 / 8


def M_gg_123_2_0():
    def func(k):
        return -77 / 1600

    return func


def N_gg_123_2_0(theta, phi):
    return (
        -3 / 2 * np.cos(2 * theta)
        - 9 / 2 * np.cos(2 * phi - theta)
        - 9 / 2 * np.cos(2 * phi + theta)
        - 3 / 2
    )


def M_gg_123_2_1():
    def func(k):
        return -11 / 800

    return func


def N_gg_123_2_1(theta, phi):
    return (
        -9 / 4 * np.cos(2 * theta)
        - 9 / 8 * np.cos(2 * phi - theta)
        - 27 / 4 * np.cos(2 * phi + theta)
        - 45 / 8 * np.cos(2 * phi + 3 * theta)
        - 9 / 4
    )


def M_gg_123_2_2():
    def func(k):
        return -11 / 800

    return func


def N_gg_123_2_2(theta, phi):
    return (
        -9 / 4 * np.cos(2 * theta)
        - 45 / 8 * np.cos(2 * phi - 3 * theta)
        - 27 / 4 * np.cos(2 * phi - theta)
        - 9 / 8 * np.cos(2 * phi + theta)
        - 9 / 4
    )


def M_gg_123_2_3():
    def func(k):
        return -11 / 2800

    return func


def N_gg_123_2_3(theta, phi):
    return (
        -7 / 3 * np.cos(2 * theta)
        - 175 / 48 * np.cos(4 * theta)
        - 35 / 16 * np.cos(2 * phi - 3 * theta)
        - 77 / 16 * np.cos(2 * phi - theta)
        - 77 / 16 * np.cos(2 * phi + theta)
        - 35 / 16 * np.cos(2 * phi + 3 * theta)
        + 21 / 16
    )


def M_gg_123_4_0():
    def func(k):
        return -11 / 800

    return func


def N_gg_123_4_0(theta, phi):
    return (
        -105 / 16 * np.cos(4 * phi)
        - 27 / 16 * np.cos(2 * theta)
        - 45 / 16 * np.cos(2 * phi - theta)
        - 15 / 4 * np.cos(2 * phi + theta)
        - 15 / 16 * np.cos(2 * phi + 3 * theta)
        - 105 / 16 * np.cos(4 * phi + 2 * theta)
        - 27 / 16
    )


def M_gg_123_4_1():
    def func(k):
        return -11 / 800

    return func


def N_gg_123_4_1(theta, phi):
    return (
        -105 / 16 * np.cos(4 * phi)
        - 27 / 16 * np.cos(2 * theta)
        - 15 / 16 * np.cos(2 * phi - 3 * theta)
        - 15 / 4 * np.cos(2 * phi - theta)
        - 45 / 16 * np.cos(2 * phi + theta)
        - 105 / 16 * np.cos(4 * phi - 2 * theta)
        - 27 / 16
    )


def M_gg_123_4_2():
    def func(k):
        return -11 / 2800

    return func


def N_gg_123_4_2(theta, phi):
    return (
        -2205 / 352 * np.cos(4 * phi)
        - 567 / 352 * np.cos(2 * theta)
        - 945 / 704 * np.cos(4 * theta)
        - 945 / 352 * np.cos(2 * phi - 3 * theta)
        - 315 / 352 * np.cos(2 * phi - theta)
        - 315 / 352 * np.cos(2 * phi + theta)
        - 945 / 352 * np.cos(2 * phi + 3 * theta)
        - 2205 / 704 * np.cos(4 * phi - 2 * theta)
        - 2205 / 704 * np.cos(4 * phi + 2 * theta)
        - 189 / 704
    )


def M_gg_123_6_0():
    def func(k):
        return -11 / 2800

    return func


def N_gg_123_6_0(theta, phi):
    return (
        -3675 / 704 * np.cos(4 * phi)
        - 4375 / 2112 * np.cos(2 * theta)
        - 875 / 4224 * np.cos(4 * theta)
        - 1225 / 1408 * np.cos(2 * phi - 3 * theta)
        - 1225 / 352 * np.cos(2 * phi - theta)
        - 1225 / 352 * np.cos(2 * phi + theta)
        - 1225 / 1408 * np.cos(2 * phi + 3 * theta)
        - 3675 / 1408 * np.cos(4 * phi - 2 * theta)
        - 3675 / 1408 * np.cos(4 * phi + 2 * theta)
        - 1225 / 128 * np.cos(6 * phi - theta)
        - 1225 / 128 * np.cos(6 * phi + theta)
        - 2625 / 1408
    )


def M_gg_124_0_0():
    def func(k):
        return 7 / 80

    return func


def N_gg_124_0_0(theta, phi):
    return 1


def M_gg_124_0_1():
    def func(k):
        return 1 / 50

    return func


def N_gg_124_0_1(theta, phi):
    return (15 / 4) * np.cos(2 * theta) + 5 / 4


def M_gg_124_2_0():
    def func(k):
        return 17 / 400

    return func


def N_gg_124_2_0(theta, phi):
    return (15 / 4) * np.cos(2 * phi + theta) + 5 / 4


def M_gg_124_2_1():
    def func(k):
        return 17 / 400

    return func


def N_gg_124_2_1(theta, phi):
    return (15 / 4) * np.cos(2 * phi - theta) + 5 / 4


def M_gg_124_2_2():
    def func(k):
        return 1 / 50

    return func


def N_gg_124_2_2(theta, phi):
    return (
        (75 / 28) * np.cos(2 * theta)
        + (75 / 28) * np.cos(2 * phi - theta)
        + (75 / 28) * np.cos(2 * phi + theta)
        - 25 / 28
    )


def M_gg_124_2_3():
    def func(k):
        return 1 / 450

    return func


def N_gg_124_2_3(theta, phi):
    return (
        (225 / 112) * np.cos(2 * theta)
        + (135 / 224) * np.cos(2 * phi - theta)
        + (225 / 112) * np.cos(2 * phi + theta)
        + (225 / 32) * np.cos(2 * phi + 3 * theta)
        + 135 / 112
    )


def M_gg_124_2_4():
    def func(k):
        return 1 / 450

    return func


def N_gg_124_2_4(theta, phi):
    return (
        (225 / 112) * np.cos(2 * theta)
        + (225 / 32) * np.cos(2 * phi - 3 * theta)
        + (225 / 112) * np.cos(2 * phi - theta)
        + (135 / 224) * np.cos(2 * phi + theta)
        + 135 / 112
    )


def M_gg_124_4_0():
    def func(k):
        return 1 / 180

    return func


def N_gg_124_4_0(theta, phi):
    return (
        (45 / 16) * np.cos(2 * phi + theta)
        + (315 / 64) * np.cos(4 * phi + 2 * theta)
        + 81 / 64
    )


def M_gg_124_4_1():
    def func(k):
        return 1 / 50

    return func


def N_gg_124_4_1(theta, phi):
    return (
        (225 / 32) * np.cos(4 * phi)
        + (135 / 224) * np.cos(2 * theta)
        + (225 / 112) * np.cos(2 * phi - theta)
        + (225 / 112) * np.cos(2 * phi + theta)
        + 135 / 112
    )


def M_gg_124_4_2():
    def func(k):
        return 1 / 450

    return func


def N_gg_124_4_2(theta, phi):
    return (
        (675 / 352) * np.cos(4 * phi)
        + (6075 / 2464) * np.cos(2 * theta)
        + (6075 / 2464) * np.cos(2 * phi - theta)
        - 225 / 308 * np.cos(2 * phi + theta)
        + (675 / 352) * np.cos(2 * phi + 3 * theta)
        + (1575 / 352) * np.cos(4 * phi + 2 * theta)
        - 2025 / 2464
    )


def M_gg_124_4_3():
    def func(k):
        return 1 / 180

    return func


def N_gg_124_4_3(theta, phi):
    return (
        (45 / 16) * np.cos(2 * phi - theta)
        + (315 / 64) * np.cos(4 * phi - 2 * theta)
        + 81 / 64
    )


def M_gg_124_4_4():
    def func(k):
        return 1 / 450

    return func


def N_gg_124_4_4(theta, phi):
    return (
        (675 / 352) * np.cos(4 * phi)
        + (6075 / 2464) * np.cos(2 * theta)
        + (675 / 352) * np.cos(2 * phi - 3 * theta)
        - 225 / 308 * np.cos(2 * phi - theta)
        + (6075 / 2464) * np.cos(2 * phi + theta)
        + (1575 / 352) * np.cos(4 * phi - 2 * theta)
        - 2025 / 2464
    )


def M_gg_124_6_0():
    def func(k):
        return 1 / 450

    return func


def N_gg_124_6_0(theta, phi):
    return (
        (4725 / 1408) * np.cos(4 * phi)
        + (1125 / 1408) * np.cos(2 * theta)
        + (4725 / 2816) * np.cos(2 * phi - theta)
        + (1575 / 704) * np.cos(2 * phi + theta)
        + (1575 / 5632) * np.cos(2 * phi + 3 * theta)
        + (4725 / 2816) * np.cos(4 * phi + 2 * theta)
        + (4725 / 512) * np.cos(6 * phi + theta)
        + 3375 / 2816
    )


def M_gg_124_6_1():
    def func(k):
        return 1 / 450

    return func


def N_gg_124_6_1(theta, phi):
    return (
        (4725 / 1408) * np.cos(4 * phi)
        + (1125 / 1408) * np.cos(2 * theta)
        + (1575 / 5632) * np.cos(2 * phi - 3 * theta)
        + (1575 / 704) * np.cos(2 * phi - theta)
        + (4725 / 2816) * np.cos(2 * phi + theta)
        + (4725 / 2816) * np.cos(4 * phi - 2 * theta)
        + (4725 / 512) * np.cos(6 * phi - theta)
        + 3375 / 2816
    )


def M_gg_125_0_0():
    def func(k):
        return 21 / 800

    return func


def N_gg_125_0_0(theta, phi):
    return -3 * np.cos(2 * theta) - 3


def M_gg_125_0_1():
    def func(k):
        return 3 / 1400

    return func


def N_gg_125_0_1(theta, phi):
    return -7 * np.cos(2 * theta) - 35 / 8 * np.cos(4 * theta) - 21 / 8


def M_gg_125_2_0():
    def func(k):
        return 21 / 800

    return func


def N_gg_125_2_0(theta, phi):
    return (
        -3 / 2 * np.cos(2 * theta)
        - 9 / 2 * np.cos(2 * phi - theta)
        - 9 / 2 * np.cos(2 * phi + theta)
        - 3 / 2
    )


def M_gg_125_2_1():
    def func(k):
        return 3 / 400

    return func


def N_gg_125_2_1(theta, phi):
    return (
        -9 / 4 * np.cos(2 * theta)
        - 9 / 8 * np.cos(2 * phi - theta)
        - 27 / 4 * np.cos(2 * phi + theta)
        - 45 / 8 * np.cos(2 * phi + 3 * theta)
        - 9 / 4
    )


def M_gg_125_2_2():
    def func(k):
        return 3 / 400

    return func


def N_gg_125_2_2(theta, phi):
    return (
        -9 / 4 * np.cos(2 * theta)
        - 45 / 8 * np.cos(2 * phi - 3 * theta)
        - 27 / 4 * np.cos(2 * phi - theta)
        - 9 / 8 * np.cos(2 * phi + theta)
        - 9 / 4
    )


def M_gg_125_2_3():
    def func(k):
        return 3 / 1400

    return func


def N_gg_125_2_3(theta, phi):
    return (
        -7 / 3 * np.cos(2 * theta)
        - 175 / 48 * np.cos(4 * theta)
        - 35 / 16 * np.cos(2 * phi - 3 * theta)
        - 77 / 16 * np.cos(2 * phi - theta)
        - 77 / 16 * np.cos(2 * phi + theta)
        - 35 / 16 * np.cos(2 * phi + 3 * theta)
        + 21 / 16
    )


def M_gg_125_4_0():
    def func(k):
        return 3 / 400

    return func


def N_gg_125_4_0(theta, phi):
    return (
        -105 / 16 * np.cos(4 * phi)
        - 27 / 16 * np.cos(2 * theta)
        - 45 / 16 * np.cos(2 * phi - theta)
        - 15 / 4 * np.cos(2 * phi + theta)
        - 15 / 16 * np.cos(2 * phi + 3 * theta)
        - 105 / 16 * np.cos(4 * phi + 2 * theta)
        - 27 / 16
    )


def M_gg_125_4_1():
    def func(k):
        return 3 / 400

    return func


def N_gg_125_4_1(theta, phi):
    return (
        -105 / 16 * np.cos(4 * phi)
        - 27 / 16 * np.cos(2 * theta)
        - 15 / 16 * np.cos(2 * phi - 3 * theta)
        - 15 / 4 * np.cos(2 * phi - theta)
        - 45 / 16 * np.cos(2 * phi + theta)
        - 105 / 16 * np.cos(4 * phi - 2 * theta)
        - 27 / 16
    )


def M_gg_125_4_2():
    def func(k):
        return 3 / 1400

    return func


def N_gg_125_4_2(theta, phi):
    return (
        -2205 / 352 * np.cos(4 * phi)
        - 567 / 352 * np.cos(2 * theta)
        - 945 / 704 * np.cos(4 * theta)
        - 945 / 352 * np.cos(2 * phi - 3 * theta)
        - 315 / 352 * np.cos(2 * phi - theta)
        - 315 / 352 * np.cos(2 * phi + theta)
        - 945 / 352 * np.cos(2 * phi + 3 * theta)
        - 2205 / 704 * np.cos(4 * phi - 2 * theta)
        - 2205 / 704 * np.cos(4 * phi + 2 * theta)
        - 189 / 704
    )


def M_gg_125_6_0():
    def func(k):
        return 3 / 1400

    return func


def N_gg_125_6_0(theta, phi):
    return (
        -3675 / 704 * np.cos(4 * phi)
        - 4375 / 2112 * np.cos(2 * theta)
        - 875 / 4224 * np.cos(4 * theta)
        - 1225 / 1408 * np.cos(2 * phi - 3 * theta)
        - 1225 / 352 * np.cos(2 * phi - theta)
        - 1225 / 352 * np.cos(2 * phi + theta)
        - 1225 / 1408 * np.cos(2 * phi + 3 * theta)
        - 3675 / 1408 * np.cos(4 * phi - 2 * theta)
        - 3675 / 1408 * np.cos(4 * phi + 2 * theta)
        - 1225 / 128 * np.cos(6 * phi - theta)
        - 1225 / 128 * np.cos(6 * phi + theta)
        - 2625 / 1408
    )


def M_gg_126_0_0():
    def func(k):
        return 7 / 96

    return func


def N_gg_126_0_0(theta, phi):
    return 1


def M_gg_126_0_1():
    def func(k):
        return 1 / 60

    return func


def N_gg_126_0_1(theta, phi):
    return (15 / 4) * np.cos(2 * theta) + 5 / 4


def M_gg_126_2_0():
    def func(k):
        return 17 / 480

    return func


def N_gg_126_2_0(theta, phi):
    return (15 / 4) * np.cos(2 * phi + theta) + 5 / 4


def M_gg_126_2_1():
    def func(k):
        return 17 / 480

    return func


def N_gg_126_2_1(theta, phi):
    return (15 / 4) * np.cos(2 * phi - theta) + 5 / 4


def M_gg_126_2_2():
    def func(k):
        return 1 / 60

    return func


def N_gg_126_2_2(theta, phi):
    return (
        (75 / 28) * np.cos(2 * theta)
        + (75 / 28) * np.cos(2 * phi - theta)
        + (75 / 28) * np.cos(2 * phi + theta)
        - 25 / 28
    )


def M_gg_126_2_3():
    def func(k):
        return 1 / 540

    return func


def N_gg_126_2_3(theta, phi):
    return (
        (225 / 112) * np.cos(2 * theta)
        + (135 / 224) * np.cos(2 * phi - theta)
        + (225 / 112) * np.cos(2 * phi + theta)
        + (225 / 32) * np.cos(2 * phi + 3 * theta)
        + 135 / 112
    )


def M_gg_126_2_4():
    def func(k):
        return 1 / 540

    return func


def N_gg_126_2_4(theta, phi):
    return (
        (225 / 112) * np.cos(2 * theta)
        + (225 / 32) * np.cos(2 * phi - 3 * theta)
        + (225 / 112) * np.cos(2 * phi - theta)
        + (135 / 224) * np.cos(2 * phi + theta)
        + 135 / 112
    )


def M_gg_126_4_0():
    def func(k):
        return 1 / 216

    return func


def N_gg_126_4_0(theta, phi):
    return (
        (45 / 16) * np.cos(2 * phi + theta)
        + (315 / 64) * np.cos(4 * phi + 2 * theta)
        + 81 / 64
    )


def M_gg_126_4_1():
    def func(k):
        return 1 / 60

    return func


def N_gg_126_4_1(theta, phi):
    return (
        (225 / 32) * np.cos(4 * phi)
        + (135 / 224) * np.cos(2 * theta)
        + (225 / 112) * np.cos(2 * phi - theta)
        + (225 / 112) * np.cos(2 * phi + theta)
        + 135 / 112
    )


def M_gg_126_4_2():
    def func(k):
        return 1 / 540

    return func


def N_gg_126_4_2(theta, phi):
    return (
        (675 / 352) * np.cos(4 * phi)
        + (6075 / 2464) * np.cos(2 * theta)
        + (6075 / 2464) * np.cos(2 * phi - theta)
        - 225 / 308 * np.cos(2 * phi + theta)
        + (675 / 352) * np.cos(2 * phi + 3 * theta)
        + (1575 / 352) * np.cos(4 * phi + 2 * theta)
        - 2025 / 2464
    )


def M_gg_126_4_3():
    def func(k):
        return 1 / 216

    return func


def N_gg_126_4_3(theta, phi):
    return (
        (45 / 16) * np.cos(2 * phi - theta)
        + (315 / 64) * np.cos(4 * phi - 2 * theta)
        + 81 / 64
    )


def M_gg_126_4_4():
    def func(k):
        return 1 / 540

    return func


def N_gg_126_4_4(theta, phi):
    return (
        (675 / 352) * np.cos(4 * phi)
        + (6075 / 2464) * np.cos(2 * theta)
        + (675 / 352) * np.cos(2 * phi - 3 * theta)
        - 225 / 308 * np.cos(2 * phi - theta)
        + (6075 / 2464) * np.cos(2 * phi + theta)
        + (1575 / 352) * np.cos(4 * phi - 2 * theta)
        - 2025 / 2464
    )


def M_gg_126_6_0():
    def func(k):
        return 1 / 540

    return func


def N_gg_126_6_0(theta, phi):
    return (
        (4725 / 1408) * np.cos(4 * phi)
        + (1125 / 1408) * np.cos(2 * theta)
        + (4725 / 2816) * np.cos(2 * phi - theta)
        + (1575 / 704) * np.cos(2 * phi + theta)
        + (1575 / 5632) * np.cos(2 * phi + 3 * theta)
        + (4725 / 2816) * np.cos(4 * phi + 2 * theta)
        + (4725 / 512) * np.cos(6 * phi + theta)
        + 3375 / 2816
    )


def M_gg_126_6_1():
    def func(k):
        return 1 / 540

    return func


def N_gg_126_6_1(theta, phi):
    return (
        (4725 / 1408) * np.cos(4 * phi)
        + (1125 / 1408) * np.cos(2 * theta)
        + (1575 / 5632) * np.cos(2 * phi - 3 * theta)
        + (1575 / 704) * np.cos(2 * phi - theta)
        + (4725 / 2816) * np.cos(2 * phi + theta)
        + (4725 / 2816) * np.cos(4 * phi - 2 * theta)
        + (4725 / 512) * np.cos(6 * phi - theta)
        + 3375 / 2816
    )


def M_gg_127_0_0():
    def func(k):
        return 7 / 320

    return func


def N_gg_127_0_0(theta, phi):
    return -3 * np.cos(2 * theta) - 3


def M_gg_127_0_1():
    def func(k):
        return 1 / 560

    return func


def N_gg_127_0_1(theta, phi):
    return -7 * np.cos(2 * theta) - 35 / 8 * np.cos(4 * theta) - 21 / 8


def M_gg_127_2_0():
    def func(k):
        return 7 / 320

    return func


def N_gg_127_2_0(theta, phi):
    return (
        -3 / 2 * np.cos(2 * theta)
        - 9 / 2 * np.cos(2 * phi - theta)
        - 9 / 2 * np.cos(2 * phi + theta)
        - 3 / 2
    )


def M_gg_127_2_1():
    def func(k):
        return 1 / 160

    return func


def N_gg_127_2_1(theta, phi):
    return (
        -9 / 4 * np.cos(2 * theta)
        - 9 / 8 * np.cos(2 * phi - theta)
        - 27 / 4 * np.cos(2 * phi + theta)
        - 45 / 8 * np.cos(2 * phi + 3 * theta)
        - 9 / 4
    )


def M_gg_127_2_2():
    def func(k):
        return 1 / 160

    return func


def N_gg_127_2_2(theta, phi):
    return (
        -9 / 4 * np.cos(2 * theta)
        - 45 / 8 * np.cos(2 * phi - 3 * theta)
        - 27 / 4 * np.cos(2 * phi - theta)
        - 9 / 8 * np.cos(2 * phi + theta)
        - 9 / 4
    )


def M_gg_127_2_3():
    def func(k):
        return 1 / 560

    return func


def N_gg_127_2_3(theta, phi):
    return (
        -7 / 3 * np.cos(2 * theta)
        - 175 / 48 * np.cos(4 * theta)
        - 35 / 16 * np.cos(2 * phi - 3 * theta)
        - 77 / 16 * np.cos(2 * phi - theta)
        - 77 / 16 * np.cos(2 * phi + theta)
        - 35 / 16 * np.cos(2 * phi + 3 * theta)
        + 21 / 16
    )


def M_gg_127_4_0():
    def func(k):
        return 1 / 160

    return func


def N_gg_127_4_0(theta, phi):
    return (
        -105 / 16 * np.cos(4 * phi)
        - 27 / 16 * np.cos(2 * theta)
        - 45 / 16 * np.cos(2 * phi - theta)
        - 15 / 4 * np.cos(2 * phi + theta)
        - 15 / 16 * np.cos(2 * phi + 3 * theta)
        - 105 / 16 * np.cos(4 * phi + 2 * theta)
        - 27 / 16
    )


def M_gg_127_4_1():
    def func(k):
        return 1 / 160

    return func


def N_gg_127_4_1(theta, phi):
    return (
        -105 / 16 * np.cos(4 * phi)
        - 27 / 16 * np.cos(2 * theta)
        - 15 / 16 * np.cos(2 * phi - 3 * theta)
        - 15 / 4 * np.cos(2 * phi - theta)
        - 45 / 16 * np.cos(2 * phi + theta)
        - 105 / 16 * np.cos(4 * phi - 2 * theta)
        - 27 / 16
    )


def M_gg_127_4_2():
    def func(k):
        return 1 / 560

    return func


def N_gg_127_4_2(theta, phi):
    return (
        -2205 / 352 * np.cos(4 * phi)
        - 567 / 352 * np.cos(2 * theta)
        - 945 / 704 * np.cos(4 * theta)
        - 945 / 352 * np.cos(2 * phi - 3 * theta)
        - 315 / 352 * np.cos(2 * phi - theta)
        - 315 / 352 * np.cos(2 * phi + theta)
        - 945 / 352 * np.cos(2 * phi + 3 * theta)
        - 2205 / 704 * np.cos(4 * phi - 2 * theta)
        - 2205 / 704 * np.cos(4 * phi + 2 * theta)
        - 189 / 704
    )


def M_gg_127_6_0():
    def func(k):
        return 1 / 560

    return func


def N_gg_127_6_0(theta, phi):
    return (
        -3675 / 704 * np.cos(4 * phi)
        - 4375 / 2112 * np.cos(2 * theta)
        - 875 / 4224 * np.cos(4 * theta)
        - 1225 / 1408 * np.cos(2 * phi - 3 * theta)
        - 1225 / 352 * np.cos(2 * phi - theta)
        - 1225 / 352 * np.cos(2 * phi + theta)
        - 1225 / 1408 * np.cos(2 * phi + 3 * theta)
        - 3675 / 1408 * np.cos(4 * phi - 2 * theta)
        - 3675 / 1408 * np.cos(4 * phi + 2 * theta)
        - 1225 / 128 * np.cos(6 * phi - theta)
        - 1225 / 128 * np.cos(6 * phi + theta)
        - 2625 / 1408
    )


def M_gv_0_1_0():
    def func(k):
        return (1 / 3) / k

    return func


def N_gv_0_1_0(theta, phi):
    return -3 * np.cos(phi + (1 / 2) * theta)


def M_gv_1_1_0():
    def func(k):
        return (2 / 3) / k

    return func


def N_gv_1_1_0(theta, phi):
    return -3 * np.cos(phi + (1 / 2) * theta)


def M_gv_2_1_0():
    def func(k):
        return (2 / 3) / k

    return func


def N_gv_2_1_0(theta, phi):
    return -3 * np.cos(phi + (1 / 2) * theta)


def M_gv_3_1_0():
    def func(k):
        return 2 * k

    return func


def N_gv_3_1_0(theta, phi):
    return -3 * np.cos(phi + (1 / 2) * theta)


def M_gv_4_1_0():
    def func(k):
        return 2 * k

    return func


def N_gv_4_1_0(theta, phi):
    return -3 * np.cos(phi + (1 / 2) * theta)


def M_gv_5_1_0():
    def func(k):
        return -(1 / 3) / k

    return func


def N_gv_5_1_0(theta, phi):
    return -3 * np.cos(phi + (1 / 2) * theta)


def M_gv_6_1_0():
    def func(k):
        return -(1 / 3) / k

    return func


def N_gv_6_1_0(theta, phi):
    return -3 * np.cos(phi + (1 / 2) * theta)


def M_gv_7_1_0():
    def func(k):
        return -(1 / 3) / k

    return func


def N_gv_7_1_0(theta, phi):
    return -3 * np.cos(phi + (1 / 2) * theta)


def M_gv_8_1_0():
    def func(k):
        return -(1 / 3) / k

    return func


def N_gv_8_1_0(theta, phi):
    return -3 * np.cos(phi + (1 / 2) * theta)


def M_gv_9_1_0():
    def func(k):
        return -(1 / 3) / k

    return func


def N_gv_9_1_0(theta, phi):
    return -3 * np.cos(phi + (1 / 2) * theta)


def M_gv_10_1_0():
    def func(k):
        return (1 / 3) / k

    return func


def N_gv_10_1_0(theta, phi):
    return -3 * np.cos(phi + (1 / 2) * theta)


def M_gv_11_1_0():
    def func(k):
        return (1 / 5) / k

    return func


def N_gv_11_1_0(theta, phi):
    return -3 * np.cos(phi + (1 / 2) * theta)


def M_gv_11_3_0():
    def func(k):
        return (2 / 35) / k

    return func


def N_gv_11_3_0(theta, phi):
    return -21 / 8 * np.cos(phi + (1 / 2) * theta) - 35 / 8 * np.cos(
        3 * phi + (3 / 2) * theta
    )


def M_gv_12_1_0():
    def func(k):
        return (2 / 3) * k

    return func


def N_gv_12_1_0(theta, phi):
    return -3 * np.cos(phi + (1 / 2) * theta)


def M_gv_13_1_0():
    def func(k):
        return (2 / 5) * k

    return func


def N_gv_13_1_0(theta, phi):
    return -3 * np.cos(phi + (1 / 2) * theta)


def M_gv_13_3_0():
    def func(k):
        return (4 / 35) * k

    return func


def N_gv_13_3_0(theta, phi):
    return -21 / 8 * np.cos(phi + (1 / 2) * theta) - 35 / 8 * np.cos(
        3 * phi + (3 / 2) * theta
    )


def M_gv_14_1_0():
    def func(k):
        return -1 / 3 * k

    return func


def N_gv_14_1_0(theta, phi):
    return -3 * np.cos(phi + (1 / 2) * theta)


def M_gv_15_1_0():
    def func(k):
        return -1 / 3 * k

    return func


def N_gv_15_1_0(theta, phi):
    return -3 * np.cos(phi + (1 / 2) * theta)


def M_gv_16_1_0():
    def func(k):
        return -1 / 3 * k

    return func


def N_gv_16_1_0(theta, phi):
    return -3 * np.cos(phi + (1 / 2) * theta)


def M_gv_17_1_0():
    def func(k):
        return -2 / 3 * k

    return func


def N_gv_17_1_0(theta, phi):
    return -3 * np.cos(phi + (1 / 2) * theta)


def M_gv_18_1_0():
    def func(k):
        return -2 * k**3

    return func


def N_gv_18_1_0(theta, phi):
    return -3 * np.cos(phi + (1 / 2) * theta)


def M_gv_19_1_0():
    def func(k):
        return -1 / 6 * k

    return func


def N_gv_19_1_0(theta, phi):
    return -3 * np.cos(phi + (1 / 2) * theta)


def M_gv_20_1_0():
    def func(k):
        return -1 / 6 * k

    return func


def N_gv_20_1_0(theta, phi):
    return -3 * np.cos(phi + (1 / 2) * theta)


def M_gv_21_1_0():
    def func(k):
        return -1 / 6 * k

    return func


def N_gv_21_1_0(theta, phi):
    return -3 * np.cos(phi + (1 / 2) * theta)


def M_gv_22_1_0():
    def func(k):
        return -1 / 3 * k

    return func


def N_gv_22_1_0(theta, phi):
    return -3 * np.cos(phi + (1 / 2) * theta)


def M_gv_23_1_0():
    def func(k):
        return (1 / 3) / k

    return func


def N_gv_23_1_0(theta, phi):
    return -3 * np.cos(phi + (1 / 2) * theta)


def M_gv_24_1_0():
    def func(k):
        return (1 / 3) / k

    return func


def N_gv_24_1_0(theta, phi):
    return -3 * np.cos(phi + (1 / 2) * theta)


def M_gv_25_1_0():
    def func(k):
        return (1 / 5) / k

    return func


def N_gv_25_1_0(theta, phi):
    return -3 * np.cos(phi + (1 / 2) * theta)


def M_gv_25_3_0():
    def func(k):
        return (2 / 35) / k

    return func


def N_gv_25_3_0(theta, phi):
    return -21 / 8 * np.cos(phi + (1 / 2) * theta) - 35 / 8 * np.cos(
        3 * phi + (3 / 2) * theta
    )


def M_gv_26_1_0():
    def func(k):
        return (1 / 5) / k

    return func


def N_gv_26_1_0(theta, phi):
    return -3 * np.cos(phi + (1 / 2) * theta)


def M_gv_26_3_0():
    def func(k):
        return (2 / 35) / k

    return func


def N_gv_26_3_0(theta, phi):
    return -21 / 8 * np.cos(phi + (1 / 2) * theta) - 35 / 8 * np.cos(
        3 * phi + (3 / 2) * theta
    )


def M_gv_27_1_0():
    def func(k):
        return -3 / 10 * k

    return func


def N_gv_27_1_0(theta, phi):
    return -3 * np.cos(phi + (1 / 2) * theta)


def M_gv_27_3_0():
    def func(k):
        return -3 / 35 * k

    return func


def N_gv_27_3_0(theta, phi):
    return -21 / 8 * np.cos(phi + (1 / 2) * theta) - 35 / 8 * np.cos(
        3 * phi + (3 / 2) * theta
    )


def M_gv_28_1_0():
    def func(k):
        return -3 / 5 * k

    return func


def N_gv_28_1_0(theta, phi):
    return -3 * np.cos(phi + (1 / 2) * theta)


def M_gv_28_3_0():
    def func(k):
        return -6 / 35 * k

    return func


def N_gv_28_3_0(theta, phi):
    return -21 / 8 * np.cos(phi + (1 / 2) * theta) - 35 / 8 * np.cos(
        3 * phi + (3 / 2) * theta
    )


def M_gv_29_1_0():
    def func(k):
        return -3 / 5 * k

    return func


def N_gv_29_1_0(theta, phi):
    return -3 * np.cos(phi + (1 / 2) * theta)


def M_gv_29_3_0():
    def func(k):
        return -6 / 35 * k

    return func


def N_gv_29_3_0(theta, phi):
    return -21 / 8 * np.cos(phi + (1 / 2) * theta) - 35 / 8 * np.cos(
        3 * phi + (3 / 2) * theta
    )


def M_gv_30_1_0():
    def func(k):
        return -9 / 5 * k**3

    return func


def N_gv_30_1_0(theta, phi):
    return -3 * np.cos(phi + (1 / 2) * theta)


def M_gv_30_3_0():
    def func(k):
        return -18 / 35 * k**3

    return func


def N_gv_30_3_0(theta, phi):
    return -21 / 8 * np.cos(phi + (1 / 2) * theta) - 35 / 8 * np.cos(
        3 * phi + (3 / 2) * theta
    )


def M_gv_31_1_0():
    def func(k):
        return -9 / 5 * k**3

    return func


def N_gv_31_1_0(theta, phi):
    return -3 * np.cos(phi + (1 / 2) * theta)


def M_gv_31_3_0():
    def func(k):
        return -18 / 35 * k**3

    return func


def N_gv_31_3_0(theta, phi):
    return -21 / 8 * np.cos(phi + (1 / 2) * theta) - 35 / 8 * np.cos(
        3 * phi + (3 / 2) * theta
    )


def M_gv_32_1_0():
    def func(k):
        return (3 / 10) * k

    return func


def N_gv_32_1_0(theta, phi):
    return -3 * np.cos(phi + (1 / 2) * theta)


def M_gv_32_3_0():
    def func(k):
        return (3 / 35) * k

    return func


def N_gv_32_3_0(theta, phi):
    return -21 / 8 * np.cos(phi + (1 / 2) * theta) - 35 / 8 * np.cos(
        3 * phi + (3 / 2) * theta
    )


def M_gv_33_1_0():
    def func(k):
        return (3 / 10) * k

    return func


def N_gv_33_1_0(theta, phi):
    return -3 * np.cos(phi + (1 / 2) * theta)


def M_gv_33_3_0():
    def func(k):
        return (3 / 35) * k

    return func


def N_gv_33_3_0(theta, phi):
    return -21 / 8 * np.cos(phi + (1 / 2) * theta) - 35 / 8 * np.cos(
        3 * phi + (3 / 2) * theta
    )


def M_gv_34_1_0():
    def func(k):
        return (3 / 10) * k

    return func


def N_gv_34_1_0(theta, phi):
    return -3 * np.cos(phi + (1 / 2) * theta)


def M_gv_34_3_0():
    def func(k):
        return (3 / 35) * k

    return func


def N_gv_34_3_0(theta, phi):
    return -21 / 8 * np.cos(phi + (1 / 2) * theta) - 35 / 8 * np.cos(
        3 * phi + (3 / 2) * theta
    )


def M_gv_35_1_0():
    def func(k):
        return (3 / 10) * k

    return func


def N_gv_35_1_0(theta, phi):
    return -3 * np.cos(phi + (1 / 2) * theta)


def M_gv_35_3_0():
    def func(k):
        return (3 / 35) * k

    return func


def N_gv_35_3_0(theta, phi):
    return -21 / 8 * np.cos(phi + (1 / 2) * theta) - 35 / 8 * np.cos(
        3 * phi + (3 / 2) * theta
    )


def M_gv_36_1_0():
    def func(k):
        return (3 / 10) * k

    return func


def N_gv_36_1_0(theta, phi):
    return -3 * np.cos(phi + (1 / 2) * theta)


def M_gv_36_3_0():
    def func(k):
        return (3 / 35) * k

    return func


def N_gv_36_3_0(theta, phi):
    return -21 / 8 * np.cos(phi + (1 / 2) * theta) - 35 / 8 * np.cos(
        3 * phi + (3 / 2) * theta
    )


def M_gv_37_1_0():
    def func(k):
        return -1 / 5 * k

    return func


def N_gv_37_1_0(theta, phi):
    return -3 * np.cos(phi + (1 / 2) * theta)


def M_gv_37_3_0():
    def func(k):
        return -2 / 35 * k

    return func


def N_gv_37_3_0(theta, phi):
    return -21 / 8 * np.cos(phi + (1 / 2) * theta) - 35 / 8 * np.cos(
        3 * phi + (3 / 2) * theta
    )


def M_gv_38_1_0():
    def func(k):
        return -1 / 7 * k

    return func


def N_gv_38_1_0(theta, phi):
    return -3 * np.cos(phi + (1 / 2) * theta)


def M_gv_38_3_0():
    def func(k):
        return -4 / 63 * k

    return func


def N_gv_38_3_0(theta, phi):
    return -21 / 8 * np.cos(phi + (1 / 2) * theta) - 35 / 8 * np.cos(
        3 * phi + (3 / 2) * theta
    )


def M_gv_38_5_0():
    def func(k):
        return -8 / 693 * k

    return func


def N_gv_38_5_0(theta, phi):
    return (
        -165 / 64 * np.cos(phi + (1 / 2) * theta)
        - 385 / 128 * np.cos(3 * phi + (3 / 2) * theta)
        - 693 / 128 * np.cos(5 * phi + (5 / 2) * theta)
    )


def M_gv_39_1_0():
    def func(k):
        return -2 / 5 * k**3

    return func


def N_gv_39_1_0(theta, phi):
    return -3 * np.cos(phi + (1 / 2) * theta)


def M_gv_39_3_0():
    def func(k):
        return -4 / 35 * k**3

    return func


def N_gv_39_3_0(theta, phi):
    return -21 / 8 * np.cos(phi + (1 / 2) * theta) - 35 / 8 * np.cos(
        3 * phi + (3 / 2) * theta
    )


def M_gv_40_1_0():
    def func(k):
        return -2 / 7 * k**3

    return func


def N_gv_40_1_0(theta, phi):
    return -3 * np.cos(phi + (1 / 2) * theta)


def M_gv_40_3_0():
    def func(k):
        return -8 / 63 * k**3

    return func


def N_gv_40_3_0(theta, phi):
    return -21 / 8 * np.cos(phi + (1 / 2) * theta) - 35 / 8 * np.cos(
        3 * phi + (3 / 2) * theta
    )


def M_gv_40_5_0():
    def func(k):
        return -16 / 693 * k**3

    return func


def N_gv_40_5_0(theta, phi):
    return (
        -165 / 64 * np.cos(phi + (1 / 2) * theta)
        - 385 / 128 * np.cos(3 * phi + (3 / 2) * theta)
        - 693 / 128 * np.cos(5 * phi + (5 / 2) * theta)
    )


def M_gv_41_1_0():
    def func(k):
        return (1 / 10) * k**3

    return func


def N_gv_41_1_0(theta, phi):
    return -3 * np.cos(phi + (1 / 2) * theta)


def M_gv_41_3_0():
    def func(k):
        return (1 / 35) * k**3

    return func


def N_gv_41_3_0(theta, phi):
    return -21 / 8 * np.cos(phi + (1 / 2) * theta) - 35 / 8 * np.cos(
        3 * phi + (3 / 2) * theta
    )


def M_gv_42_1_0():
    def func(k):
        return (1 / 10) * k**3

    return func


def N_gv_42_1_0(theta, phi):
    return -3 * np.cos(phi + (1 / 2) * theta)


def M_gv_42_3_0():
    def func(k):
        return (1 / 35) * k**3

    return func


def N_gv_42_3_0(theta, phi):
    return -21 / 8 * np.cos(phi + (1 / 2) * theta) - 35 / 8 * np.cos(
        3 * phi + (3 / 2) * theta
    )


def M_gv_43_1_0():
    def func(k):
        return (1 / 10) * k**3

    return func


def N_gv_43_1_0(theta, phi):
    return -3 * np.cos(phi + (1 / 2) * theta)


def M_gv_43_3_0():
    def func(k):
        return (1 / 35) * k**3

    return func


def N_gv_43_3_0(theta, phi):
    return -21 / 8 * np.cos(phi + (1 / 2) * theta) - 35 / 8 * np.cos(
        3 * phi + (3 / 2) * theta
    )


def M_gv_44_1_0():
    def func(k):
        return (1 / 5) * k**3

    return func


def N_gv_44_1_0(theta, phi):
    return -3 * np.cos(phi + (1 / 2) * theta)


def M_gv_44_3_0():
    def func(k):
        return (2 / 35) * k**3

    return func


def N_gv_44_3_0(theta, phi):
    return -21 / 8 * np.cos(phi + (1 / 2) * theta) - 35 / 8 * np.cos(
        3 * phi + (3 / 2) * theta
    )


def M_gv_45_1_0():
    def func(k):
        return (3 / 5) * k**5

    return func


def N_gv_45_1_0(theta, phi):
    return -3 * np.cos(phi + (1 / 2) * theta)


def M_gv_45_3_0():
    def func(k):
        return (6 / 35) * k**5

    return func


def N_gv_45_3_0(theta, phi):
    return -21 / 8 * np.cos(phi + (1 / 2) * theta) - 35 / 8 * np.cos(
        3 * phi + (3 / 2) * theta
    )


def M_gv_46_1_0():
    def func(k):
        return (1 / 20) * k**3

    return func


def N_gv_46_1_0(theta, phi):
    return -3 * np.cos(phi + (1 / 2) * theta)


def M_gv_46_3_0():
    def func(k):
        return (1 / 70) * k**3

    return func


def N_gv_46_3_0(theta, phi):
    return -21 / 8 * np.cos(phi + (1 / 2) * theta) - 35 / 8 * np.cos(
        3 * phi + (3 / 2) * theta
    )


def M_gv_47_1_0():
    def func(k):
        return (1 / 20) * k**3

    return func


def N_gv_47_1_0(theta, phi):
    return -3 * np.cos(phi + (1 / 2) * theta)


def M_gv_47_3_0():
    def func(k):
        return (1 / 70) * k**3

    return func


def N_gv_47_3_0(theta, phi):
    return -21 / 8 * np.cos(phi + (1 / 2) * theta) - 35 / 8 * np.cos(
        3 * phi + (3 / 2) * theta
    )


def M_gv_48_1_0():
    def func(k):
        return (1 / 20) * k**3

    return func


def N_gv_48_1_0(theta, phi):
    return -3 * np.cos(phi + (1 / 2) * theta)


def M_gv_48_3_0():
    def func(k):
        return (1 / 70) * k**3

    return func


def N_gv_48_3_0(theta, phi):
    return -21 / 8 * np.cos(phi + (1 / 2) * theta) - 35 / 8 * np.cos(
        3 * phi + (3 / 2) * theta
    )


def M_gv_49_1_0():
    def func(k):
        return (1 / 10) * k**3

    return func


def N_gv_49_1_0(theta, phi):
    return -3 * np.cos(phi + (1 / 2) * theta)


def M_gv_49_3_0():
    def func(k):
        return (1 / 35) * k**3

    return func


def N_gv_49_3_0(theta, phi):
    return -21 / 8 * np.cos(phi + (1 / 2) * theta) - 35 / 8 * np.cos(
        3 * phi + (3 / 2) * theta
    )


def M_gv_50_1_0():
    def func(k):
        return (1 / 10) * k**3

    return func


def N_gv_50_1_0(theta, phi):
    return -3 * np.cos(phi + (1 / 2) * theta)


def M_gv_50_3_0():
    def func(k):
        return (1 / 35) * k**3

    return func


def N_gv_50_3_0(theta, phi):
    return -21 / 8 * np.cos(phi + (1 / 2) * theta) - 35 / 8 * np.cos(
        3 * phi + (3 / 2) * theta
    )


def M_gv_51_1_0():
    def func(k):
        return (1 / 10) * k**3

    return func


def N_gv_51_1_0(theta, phi):
    return -3 * np.cos(phi + (1 / 2) * theta)


def M_gv_51_3_0():
    def func(k):
        return (1 / 35) * k**3

    return func


def N_gv_51_3_0(theta, phi):
    return -21 / 8 * np.cos(phi + (1 / 2) * theta) - 35 / 8 * np.cos(
        3 * phi + (3 / 2) * theta
    )


def M_gv_52_1_0():
    def func(k):
        return (1 / 10) * k**3

    return func


def N_gv_52_1_0(theta, phi):
    return -3 * np.cos(phi + (1 / 2) * theta)


def M_gv_52_3_0():
    def func(k):
        return (1 / 35) * k**3

    return func


def N_gv_52_3_0(theta, phi):
    return -21 / 8 * np.cos(phi + (1 / 2) * theta) - 35 / 8 * np.cos(
        3 * phi + (3 / 2) * theta
    )


def M_gv_53_1_0():
    def func(k):
        return (1 / 5) * k**3

    return func


def N_gv_53_1_0(theta, phi):
    return -3 * np.cos(phi + (1 / 2) * theta)


def M_gv_53_3_0():
    def func(k):
        return (2 / 35) * k**3

    return func


def N_gv_53_3_0(theta, phi):
    return -21 / 8 * np.cos(phi + (1 / 2) * theta) - 35 / 8 * np.cos(
        3 * phi + (3 / 2) * theta
    )


def M_gv_54_1_0():
    def func(k):
        return (3 / 5) * k**5

    return func


def N_gv_54_1_0(theta, phi):
    return -3 * np.cos(phi + (1 / 2) * theta)


def M_gv_54_3_0():
    def func(k):
        return (6 / 35) * k**5

    return func


def N_gv_54_3_0(theta, phi):
    return -21 / 8 * np.cos(phi + (1 / 2) * theta) - 35 / 8 * np.cos(
        3 * phi + (3 / 2) * theta
    )


def M_gv_55_1_0():
    def func(k):
        return (1 / 20) * k**3

    return func


def N_gv_55_1_0(theta, phi):
    return -3 * np.cos(phi + (1 / 2) * theta)


def M_gv_55_3_0():
    def func(k):
        return (1 / 70) * k**3

    return func


def N_gv_55_3_0(theta, phi):
    return -21 / 8 * np.cos(phi + (1 / 2) * theta) - 35 / 8 * np.cos(
        3 * phi + (3 / 2) * theta
    )


def M_gv_56_1_0():
    def func(k):
        return (1 / 20) * k**3

    return func


def N_gv_56_1_0(theta, phi):
    return -3 * np.cos(phi + (1 / 2) * theta)


def M_gv_56_3_0():
    def func(k):
        return (1 / 70) * k**3

    return func


def N_gv_56_3_0(theta, phi):
    return -21 / 8 * np.cos(phi + (1 / 2) * theta) - 35 / 8 * np.cos(
        3 * phi + (3 / 2) * theta
    )


def M_gv_57_1_0():
    def func(k):
        return (1 / 20) * k**3

    return func


def N_gv_57_1_0(theta, phi):
    return -3 * np.cos(phi + (1 / 2) * theta)


def M_gv_57_3_0():
    def func(k):
        return (1 / 70) * k**3

    return func


def N_gv_57_3_0(theta, phi):
    return -21 / 8 * np.cos(phi + (1 / 2) * theta) - 35 / 8 * np.cos(
        3 * phi + (3 / 2) * theta
    )


def M_gv_58_1_0():
    def func(k):
        return (1 / 10) * k**3

    return func


def N_gv_58_1_0(theta, phi):
    return -3 * np.cos(phi + (1 / 2) * theta)


def M_gv_58_3_0():
    def func(k):
        return (1 / 35) * k**3

    return func


def N_gv_58_3_0(theta, phi):
    return -21 / 8 * np.cos(phi + (1 / 2) * theta) - 35 / 8 * np.cos(
        3 * phi + (3 / 2) * theta
    )


def M_gv_59_1_0():
    def func(k):
        return (1 / 9) / k

    return func


def N_gv_59_1_0(theta, phi):
    return -3 * np.cos(phi + (1 / 2) * theta)


def M_gv_59_1_1():
    def func(k):
        return (2 / 45) / k

    return func


def N_gv_59_1_1(theta, phi):
    return -9 / 2 * np.cos(phi - 3 / 2 * theta) - 3 / 2 * np.cos(phi + (1 / 2) * theta)


def M_gv_59_3_0():
    def func(k):
        return (2 / 45) / k

    return func


def N_gv_59_3_0(theta, phi):
    return (
        -9 / 8 * np.cos(phi - 3 / 2 * theta)
        - 9 / 4 * np.cos(phi + (1 / 2) * theta)
        - 45 / 8 * np.cos(3 * phi - 1 / 2 * theta)
    )


def M_gv_60_1_0():
    def func(k):
        return (2 / 9) / k

    return func


def N_gv_60_1_0(theta, phi):
    return -3 * np.cos(phi + (1 / 2) * theta)


def M_gv_60_1_1():
    def func(k):
        return (4 / 45) / k

    return func


def N_gv_60_1_1(theta, phi):
    return -9 / 2 * np.cos(phi - 3 / 2 * theta) - 3 / 2 * np.cos(phi + (1 / 2) * theta)


def M_gv_60_3_0():
    def func(k):
        return (4 / 45) / k

    return func


def N_gv_60_3_0(theta, phi):
    return (
        -9 / 8 * np.cos(phi - 3 / 2 * theta)
        - 9 / 4 * np.cos(phi + (1 / 2) * theta)
        - 45 / 8 * np.cos(3 * phi - 1 / 2 * theta)
    )


def M_gv_61_1_0():
    def func(k):
        return (2 / 9) / k

    return func


def N_gv_61_1_0(theta, phi):
    return -3 * np.cos(phi + (1 / 2) * theta)


def M_gv_61_1_1():
    def func(k):
        return (4 / 45) / k

    return func


def N_gv_61_1_1(theta, phi):
    return -9 / 2 * np.cos(phi - 3 / 2 * theta) - 3 / 2 * np.cos(phi + (1 / 2) * theta)


def M_gv_61_3_0():
    def func(k):
        return (4 / 45) / k

    return func


def N_gv_61_3_0(theta, phi):
    return (
        -9 / 8 * np.cos(phi - 3 / 2 * theta)
        - 9 / 4 * np.cos(phi + (1 / 2) * theta)
        - 45 / 8 * np.cos(3 * phi - 1 / 2 * theta)
    )


def M_gv_62_1_0():
    def func(k):
        return (1 / 3) / k

    return func


def N_gv_62_1_0(theta, phi):
    return -3 / 2 * np.cos(phi - 3 / 2 * theta) - 3 / 2 * np.cos(phi + (1 / 2) * theta)


def M_gv_63_1_0():
    def func(k):
        return (1 / 9) / k

    return func


def N_gv_63_1_0(theta, phi):
    return -3 * np.cos(phi + (1 / 2) * theta)


def M_gv_63_1_1():
    def func(k):
        return (2 / 45) / k

    return func


def N_gv_63_1_1(theta, phi):
    return -9 / 2 * np.cos(phi - 3 / 2 * theta) - 3 / 2 * np.cos(phi + (1 / 2) * theta)


def M_gv_63_3_0():
    def func(k):
        return (2 / 45) / k

    return func


def N_gv_63_3_0(theta, phi):
    return (
        -9 / 8 * np.cos(phi - 3 / 2 * theta)
        - 9 / 4 * np.cos(phi + (1 / 2) * theta)
        - 45 / 8 * np.cos(3 * phi - 1 / 2 * theta)
    )


def M_gv_64_1_0():
    def func(k):
        return (2 / 3) * k

    return func


def N_gv_64_1_0(theta, phi):
    return -3 * np.cos(phi + (1 / 2) * theta)


def M_gv_64_1_1():
    def func(k):
        return (4 / 15) * k

    return func


def N_gv_64_1_1(theta, phi):
    return -9 / 2 * np.cos(phi - 3 / 2 * theta) - 3 / 2 * np.cos(phi + (1 / 2) * theta)


def M_gv_64_3_0():
    def func(k):
        return (4 / 15) * k

    return func


def N_gv_64_3_0(theta, phi):
    return (
        -9 / 8 * np.cos(phi - 3 / 2 * theta)
        - 9 / 4 * np.cos(phi + (1 / 2) * theta)
        - 45 / 8 * np.cos(3 * phi - 1 / 2 * theta)
    )


def M_gv_65_1_0():
    def func(k):
        return (2 / 3) * k

    return func


def N_gv_65_1_0(theta, phi):
    return -3 * np.cos(phi + (1 / 2) * theta)


def M_gv_65_1_1():
    def func(k):
        return (4 / 15) * k

    return func


def N_gv_65_1_1(theta, phi):
    return -9 / 2 * np.cos(phi - 3 / 2 * theta) - 3 / 2 * np.cos(phi + (1 / 2) * theta)


def M_gv_65_3_0():
    def func(k):
        return (4 / 15) * k

    return func


def N_gv_65_3_0(theta, phi):
    return (
        -9 / 8 * np.cos(phi - 3 / 2 * theta)
        - 9 / 4 * np.cos(phi + (1 / 2) * theta)
        - 45 / 8 * np.cos(3 * phi - 1 / 2 * theta)
    )


def M_gv_66_1_0():
    def func(k):
        return (1 / 6) / k

    return func


def N_gv_66_1_0(theta, phi):
    return -3 * np.cos(phi + (1 / 2) * theta)


def M_gv_66_1_1():
    def func(k):
        return (1 / 15) / k

    return func


def N_gv_66_1_1(theta, phi):
    return -9 / 2 * np.cos(phi - 3 / 2 * theta) - 3 / 2 * np.cos(phi + (1 / 2) * theta)


def M_gv_66_3_0():
    def func(k):
        return (1 / 15) / k

    return func


def N_gv_66_3_0(theta, phi):
    return (
        -9 / 8 * np.cos(phi - 3 / 2 * theta)
        - 9 / 4 * np.cos(phi + (1 / 2) * theta)
        - 45 / 8 * np.cos(3 * phi - 1 / 2 * theta)
    )


def M_gv_67_1_0():
    def func(k):
        return (1 / 10) / k

    return func


def N_gv_67_1_0(theta, phi):
    return -3 * np.cos(phi + (1 / 2) * theta)


def M_gv_67_1_1():
    def func(k):
        return (8 / 175) / k

    return func


def N_gv_67_1_1(theta, phi):
    return -9 / 2 * np.cos(phi - 3 / 2 * theta) - 3 / 2 * np.cos(phi + (1 / 2) * theta)


def M_gv_67_1_2():
    def func(k):
        return (4 / 525) / k

    return func


def N_gv_67_1_2(theta, phi):
    return (
        -9 / 8 * np.cos(phi - 3 / 2 * theta)
        - 9 / 4 * np.cos(phi + (1 / 2) * theta)
        - 45 / 8 * np.cos(phi + (5 / 2) * theta)
    )


def M_gv_67_3_0():
    def func(k):
        return (2 / 105) / k

    return func


def N_gv_67_3_0(theta, phi):
    return -21 / 8 * np.cos(phi + (1 / 2) * theta) - 35 / 8 * np.cos(
        3 * phi + (3 / 2) * theta
    )


def M_gv_67_3_1():
    def func(k):
        return (8 / 175) / k

    return func


def N_gv_67_3_1(theta, phi):
    return (
        -9 / 8 * np.cos(phi - 3 / 2 * theta)
        - 9 / 4 * np.cos(phi + (1 / 2) * theta)
        - 45 / 8 * np.cos(3 * phi - 1 / 2 * theta)
    )


def M_gv_67_3_2():
    def func(k):
        return (4 / 525) / k

    return func


def N_gv_67_3_2(theta, phi):
    return (
        -21 / 8 * np.cos(phi - 3 / 2 * theta)
        + (21 / 16) * np.cos(phi + (1 / 2) * theta)
        - 35 / 16 * np.cos(phi + (5 / 2) * theta)
        - 35 / 16 * np.cos(3 * phi - 1 / 2 * theta)
        - 175 / 48 * np.cos(3 * phi + (3 / 2) * theta)
    )


def M_gv_67_3_3():
    def func(k):
        return (4 / 945) / k

    return func


def N_gv_67_3_3(theta, phi):
    return (
        -45 / 16 * np.cos(phi - 3 / 2 * theta)
        - 27 / 16 * np.cos(phi + (1 / 2) * theta)
        - 105 / 16 * np.cos(3 * phi - 5 / 2 * theta)
        - 15 / 16 * np.cos(3 * phi - 1 / 2 * theta)
    )


def M_gv_67_5_0():
    def func(k):
        return (4 / 525) / k

    return func


def N_gv_67_5_0(theta, phi):
    return (
        -75 / 64 * np.cos(phi - 3 / 2 * theta)
        - 75 / 32 * np.cos(phi + (1 / 2) * theta)
        - 25 / 64 * np.cos(phi + (5 / 2) * theta)
        - 175 / 64 * np.cos(3 * phi - 1 / 2 * theta)
        - 175 / 96 * np.cos(3 * phi + (3 / 2) * theta)
        - 525 / 64 * np.cos(5 * phi + (1 / 2) * theta)
    )


def M_gv_67_5_1():
    def func(k):
        return (4 / 945) / k

    return func


def N_gv_67_5_1(theta, phi):
    return (
        -45 / 32 * np.cos(phi - 3 / 2 * theta)
        - 135 / 64 * np.cos(phi + (1 / 2) * theta)
        - 105 / 128 * np.cos(3 * phi - 5 / 2 * theta)
        - 105 / 32 * np.cos(3 * phi - 1 / 2 * theta)
        - 945 / 128 * np.cos(5 * phi - 3 / 2 * theta)
    )


def M_gv_68_1_0():
    def func(k):
        return -(1 / 6) / k

    return func


def N_gv_68_1_0(theta, phi):
    return -3 * np.cos(phi + (1 / 2) * theta)


def M_gv_68_1_1():
    def func(k):
        return -(1 / 15) / k

    return func


def N_gv_68_1_1(theta, phi):
    return -9 / 2 * np.cos(phi - 3 / 2 * theta) - 3 / 2 * np.cos(phi + (1 / 2) * theta)


def M_gv_68_3_0():
    def func(k):
        return -(1 / 15) / k

    return func


def N_gv_68_3_0(theta, phi):
    return (
        -9 / 8 * np.cos(phi - 3 / 2 * theta)
        - 9 / 4 * np.cos(phi + (1 / 2) * theta)
        - 45 / 8 * np.cos(3 * phi - 1 / 2 * theta)
    )


def M_gv_69_1_0():
    def func(k):
        return -(1 / 10) / k

    return func


def N_gv_69_1_0(theta, phi):
    return -3 * np.cos(phi + (1 / 2) * theta)


def M_gv_69_1_1():
    def func(k):
        return -(8 / 175) / k

    return func


def N_gv_69_1_1(theta, phi):
    return -9 / 2 * np.cos(phi - 3 / 2 * theta) - 3 / 2 * np.cos(phi + (1 / 2) * theta)


def M_gv_69_1_2():
    def func(k):
        return -(4 / 525) / k

    return func


def N_gv_69_1_2(theta, phi):
    return (
        -9 / 8 * np.cos(phi - 3 / 2 * theta)
        - 9 / 4 * np.cos(phi + (1 / 2) * theta)
        - 45 / 8 * np.cos(phi + (5 / 2) * theta)
    )


def M_gv_69_3_0():
    def func(k):
        return -(2 / 105) / k

    return func


def N_gv_69_3_0(theta, phi):
    return -21 / 8 * np.cos(phi + (1 / 2) * theta) - 35 / 8 * np.cos(
        3 * phi + (3 / 2) * theta
    )


def M_gv_69_3_1():
    def func(k):
        return -(8 / 175) / k

    return func


def N_gv_69_3_1(theta, phi):
    return (
        -9 / 8 * np.cos(phi - 3 / 2 * theta)
        - 9 / 4 * np.cos(phi + (1 / 2) * theta)
        - 45 / 8 * np.cos(3 * phi - 1 / 2 * theta)
    )


def M_gv_69_3_2():
    def func(k):
        return -(4 / 525) / k

    return func


def N_gv_69_3_2(theta, phi):
    return (
        -21 / 8 * np.cos(phi - 3 / 2 * theta)
        + (21 / 16) * np.cos(phi + (1 / 2) * theta)
        - 35 / 16 * np.cos(phi + (5 / 2) * theta)
        - 35 / 16 * np.cos(3 * phi - 1 / 2 * theta)
        - 175 / 48 * np.cos(3 * phi + (3 / 2) * theta)
    )


def M_gv_69_3_3():
    def func(k):
        return -(4 / 945) / k

    return func


def N_gv_69_3_3(theta, phi):
    return (
        -45 / 16 * np.cos(phi - 3 / 2 * theta)
        - 27 / 16 * np.cos(phi + (1 / 2) * theta)
        - 105 / 16 * np.cos(3 * phi - 5 / 2 * theta)
        - 15 / 16 * np.cos(3 * phi - 1 / 2 * theta)
    )


def M_gv_69_5_0():
    def func(k):
        return -(4 / 525) / k

    return func


def N_gv_69_5_0(theta, phi):
    return (
        -75 / 64 * np.cos(phi - 3 / 2 * theta)
        - 75 / 32 * np.cos(phi + (1 / 2) * theta)
        - 25 / 64 * np.cos(phi + (5 / 2) * theta)
        - 175 / 64 * np.cos(3 * phi - 1 / 2 * theta)
        - 175 / 96 * np.cos(3 * phi + (3 / 2) * theta)
        - 525 / 64 * np.cos(5 * phi + (1 / 2) * theta)
    )


def M_gv_69_5_1():
    def func(k):
        return -(4 / 945) / k

    return func


def N_gv_69_5_1(theta, phi):
    return (
        -45 / 32 * np.cos(phi - 3 / 2 * theta)
        - 135 / 64 * np.cos(phi + (1 / 2) * theta)
        - 105 / 128 * np.cos(3 * phi - 5 / 2 * theta)
        - 105 / 32 * np.cos(3 * phi - 1 / 2 * theta)
        - 945 / 128 * np.cos(5 * phi - 3 / 2 * theta)
    )


def M_gv_70_1_0():
    def func(k):
        return (1 / 3) * k

    return func


def N_gv_70_1_0(theta, phi):
    return -3 * np.cos(phi + (1 / 2) * theta)


def M_gv_70_1_1():
    def func(k):
        return (2 / 15) * k

    return func


def N_gv_70_1_1(theta, phi):
    return -9 / 2 * np.cos(phi - 3 / 2 * theta) - 3 / 2 * np.cos(phi + (1 / 2) * theta)


def M_gv_70_3_0():
    def func(k):
        return (2 / 15) * k

    return func


def N_gv_70_3_0(theta, phi):
    return (
        -9 / 8 * np.cos(phi - 3 / 2 * theta)
        - 9 / 4 * np.cos(phi + (1 / 2) * theta)
        - 45 / 8 * np.cos(3 * phi - 1 / 2 * theta)
    )


def M_gv_71_1_0():
    def func(k):
        return (1 / 5) * k

    return func


def N_gv_71_1_0(theta, phi):
    return -3 * np.cos(phi + (1 / 2) * theta)


def M_gv_71_1_1():
    def func(k):
        return (16 / 175) * k

    return func


def N_gv_71_1_1(theta, phi):
    return -9 / 2 * np.cos(phi - 3 / 2 * theta) - 3 / 2 * np.cos(phi + (1 / 2) * theta)


def M_gv_71_1_2():
    def func(k):
        return (8 / 525) * k

    return func


def N_gv_71_1_2(theta, phi):
    return (
        -9 / 8 * np.cos(phi - 3 / 2 * theta)
        - 9 / 4 * np.cos(phi + (1 / 2) * theta)
        - 45 / 8 * np.cos(phi + (5 / 2) * theta)
    )


def M_gv_71_3_0():
    def func(k):
        return (4 / 105) * k

    return func


def N_gv_71_3_0(theta, phi):
    return -21 / 8 * np.cos(phi + (1 / 2) * theta) - 35 / 8 * np.cos(
        3 * phi + (3 / 2) * theta
    )


def M_gv_71_3_1():
    def func(k):
        return (16 / 175) * k

    return func


def N_gv_71_3_1(theta, phi):
    return (
        -9 / 8 * np.cos(phi - 3 / 2 * theta)
        - 9 / 4 * np.cos(phi + (1 / 2) * theta)
        - 45 / 8 * np.cos(3 * phi - 1 / 2 * theta)
    )


def M_gv_71_3_2():
    def func(k):
        return (8 / 525) * k

    return func


def N_gv_71_3_2(theta, phi):
    return (
        -21 / 8 * np.cos(phi - 3 / 2 * theta)
        + (21 / 16) * np.cos(phi + (1 / 2) * theta)
        - 35 / 16 * np.cos(phi + (5 / 2) * theta)
        - 35 / 16 * np.cos(3 * phi - 1 / 2 * theta)
        - 175 / 48 * np.cos(3 * phi + (3 / 2) * theta)
    )


def M_gv_71_3_3():
    def func(k):
        return (8 / 945) * k

    return func


def N_gv_71_3_3(theta, phi):
    return (
        -45 / 16 * np.cos(phi - 3 / 2 * theta)
        - 27 / 16 * np.cos(phi + (1 / 2) * theta)
        - 105 / 16 * np.cos(3 * phi - 5 / 2 * theta)
        - 15 / 16 * np.cos(3 * phi - 1 / 2 * theta)
    )


def M_gv_71_5_0():
    def func(k):
        return (8 / 525) * k

    return func


def N_gv_71_5_0(theta, phi):
    return (
        -75 / 64 * np.cos(phi - 3 / 2 * theta)
        - 75 / 32 * np.cos(phi + (1 / 2) * theta)
        - 25 / 64 * np.cos(phi + (5 / 2) * theta)
        - 175 / 64 * np.cos(3 * phi - 1 / 2 * theta)
        - 175 / 96 * np.cos(3 * phi + (3 / 2) * theta)
        - 525 / 64 * np.cos(5 * phi + (1 / 2) * theta)
    )


def M_gv_71_5_1():
    def func(k):
        return (8 / 945) * k

    return func


def N_gv_71_5_1(theta, phi):
    return (
        -45 / 32 * np.cos(phi - 3 / 2 * theta)
        - 135 / 64 * np.cos(phi + (1 / 2) * theta)
        - 105 / 128 * np.cos(3 * phi - 5 / 2 * theta)
        - 105 / 32 * np.cos(3 * phi - 1 / 2 * theta)
        - 945 / 128 * np.cos(5 * phi - 3 / 2 * theta)
    )


def M_gv_72_1_0():
    def func(k):
        return -1 / 6 * k

    return func


def N_gv_72_1_0(theta, phi):
    return -3 * np.cos(phi + (1 / 2) * theta)


def M_gv_72_1_1():
    def func(k):
        return -1 / 15 * k

    return func


def N_gv_72_1_1(theta, phi):
    return -9 / 2 * np.cos(phi - 3 / 2 * theta) - 3 / 2 * np.cos(phi + (1 / 2) * theta)


def M_gv_72_3_0():
    def func(k):
        return -1 / 15 * k

    return func


def N_gv_72_3_0(theta, phi):
    return (
        -9 / 8 * np.cos(phi - 3 / 2 * theta)
        - 9 / 4 * np.cos(phi + (1 / 2) * theta)
        - 45 / 8 * np.cos(3 * phi - 1 / 2 * theta)
    )


def M_gv_73_1_0():
    def func(k):
        return -1 / 3 * k

    return func


def N_gv_73_1_0(theta, phi):
    return -3 * np.cos(phi + (1 / 2) * theta)


def M_gv_73_1_1():
    def func(k):
        return -2 / 15 * k

    return func


def N_gv_73_1_1(theta, phi):
    return -9 / 2 * np.cos(phi - 3 / 2 * theta) - 3 / 2 * np.cos(phi + (1 / 2) * theta)


def M_gv_73_3_0():
    def func(k):
        return -2 / 15 * k

    return func


def N_gv_73_3_0(theta, phi):
    return (
        -9 / 8 * np.cos(phi - 3 / 2 * theta)
        - 9 / 4 * np.cos(phi + (1 / 2) * theta)
        - 45 / 8 * np.cos(3 * phi - 1 / 2 * theta)
    )


def M_gv_74_1_0():
    def func(k):
        return -1 / 3 * k

    return func


def N_gv_74_1_0(theta, phi):
    return -3 * np.cos(phi + (1 / 2) * theta)


def M_gv_74_1_1():
    def func(k):
        return -2 / 15 * k

    return func


def N_gv_74_1_1(theta, phi):
    return -9 / 2 * np.cos(phi - 3 / 2 * theta) - 3 / 2 * np.cos(phi + (1 / 2) * theta)


def M_gv_74_3_0():
    def func(k):
        return -2 / 15 * k

    return func


def N_gv_74_3_0(theta, phi):
    return (
        -9 / 8 * np.cos(phi - 3 / 2 * theta)
        - 9 / 4 * np.cos(phi + (1 / 2) * theta)
        - 45 / 8 * np.cos(3 * phi - 1 / 2 * theta)
    )


def M_gv_75_1_0():
    def func(k):
        return -(k**3)

    return func


def N_gv_75_1_0(theta, phi):
    return -3 * np.cos(phi + (1 / 2) * theta)


def M_gv_75_1_1():
    def func(k):
        return -2 / 5 * k**3

    return func


def N_gv_75_1_1(theta, phi):
    return -9 / 2 * np.cos(phi - 3 / 2 * theta) - 3 / 2 * np.cos(phi + (1 / 2) * theta)


def M_gv_75_3_0():
    def func(k):
        return -2 / 5 * k**3

    return func


def N_gv_75_3_0(theta, phi):
    return (
        -9 / 8 * np.cos(phi - 3 / 2 * theta)
        - 9 / 4 * np.cos(phi + (1 / 2) * theta)
        - 45 / 8 * np.cos(3 * phi - 1 / 2 * theta)
    )


def M_gv_76_1_0():
    def func(k):
        return -(k**3)

    return func


def N_gv_76_1_0(theta, phi):
    return -3 * np.cos(phi + (1 / 2) * theta)


def M_gv_76_1_1():
    def func(k):
        return -2 / 5 * k**3

    return func


def N_gv_76_1_1(theta, phi):
    return -9 / 2 * np.cos(phi - 3 / 2 * theta) - 3 / 2 * np.cos(phi + (1 / 2) * theta)


def M_gv_76_3_0():
    def func(k):
        return -2 / 5 * k**3

    return func


def N_gv_76_3_0(theta, phi):
    return (
        -9 / 8 * np.cos(phi - 3 / 2 * theta)
        - 9 / 4 * np.cos(phi + (1 / 2) * theta)
        - 45 / 8 * np.cos(3 * phi - 1 / 2 * theta)
    )


def M_gv_77_1_0():
    def func(k):
        return (1 / 6) * k

    return func


def N_gv_77_1_0(theta, phi):
    return -3 * np.cos(phi + (1 / 2) * theta)


def M_gv_77_1_1():
    def func(k):
        return (1 / 15) * k

    return func


def N_gv_77_1_1(theta, phi):
    return -9 / 2 * np.cos(phi - 3 / 2 * theta) - 3 / 2 * np.cos(phi + (1 / 2) * theta)


def M_gv_77_3_0():
    def func(k):
        return (1 / 15) * k

    return func


def N_gv_77_3_0(theta, phi):
    return (
        -9 / 8 * np.cos(phi - 3 / 2 * theta)
        - 9 / 4 * np.cos(phi + (1 / 2) * theta)
        - 45 / 8 * np.cos(3 * phi - 1 / 2 * theta)
    )


def M_gv_78_1_0():
    def func(k):
        return (1 / 6) * k

    return func


def N_gv_78_1_0(theta, phi):
    return -3 * np.cos(phi + (1 / 2) * theta)


def M_gv_78_1_1():
    def func(k):
        return (1 / 15) * k

    return func


def N_gv_78_1_1(theta, phi):
    return -9 / 2 * np.cos(phi - 3 / 2 * theta) - 3 / 2 * np.cos(phi + (1 / 2) * theta)


def M_gv_78_3_0():
    def func(k):
        return (1 / 15) * k

    return func


def N_gv_78_3_0(theta, phi):
    return (
        -9 / 8 * np.cos(phi - 3 / 2 * theta)
        - 9 / 4 * np.cos(phi + (1 / 2) * theta)
        - 45 / 8 * np.cos(3 * phi - 1 / 2 * theta)
    )


def M_gv_79_1_0():
    def func(k):
        return (1 / 6) * k

    return func


def N_gv_79_1_0(theta, phi):
    return -3 * np.cos(phi + (1 / 2) * theta)


def M_gv_79_1_1():
    def func(k):
        return (1 / 15) * k

    return func


def N_gv_79_1_1(theta, phi):
    return -9 / 2 * np.cos(phi - 3 / 2 * theta) - 3 / 2 * np.cos(phi + (1 / 2) * theta)


def M_gv_79_3_0():
    def func(k):
        return (1 / 15) * k

    return func


def N_gv_79_3_0(theta, phi):
    return (
        -9 / 8 * np.cos(phi - 3 / 2 * theta)
        - 9 / 4 * np.cos(phi + (1 / 2) * theta)
        - 45 / 8 * np.cos(3 * phi - 1 / 2 * theta)
    )


def M_gv_80_1_0():
    def func(k):
        return (1 / 6) * k

    return func


def N_gv_80_1_0(theta, phi):
    return -3 * np.cos(phi + (1 / 2) * theta)


def M_gv_80_1_1():
    def func(k):
        return (1 / 15) * k

    return func


def N_gv_80_1_1(theta, phi):
    return -9 / 2 * np.cos(phi - 3 / 2 * theta) - 3 / 2 * np.cos(phi + (1 / 2) * theta)


def M_gv_80_3_0():
    def func(k):
        return (1 / 15) * k

    return func


def N_gv_80_3_0(theta, phi):
    return (
        -9 / 8 * np.cos(phi - 3 / 2 * theta)
        - 9 / 4 * np.cos(phi + (1 / 2) * theta)
        - 45 / 8 * np.cos(3 * phi - 1 / 2 * theta)
    )


def M_gv_81_1_0():
    def func(k):
        return (1 / 6) * k

    return func


def N_gv_81_1_0(theta, phi):
    return -3 * np.cos(phi + (1 / 2) * theta)


def M_gv_81_1_1():
    def func(k):
        return (1 / 15) * k

    return func


def N_gv_81_1_1(theta, phi):
    return -9 / 2 * np.cos(phi - 3 / 2 * theta) - 3 / 2 * np.cos(phi + (1 / 2) * theta)


def M_gv_81_3_0():
    def func(k):
        return (1 / 15) * k

    return func


def N_gv_81_3_0(theta, phi):
    return (
        -9 / 8 * np.cos(phi - 3 / 2 * theta)
        - 9 / 4 * np.cos(phi + (1 / 2) * theta)
        - 45 / 8 * np.cos(3 * phi - 1 / 2 * theta)
    )


def M_gv_82_1_0():
    def func(k):
        return (1 / 3) * k

    return func


def N_gv_82_1_0(theta, phi):
    return -3 * np.cos(phi + (1 / 2) * theta)


def M_gv_82_1_1():
    def func(k):
        return (2 / 15) * k

    return func


def N_gv_82_1_1(theta, phi):
    return -9 / 2 * np.cos(phi - 3 / 2 * theta) - 3 / 2 * np.cos(phi + (1 / 2) * theta)


def M_gv_82_3_0():
    def func(k):
        return (2 / 15) * k

    return func


def N_gv_82_3_0(theta, phi):
    return (
        -9 / 8 * np.cos(phi - 3 / 2 * theta)
        - 9 / 4 * np.cos(phi + (1 / 2) * theta)
        - 45 / 8 * np.cos(3 * phi - 1 / 2 * theta)
    )


def M_gv_83_1_0():
    def func(k):
        return k**3

    return func


def N_gv_83_1_0(theta, phi):
    return -3 * np.cos(phi + (1 / 2) * theta)


def M_gv_83_1_1():
    def func(k):
        return (2 / 5) * k**3

    return func


def N_gv_83_1_1(theta, phi):
    return -9 / 2 * np.cos(phi - 3 / 2 * theta) - 3 / 2 * np.cos(phi + (1 / 2) * theta)


def M_gv_83_3_0():
    def func(k):
        return (2 / 5) * k**3

    return func


def N_gv_83_3_0(theta, phi):
    return (
        -9 / 8 * np.cos(phi - 3 / 2 * theta)
        - 9 / 4 * np.cos(phi + (1 / 2) * theta)
        - 45 / 8 * np.cos(3 * phi - 1 / 2 * theta)
    )


def M_gv_84_1_0():
    def func(k):
        return (1 / 6) / k

    return func


def N_gv_84_1_0(theta, phi):
    return 6 * np.cos(phi + (1 / 2) * theta)


def M_gv_84_1_1():
    def func(k):
        return (1 / 15) / k

    return func


def N_gv_84_1_1(theta, phi):
    return 9 * np.cos(phi - 3 / 2 * theta) + 3 * np.cos(phi + (1 / 2) * theta)


def M_gv_84_3_0():
    def func(k):
        return (1 / 15) / k

    return func


def N_gv_84_3_0(theta, phi):
    return (
        (9 / 4) * np.cos(phi - 3 / 2 * theta)
        + (9 / 2) * np.cos(phi + (1 / 2) * theta)
        + (45 / 4) * np.cos(3 * phi - 1 / 2 * theta)
    )


def M_gv_85_1_0():
    def func(k):
        return (19 / 90) / k

    return func


def N_gv_85_1_0(theta, phi):
    return -3 * np.cos(phi - 3 / 2 * theta) - 3 * np.cos(phi + (1 / 2) * theta)


def M_gv_85_1_1():
    def func(k):
        return (2 / 45) / k

    return func


def N_gv_85_1_1(theta, phi):
    return (
        -3 / 2 * np.cos(phi - 3 / 2 * theta)
        - 6 * np.cos(phi + (1 / 2) * theta)
        - 9 / 2 * np.cos(phi + (5 / 2) * theta)
    )


def M_gv_85_3_0():
    def func(k):
        return (2 / 45) / k

    return func


def N_gv_85_3_0(theta, phi):
    return (
        -9 / 4 * np.cos(phi - 3 / 2 * theta)
        - 27 / 8 * np.cos(phi + (1 / 2) * theta)
        - 9 / 8 * np.cos(phi + (5 / 2) * theta)
        - 45 / 8 * np.cos(3 * phi - 1 / 2 * theta)
        - 45 / 8 * np.cos(3 * phi + (3 / 2) * theta)
    )


def M_gv_85_3_1():
    def func(k):
        return (1 / 35) / k

    return func


def N_gv_85_3_1(theta, phi):
    return (
        -21 / 8 * np.cos(phi - 3 / 2 * theta)
        - 21 / 8 * np.cos(phi + (1 / 2) * theta)
        - 35 / 8 * np.cos(3 * phi - 5 / 2 * theta)
        - 35 / 8 * np.cos(3 * phi - 1 / 2 * theta)
    )


def M_gv_86_1_0():
    def func(k):
        return -2 / 15 * k

    return func


def N_gv_86_1_0(theta, phi):
    return -3 * np.cos(phi + (1 / 2) * theta)


def M_gv_86_1_1():
    def func(k):
        return -31 / 525 * k

    return func


def N_gv_86_1_1(theta, phi):
    return -9 / 2 * np.cos(phi - 3 / 2 * theta) - 3 / 2 * np.cos(phi + (1 / 2) * theta)


def M_gv_86_1_2():
    def func(k):
        return -2 / 175 * k

    return func


def N_gv_86_1_2(theta, phi):
    return (
        -9 / 8 * np.cos(phi - 3 / 2 * theta)
        - 9 / 4 * np.cos(phi + (1 / 2) * theta)
        - 45 / 8 * np.cos(phi + (5 / 2) * theta)
    )


def M_gv_86_3_0():
    def func(k):
        return -1 / 35 * k

    return func


def N_gv_86_3_0(theta, phi):
    return -21 / 8 * np.cos(phi + (1 / 2) * theta) - 35 / 8 * np.cos(
        3 * phi + (3 / 2) * theta
    )


def M_gv_86_3_1():
    def func(k):
        return -31 / 525 * k

    return func


def N_gv_86_3_1(theta, phi):
    return (
        -9 / 8 * np.cos(phi - 3 / 2 * theta)
        - 9 / 4 * np.cos(phi + (1 / 2) * theta)
        - 45 / 8 * np.cos(3 * phi - 1 / 2 * theta)
    )


def M_gv_86_3_2():
    def func(k):
        return -2 / 175 * k

    return func


def N_gv_86_3_2(theta, phi):
    return (
        -21 / 8 * np.cos(phi - 3 / 2 * theta)
        + (21 / 16) * np.cos(phi + (1 / 2) * theta)
        - 35 / 16 * np.cos(phi + (5 / 2) * theta)
        - 35 / 16 * np.cos(3 * phi - 1 / 2 * theta)
        - 175 / 48 * np.cos(3 * phi + (3 / 2) * theta)
    )


def M_gv_86_3_3():
    def func(k):
        return -4 / 945 * k

    return func


def N_gv_86_3_3(theta, phi):
    return (
        -45 / 16 * np.cos(phi - 3 / 2 * theta)
        - 27 / 16 * np.cos(phi + (1 / 2) * theta)
        - 105 / 16 * np.cos(3 * phi - 5 / 2 * theta)
        - 15 / 16 * np.cos(3 * phi - 1 / 2 * theta)
    )


def M_gv_86_5_0():
    def func(k):
        return -2 / 175 * k

    return func


def N_gv_86_5_0(theta, phi):
    return (
        -75 / 64 * np.cos(phi - 3 / 2 * theta)
        - 75 / 32 * np.cos(phi + (1 / 2) * theta)
        - 25 / 64 * np.cos(phi + (5 / 2) * theta)
        - 175 / 64 * np.cos(3 * phi - 1 / 2 * theta)
        - 175 / 96 * np.cos(3 * phi + (3 / 2) * theta)
        - 525 / 64 * np.cos(5 * phi + (1 / 2) * theta)
    )


def M_gv_86_5_1():
    def func(k):
        return -4 / 945 * k

    return func


def N_gv_86_5_1(theta, phi):
    return (
        -45 / 32 * np.cos(phi - 3 / 2 * theta)
        - 135 / 64 * np.cos(phi + (1 / 2) * theta)
        - 105 / 128 * np.cos(3 * phi - 5 / 2 * theta)
        - 105 / 32 * np.cos(3 * phi - 1 / 2 * theta)
        - 945 / 128 * np.cos(5 * phi - 3 / 2 * theta)
    )


def M_gv_87_1_0():
    def func(k):
        return -4 / 15 * k

    return func


def N_gv_87_1_0(theta, phi):
    return -3 * np.cos(phi + (1 / 2) * theta)


def M_gv_87_1_1():
    def func(k):
        return -62 / 525 * k

    return func


def N_gv_87_1_1(theta, phi):
    return -9 / 2 * np.cos(phi - 3 / 2 * theta) - 3 / 2 * np.cos(phi + (1 / 2) * theta)


def M_gv_87_1_2():
    def func(k):
        return -4 / 175 * k

    return func


def N_gv_87_1_2(theta, phi):
    return (
        -9 / 8 * np.cos(phi - 3 / 2 * theta)
        - 9 / 4 * np.cos(phi + (1 / 2) * theta)
        - 45 / 8 * np.cos(phi + (5 / 2) * theta)
    )


def M_gv_87_3_0():
    def func(k):
        return -2 / 35 * k

    return func


def N_gv_87_3_0(theta, phi):
    return -21 / 8 * np.cos(phi + (1 / 2) * theta) - 35 / 8 * np.cos(
        3 * phi + (3 / 2) * theta
    )


def M_gv_87_3_1():
    def func(k):
        return -62 / 525 * k

    return func


def N_gv_87_3_1(theta, phi):
    return (
        -9 / 8 * np.cos(phi - 3 / 2 * theta)
        - 9 / 4 * np.cos(phi + (1 / 2) * theta)
        - 45 / 8 * np.cos(3 * phi - 1 / 2 * theta)
    )


def M_gv_87_3_2():
    def func(k):
        return -4 / 175 * k

    return func


def N_gv_87_3_2(theta, phi):
    return (
        -21 / 8 * np.cos(phi - 3 / 2 * theta)
        + (21 / 16) * np.cos(phi + (1 / 2) * theta)
        - 35 / 16 * np.cos(phi + (5 / 2) * theta)
        - 35 / 16 * np.cos(3 * phi - 1 / 2 * theta)
        - 175 / 48 * np.cos(3 * phi + (3 / 2) * theta)
    )


def M_gv_87_3_3():
    def func(k):
        return -8 / 945 * k

    return func


def N_gv_87_3_3(theta, phi):
    return (
        -45 / 16 * np.cos(phi - 3 / 2 * theta)
        - 27 / 16 * np.cos(phi + (1 / 2) * theta)
        - 105 / 16 * np.cos(3 * phi - 5 / 2 * theta)
        - 15 / 16 * np.cos(3 * phi - 1 / 2 * theta)
    )


def M_gv_87_5_0():
    def func(k):
        return -4 / 175 * k

    return func


def N_gv_87_5_0(theta, phi):
    return (
        -75 / 64 * np.cos(phi - 3 / 2 * theta)
        - 75 / 32 * np.cos(phi + (1 / 2) * theta)
        - 25 / 64 * np.cos(phi + (5 / 2) * theta)
        - 175 / 64 * np.cos(3 * phi - 1 / 2 * theta)
        - 175 / 96 * np.cos(3 * phi + (3 / 2) * theta)
        - 525 / 64 * np.cos(5 * phi + (1 / 2) * theta)
    )


def M_gv_87_5_1():
    def func(k):
        return -8 / 945 * k

    return func


def N_gv_87_5_1(theta, phi):
    return (
        -45 / 32 * np.cos(phi - 3 / 2 * theta)
        - 135 / 64 * np.cos(phi + (1 / 2) * theta)
        - 105 / 128 * np.cos(3 * phi - 5 / 2 * theta)
        - 105 / 32 * np.cos(3 * phi - 1 / 2 * theta)
        - 945 / 128 * np.cos(5 * phi - 3 / 2 * theta)
    )


def M_gv_88_1_0():
    def func(k):
        return -4 / 15 * k

    return func


def N_gv_88_1_0(theta, phi):
    return -3 * np.cos(phi + (1 / 2) * theta)


def M_gv_88_1_1():
    def func(k):
        return -62 / 525 * k

    return func


def N_gv_88_1_1(theta, phi):
    return -9 / 2 * np.cos(phi - 3 / 2 * theta) - 3 / 2 * np.cos(phi + (1 / 2) * theta)


def M_gv_88_1_2():
    def func(k):
        return -4 / 175 * k

    return func


def N_gv_88_1_2(theta, phi):
    return (
        -9 / 8 * np.cos(phi - 3 / 2 * theta)
        - 9 / 4 * np.cos(phi + (1 / 2) * theta)
        - 45 / 8 * np.cos(phi + (5 / 2) * theta)
    )


def M_gv_88_3_0():
    def func(k):
        return -2 / 35 * k

    return func


def N_gv_88_3_0(theta, phi):
    return -21 / 8 * np.cos(phi + (1 / 2) * theta) - 35 / 8 * np.cos(
        3 * phi + (3 / 2) * theta
    )


def M_gv_88_3_1():
    def func(k):
        return -62 / 525 * k

    return func


def N_gv_88_3_1(theta, phi):
    return (
        -9 / 8 * np.cos(phi - 3 / 2 * theta)
        - 9 / 4 * np.cos(phi + (1 / 2) * theta)
        - 45 / 8 * np.cos(3 * phi - 1 / 2 * theta)
    )


def M_gv_88_3_2():
    def func(k):
        return -4 / 175 * k

    return func


def N_gv_88_3_2(theta, phi):
    return (
        -21 / 8 * np.cos(phi - 3 / 2 * theta)
        + (21 / 16) * np.cos(phi + (1 / 2) * theta)
        - 35 / 16 * np.cos(phi + (5 / 2) * theta)
        - 35 / 16 * np.cos(3 * phi - 1 / 2 * theta)
        - 175 / 48 * np.cos(3 * phi + (3 / 2) * theta)
    )


def M_gv_88_3_3():
    def func(k):
        return -8 / 945 * k

    return func


def N_gv_88_3_3(theta, phi):
    return (
        -45 / 16 * np.cos(phi - 3 / 2 * theta)
        - 27 / 16 * np.cos(phi + (1 / 2) * theta)
        - 105 / 16 * np.cos(3 * phi - 5 / 2 * theta)
        - 15 / 16 * np.cos(3 * phi - 1 / 2 * theta)
    )


def M_gv_88_5_0():
    def func(k):
        return -4 / 175 * k

    return func


def N_gv_88_5_0(theta, phi):
    return (
        -75 / 64 * np.cos(phi - 3 / 2 * theta)
        - 75 / 32 * np.cos(phi + (1 / 2) * theta)
        - 25 / 64 * np.cos(phi + (5 / 2) * theta)
        - 175 / 64 * np.cos(3 * phi - 1 / 2 * theta)
        - 175 / 96 * np.cos(3 * phi + (3 / 2) * theta)
        - 525 / 64 * np.cos(5 * phi + (1 / 2) * theta)
    )


def M_gv_88_5_1():
    def func(k):
        return -8 / 945 * k

    return func


def N_gv_88_5_1(theta, phi):
    return (
        -45 / 32 * np.cos(phi - 3 / 2 * theta)
        - 135 / 64 * np.cos(phi + (1 / 2) * theta)
        - 105 / 128 * np.cos(3 * phi - 5 / 2 * theta)
        - 105 / 32 * np.cos(3 * phi - 1 / 2 * theta)
        - 945 / 128 * np.cos(5 * phi - 3 / 2 * theta)
    )


def M_gv_89_1_0():
    def func(k):
        return -4 / 5 * k**3

    return func


def N_gv_89_1_0(theta, phi):
    return -3 * np.cos(phi + (1 / 2) * theta)


def M_gv_89_1_1():
    def func(k):
        return -62 / 175 * k**3

    return func


def N_gv_89_1_1(theta, phi):
    return -9 / 2 * np.cos(phi - 3 / 2 * theta) - 3 / 2 * np.cos(phi + (1 / 2) * theta)


def M_gv_89_1_2():
    def func(k):
        return -12 / 175 * k**3

    return func


def N_gv_89_1_2(theta, phi):
    return (
        -9 / 8 * np.cos(phi - 3 / 2 * theta)
        - 9 / 4 * np.cos(phi + (1 / 2) * theta)
        - 45 / 8 * np.cos(phi + (5 / 2) * theta)
    )


def M_gv_89_3_0():
    def func(k):
        return -6 / 35 * k**3

    return func


def N_gv_89_3_0(theta, phi):
    return -21 / 8 * np.cos(phi + (1 / 2) * theta) - 35 / 8 * np.cos(
        3 * phi + (3 / 2) * theta
    )


def M_gv_89_3_1():
    def func(k):
        return -62 / 175 * k**3

    return func


def N_gv_89_3_1(theta, phi):
    return (
        -9 / 8 * np.cos(phi - 3 / 2 * theta)
        - 9 / 4 * np.cos(phi + (1 / 2) * theta)
        - 45 / 8 * np.cos(3 * phi - 1 / 2 * theta)
    )


def M_gv_89_3_2():
    def func(k):
        return -12 / 175 * k**3

    return func


def N_gv_89_3_2(theta, phi):
    return (
        -21 / 8 * np.cos(phi - 3 / 2 * theta)
        + (21 / 16) * np.cos(phi + (1 / 2) * theta)
        - 35 / 16 * np.cos(phi + (5 / 2) * theta)
        - 35 / 16 * np.cos(3 * phi - 1 / 2 * theta)
        - 175 / 48 * np.cos(3 * phi + (3 / 2) * theta)
    )


def M_gv_89_3_3():
    def func(k):
        return -8 / 315 * k**3

    return func


def N_gv_89_3_3(theta, phi):
    return (
        -45 / 16 * np.cos(phi - 3 / 2 * theta)
        - 27 / 16 * np.cos(phi + (1 / 2) * theta)
        - 105 / 16 * np.cos(3 * phi - 5 / 2 * theta)
        - 15 / 16 * np.cos(3 * phi - 1 / 2 * theta)
    )


def M_gv_89_5_0():
    def func(k):
        return -12 / 175 * k**3

    return func


def N_gv_89_5_0(theta, phi):
    return (
        -75 / 64 * np.cos(phi - 3 / 2 * theta)
        - 75 / 32 * np.cos(phi + (1 / 2) * theta)
        - 25 / 64 * np.cos(phi + (5 / 2) * theta)
        - 175 / 64 * np.cos(3 * phi - 1 / 2 * theta)
        - 175 / 96 * np.cos(3 * phi + (3 / 2) * theta)
        - 525 / 64 * np.cos(5 * phi + (1 / 2) * theta)
    )


def M_gv_89_5_1():
    def func(k):
        return -8 / 315 * k**3

    return func


def N_gv_89_5_1(theta, phi):
    return (
        -45 / 32 * np.cos(phi - 3 / 2 * theta)
        - 135 / 64 * np.cos(phi + (1 / 2) * theta)
        - 105 / 128 * np.cos(3 * phi - 5 / 2 * theta)
        - 105 / 32 * np.cos(3 * phi - 1 / 2 * theta)
        - 945 / 128 * np.cos(5 * phi - 3 / 2 * theta)
    )


def M_gv_90_1_0():
    def func(k):
        return -4 / 5 * k**3

    return func


def N_gv_90_1_0(theta, phi):
    return -3 * np.cos(phi + (1 / 2) * theta)


def M_gv_90_1_1():
    def func(k):
        return -62 / 175 * k**3

    return func


def N_gv_90_1_1(theta, phi):
    return -9 / 2 * np.cos(phi - 3 / 2 * theta) - 3 / 2 * np.cos(phi + (1 / 2) * theta)


def M_gv_90_1_2():
    def func(k):
        return -12 / 175 * k**3

    return func


def N_gv_90_1_2(theta, phi):
    return (
        -9 / 8 * np.cos(phi - 3 / 2 * theta)
        - 9 / 4 * np.cos(phi + (1 / 2) * theta)
        - 45 / 8 * np.cos(phi + (5 / 2) * theta)
    )


def M_gv_90_3_0():
    def func(k):
        return -6 / 35 * k**3

    return func


def N_gv_90_3_0(theta, phi):
    return -21 / 8 * np.cos(phi + (1 / 2) * theta) - 35 / 8 * np.cos(
        3 * phi + (3 / 2) * theta
    )


def M_gv_90_3_1():
    def func(k):
        return -62 / 175 * k**3

    return func


def N_gv_90_3_1(theta, phi):
    return (
        -9 / 8 * np.cos(phi - 3 / 2 * theta)
        - 9 / 4 * np.cos(phi + (1 / 2) * theta)
        - 45 / 8 * np.cos(3 * phi - 1 / 2 * theta)
    )


def M_gv_90_3_2():
    def func(k):
        return -12 / 175 * k**3

    return func


def N_gv_90_3_2(theta, phi):
    return (
        -21 / 8 * np.cos(phi - 3 / 2 * theta)
        + (21 / 16) * np.cos(phi + (1 / 2) * theta)
        - 35 / 16 * np.cos(phi + (5 / 2) * theta)
        - 35 / 16 * np.cos(3 * phi - 1 / 2 * theta)
        - 175 / 48 * np.cos(3 * phi + (3 / 2) * theta)
    )


def M_gv_90_3_3():
    def func(k):
        return -8 / 315 * k**3

    return func


def N_gv_90_3_3(theta, phi):
    return (
        -45 / 16 * np.cos(phi - 3 / 2 * theta)
        - 27 / 16 * np.cos(phi + (1 / 2) * theta)
        - 105 / 16 * np.cos(3 * phi - 5 / 2 * theta)
        - 15 / 16 * np.cos(3 * phi - 1 / 2 * theta)
    )


def M_gv_90_5_0():
    def func(k):
        return -12 / 175 * k**3

    return func


def N_gv_90_5_0(theta, phi):
    return (
        -75 / 64 * np.cos(phi - 3 / 2 * theta)
        - 75 / 32 * np.cos(phi + (1 / 2) * theta)
        - 25 / 64 * np.cos(phi + (5 / 2) * theta)
        - 175 / 64 * np.cos(3 * phi - 1 / 2 * theta)
        - 175 / 96 * np.cos(3 * phi + (3 / 2) * theta)
        - 525 / 64 * np.cos(5 * phi + (1 / 2) * theta)
    )


def M_gv_90_5_1():
    def func(k):
        return -8 / 315 * k**3

    return func


def N_gv_90_5_1(theta, phi):
    return (
        -45 / 32 * np.cos(phi - 3 / 2 * theta)
        - 135 / 64 * np.cos(phi + (1 / 2) * theta)
        - 105 / 128 * np.cos(3 * phi - 5 / 2 * theta)
        - 105 / 32 * np.cos(3 * phi - 1 / 2 * theta)
        - 945 / 128 * np.cos(5 * phi - 3 / 2 * theta)
    )


def M_gv_91_1_0():
    def func(k):
        return -2 / 15 * k

    return func


def N_gv_91_1_0(theta, phi):
    return -3 * np.cos(phi + (1 / 2) * theta)


def M_gv_91_1_1():
    def func(k):
        return -31 / 525 * k

    return func


def N_gv_91_1_1(theta, phi):
    return -9 / 2 * np.cos(phi - 3 / 2 * theta) - 3 / 2 * np.cos(phi + (1 / 2) * theta)


def M_gv_91_1_2():
    def func(k):
        return -2 / 175 * k

    return func


def N_gv_91_1_2(theta, phi):
    return (
        -9 / 8 * np.cos(phi - 3 / 2 * theta)
        - 9 / 4 * np.cos(phi + (1 / 2) * theta)
        - 45 / 8 * np.cos(phi + (5 / 2) * theta)
    )


def M_gv_91_3_0():
    def func(k):
        return -1 / 35 * k

    return func


def N_gv_91_3_0(theta, phi):
    return -21 / 8 * np.cos(phi + (1 / 2) * theta) - 35 / 8 * np.cos(
        3 * phi + (3 / 2) * theta
    )


def M_gv_91_3_1():
    def func(k):
        return -31 / 525 * k

    return func


def N_gv_91_3_1(theta, phi):
    return (
        -9 / 8 * np.cos(phi - 3 / 2 * theta)
        - 9 / 4 * np.cos(phi + (1 / 2) * theta)
        - 45 / 8 * np.cos(3 * phi - 1 / 2 * theta)
    )


def M_gv_91_3_2():
    def func(k):
        return -2 / 175 * k

    return func


def N_gv_91_3_2(theta, phi):
    return (
        -21 / 8 * np.cos(phi - 3 / 2 * theta)
        + (21 / 16) * np.cos(phi + (1 / 2) * theta)
        - 35 / 16 * np.cos(phi + (5 / 2) * theta)
        - 35 / 16 * np.cos(3 * phi - 1 / 2 * theta)
        - 175 / 48 * np.cos(3 * phi + (3 / 2) * theta)
    )


def M_gv_91_3_3():
    def func(k):
        return -4 / 945 * k

    return func


def N_gv_91_3_3(theta, phi):
    return (
        -45 / 16 * np.cos(phi - 3 / 2 * theta)
        - 27 / 16 * np.cos(phi + (1 / 2) * theta)
        - 105 / 16 * np.cos(3 * phi - 5 / 2 * theta)
        - 15 / 16 * np.cos(3 * phi - 1 / 2 * theta)
    )


def M_gv_91_5_0():
    def func(k):
        return -2 / 175 * k

    return func


def N_gv_91_5_0(theta, phi):
    return (
        -75 / 64 * np.cos(phi - 3 / 2 * theta)
        - 75 / 32 * np.cos(phi + (1 / 2) * theta)
        - 25 / 64 * np.cos(phi + (5 / 2) * theta)
        - 175 / 64 * np.cos(3 * phi - 1 / 2 * theta)
        - 175 / 96 * np.cos(3 * phi + (3 / 2) * theta)
        - 525 / 64 * np.cos(5 * phi + (1 / 2) * theta)
    )


def M_gv_91_5_1():
    def func(k):
        return -4 / 945 * k

    return func


def N_gv_91_5_1(theta, phi):
    return (
        -45 / 32 * np.cos(phi - 3 / 2 * theta)
        - 135 / 64 * np.cos(phi + (1 / 2) * theta)
        - 105 / 128 * np.cos(3 * phi - 5 / 2 * theta)
        - 105 / 32 * np.cos(3 * phi - 1 / 2 * theta)
        - 945 / 128 * np.cos(5 * phi - 3 / 2 * theta)
    )


def M_gv_92_1_0():
    def func(k):
        return -4 / 15 * k

    return func


def N_gv_92_1_0(theta, phi):
    return -3 / 2 * np.cos(phi - 3 / 2 * theta) - 3 / 2 * np.cos(phi + (1 / 2) * theta)


def M_gv_92_1_1():
    def func(k):
        return -1 / 15 * k

    return func


def N_gv_92_1_1(theta, phi):
    return (
        -3 / 4 * np.cos(phi - 3 / 2 * theta)
        - 3 * np.cos(phi + (1 / 2) * theta)
        - 9 / 4 * np.cos(phi + (5 / 2) * theta)
    )


def M_gv_92_3_0():
    def func(k):
        return -1 / 15 * k

    return func


def N_gv_92_3_0(theta, phi):
    return (
        -9 / 8 * np.cos(phi - 3 / 2 * theta)
        - 27 / 16 * np.cos(phi + (1 / 2) * theta)
        - 9 / 16 * np.cos(phi + (5 / 2) * theta)
        - 45 / 16 * np.cos(3 * phi - 1 / 2 * theta)
        - 45 / 16 * np.cos(3 * phi + (3 / 2) * theta)
    )


def M_gv_92_3_1():
    def func(k):
        return -1 / 35 * k

    return func


def N_gv_92_3_1(theta, phi):
    return (
        -21 / 16 * np.cos(phi - 3 / 2 * theta)
        - 21 / 16 * np.cos(phi + (1 / 2) * theta)
        - 35 / 16 * np.cos(3 * phi - 5 / 2 * theta)
        - 35 / 16 * np.cos(3 * phi - 1 / 2 * theta)
    )


def M_gv_93_1_0():
    def func(k):
        return (1 / 18) / k

    return func


def N_gv_93_1_0(theta, phi):
    return (
        -1 / 8 * np.cos(phi - 3 / 2 * theta)
        - 1 / 2 * np.cos(phi + (1 / 2) * theta)
        - 1 / 8 * np.cos(phi + (5 / 2) * theta)
    )


def M_gv_93_1_1():
    def func(k):
        return (1 / 45) / k

    return func


def N_gv_93_1_1(theta, phi):
    return (
        -3 / 16 * np.cos(phi - 7 / 2 * theta)
        - 13 / 16 * np.cos(phi - 3 / 2 * theta)
        - 7 / 16 * np.cos(phi + (1 / 2) * theta)
        - 1 / 16 * np.cos(phi + (5 / 2) * theta)
    )


def M_gv_93_3_0():
    def func(k):
        return (1 / 45) / k

    return func


def N_gv_93_3_0(theta, phi):
    return (
        -3 / 64 * np.cos(phi - 7 / 2 * theta)
        - 9 / 32 * np.cos(phi - 3 / 2 * theta)
        - 27 / 64 * np.cos(phi + (1 / 2) * theta)
        - 3 / 32 * np.cos(phi + (5 / 2) * theta)
        - 15 / 64 * np.cos(3 * phi - 5 / 2 * theta)
        - 15 / 16 * np.cos(3 * phi - 1 / 2 * theta)
        - 15 / 64 * np.cos(3 * phi + (3 / 2) * theta)
    )


def M_gv_94_1_0():
    def func(k):
        return (1 / 30) / k

    return func


def N_gv_94_1_0(theta, phi):
    return -3 / 4 * np.cos(phi - 3 / 2 * theta) - 3 / 4 * np.cos(phi + (1 / 2) * theta)


def M_gv_94_1_1():
    def func(k):
        return (1 / 75) / k

    return func


def N_gv_94_1_1(theta, phi):
    return (
        -3 / 8 * np.cos(phi - 3 / 2 * theta)
        - 3 / 2 * np.cos(phi + (1 / 2) * theta)
        - 9 / 8 * np.cos(phi + (5 / 2) * theta)
    )


def M_gv_94_1_2():
    def func(k):
        return (2 / 525) / k

    return func


def N_gv_94_1_2(theta, phi):
    return (
        -45 / 32 * np.cos(phi - 7 / 2 * theta)
        - 63 / 32 * np.cos(phi - 3 / 2 * theta)
        - 27 / 32 * np.cos(phi + (1 / 2) * theta)
        - 9 / 32 * np.cos(phi + (5 / 2) * theta)
    )


def M_gv_94_3_0():
    def func(k):
        return (1 / 75) / k

    return func


def N_gv_94_3_0(theta, phi):
    return (
        -9 / 16 * np.cos(phi - 3 / 2 * theta)
        - 27 / 32 * np.cos(phi + (1 / 2) * theta)
        - 9 / 32 * np.cos(phi + (5 / 2) * theta)
        - 45 / 32 * np.cos(3 * phi - 1 / 2 * theta)
        - 45 / 32 * np.cos(3 * phi + (3 / 2) * theta)
    )


def M_gv_94_3_1():
    def func(k):
        return (1 / 105) / k

    return func


def N_gv_94_3_1(theta, phi):
    return (
        -21 / 32 * np.cos(phi - 3 / 2 * theta)
        - 21 / 32 * np.cos(phi + (1 / 2) * theta)
        - 35 / 32 * np.cos(3 * phi - 5 / 2 * theta)
        - 35 / 32 * np.cos(3 * phi - 1 / 2 * theta)
    )


def M_gv_94_3_2():
    def func(k):
        return (2 / 525) / k

    return func


def N_gv_94_3_2(theta, phi):
    return (
        -35 / 64 * np.cos(phi - 7 / 2 * theta)
        - 7 / 32 * np.cos(phi - 3 / 2 * theta)
        - 21 / 64 * np.cos(phi + (1 / 2) * theta)
        - 21 / 32 * np.cos(phi + (5 / 2) * theta)
        - 175 / 192 * np.cos(3 * phi - 5 / 2 * theta)
        - 35 / 24 * np.cos(3 * phi - 1 / 2 * theta)
        - 35 / 64 * np.cos(3 * phi + (3 / 2) * theta)
    )


def M_gv_94_5_0():
    def func(k):
        return (2 / 525) / k

    return func


def N_gv_94_5_0(theta, phi):
    return (
        -25 / 256 * np.cos(phi - 7 / 2 * theta)
        - 175 / 256 * np.cos(phi - 3 / 2 * theta)
        - 225 / 256 * np.cos(phi + (1 / 2) * theta)
        - 75 / 256 * np.cos(phi + (5 / 2) * theta)
        - 175 / 384 * np.cos(3 * phi - 5 / 2 * theta)
        - 875 / 768 * np.cos(3 * phi - 1 / 2 * theta)
        - 175 / 256 * np.cos(3 * phi + (3 / 2) * theta)
        - 525 / 256 * np.cos(5 * phi - 3 / 2 * theta)
        - 525 / 256 * np.cos(5 * phi + (1 / 2) * theta)
    )


def M_gv_95_1_0():
    def func(k):
        return (1 / 200) / k

    return func


def N_gv_95_1_0(theta, phi):
    return -3 * np.cos(phi + (1 / 2) * theta)


def M_gv_95_1_1():
    def func(k):
        return (1 / 350) / k

    return func


def N_gv_95_1_1(theta, phi):
    return -9 / 2 * np.cos(phi - 3 / 2 * theta) - 3 / 2 * np.cos(phi + (1 / 2) * theta)


def M_gv_95_1_2():
    def func(k):
        return (1 / 1225) / k

    return func


def N_gv_95_1_2(theta, phi):
    return (
        -9 / 8 * np.cos(phi - 3 / 2 * theta)
        - 9 / 4 * np.cos(phi + (1 / 2) * theta)
        - 45 / 8 * np.cos(phi + (5 / 2) * theta)
    )


def M_gv_95_1_3():
    def func(k):
        return (2 / 11025) / k

    return func


def N_gv_95_1_3(theta, phi):
    return (
        -105 / 16 * np.cos(phi - 7 / 2 * theta)
        - 45 / 16 * np.cos(phi - 3 / 2 * theta)
        - 27 / 16 * np.cos(phi + (1 / 2) * theta)
        - 15 / 16 * np.cos(phi + (5 / 2) * theta)
    )


def M_gv_95_3_0():
    def func(k):
        return (1 / 700) / k

    return func


def N_gv_95_3_0(theta, phi):
    return -21 / 8 * np.cos(phi + (1 / 2) * theta) - 35 / 8 * np.cos(
        3 * phi + (3 / 2) * theta
    )


def M_gv_95_3_1():
    def func(k):
        return (1 / 350) / k

    return func


def N_gv_95_3_1(theta, phi):
    return (
        -9 / 8 * np.cos(phi - 3 / 2 * theta)
        - 9 / 4 * np.cos(phi + (1 / 2) * theta)
        - 45 / 8 * np.cos(3 * phi - 1 / 2 * theta)
    )


def M_gv_95_3_2():
    def func(k):
        return (1 / 1225) / k

    return func


def N_gv_95_3_2(theta, phi):
    return (
        -21 / 8 * np.cos(phi - 3 / 2 * theta)
        + (21 / 16) * np.cos(phi + (1 / 2) * theta)
        - 35 / 16 * np.cos(phi + (5 / 2) * theta)
        - 35 / 16 * np.cos(3 * phi - 1 / 2 * theta)
        - 175 / 48 * np.cos(3 * phi + (3 / 2) * theta)
    )


def M_gv_95_3_3():
    def func(k):
        return (1 / 1575) / k

    return func


def N_gv_95_3_3(theta, phi):
    return (
        -45 / 16 * np.cos(phi - 3 / 2 * theta)
        - 27 / 16 * np.cos(phi + (1 / 2) * theta)
        - 105 / 16 * np.cos(3 * phi - 5 / 2 * theta)
        - 15 / 16 * np.cos(3 * phi - 1 / 2 * theta)
    )


def M_gv_95_3_4():
    def func(k):
        return (2 / 11025) / k

    return func


def N_gv_95_3_4(theta, phi):
    return (
        -2205 / 704 * np.cos(phi - 7 / 2 * theta)
        + (315 / 176) * np.cos(phi - 3 / 2 * theta)
        - 189 / 704 * np.cos(phi + (1 / 2) * theta)
        - 945 / 352 * np.cos(phi + (5 / 2) * theta)
        - 2205 / 704 * np.cos(3 * phi - 5 / 2 * theta)
        - 945 / 352 * np.cos(3 * phi - 1 / 2 * theta)
        - 945 / 704 * np.cos(3 * phi + (3 / 2) * theta)
    )


def M_gv_95_5_0():
    def func(k):
        return (1 / 1225) / k

    return func


def N_gv_95_5_0(theta, phi):
    return (
        -75 / 64 * np.cos(phi - 3 / 2 * theta)
        - 75 / 32 * np.cos(phi + (1 / 2) * theta)
        - 25 / 64 * np.cos(phi + (5 / 2) * theta)
        - 175 / 64 * np.cos(3 * phi - 1 / 2 * theta)
        - 175 / 96 * np.cos(3 * phi + (3 / 2) * theta)
        - 525 / 64 * np.cos(5 * phi + (1 / 2) * theta)
    )


def M_gv_95_5_1():
    def func(k):
        return (1 / 1575) / k

    return func


def N_gv_95_5_1(theta, phi):
    return (
        -45 / 32 * np.cos(phi - 3 / 2 * theta)
        - 135 / 64 * np.cos(phi + (1 / 2) * theta)
        - 105 / 128 * np.cos(3 * phi - 5 / 2 * theta)
        - 105 / 32 * np.cos(3 * phi - 1 / 2 * theta)
        - 945 / 128 * np.cos(5 * phi - 3 / 2 * theta)
    )


def M_gv_95_5_2():
    def func(k):
        return (2 / 11025) / k

    return func


def N_gv_95_5_2(theta, phi):
    return (
        -1575 / 1664 * np.cos(phi - 7 / 2 * theta)
        - 1935 / 1664 * np.cos(phi - 3 / 2 * theta)
        + (1485 / 1664) * np.cos(phi + (1 / 2) * theta)
        - 3375 / 1664 * np.cos(phi + (5 / 2) * theta)
        - 2205 / 832 * np.cos(3 * phi - 5 / 2 * theta)
        + (2835 / 1664) * np.cos(3 * phi - 1 / 2 * theta)
        - 4725 / 1664 * np.cos(3 * phi + (3 / 2) * theta)
        - 6615 / 1664 * np.cos(5 * phi - 3 / 2 * theta)
        - 4725 / 1664 * np.cos(5 * phi + (1 / 2) * theta)
    )


def M_gv_95_7_0():
    def func(k):
        return (2 / 11025) / k

    return func


def N_gv_95_7_0(theta, phi):
    return (
        -18375 / 146432 * np.cos(phi - 7 / 2 * theta)
        - 55125 / 36608 * np.cos(phi - 3 / 2 * theta)
        - 165375 / 73216 * np.cos(phi + (1 / 2) * theta)
        - 18375 / 36608 * np.cos(phi + (5 / 2) * theta)
        - 99225 / 146432 * np.cos(3 * phi - 5 / 2 * theta)
        - 99225 / 36608 * np.cos(3 * phi - 1 / 2 * theta)
        - 99225 / 73216 * np.cos(3 * phi + (3 / 2) * theta)
        - 33075 / 13312 * np.cos(5 * phi - 3 / 2 * theta)
        - 11025 / 3328 * np.cos(5 * phi + (1 / 2) * theta)
        - 11025 / 1024 * np.cos(7 * phi - 1 / 2 * theta)
    )


def M_gv_96_1_0():
    def func(k):
        return (1 / 6) * k**3

    return func


def N_gv_96_1_0(theta, phi):
    return -3 * np.cos(phi + (1 / 2) * theta)


def M_gv_96_1_1():
    def func(k):
        return (1 / 15) * k**3

    return func


def N_gv_96_1_1(theta, phi):
    return -9 / 2 * np.cos(phi - 3 / 2 * theta) - 3 / 2 * np.cos(phi + (1 / 2) * theta)


def M_gv_96_3_0():
    def func(k):
        return (1 / 15) * k**3

    return func


def N_gv_96_3_0(theta, phi):
    return (
        -9 / 8 * np.cos(phi - 3 / 2 * theta)
        - 9 / 4 * np.cos(phi + (1 / 2) * theta)
        - 45 / 8 * np.cos(3 * phi - 1 / 2 * theta)
    )


def M_gv_97_1_0():
    def func(k):
        return (1 / 6) * k**3

    return func


def N_gv_97_1_0(theta, phi):
    return -3 * np.cos(phi + (1 / 2) * theta)


def M_gv_97_1_1():
    def func(k):
        return (1 / 15) * k**3

    return func


def N_gv_97_1_1(theta, phi):
    return -9 / 2 * np.cos(phi - 3 / 2 * theta) - 3 / 2 * np.cos(phi + (1 / 2) * theta)


def M_gv_97_3_0():
    def func(k):
        return (1 / 15) * k**3

    return func


def N_gv_97_3_0(theta, phi):
    return (
        -9 / 8 * np.cos(phi - 3 / 2 * theta)
        - 9 / 4 * np.cos(phi + (1 / 2) * theta)
        - 45 / 8 * np.cos(3 * phi - 1 / 2 * theta)
    )


def M_gv_98_1_0():
    def func(k):
        return (1 / 6) * k**3

    return func


def N_gv_98_1_0(theta, phi):
    return -3 * np.cos(phi + (1 / 2) * theta)


def M_gv_98_1_1():
    def func(k):
        return (1 / 15) * k**3

    return func


def N_gv_98_1_1(theta, phi):
    return -9 / 2 * np.cos(phi - 3 / 2 * theta) - 3 / 2 * np.cos(phi + (1 / 2) * theta)


def M_gv_98_3_0():
    def func(k):
        return (1 / 15) * k**3

    return func


def N_gv_98_3_0(theta, phi):
    return (
        -9 / 8 * np.cos(phi - 3 / 2 * theta)
        - 9 / 4 * np.cos(phi + (1 / 2) * theta)
        - 45 / 8 * np.cos(3 * phi - 1 / 2 * theta)
    )


def M_gv_99_1_0():
    def func(k):
        return (1 / 3) * k**3

    return func


def N_gv_99_1_0(theta, phi):
    return -3 * np.cos(phi + (1 / 2) * theta)


def M_gv_99_1_1():
    def func(k):
        return (2 / 15) * k**3

    return func


def N_gv_99_1_1(theta, phi):
    return -9 / 2 * np.cos(phi - 3 / 2 * theta) - 3 / 2 * np.cos(phi + (1 / 2) * theta)


def M_gv_99_3_0():
    def func(k):
        return (2 / 15) * k**3

    return func


def N_gv_99_3_0(theta, phi):
    return (
        -9 / 8 * np.cos(phi - 3 / 2 * theta)
        - 9 / 4 * np.cos(phi + (1 / 2) * theta)
        - 45 / 8 * np.cos(3 * phi - 1 / 2 * theta)
    )


def M_gv_100_1_0():
    def func(k):
        return k**5

    return func


def N_gv_100_1_0(theta, phi):
    return -3 * np.cos(phi + (1 / 2) * theta)


def M_gv_100_1_1():
    def func(k):
        return (2 / 5) * k**5

    return func


def N_gv_100_1_1(theta, phi):
    return -9 / 2 * np.cos(phi - 3 / 2 * theta) - 3 / 2 * np.cos(phi + (1 / 2) * theta)


def M_gv_100_3_0():
    def func(k):
        return (2 / 5) * k**5

    return func


def N_gv_100_3_0(theta, phi):
    return (
        -9 / 8 * np.cos(phi - 3 / 2 * theta)
        - 9 / 4 * np.cos(phi + (1 / 2) * theta)
        - 45 / 8 * np.cos(3 * phi - 1 / 2 * theta)
    )


def M_gv_101_1_0():
    def func(k):
        return (1 / 12) * k**3

    return func


def N_gv_101_1_0(theta, phi):
    return -3 * np.cos(phi + (1 / 2) * theta)


def M_gv_101_1_1():
    def func(k):
        return (1 / 30) * k**3

    return func


def N_gv_101_1_1(theta, phi):
    return -9 / 2 * np.cos(phi - 3 / 2 * theta) - 3 / 2 * np.cos(phi + (1 / 2) * theta)


def M_gv_101_3_0():
    def func(k):
        return (1 / 30) * k**3

    return func


def N_gv_101_3_0(theta, phi):
    return (
        -9 / 8 * np.cos(phi - 3 / 2 * theta)
        - 9 / 4 * np.cos(phi + (1 / 2) * theta)
        - 45 / 8 * np.cos(3 * phi - 1 / 2 * theta)
    )


def M_gv_102_1_0():
    def func(k):
        return (1 / 12) * k**3

    return func


def N_gv_102_1_0(theta, phi):
    return -3 * np.cos(phi + (1 / 2) * theta)


def M_gv_102_1_1():
    def func(k):
        return (1 / 30) * k**3

    return func


def N_gv_102_1_1(theta, phi):
    return -9 / 2 * np.cos(phi - 3 / 2 * theta) - 3 / 2 * np.cos(phi + (1 / 2) * theta)


def M_gv_102_3_0():
    def func(k):
        return (1 / 30) * k**3

    return func


def N_gv_102_3_0(theta, phi):
    return (
        -9 / 8 * np.cos(phi - 3 / 2 * theta)
        - 9 / 4 * np.cos(phi + (1 / 2) * theta)
        - 45 / 8 * np.cos(3 * phi - 1 / 2 * theta)
    )


def M_gv_103_1_0():
    def func(k):
        return (1 / 12) * k**3

    return func


def N_gv_103_1_0(theta, phi):
    return -3 * np.cos(phi + (1 / 2) * theta)


def M_gv_103_1_1():
    def func(k):
        return (1 / 30) * k**3

    return func


def N_gv_103_1_1(theta, phi):
    return -9 / 2 * np.cos(phi - 3 / 2 * theta) - 3 / 2 * np.cos(phi + (1 / 2) * theta)


def M_gv_103_3_0():
    def func(k):
        return (1 / 30) * k**3

    return func


def N_gv_103_3_0(theta, phi):
    return (
        -9 / 8 * np.cos(phi - 3 / 2 * theta)
        - 9 / 4 * np.cos(phi + (1 / 2) * theta)
        - 45 / 8 * np.cos(3 * phi - 1 / 2 * theta)
    )


def M_gv_104_1_0():
    def func(k):
        return (1 / 6) * k**3

    return func


def N_gv_104_1_0(theta, phi):
    return -3 * np.cos(phi + (1 / 2) * theta)


def M_gv_104_1_1():
    def func(k):
        return (1 / 15) * k**3

    return func


def N_gv_104_1_1(theta, phi):
    return -9 / 2 * np.cos(phi - 3 / 2 * theta) - 3 / 2 * np.cos(phi + (1 / 2) * theta)


def M_gv_104_3_0():
    def func(k):
        return (1 / 15) * k**3

    return func


def N_gv_104_3_0(theta, phi):
    return (
        -9 / 8 * np.cos(phi - 3 / 2 * theta)
        - 9 / 4 * np.cos(phi + (1 / 2) * theta)
        - 45 / 8 * np.cos(3 * phi - 1 / 2 * theta)
    )


def M_gv_105_1_0():
    def func(k):
        return -1 / 9 * k

    return func


def N_gv_105_1_0(theta, phi):
    return -3 * np.cos(phi + (1 / 2) * theta)


def M_gv_105_1_1():
    def func(k):
        return -2 / 45 * k

    return func


def N_gv_105_1_1(theta, phi):
    return -9 / 2 * np.cos(phi - 3 / 2 * theta) - 3 / 2 * np.cos(phi + (1 / 2) * theta)


def M_gv_105_3_0():
    def func(k):
        return -2 / 45 * k

    return func


def N_gv_105_3_0(theta, phi):
    return (
        -9 / 8 * np.cos(phi - 3 / 2 * theta)
        - 9 / 4 * np.cos(phi + (1 / 2) * theta)
        - 45 / 8 * np.cos(3 * phi - 1 / 2 * theta)
    )


def M_gv_106_1_0():
    def func(k):
        return -1 / 15 * k

    return func


def N_gv_106_1_0(theta, phi):
    return -3 * np.cos(phi + (1 / 2) * theta)


def M_gv_106_1_1():
    def func(k):
        return -17 / 525 * k

    return func


def N_gv_106_1_1(theta, phi):
    return -9 / 2 * np.cos(phi - 3 / 2 * theta) - 3 / 2 * np.cos(phi + (1 / 2) * theta)


def M_gv_106_1_2():
    def func(k):
        return -2 / 525 * k

    return func


def N_gv_106_1_2(theta, phi):
    return (
        -9 / 8 * np.cos(phi - 3 / 2 * theta)
        - 9 / 4 * np.cos(phi + (1 / 2) * theta)
        - 45 / 8 * np.cos(phi + (5 / 2) * theta)
    )


def M_gv_106_3_0():
    def func(k):
        return -1 / 105 * k

    return func


def N_gv_106_3_0(theta, phi):
    return -21 / 8 * np.cos(phi + (1 / 2) * theta) - 35 / 8 * np.cos(
        3 * phi + (3 / 2) * theta
    )


def M_gv_106_3_1():
    def func(k):
        return -17 / 525 * k

    return func


def N_gv_106_3_1(theta, phi):
    return (
        -9 / 8 * np.cos(phi - 3 / 2 * theta)
        - 9 / 4 * np.cos(phi + (1 / 2) * theta)
        - 45 / 8 * np.cos(3 * phi - 1 / 2 * theta)
    )


def M_gv_106_3_2():
    def func(k):
        return -2 / 525 * k

    return func


def N_gv_106_3_2(theta, phi):
    return (
        -21 / 8 * np.cos(phi - 3 / 2 * theta)
        + (21 / 16) * np.cos(phi + (1 / 2) * theta)
        - 35 / 16 * np.cos(phi + (5 / 2) * theta)
        - 35 / 16 * np.cos(3 * phi - 1 / 2 * theta)
        - 175 / 48 * np.cos(3 * phi + (3 / 2) * theta)
    )


def M_gv_106_3_3():
    def func(k):
        return -4 / 945 * k

    return func


def N_gv_106_3_3(theta, phi):
    return (
        -45 / 16 * np.cos(phi - 3 / 2 * theta)
        - 27 / 16 * np.cos(phi + (1 / 2) * theta)
        - 105 / 16 * np.cos(3 * phi - 5 / 2 * theta)
        - 15 / 16 * np.cos(3 * phi - 1 / 2 * theta)
    )


def M_gv_106_5_0():
    def func(k):
        return -2 / 525 * k

    return func


def N_gv_106_5_0(theta, phi):
    return (
        -75 / 64 * np.cos(phi - 3 / 2 * theta)
        - 75 / 32 * np.cos(phi + (1 / 2) * theta)
        - 25 / 64 * np.cos(phi + (5 / 2) * theta)
        - 175 / 64 * np.cos(3 * phi - 1 / 2 * theta)
        - 175 / 96 * np.cos(3 * phi + (3 / 2) * theta)
        - 525 / 64 * np.cos(5 * phi + (1 / 2) * theta)
    )


def M_gv_106_5_1():
    def func(k):
        return -4 / 945 * k

    return func


def N_gv_106_5_1(theta, phi):
    return (
        -45 / 32 * np.cos(phi - 3 / 2 * theta)
        - 135 / 64 * np.cos(phi + (1 / 2) * theta)
        - 105 / 128 * np.cos(3 * phi - 5 / 2 * theta)
        - 105 / 32 * np.cos(3 * phi - 1 / 2 * theta)
        - 945 / 128 * np.cos(5 * phi - 3 / 2 * theta)
    )


def M_gv_107_1_0():
    def func(k):
        return -2 / 9 * k**3

    return func


def N_gv_107_1_0(theta, phi):
    return -3 * np.cos(phi + (1 / 2) * theta)


def M_gv_107_1_1():
    def func(k):
        return -4 / 45 * k**3

    return func


def N_gv_107_1_1(theta, phi):
    return -9 / 2 * np.cos(phi - 3 / 2 * theta) - 3 / 2 * np.cos(phi + (1 / 2) * theta)


def M_gv_107_3_0():
    def func(k):
        return -4 / 45 * k**3

    return func


def N_gv_107_3_0(theta, phi):
    return (
        -9 / 8 * np.cos(phi - 3 / 2 * theta)
        - 9 / 4 * np.cos(phi + (1 / 2) * theta)
        - 45 / 8 * np.cos(3 * phi - 1 / 2 * theta)
    )


def M_gv_108_1_0():
    def func(k):
        return -2 / 15 * k**3

    return func


def N_gv_108_1_0(theta, phi):
    return -3 * np.cos(phi + (1 / 2) * theta)


def M_gv_108_1_1():
    def func(k):
        return -34 / 525 * k**3

    return func


def N_gv_108_1_1(theta, phi):
    return -9 / 2 * np.cos(phi - 3 / 2 * theta) - 3 / 2 * np.cos(phi + (1 / 2) * theta)


def M_gv_108_1_2():
    def func(k):
        return -4 / 525 * k**3

    return func


def N_gv_108_1_2(theta, phi):
    return (
        -9 / 8 * np.cos(phi - 3 / 2 * theta)
        - 9 / 4 * np.cos(phi + (1 / 2) * theta)
        - 45 / 8 * np.cos(phi + (5 / 2) * theta)
    )


def M_gv_108_3_0():
    def func(k):
        return -2 / 105 * k**3

    return func


def N_gv_108_3_0(theta, phi):
    return -21 / 8 * np.cos(phi + (1 / 2) * theta) - 35 / 8 * np.cos(
        3 * phi + (3 / 2) * theta
    )


def M_gv_108_3_1():
    def func(k):
        return -34 / 525 * k**3

    return func


def N_gv_108_3_1(theta, phi):
    return (
        -9 / 8 * np.cos(phi - 3 / 2 * theta)
        - 9 / 4 * np.cos(phi + (1 / 2) * theta)
        - 45 / 8 * np.cos(3 * phi - 1 / 2 * theta)
    )


def M_gv_108_3_2():
    def func(k):
        return -4 / 525 * k**3

    return func


def N_gv_108_3_2(theta, phi):
    return (
        -21 / 8 * np.cos(phi - 3 / 2 * theta)
        + (21 / 16) * np.cos(phi + (1 / 2) * theta)
        - 35 / 16 * np.cos(phi + (5 / 2) * theta)
        - 35 / 16 * np.cos(3 * phi - 1 / 2 * theta)
        - 175 / 48 * np.cos(3 * phi + (3 / 2) * theta)
    )


def M_gv_108_3_3():
    def func(k):
        return -8 / 945 * k**3

    return func


def N_gv_108_3_3(theta, phi):
    return (
        -45 / 16 * np.cos(phi - 3 / 2 * theta)
        - 27 / 16 * np.cos(phi + (1 / 2) * theta)
        - 105 / 16 * np.cos(3 * phi - 5 / 2 * theta)
        - 15 / 16 * np.cos(3 * phi - 1 / 2 * theta)
    )


def M_gv_108_5_0():
    def func(k):
        return -4 / 525 * k**3

    return func


def N_gv_108_5_0(theta, phi):
    return (
        -75 / 64 * np.cos(phi - 3 / 2 * theta)
        - 75 / 32 * np.cos(phi + (1 / 2) * theta)
        - 25 / 64 * np.cos(phi + (5 / 2) * theta)
        - 175 / 64 * np.cos(3 * phi - 1 / 2 * theta)
        - 175 / 96 * np.cos(3 * phi + (3 / 2) * theta)
        - 525 / 64 * np.cos(5 * phi + (1 / 2) * theta)
    )


def M_gv_108_5_1():
    def func(k):
        return -8 / 945 * k**3

    return func


def N_gv_108_5_1(theta, phi):
    return (
        -45 / 32 * np.cos(phi - 3 / 2 * theta)
        - 135 / 64 * np.cos(phi + (1 / 2) * theta)
        - 105 / 128 * np.cos(3 * phi - 5 / 2 * theta)
        - 105 / 32 * np.cos(3 * phi - 1 / 2 * theta)
        - 945 / 128 * np.cos(5 * phi - 3 / 2 * theta)
    )


def M_gv_109_1_0():
    def func(k):
        return -1 / 18 * k

    return func


def N_gv_109_1_0(theta, phi):
    return -3 * np.cos(phi + (1 / 2) * theta)


def M_gv_109_1_1():
    def func(k):
        return -1 / 45 * k

    return func


def N_gv_109_1_1(theta, phi):
    return -9 / 2 * np.cos(phi - 3 / 2 * theta) - 3 / 2 * np.cos(phi + (1 / 2) * theta)


def M_gv_109_3_0():
    def func(k):
        return -1 / 45 * k

    return func


def N_gv_109_3_0(theta, phi):
    return (
        -9 / 8 * np.cos(phi - 3 / 2 * theta)
        - 9 / 4 * np.cos(phi + (1 / 2) * theta)
        - 45 / 8 * np.cos(3 * phi - 1 / 2 * theta)
    )


def M_gv_110_1_0():
    def func(k):
        return -1 / 18 * k

    return func


def N_gv_110_1_0(theta, phi):
    return -3 * np.cos(phi + (1 / 2) * theta)


def M_gv_110_1_1():
    def func(k):
        return -1 / 45 * k

    return func


def N_gv_110_1_1(theta, phi):
    return -9 / 2 * np.cos(phi - 3 / 2 * theta) - 3 / 2 * np.cos(phi + (1 / 2) * theta)


def M_gv_110_3_0():
    def func(k):
        return -1 / 45 * k

    return func


def N_gv_110_3_0(theta, phi):
    return (
        -9 / 8 * np.cos(phi - 3 / 2 * theta)
        - 9 / 4 * np.cos(phi + (1 / 2) * theta)
        - 45 / 8 * np.cos(3 * phi - 1 / 2 * theta)
    )


def M_gv_111_1_0():
    def func(k):
        return -1 / 30 * k

    return func


def N_gv_111_1_0(theta, phi):
    return -3 * np.cos(phi + (1 / 2) * theta)


def M_gv_111_1_1():
    def func(k):
        return -17 / 1050 * k

    return func


def N_gv_111_1_1(theta, phi):
    return -9 / 2 * np.cos(phi - 3 / 2 * theta) - 3 / 2 * np.cos(phi + (1 / 2) * theta)


def M_gv_111_1_2():
    def func(k):
        return -1 / 525 * k

    return func


def N_gv_111_1_2(theta, phi):
    return (
        -9 / 8 * np.cos(phi - 3 / 2 * theta)
        - 9 / 4 * np.cos(phi + (1 / 2) * theta)
        - 45 / 8 * np.cos(phi + (5 / 2) * theta)
    )


def M_gv_111_3_0():
    def func(k):
        return -1 / 210 * k

    return func


def N_gv_111_3_0(theta, phi):
    return -21 / 8 * np.cos(phi + (1 / 2) * theta) - 35 / 8 * np.cos(
        3 * phi + (3 / 2) * theta
    )


def M_gv_111_3_1():
    def func(k):
        return -17 / 1050 * k

    return func


def N_gv_111_3_1(theta, phi):
    return (
        -9 / 8 * np.cos(phi - 3 / 2 * theta)
        - 9 / 4 * np.cos(phi + (1 / 2) * theta)
        - 45 / 8 * np.cos(3 * phi - 1 / 2 * theta)
    )


def M_gv_111_3_2():
    def func(k):
        return -1 / 525 * k

    return func


def N_gv_111_3_2(theta, phi):
    return (
        -21 / 8 * np.cos(phi - 3 / 2 * theta)
        + (21 / 16) * np.cos(phi + (1 / 2) * theta)
        - 35 / 16 * np.cos(phi + (5 / 2) * theta)
        - 35 / 16 * np.cos(3 * phi - 1 / 2 * theta)
        - 175 / 48 * np.cos(3 * phi + (3 / 2) * theta)
    )


def M_gv_111_3_3():
    def func(k):
        return -2 / 945 * k

    return func


def N_gv_111_3_3(theta, phi):
    return (
        -45 / 16 * np.cos(phi - 3 / 2 * theta)
        - 27 / 16 * np.cos(phi + (1 / 2) * theta)
        - 105 / 16 * np.cos(3 * phi - 5 / 2 * theta)
        - 15 / 16 * np.cos(3 * phi - 1 / 2 * theta)
    )


def M_gv_111_5_0():
    def func(k):
        return -1 / 525 * k

    return func


def N_gv_111_5_0(theta, phi):
    return (
        -75 / 64 * np.cos(phi - 3 / 2 * theta)
        - 75 / 32 * np.cos(phi + (1 / 2) * theta)
        - 25 / 64 * np.cos(phi + (5 / 2) * theta)
        - 175 / 64 * np.cos(3 * phi - 1 / 2 * theta)
        - 175 / 96 * np.cos(3 * phi + (3 / 2) * theta)
        - 525 / 64 * np.cos(5 * phi + (1 / 2) * theta)
    )


def M_gv_111_5_1():
    def func(k):
        return -2 / 945 * k

    return func


def N_gv_111_5_1(theta, phi):
    return (
        -45 / 32 * np.cos(phi - 3 / 2 * theta)
        - 135 / 64 * np.cos(phi + (1 / 2) * theta)
        - 105 / 128 * np.cos(3 * phi - 5 / 2 * theta)
        - 105 / 32 * np.cos(3 * phi - 1 / 2 * theta)
        - 945 / 128 * np.cos(5 * phi - 3 / 2 * theta)
    )


def M_gv_112_1_0():
    def func(k):
        return -1 / 30 * k

    return func


def N_gv_112_1_0(theta, phi):
    return -3 * np.cos(phi + (1 / 2) * theta)


def M_gv_112_1_1():
    def func(k):
        return -17 / 1050 * k

    return func


def N_gv_112_1_1(theta, phi):
    return -9 / 2 * np.cos(phi - 3 / 2 * theta) - 3 / 2 * np.cos(phi + (1 / 2) * theta)


def M_gv_112_1_2():
    def func(k):
        return -1 / 525 * k

    return func


def N_gv_112_1_2(theta, phi):
    return (
        -9 / 8 * np.cos(phi - 3 / 2 * theta)
        - 9 / 4 * np.cos(phi + (1 / 2) * theta)
        - 45 / 8 * np.cos(phi + (5 / 2) * theta)
    )


def M_gv_112_3_0():
    def func(k):
        return -1 / 210 * k

    return func


def N_gv_112_3_0(theta, phi):
    return -21 / 8 * np.cos(phi + (1 / 2) * theta) - 35 / 8 * np.cos(
        3 * phi + (3 / 2) * theta
    )


def M_gv_112_3_1():
    def func(k):
        return -17 / 1050 * k

    return func


def N_gv_112_3_1(theta, phi):
    return (
        -9 / 8 * np.cos(phi - 3 / 2 * theta)
        - 9 / 4 * np.cos(phi + (1 / 2) * theta)
        - 45 / 8 * np.cos(3 * phi - 1 / 2 * theta)
    )


def M_gv_112_3_2():
    def func(k):
        return -1 / 525 * k

    return func


def N_gv_112_3_2(theta, phi):
    return (
        -21 / 8 * np.cos(phi - 3 / 2 * theta)
        + (21 / 16) * np.cos(phi + (1 / 2) * theta)
        - 35 / 16 * np.cos(phi + (5 / 2) * theta)
        - 35 / 16 * np.cos(3 * phi - 1 / 2 * theta)
        - 175 / 48 * np.cos(3 * phi + (3 / 2) * theta)
    )


def M_gv_112_3_3():
    def func(k):
        return -2 / 945 * k

    return func


def N_gv_112_3_3(theta, phi):
    return (
        -45 / 16 * np.cos(phi - 3 / 2 * theta)
        - 27 / 16 * np.cos(phi + (1 / 2) * theta)
        - 105 / 16 * np.cos(3 * phi - 5 / 2 * theta)
        - 15 / 16 * np.cos(3 * phi - 1 / 2 * theta)
    )


def M_gv_112_5_0():
    def func(k):
        return -1 / 525 * k

    return func


def N_gv_112_5_0(theta, phi):
    return (
        -75 / 64 * np.cos(phi - 3 / 2 * theta)
        - 75 / 32 * np.cos(phi + (1 / 2) * theta)
        - 25 / 64 * np.cos(phi + (5 / 2) * theta)
        - 175 / 64 * np.cos(3 * phi - 1 / 2 * theta)
        - 175 / 96 * np.cos(3 * phi + (3 / 2) * theta)
        - 525 / 64 * np.cos(5 * phi + (1 / 2) * theta)
    )


def M_gv_112_5_1():
    def func(k):
        return -2 / 945 * k

    return func


def N_gv_112_5_1(theta, phi):
    return (
        -45 / 32 * np.cos(phi - 3 / 2 * theta)
        - 135 / 64 * np.cos(phi + (1 / 2) * theta)
        - 105 / 128 * np.cos(3 * phi - 5 / 2 * theta)
        - 105 / 32 * np.cos(3 * phi - 1 / 2 * theta)
        - 945 / 128 * np.cos(5 * phi - 3 / 2 * theta)
    )


def M_gv_113_1_0():
    def func(k):
        return -(77 / 240) / k

    return func


def N_gv_113_1_0(theta, phi):
    return -3 * np.cos(phi + (1 / 2) * theta)


def M_gv_113_1_1():
    def func(k):
        return -(187 / 1200) / k

    return func


def N_gv_113_1_1(theta, phi):
    return -9 / 2 * np.cos(phi - 3 / 2 * theta) - 3 / 2 * np.cos(phi + (1 / 2) * theta)


def M_gv_113_1_2():
    def func(k):
        return -(11 / 600) / k

    return func


def N_gv_113_1_2(theta, phi):
    return (
        -9 / 8 * np.cos(phi - 3 / 2 * theta)
        - 9 / 4 * np.cos(phi + (1 / 2) * theta)
        - 45 / 8 * np.cos(phi + (5 / 2) * theta)
    )


def M_gv_113_3_0():
    def func(k):
        return -(11 / 240) / k

    return func


def N_gv_113_3_0(theta, phi):
    return -21 / 8 * np.cos(phi + (1 / 2) * theta) - 35 / 8 * np.cos(
        3 * phi + (3 / 2) * theta
    )


def M_gv_113_3_1():
    def func(k):
        return -(187 / 1200) / k

    return func


def N_gv_113_3_1(theta, phi):
    return (
        -9 / 8 * np.cos(phi - 3 / 2 * theta)
        - 9 / 4 * np.cos(phi + (1 / 2) * theta)
        - 45 / 8 * np.cos(3 * phi - 1 / 2 * theta)
    )


def M_gv_113_3_2():
    def func(k):
        return -(11 / 600) / k

    return func


def N_gv_113_3_2(theta, phi):
    return (
        -21 / 8 * np.cos(phi - 3 / 2 * theta)
        + (21 / 16) * np.cos(phi + (1 / 2) * theta)
        - 35 / 16 * np.cos(phi + (5 / 2) * theta)
        - 35 / 16 * np.cos(3 * phi - 1 / 2 * theta)
        - 175 / 48 * np.cos(3 * phi + (3 / 2) * theta)
    )


def M_gv_113_3_3():
    def func(k):
        return -(11 / 540) / k

    return func


def N_gv_113_3_3(theta, phi):
    return (
        -45 / 16 * np.cos(phi - 3 / 2 * theta)
        - 27 / 16 * np.cos(phi + (1 / 2) * theta)
        - 105 / 16 * np.cos(3 * phi - 5 / 2 * theta)
        - 15 / 16 * np.cos(3 * phi - 1 / 2 * theta)
    )


def M_gv_113_5_0():
    def func(k):
        return -(11 / 600) / k

    return func


def N_gv_113_5_0(theta, phi):
    return (
        -75 / 64 * np.cos(phi - 3 / 2 * theta)
        - 75 / 32 * np.cos(phi + (1 / 2) * theta)
        - 25 / 64 * np.cos(phi + (5 / 2) * theta)
        - 175 / 64 * np.cos(3 * phi - 1 / 2 * theta)
        - 175 / 96 * np.cos(3 * phi + (3 / 2) * theta)
        - 525 / 64 * np.cos(5 * phi + (1 / 2) * theta)
    )


def M_gv_113_5_1():
    def func(k):
        return -(11 / 540) / k

    return func


def N_gv_113_5_1(theta, phi):
    return (
        -45 / 32 * np.cos(phi - 3 / 2 * theta)
        - 135 / 64 * np.cos(phi + (1 / 2) * theta)
        - 105 / 128 * np.cos(3 * phi - 5 / 2 * theta)
        - 105 / 32 * np.cos(3 * phi - 1 / 2 * theta)
        - 945 / 128 * np.cos(5 * phi - 3 / 2 * theta)
    )


def M_gv_114_1_0():
    def func(k):
        return -(77 / 480) / k

    return func


def N_gv_114_1_0(theta, phi):
    return 3 * np.cos(phi - 3 / 2 * theta) + 3 * np.cos(phi + (1 / 2) * theta)


def M_gv_114_1_1():
    def func(k):
        return -(77 / 1200) / k

    return func


def N_gv_114_1_1(theta, phi):
    return (
        (3 / 2) * np.cos(phi - 3 / 2 * theta)
        + 6 * np.cos(phi + (1 / 2) * theta)
        + (9 / 2) * np.cos(phi + (5 / 2) * theta)
    )


def M_gv_114_1_2():
    def func(k):
        return -(11 / 600) / k

    return func


def N_gv_114_1_2(theta, phi):
    return (
        (45 / 8) * np.cos(phi - 7 / 2 * theta)
        + (63 / 8) * np.cos(phi - 3 / 2 * theta)
        + (27 / 8) * np.cos(phi + (1 / 2) * theta)
        + (9 / 8) * np.cos(phi + (5 / 2) * theta)
    )


def M_gv_114_3_0():
    def func(k):
        return -(77 / 1200) / k

    return func


def N_gv_114_3_0(theta, phi):
    return (
        (9 / 4) * np.cos(phi - 3 / 2 * theta)
        + (27 / 8) * np.cos(phi + (1 / 2) * theta)
        + (9 / 8) * np.cos(phi + (5 / 2) * theta)
        + (45 / 8) * np.cos(3 * phi - 1 / 2 * theta)
        + (45 / 8) * np.cos(3 * phi + (3 / 2) * theta)
    )


def M_gv_114_3_1():
    def func(k):
        return -(11 / 240) / k

    return func


def N_gv_114_3_1(theta, phi):
    return (
        (21 / 8) * np.cos(phi - 3 / 2 * theta)
        + (21 / 8) * np.cos(phi + (1 / 2) * theta)
        + (35 / 8) * np.cos(3 * phi - 5 / 2 * theta)
        + (35 / 8) * np.cos(3 * phi - 1 / 2 * theta)
    )


def M_gv_114_3_2():
    def func(k):
        return -(11 / 600) / k

    return func


def N_gv_114_3_2(theta, phi):
    return (
        (35 / 16) * np.cos(phi - 7 / 2 * theta)
        + (7 / 8) * np.cos(phi - 3 / 2 * theta)
        + (21 / 16) * np.cos(phi + (1 / 2) * theta)
        + (21 / 8) * np.cos(phi + (5 / 2) * theta)
        + (175 / 48) * np.cos(3 * phi - 5 / 2 * theta)
        + (35 / 6) * np.cos(3 * phi - 1 / 2 * theta)
        + (35 / 16) * np.cos(3 * phi + (3 / 2) * theta)
    )


def M_gv_114_5_0():
    def func(k):
        return -(11 / 600) / k

    return func


def N_gv_114_5_0(theta, phi):
    return (
        (25 / 64) * np.cos(phi - 7 / 2 * theta)
        + (175 / 64) * np.cos(phi - 3 / 2 * theta)
        + (225 / 64) * np.cos(phi + (1 / 2) * theta)
        + (75 / 64) * np.cos(phi + (5 / 2) * theta)
        + (175 / 96) * np.cos(3 * phi - 5 / 2 * theta)
        + (875 / 192) * np.cos(3 * phi - 1 / 2 * theta)
        + (175 / 64) * np.cos(3 * phi + (3 / 2) * theta)
        + (525 / 64) * np.cos(5 * phi - 3 / 2 * theta)
        + (525 / 64) * np.cos(5 * phi + (1 / 2) * theta)
    )


def M_gv_115_1_0():
    def func(k):
        return (7 / 40) / k

    return func


def N_gv_115_1_0(theta, phi):
    return -3 * np.cos(phi + (1 / 2) * theta)


def M_gv_115_1_1():
    def func(k):
        return (17 / 200) / k

    return func


def N_gv_115_1_1(theta, phi):
    return -9 / 2 * np.cos(phi - 3 / 2 * theta) - 3 / 2 * np.cos(phi + (1 / 2) * theta)


def M_gv_115_1_2():
    def func(k):
        return (1 / 100) / k

    return func


def N_gv_115_1_2(theta, phi):
    return (
        -9 / 8 * np.cos(phi - 3 / 2 * theta)
        - 9 / 4 * np.cos(phi + (1 / 2) * theta)
        - 45 / 8 * np.cos(phi + (5 / 2) * theta)
    )


def M_gv_115_3_0():
    def func(k):
        return (1 / 40) / k

    return func


def N_gv_115_3_0(theta, phi):
    return -21 / 8 * np.cos(phi + (1 / 2) * theta) - 35 / 8 * np.cos(
        3 * phi + (3 / 2) * theta
    )


def M_gv_115_3_1():
    def func(k):
        return (17 / 200) / k

    return func


def N_gv_115_3_1(theta, phi):
    return (
        -9 / 8 * np.cos(phi - 3 / 2 * theta)
        - 9 / 4 * np.cos(phi + (1 / 2) * theta)
        - 45 / 8 * np.cos(3 * phi - 1 / 2 * theta)
    )


def M_gv_115_3_2():
    def func(k):
        return (1 / 100) / k

    return func


def N_gv_115_3_2(theta, phi):
    return (
        -21 / 8 * np.cos(phi - 3 / 2 * theta)
        + (21 / 16) * np.cos(phi + (1 / 2) * theta)
        - 35 / 16 * np.cos(phi + (5 / 2) * theta)
        - 35 / 16 * np.cos(3 * phi - 1 / 2 * theta)
        - 175 / 48 * np.cos(3 * phi + (3 / 2) * theta)
    )


def M_gv_115_3_3():
    def func(k):
        return (1 / 90) / k

    return func


def N_gv_115_3_3(theta, phi):
    return (
        -45 / 16 * np.cos(phi - 3 / 2 * theta)
        - 27 / 16 * np.cos(phi + (1 / 2) * theta)
        - 105 / 16 * np.cos(3 * phi - 5 / 2 * theta)
        - 15 / 16 * np.cos(3 * phi - 1 / 2 * theta)
    )


def M_gv_115_5_0():
    def func(k):
        return (1 / 100) / k

    return func


def N_gv_115_5_0(theta, phi):
    return (
        -75 / 64 * np.cos(phi - 3 / 2 * theta)
        - 75 / 32 * np.cos(phi + (1 / 2) * theta)
        - 25 / 64 * np.cos(phi + (5 / 2) * theta)
        - 175 / 64 * np.cos(3 * phi - 1 / 2 * theta)
        - 175 / 96 * np.cos(3 * phi + (3 / 2) * theta)
        - 525 / 64 * np.cos(5 * phi + (1 / 2) * theta)
    )


def M_gv_115_5_1():
    def func(k):
        return (1 / 90) / k

    return func


def N_gv_115_5_1(theta, phi):
    return (
        -45 / 32 * np.cos(phi - 3 / 2 * theta)
        - 135 / 64 * np.cos(phi + (1 / 2) * theta)
        - 105 / 128 * np.cos(3 * phi - 5 / 2 * theta)
        - 105 / 32 * np.cos(3 * phi - 1 / 2 * theta)
        - 945 / 128 * np.cos(5 * phi - 3 / 2 * theta)
    )


def M_gv_116_1_0():
    def func(k):
        return (7 / 80) / k

    return func


def N_gv_116_1_0(theta, phi):
    return 3 * np.cos(phi - 3 / 2 * theta) + 3 * np.cos(phi + (1 / 2) * theta)


def M_gv_116_1_1():
    def func(k):
        return (7 / 200) / k

    return func


def N_gv_116_1_1(theta, phi):
    return (
        (3 / 2) * np.cos(phi - 3 / 2 * theta)
        + 6 * np.cos(phi + (1 / 2) * theta)
        + (9 / 2) * np.cos(phi + (5 / 2) * theta)
    )


def M_gv_116_1_2():
    def func(k):
        return (1 / 100) / k

    return func


def N_gv_116_1_2(theta, phi):
    return (
        (45 / 8) * np.cos(phi - 7 / 2 * theta)
        + (63 / 8) * np.cos(phi - 3 / 2 * theta)
        + (27 / 8) * np.cos(phi + (1 / 2) * theta)
        + (9 / 8) * np.cos(phi + (5 / 2) * theta)
    )


def M_gv_116_3_0():
    def func(k):
        return (7 / 200) / k

    return func


def N_gv_116_3_0(theta, phi):
    return (
        (9 / 4) * np.cos(phi - 3 / 2 * theta)
        + (27 / 8) * np.cos(phi + (1 / 2) * theta)
        + (9 / 8) * np.cos(phi + (5 / 2) * theta)
        + (45 / 8) * np.cos(3 * phi - 1 / 2 * theta)
        + (45 / 8) * np.cos(3 * phi + (3 / 2) * theta)
    )


def M_gv_116_3_1():
    def func(k):
        return (1 / 40) / k

    return func


def N_gv_116_3_1(theta, phi):
    return (
        (21 / 8) * np.cos(phi - 3 / 2 * theta)
        + (21 / 8) * np.cos(phi + (1 / 2) * theta)
        + (35 / 8) * np.cos(3 * phi - 5 / 2 * theta)
        + (35 / 8) * np.cos(3 * phi - 1 / 2 * theta)
    )


def M_gv_116_3_2():
    def func(k):
        return (1 / 100) / k

    return func


def N_gv_116_3_2(theta, phi):
    return (
        (35 / 16) * np.cos(phi - 7 / 2 * theta)
        + (7 / 8) * np.cos(phi - 3 / 2 * theta)
        + (21 / 16) * np.cos(phi + (1 / 2) * theta)
        + (21 / 8) * np.cos(phi + (5 / 2) * theta)
        + (175 / 48) * np.cos(3 * phi - 5 / 2 * theta)
        + (35 / 6) * np.cos(3 * phi - 1 / 2 * theta)
        + (35 / 16) * np.cos(3 * phi + (3 / 2) * theta)
    )


def M_gv_116_5_0():
    def func(k):
        return (1 / 100) / k

    return func


def N_gv_116_5_0(theta, phi):
    return (
        (25 / 64) * np.cos(phi - 7 / 2 * theta)
        + (175 / 64) * np.cos(phi - 3 / 2 * theta)
        + (225 / 64) * np.cos(phi + (1 / 2) * theta)
        + (75 / 64) * np.cos(phi + (5 / 2) * theta)
        + (175 / 96) * np.cos(3 * phi - 5 / 2 * theta)
        + (875 / 192) * np.cos(3 * phi - 1 / 2 * theta)
        + (175 / 64) * np.cos(3 * phi + (3 / 2) * theta)
        + (525 / 64) * np.cos(5 * phi - 3 / 2 * theta)
        + (525 / 64) * np.cos(5 * phi + (1 / 2) * theta)
    )


def M_gv_117_1_0():
    def func(k):
        return (7 / 48) / k

    return func


def N_gv_117_1_0(theta, phi):
    return -3 * np.cos(phi + (1 / 2) * theta)


def M_gv_117_1_1():
    def func(k):
        return (17 / 240) / k

    return func


def N_gv_117_1_1(theta, phi):
    return -9 / 2 * np.cos(phi - 3 / 2 * theta) - 3 / 2 * np.cos(phi + (1 / 2) * theta)


def M_gv_117_1_2():
    def func(k):
        return (1 / 120) / k

    return func


def N_gv_117_1_2(theta, phi):
    return (
        -9 / 8 * np.cos(phi - 3 / 2 * theta)
        - 9 / 4 * np.cos(phi + (1 / 2) * theta)
        - 45 / 8 * np.cos(phi + (5 / 2) * theta)
    )


def M_gv_117_3_0():
    def func(k):
        return (1 / 48) / k

    return func


def N_gv_117_3_0(theta, phi):
    return -21 / 8 * np.cos(phi + (1 / 2) * theta) - 35 / 8 * np.cos(
        3 * phi + (3 / 2) * theta
    )


def M_gv_117_3_1():
    def func(k):
        return (17 / 240) / k

    return func


def N_gv_117_3_1(theta, phi):
    return (
        -9 / 8 * np.cos(phi - 3 / 2 * theta)
        - 9 / 4 * np.cos(phi + (1 / 2) * theta)
        - 45 / 8 * np.cos(3 * phi - 1 / 2 * theta)
    )


def M_gv_117_3_2():
    def func(k):
        return (1 / 120) / k

    return func


def N_gv_117_3_2(theta, phi):
    return (
        -21 / 8 * np.cos(phi - 3 / 2 * theta)
        + (21 / 16) * np.cos(phi + (1 / 2) * theta)
        - 35 / 16 * np.cos(phi + (5 / 2) * theta)
        - 35 / 16 * np.cos(3 * phi - 1 / 2 * theta)
        - 175 / 48 * np.cos(3 * phi + (3 / 2) * theta)
    )


def M_gv_117_3_3():
    def func(k):
        return (1 / 108) / k

    return func


def N_gv_117_3_3(theta, phi):
    return (
        -45 / 16 * np.cos(phi - 3 / 2 * theta)
        - 27 / 16 * np.cos(phi + (1 / 2) * theta)
        - 105 / 16 * np.cos(3 * phi - 5 / 2 * theta)
        - 15 / 16 * np.cos(3 * phi - 1 / 2 * theta)
    )


def M_gv_117_5_0():
    def func(k):
        return (1 / 120) / k

    return func


def N_gv_117_5_0(theta, phi):
    return (
        -75 / 64 * np.cos(phi - 3 / 2 * theta)
        - 75 / 32 * np.cos(phi + (1 / 2) * theta)
        - 25 / 64 * np.cos(phi + (5 / 2) * theta)
        - 175 / 64 * np.cos(3 * phi - 1 / 2 * theta)
        - 175 / 96 * np.cos(3 * phi + (3 / 2) * theta)
        - 525 / 64 * np.cos(5 * phi + (1 / 2) * theta)
    )


def M_gv_117_5_1():
    def func(k):
        return (1 / 108) / k

    return func


def N_gv_117_5_1(theta, phi):
    return (
        -45 / 32 * np.cos(phi - 3 / 2 * theta)
        - 135 / 64 * np.cos(phi + (1 / 2) * theta)
        - 105 / 128 * np.cos(3 * phi - 5 / 2 * theta)
        - 105 / 32 * np.cos(3 * phi - 1 / 2 * theta)
        - 945 / 128 * np.cos(5 * phi - 3 / 2 * theta)
    )


def M_gv_118_1_0():
    def func(k):
        return (7 / 96) / k

    return func


def N_gv_118_1_0(theta, phi):
    return 3 * np.cos(phi - 3 / 2 * theta) + 3 * np.cos(phi + (1 / 2) * theta)


def M_gv_118_1_1():
    def func(k):
        return (7 / 240) / k

    return func


def N_gv_118_1_1(theta, phi):
    return (
        (3 / 2) * np.cos(phi - 3 / 2 * theta)
        + 6 * np.cos(phi + (1 / 2) * theta)
        + (9 / 2) * np.cos(phi + (5 / 2) * theta)
    )


def M_gv_118_1_2():
    def func(k):
        return (1 / 120) / k

    return func


def N_gv_118_1_2(theta, phi):
    return (
        (45 / 8) * np.cos(phi - 7 / 2 * theta)
        + (63 / 8) * np.cos(phi - 3 / 2 * theta)
        + (27 / 8) * np.cos(phi + (1 / 2) * theta)
        + (9 / 8) * np.cos(phi + (5 / 2) * theta)
    )


def M_gv_118_3_0():
    def func(k):
        return (7 / 240) / k

    return func


def N_gv_118_3_0(theta, phi):
    return (
        (9 / 4) * np.cos(phi - 3 / 2 * theta)
        + (27 / 8) * np.cos(phi + (1 / 2) * theta)
        + (9 / 8) * np.cos(phi + (5 / 2) * theta)
        + (45 / 8) * np.cos(3 * phi - 1 / 2 * theta)
        + (45 / 8) * np.cos(3 * phi + (3 / 2) * theta)
    )


def M_gv_118_3_1():
    def func(k):
        return (1 / 48) / k

    return func


def N_gv_118_3_1(theta, phi):
    return (
        (21 / 8) * np.cos(phi - 3 / 2 * theta)
        + (21 / 8) * np.cos(phi + (1 / 2) * theta)
        + (35 / 8) * np.cos(3 * phi - 5 / 2 * theta)
        + (35 / 8) * np.cos(3 * phi - 1 / 2 * theta)
    )


def M_gv_118_3_2():
    def func(k):
        return (1 / 120) / k

    return func


def N_gv_118_3_2(theta, phi):
    return (
        (35 / 16) * np.cos(phi - 7 / 2 * theta)
        + (7 / 8) * np.cos(phi - 3 / 2 * theta)
        + (21 / 16) * np.cos(phi + (1 / 2) * theta)
        + (21 / 8) * np.cos(phi + (5 / 2) * theta)
        + (175 / 48) * np.cos(3 * phi - 5 / 2 * theta)
        + (35 / 6) * np.cos(3 * phi - 1 / 2 * theta)
        + (35 / 16) * np.cos(3 * phi + (3 / 2) * theta)
    )


def M_gv_118_5_0():
    def func(k):
        return (1 / 120) / k

    return func


def N_gv_118_5_0(theta, phi):
    return (
        (25 / 64) * np.cos(phi - 7 / 2 * theta)
        + (175 / 64) * np.cos(phi - 3 / 2 * theta)
        + (225 / 64) * np.cos(phi + (1 / 2) * theta)
        + (75 / 64) * np.cos(phi + (5 / 2) * theta)
        + (175 / 96) * np.cos(3 * phi - 5 / 2 * theta)
        + (875 / 192) * np.cos(3 * phi - 1 / 2 * theta)
        + (175 / 64) * np.cos(3 * phi + (3 / 2) * theta)
        + (525 / 64) * np.cos(5 * phi - 3 / 2 * theta)
        + (525 / 64) * np.cos(5 * phi + (1 / 2) * theta)
    )


def M_vv_0_0_0():
    def func(k):
        return (1 / 9) / k**2

    return func


def N_vv_0_0_0(theta, phi):
    return 3 * np.cos(theta)


def M_vv_0_2_0():
    def func(k):
        return (1 / 9) / k**2

    return func


def N_vv_0_2_0(theta, phi):
    return (9 / 2) * np.cos(2 * phi) + (3 / 2) * np.cos(theta)


def M_vv_1_0_0():
    def func(k):
        return (2 / 9) / k**2

    return func


def N_vv_1_0_0(theta, phi):
    return 3 * np.cos(theta)


def M_vv_1_2_0():
    def func(k):
        return (2 / 9) / k**2

    return func


def N_vv_1_2_0(theta, phi):
    return (9 / 2) * np.cos(2 * phi) + (3 / 2) * np.cos(theta)


def M_vv_2_0_0():
    def func(k):
        return (4 / 9) / k**2

    return func


def N_vv_2_0_0(theta, phi):
    return 3 * np.cos(theta)


def M_vv_2_2_0():
    def func(k):
        return (4 / 9) / k**2

    return func


def N_vv_2_2_0(theta, phi):
    return (9 / 2) * np.cos(2 * phi) + (3 / 2) * np.cos(theta)


def M_vv_3_0_0():
    def func(k):
        return k ** (-2)

    return func


def N_vv_3_0_0(theta, phi):
    return np.cos(theta)


def M_vv_4_0_0():
    def func(k):
        return (1 / 9) / k**2

    return func


def N_vv_4_0_0(theta, phi):
    return 3 * np.cos(theta)


def M_vv_4_2_0():
    def func(k):
        return (1 / 9) / k**2

    return func


def N_vv_4_2_0(theta, phi):
    return (9 / 2) * np.cos(2 * phi) + (3 / 2) * np.cos(theta)


def M_vv_5_0_0():
    def func(k):
        return 4 / 3

    return func


def N_vv_5_0_0(theta, phi):
    return 3 * np.cos(theta)


def M_vv_5_2_0():
    def func(k):
        return 4 / 3

    return func


def N_vv_5_2_0(theta, phi):
    return (9 / 2) * np.cos(2 * phi) + (3 / 2) * np.cos(theta)


def M_vv_6_0_0():
    def func(k):
        return 2 / 3

    return func


def N_vv_6_0_0(theta, phi):
    return 3 * np.cos(theta)


def M_vv_6_2_0():
    def func(k):
        return 2 / 3

    return func


def N_vv_6_2_0(theta, phi):
    return (9 / 2) * np.cos(2 * phi) + (3 / 2) * np.cos(theta)


def M_vv_7_0_0():
    def func(k):
        return (2 / 9) / k**2

    return func


def N_vv_7_0_0(theta, phi):
    return 3 * np.cos(theta)


def M_vv_7_2_0():
    def func(k):
        return (2 / 9) / k**2

    return func


def N_vv_7_2_0(theta, phi):
    return (9 / 2) * np.cos(2 * phi) + (3 / 2) * np.cos(theta)


def M_vv_8_0_0():
    def func(k):
        return (2 / 15) / k**2

    return func


def N_vv_8_0_0(theta, phi):
    return 3 * np.cos(theta)


def M_vv_8_2_0():
    def func(k):
        return (2 / 15) / k**2

    return func


def N_vv_8_2_0(theta, phi):
    return (9 / 2) * np.cos(2 * phi) + (3 / 2) * np.cos(theta)


def M_vv_8_2_1():
    def func(k):
        return (2 / 105) / k**2

    return func


def N_vv_8_2_1(theta, phi):
    return (
        (9 / 8) * np.cos(2 * phi)
        + (9 / 4) * np.cos(theta)
        + (45 / 8) * np.cos(2 * phi + 2 * theta)
    )


def M_vv_8_2_2():
    def func(k):
        return (2 / 105) / k**2

    return func


def N_vv_8_2_2(theta, phi):
    return (
        (9 / 8) * np.cos(2 * phi)
        + (9 / 4) * np.cos(theta)
        + (45 / 8) * np.cos(2 * phi - 2 * theta)
    )


def M_vv_8_4_0():
    def func(k):
        return (2 / 105) / k**2

    return func


def N_vv_8_4_0(theta, phi):
    return (
        (45 / 16) * np.cos(2 * phi)
        + (27 / 16) * np.cos(theta)
        + (15 / 16) * np.cos(2 * phi + 2 * theta)
        + (105 / 16) * np.cos(4 * phi + theta)
    )


def M_vv_8_4_1():
    def func(k):
        return (2 / 105) / k**2

    return func


def N_vv_8_4_1(theta, phi):
    return (
        (45 / 16) * np.cos(2 * phi)
        + (27 / 16) * np.cos(theta)
        + (15 / 16) * np.cos(2 * phi - 2 * theta)
        + (105 / 16) * np.cos(4 * phi - theta)
    )


def M_vv_9_0_0():
    def func(k):
        return -(2 / 9) / k**2

    return func


def N_vv_9_0_0(theta, phi):
    return 3 * np.cos(theta)


def M_vv_9_2_0():
    def func(k):
        return -(2 / 9) / k**2

    return func


def N_vv_9_2_0(theta, phi):
    return (9 / 2) * np.cos(2 * phi) + (3 / 2) * np.cos(theta)


def M_vv_10_0_0():
    def func(k):
        return -(2 / 15) / k**2

    return func


def N_vv_10_0_0(theta, phi):
    return 3 * np.cos(theta)


def M_vv_10_2_0():
    def func(k):
        return -(2 / 15) / k**2

    return func


def N_vv_10_2_0(theta, phi):
    return (9 / 2) * np.cos(2 * phi) + (3 / 2) * np.cos(theta)


def M_vv_10_2_1():
    def func(k):
        return -(2 / 105) / k**2

    return func


def N_vv_10_2_1(theta, phi):
    return (
        (9 / 8) * np.cos(2 * phi)
        + (9 / 4) * np.cos(theta)
        + (45 / 8) * np.cos(2 * phi + 2 * theta)
    )


def M_vv_10_2_2():
    def func(k):
        return -(2 / 105) / k**2

    return func


def N_vv_10_2_2(theta, phi):
    return (
        (9 / 8) * np.cos(2 * phi)
        + (9 / 4) * np.cos(theta)
        + (45 / 8) * np.cos(2 * phi - 2 * theta)
    )


def M_vv_10_4_0():
    def func(k):
        return -(2 / 105) / k**2

    return func


def N_vv_10_4_0(theta, phi):
    return (
        (45 / 16) * np.cos(2 * phi)
        + (27 / 16) * np.cos(theta)
        + (15 / 16) * np.cos(2 * phi + 2 * theta)
        + (105 / 16) * np.cos(4 * phi + theta)
    )


def M_vv_10_4_1():
    def func(k):
        return -(2 / 105) / k**2

    return func


def N_vv_10_4_1(theta, phi):
    return (
        (45 / 16) * np.cos(2 * phi)
        + (27 / 16) * np.cos(theta)
        + (15 / 16) * np.cos(2 * phi - 2 * theta)
        + (105 / 16) * np.cos(4 * phi - theta)
    )


def M_vv_11_0_0():
    def func(k):
        return 4 / 9

    return func


def N_vv_11_0_0(theta, phi):
    return 3 * np.cos(theta)


def M_vv_11_2_0():
    def func(k):
        return 4 / 9

    return func


def N_vv_11_2_0(theta, phi):
    return (9 / 2) * np.cos(2 * phi) + (3 / 2) * np.cos(theta)


def M_vv_12_0_0():
    def func(k):
        return 4 / 15

    return func


def N_vv_12_0_0(theta, phi):
    return 3 * np.cos(theta)


def M_vv_12_2_0():
    def func(k):
        return 4 / 15

    return func


def N_vv_12_2_0(theta, phi):
    return (9 / 2) * np.cos(2 * phi) + (3 / 2) * np.cos(theta)


def M_vv_12_2_1():
    def func(k):
        return 4 / 105

    return func


def N_vv_12_2_1(theta, phi):
    return (
        (9 / 8) * np.cos(2 * phi)
        + (9 / 4) * np.cos(theta)
        + (45 / 8) * np.cos(2 * phi + 2 * theta)
    )


def M_vv_12_2_2():
    def func(k):
        return 4 / 105

    return func


def N_vv_12_2_2(theta, phi):
    return (
        (9 / 8) * np.cos(2 * phi)
        + (9 / 4) * np.cos(theta)
        + (45 / 8) * np.cos(2 * phi - 2 * theta)
    )


def M_vv_12_4_0():
    def func(k):
        return 4 / 105

    return func


def N_vv_12_4_0(theta, phi):
    return (
        (45 / 16) * np.cos(2 * phi)
        + (27 / 16) * np.cos(theta)
        + (15 / 16) * np.cos(2 * phi + 2 * theta)
        + (105 / 16) * np.cos(4 * phi + theta)
    )


def M_vv_12_4_1():
    def func(k):
        return 4 / 105

    return func


def N_vv_12_4_1(theta, phi):
    return (
        (45 / 16) * np.cos(2 * phi)
        + (27 / 16) * np.cos(theta)
        + (15 / 16) * np.cos(2 * phi - 2 * theta)
        + (105 / 16) * np.cos(4 * phi - theta)
    )


def M_vv_13_0_0():
    def func(k):
        return -2 / 9

    return func


def N_vv_13_0_0(theta, phi):
    return 3 * np.cos(theta)


def M_vv_13_2_0():
    def func(k):
        return -2 / 9

    return func


def N_vv_13_2_0(theta, phi):
    return (9 / 2) * np.cos(2 * phi) + (3 / 2) * np.cos(theta)


def M_vv_14_0_0():
    def func(k):
        return -4 / 9

    return func


def N_vv_14_0_0(theta, phi):
    return 3 * np.cos(theta)


def M_vv_14_2_0():
    def func(k):
        return -4 / 9

    return func


def N_vv_14_2_0(theta, phi):
    return (9 / 2) * np.cos(2 * phi) + (3 / 2) * np.cos(theta)


def M_vv_15_0_0():
    def func(k):
        return -4 / 9

    return func


def N_vv_15_0_0(theta, phi):
    return 3 * np.cos(theta)


def M_vv_15_2_0():
    def func(k):
        return -4 / 9

    return func


def N_vv_15_2_0(theta, phi):
    return (9 / 2) * np.cos(2 * phi) + (3 / 2) * np.cos(theta)


def M_vv_16_0_0():
    def func(k):
        return -4 / 3 * k**2

    return func


def N_vv_16_0_0(theta, phi):
    return 3 * np.cos(theta)


def M_vv_16_2_0():
    def func(k):
        return -4 / 3 * k**2

    return func


def N_vv_16_2_0(theta, phi):
    return (9 / 2) * np.cos(2 * phi) + (3 / 2) * np.cos(theta)


def M_vv_17_0_0():
    def func(k):
        return -4 / 3 * k**2

    return func


def N_vv_17_0_0(theta, phi):
    return 3 * np.cos(theta)


def M_vv_17_2_0():
    def func(k):
        return -4 / 3 * k**2

    return func


def N_vv_17_2_0(theta, phi):
    return (9 / 2) * np.cos(2 * phi) + (3 / 2) * np.cos(theta)


def M_vv_18_0_0():
    def func(k):
        return 2 / 9

    return func


def N_vv_18_0_0(theta, phi):
    return 3 * np.cos(theta)


def M_vv_18_2_0():
    def func(k):
        return 2 / 9

    return func


def N_vv_18_2_0(theta, phi):
    return (9 / 2) * np.cos(2 * phi) + (3 / 2) * np.cos(theta)


def M_vv_19_0_0():
    def func(k):
        return 2 / 9

    return func


def N_vv_19_0_0(theta, phi):
    return 3 * np.cos(theta)


def M_vv_19_2_0():
    def func(k):
        return 2 / 9

    return func


def N_vv_19_2_0(theta, phi):
    return (9 / 2) * np.cos(2 * phi) + (3 / 2) * np.cos(theta)


def M_vv_20_0_0():
    def func(k):
        return 2 / 9

    return func


def N_vv_20_0_0(theta, phi):
    return 3 * np.cos(theta)


def M_vv_20_2_0():
    def func(k):
        return 2 / 9

    return func


def N_vv_20_2_0(theta, phi):
    return (9 / 2) * np.cos(2 * phi) + (3 / 2) * np.cos(theta)


def M_vv_21_0_0():
    def func(k):
        return 2 / 9

    return func


def N_vv_21_0_0(theta, phi):
    return 3 * np.cos(theta)


def M_vv_21_2_0():
    def func(k):
        return 2 / 9

    return func


def N_vv_21_2_0(theta, phi):
    return (9 / 2) * np.cos(2 * phi) + (3 / 2) * np.cos(theta)


def M_vv_22_0_0():
    def func(k):
        return 2 / 9

    return func


def N_vv_22_0_0(theta, phi):
    return 3 * np.cos(theta)


def M_vv_22_2_0():
    def func(k):
        return 2 / 9

    return func


def N_vv_22_2_0(theta, phi):
    return (9 / 2) * np.cos(2 * phi) + (3 / 2) * np.cos(theta)


def M_vv_23_0_0():
    def func(k):
        return 4 / 9

    return func


def N_vv_23_0_0(theta, phi):
    return 3 * np.cos(theta)


def M_vv_23_2_0():
    def func(k):
        return 4 / 9

    return func


def N_vv_23_2_0(theta, phi):
    return (9 / 2) * np.cos(2 * phi) + (3 / 2) * np.cos(theta)


def M_vv_24_0_0():
    def func(k):
        return (4 / 3) * k**2

    return func


def N_vv_24_0_0(theta, phi):
    return 3 * np.cos(theta)


def M_vv_24_2_0():
    def func(k):
        return (4 / 3) * k**2

    return func


def N_vv_24_2_0(theta, phi):
    return (9 / 2) * np.cos(2 * phi) + (3 / 2) * np.cos(theta)


def M_vv_25_0_0():
    def func(k):
        return (2 / 9) / k**2

    return func


def N_vv_25_0_0(theta, phi):
    return -6 * np.cos(theta)


def M_vv_25_2_0():
    def func(k):
        return (2 / 9) / k**2

    return func


def N_vv_25_2_0(theta, phi):
    return -9 * np.cos(2 * phi) - 3 * np.cos(theta)


def M_vv_26_0_0():
    def func(k):
        return (2 / 3) / k**2

    return func


def N_vv_26_0_0(theta, phi):
    return 2 * np.cos(theta)


def M_vv_26_2_0():
    def func(k):
        return (2 / 15) / k**2

    return func


def N_vv_26_2_0(theta, phi):
    return (
        (15 / 4) * np.cos(2 * phi)
        + (5 / 2) * np.cos(theta)
        + (15 / 4) * np.cos(2 * phi + 2 * theta)
    )


def M_vv_26_2_1():
    def func(k):
        return (2 / 15) / k**2

    return func


def N_vv_26_2_1(theta, phi):
    return (
        (15 / 4) * np.cos(2 * phi)
        + (5 / 2) * np.cos(theta)
        + (15 / 4) * np.cos(2 * phi - 2 * theta)
    )


def M_vv_27_0_0():
    def func(k):
        return -1 / 5

    return func


def N_vv_27_0_0(theta, phi):
    return 3 * np.cos(theta)


def M_vv_27_2_0():
    def func(k):
        return -1 / 5

    return func


def N_vv_27_2_0(theta, phi):
    return (9 / 2) * np.cos(2 * phi) + (3 / 2) * np.cos(theta)


def M_vv_27_2_1():
    def func(k):
        return -1 / 35

    return func


def N_vv_27_2_1(theta, phi):
    return (
        (9 / 8) * np.cos(2 * phi)
        + (9 / 4) * np.cos(theta)
        + (45 / 8) * np.cos(2 * phi + 2 * theta)
    )


def M_vv_27_2_2():
    def func(k):
        return -1 / 35

    return func


def N_vv_27_2_2(theta, phi):
    return (
        (9 / 8) * np.cos(2 * phi)
        + (9 / 4) * np.cos(theta)
        + (45 / 8) * np.cos(2 * phi - 2 * theta)
    )


def M_vv_27_4_0():
    def func(k):
        return -1 / 35

    return func


def N_vv_27_4_0(theta, phi):
    return (
        (45 / 16) * np.cos(2 * phi)
        + (27 / 16) * np.cos(theta)
        + (15 / 16) * np.cos(2 * phi + 2 * theta)
        + (105 / 16) * np.cos(4 * phi + theta)
    )


def M_vv_27_4_1():
    def func(k):
        return -1 / 35

    return func


def N_vv_27_4_1(theta, phi):
    return (
        (45 / 16) * np.cos(2 * phi)
        + (27 / 16) * np.cos(theta)
        + (15 / 16) * np.cos(2 * phi - 2 * theta)
        + (105 / 16) * np.cos(4 * phi - theta)
    )


def M_vv_28_0_0():
    def func(k):
        return -2 / 5

    return func


def N_vv_28_0_0(theta, phi):
    return 3 * np.cos(theta)


def M_vv_28_2_0():
    def func(k):
        return -2 / 5

    return func


def N_vv_28_2_0(theta, phi):
    return (9 / 2) * np.cos(2 * phi) + (3 / 2) * np.cos(theta)


def M_vv_28_2_1():
    def func(k):
        return -2 / 35

    return func


def N_vv_28_2_1(theta, phi):
    return (
        (9 / 8) * np.cos(2 * phi)
        + (9 / 4) * np.cos(theta)
        + (45 / 8) * np.cos(2 * phi + 2 * theta)
    )


def M_vv_28_2_2():
    def func(k):
        return -2 / 35

    return func


def N_vv_28_2_2(theta, phi):
    return (
        (9 / 8) * np.cos(2 * phi)
        + (9 / 4) * np.cos(theta)
        + (45 / 8) * np.cos(2 * phi - 2 * theta)
    )


def M_vv_28_4_0():
    def func(k):
        return -2 / 35

    return func


def N_vv_28_4_0(theta, phi):
    return (
        (45 / 16) * np.cos(2 * phi)
        + (27 / 16) * np.cos(theta)
        + (15 / 16) * np.cos(2 * phi + 2 * theta)
        + (105 / 16) * np.cos(4 * phi + theta)
    )


def M_vv_28_4_1():
    def func(k):
        return -2 / 35

    return func


def N_vv_28_4_1(theta, phi):
    return (
        (45 / 16) * np.cos(2 * phi)
        + (27 / 16) * np.cos(theta)
        + (15 / 16) * np.cos(2 * phi - 2 * theta)
        + (105 / 16) * np.cos(4 * phi - theta)
    )


def M_vv_29_0_0():
    def func(k):
        return -4 / 5

    return func


def N_vv_29_0_0(theta, phi):
    return 3 * np.cos(theta)


def M_vv_29_2_0():
    def func(k):
        return -4 / 5

    return func


def N_vv_29_2_0(theta, phi):
    return (9 / 2) * np.cos(2 * phi) + (3 / 2) * np.cos(theta)


def M_vv_29_2_1():
    def func(k):
        return -4 / 35

    return func


def N_vv_29_2_1(theta, phi):
    return (
        (9 / 8) * np.cos(2 * phi)
        + (9 / 4) * np.cos(theta)
        + (45 / 8) * np.cos(2 * phi + 2 * theta)
    )


def M_vv_29_2_2():
    def func(k):
        return -4 / 35

    return func


def N_vv_29_2_2(theta, phi):
    return (
        (9 / 8) * np.cos(2 * phi)
        + (9 / 4) * np.cos(theta)
        + (45 / 8) * np.cos(2 * phi - 2 * theta)
    )


def M_vv_29_4_0():
    def func(k):
        return -4 / 35

    return func


def N_vv_29_4_0(theta, phi):
    return (
        (45 / 16) * np.cos(2 * phi)
        + (27 / 16) * np.cos(theta)
        + (15 / 16) * np.cos(2 * phi + 2 * theta)
        + (105 / 16) * np.cos(4 * phi + theta)
    )


def M_vv_29_4_1():
    def func(k):
        return -4 / 35

    return func


def N_vv_29_4_1(theta, phi):
    return (
        (45 / 16) * np.cos(2 * phi)
        + (27 / 16) * np.cos(theta)
        + (15 / 16) * np.cos(2 * phi - 2 * theta)
        + (105 / 16) * np.cos(4 * phi - theta)
    )


def M_vv_30_0_0():
    def func(k):
        return -6 / 5 * k**2

    return func


def N_vv_30_0_0(theta, phi):
    return 3 * np.cos(theta)


def M_vv_30_2_0():
    def func(k):
        return -6 / 5 * k**2

    return func


def N_vv_30_2_0(theta, phi):
    return (9 / 2) * np.cos(2 * phi) + (3 / 2) * np.cos(theta)


def M_vv_30_2_1():
    def func(k):
        return -6 / 35 * k**2

    return func


def N_vv_30_2_1(theta, phi):
    return (
        (9 / 8) * np.cos(2 * phi)
        + (9 / 4) * np.cos(theta)
        + (45 / 8) * np.cos(2 * phi + 2 * theta)
    )


def M_vv_30_2_2():
    def func(k):
        return -6 / 35 * k**2

    return func


def N_vv_30_2_2(theta, phi):
    return (
        (9 / 8) * np.cos(2 * phi)
        + (9 / 4) * np.cos(theta)
        + (45 / 8) * np.cos(2 * phi - 2 * theta)
    )


def M_vv_30_4_0():
    def func(k):
        return -6 / 35 * k**2

    return func


def N_vv_30_4_0(theta, phi):
    return (
        (45 / 16) * np.cos(2 * phi)
        + (27 / 16) * np.cos(theta)
        + (15 / 16) * np.cos(2 * phi + 2 * theta)
        + (105 / 16) * np.cos(4 * phi + theta)
    )


def M_vv_30_4_1():
    def func(k):
        return -6 / 35 * k**2

    return func


def N_vv_30_4_1(theta, phi):
    return (
        (45 / 16) * np.cos(2 * phi)
        + (27 / 16) * np.cos(theta)
        + (15 / 16) * np.cos(2 * phi - 2 * theta)
        + (105 / 16) * np.cos(4 * phi - theta)
    )


def M_vv_31_0_0():
    def func(k):
        return -12 / 5 * k**2

    return func


def N_vv_31_0_0(theta, phi):
    return 3 * np.cos(theta)


def M_vv_31_2_0():
    def func(k):
        return -12 / 5 * k**2

    return func


def N_vv_31_2_0(theta, phi):
    return (9 / 2) * np.cos(2 * phi) + (3 / 2) * np.cos(theta)


def M_vv_31_2_1():
    def func(k):
        return -12 / 35 * k**2

    return func


def N_vv_31_2_1(theta, phi):
    return (
        (9 / 8) * np.cos(2 * phi)
        + (9 / 4) * np.cos(theta)
        + (45 / 8) * np.cos(2 * phi + 2 * theta)
    )


def M_vv_31_2_2():
    def func(k):
        return -12 / 35 * k**2

    return func


def N_vv_31_2_2(theta, phi):
    return (
        (9 / 8) * np.cos(2 * phi)
        + (9 / 4) * np.cos(theta)
        + (45 / 8) * np.cos(2 * phi - 2 * theta)
    )


def M_vv_31_4_0():
    def func(k):
        return -12 / 35 * k**2

    return func


def N_vv_31_4_0(theta, phi):
    return (
        (45 / 16) * np.cos(2 * phi)
        + (27 / 16) * np.cos(theta)
        + (15 / 16) * np.cos(2 * phi + 2 * theta)
        + (105 / 16) * np.cos(4 * phi + theta)
    )


def M_vv_31_4_1():
    def func(k):
        return -12 / 35 * k**2

    return func


def N_vv_31_4_1(theta, phi):
    return (
        (45 / 16) * np.cos(2 * phi)
        + (27 / 16) * np.cos(theta)
        + (15 / 16) * np.cos(2 * phi - 2 * theta)
        + (105 / 16) * np.cos(4 * phi - theta)
    )


def M_vv_32_0_0():
    def func(k):
        return -1 / 5

    return func


def N_vv_32_0_0(theta, phi):
    return 3 * np.cos(theta)


def M_vv_32_2_0():
    def func(k):
        return -1 / 5

    return func


def N_vv_32_2_0(theta, phi):
    return (9 / 2) * np.cos(2 * phi) + (3 / 2) * np.cos(theta)


def M_vv_32_2_1():
    def func(k):
        return -1 / 35

    return func


def N_vv_32_2_1(theta, phi):
    return (
        (9 / 8) * np.cos(2 * phi)
        + (9 / 4) * np.cos(theta)
        + (45 / 8) * np.cos(2 * phi + 2 * theta)
    )


def M_vv_32_2_2():
    def func(k):
        return -1 / 35

    return func


def N_vv_32_2_2(theta, phi):
    return (
        (9 / 8) * np.cos(2 * phi)
        + (9 / 4) * np.cos(theta)
        + (45 / 8) * np.cos(2 * phi - 2 * theta)
    )


def M_vv_32_4_0():
    def func(k):
        return -1 / 35

    return func


def N_vv_32_4_0(theta, phi):
    return (
        (45 / 16) * np.cos(2 * phi)
        + (27 / 16) * np.cos(theta)
        + (15 / 16) * np.cos(2 * phi + 2 * theta)
        + (105 / 16) * np.cos(4 * phi + theta)
    )


def M_vv_32_4_1():
    def func(k):
        return -1 / 35

    return func


def N_vv_32_4_1(theta, phi):
    return (
        (45 / 16) * np.cos(2 * phi)
        + (27 / 16) * np.cos(theta)
        + (15 / 16) * np.cos(2 * phi - 2 * theta)
        + (105 / 16) * np.cos(4 * phi - theta)
    )


def M_vv_33_0_0():
    def func(k):
        return -1

    return func


def N_vv_33_0_0(theta, phi):
    return np.cos(theta)


def M_vv_33_2_0():
    def func(k):
        return -1 / 5

    return func


def N_vv_33_2_0(theta, phi):
    return (
        (15 / 8) * np.cos(2 * phi)
        + (5 / 4) * np.cos(theta)
        + (15 / 8) * np.cos(2 * phi + 2 * theta)
    )


def M_vv_33_2_1():
    def func(k):
        return -1 / 5

    return func


def N_vv_33_2_1(theta, phi):
    return (
        (15 / 8) * np.cos(2 * phi)
        + (5 / 4) * np.cos(theta)
        + (15 / 8) * np.cos(2 * phi - 2 * theta)
    )


def M_vv_34_0_0():
    def func(k):
        return (1 / 9) / k**2

    return func


def N_vv_34_0_0(theta, phi):
    return (5 / 8) * np.cos(theta) + (1 / 8) * np.cos(3 * theta)


def M_vv_34_2_0():
    def func(k):
        return (1 / 9) / k**2

    return func


def N_vv_34_2_0(theta, phi):
    return (
        (3 / 4) * np.cos(2 * phi)
        + (5 / 16) * np.cos(theta)
        + (1 / 16) * np.cos(3 * theta)
        + (3 / 16) * np.cos(2 * phi - 2 * theta)
        + (3 / 16) * np.cos(2 * phi + 2 * theta)
    )


def M_vv_35_0_0():
    def func(k):
        return (1 / 9) / k**2

    return func


def N_vv_35_0_0(theta, phi):
    return (1 / 2) * np.cos(theta)


def M_vv_35_0_1():
    def func(k):
        return (4 / 225) / k**2

    return func


def N_vv_35_0_1(theta, phi):
    return (25 / 16) * np.cos(theta) + (15 / 16) * np.cos(3 * theta)


def M_vv_35_2_0():
    def func(k):
        return (2 / 45) / k**2

    return func


def N_vv_35_2_0(theta, phi):
    return (
        (15 / 16) * np.cos(2 * phi)
        + (5 / 8) * np.cos(theta)
        + (15 / 16) * np.cos(2 * phi + 2 * theta)
    )


def M_vv_35_2_1():
    def func(k):
        return (2 / 45) / k**2

    return func


def N_vv_35_2_1(theta, phi):
    return (
        (15 / 16) * np.cos(2 * phi)
        + (5 / 8) * np.cos(theta)
        + (15 / 16) * np.cos(2 * phi - 2 * theta)
    )


def M_vv_35_2_2():
    def func(k):
        return (4 / 225) / k**2

    return func


def N_vv_35_2_2(theta, phi):
    return (
        (75 / 56) * np.cos(2 * phi)
        + (25 / 112) * np.cos(theta)
        + (75 / 112) * np.cos(3 * theta)
        + (75 / 112) * np.cos(2 * phi - 2 * theta)
        + (75 / 112) * np.cos(2 * phi + 2 * theta)
    )


def M_vv_35_4_0():
    def func(k):
        return (4 / 225) / k**2

    return func


def N_vv_35_4_0(theta, phi):
    return (
        (225 / 224) * np.cos(2 * phi)
        + (675 / 896) * np.cos(theta)
        + (135 / 896) * np.cos(3 * theta)
        + (225 / 448) * np.cos(2 * phi - 2 * theta)
        + (225 / 448) * np.cos(2 * phi + 2 * theta)
        + (225 / 128) * np.cos(4 * phi - theta)
        + (225 / 128) * np.cos(4 * phi + theta)
    )


def M_vv_36_0_0():
    def func(k):
        return (1 / 100) / k**2

    return func


def N_vv_36_0_0(theta, phi):
    return 3 * np.cos(theta)


def M_vv_36_0_1():
    def func(k):
        return (1 / 1225) / k**2

    return func


def N_vv_36_0_1(theta, phi):
    return (21 / 8) * np.cos(theta) + (35 / 8) * np.cos(3 * theta)


def M_vv_36_2_0():
    def func(k):
        return (1 / 100) / k**2

    return func


def N_vv_36_2_0(theta, phi):
    return (9 / 2) * np.cos(2 * phi) + (3 / 2) * np.cos(theta)


def M_vv_36_2_1():
    def func(k):
        return (1 / 350) / k**2

    return func


def N_vv_36_2_1(theta, phi):
    return (
        (9 / 8) * np.cos(2 * phi)
        + (9 / 4) * np.cos(theta)
        + (45 / 8) * np.cos(2 * phi + 2 * theta)
    )


def M_vv_36_2_2():
    def func(k):
        return (1 / 350) / k**2

    return func


def N_vv_36_2_2(theta, phi):
    return (
        (9 / 8) * np.cos(2 * phi)
        + (9 / 4) * np.cos(theta)
        + (45 / 8) * np.cos(2 * phi - 2 * theta)
    )


def M_vv_36_2_3():
    def func(k):
        return (1 / 1225) / k**2

    return func


def N_vv_36_2_3(theta, phi):
    return (
        (21 / 8) * np.cos(2 * phi)
        - 21 / 16 * np.cos(theta)
        + (175 / 48) * np.cos(3 * theta)
        + (35 / 16) * np.cos(2 * phi - 2 * theta)
        + (35 / 16) * np.cos(2 * phi + 2 * theta)
    )


def M_vv_36_4_0():
    def func(k):
        return (1 / 350) / k**2

    return func


def N_vv_36_4_0(theta, phi):
    return (
        (45 / 16) * np.cos(2 * phi)
        + (27 / 16) * np.cos(theta)
        + (15 / 16) * np.cos(2 * phi + 2 * theta)
        + (105 / 16) * np.cos(4 * phi + theta)
    )


def M_vv_36_4_1():
    def func(k):
        return (1 / 350) / k**2

    return func


def N_vv_36_4_1(theta, phi):
    return (
        (45 / 16) * np.cos(2 * phi)
        + (27 / 16) * np.cos(theta)
        + (15 / 16) * np.cos(2 * phi - 2 * theta)
        + (105 / 16) * np.cos(4 * phi - theta)
    )


def M_vv_36_4_2():
    def func(k):
        return (1 / 1225) / k**2

    return func


def N_vv_36_4_2(theta, phi):
    return (
        -315 / 176 * np.cos(2 * phi)
        + (189 / 704) * np.cos(theta)
        + (945 / 704) * np.cos(3 * theta)
        + (945 / 352) * np.cos(2 * phi - 2 * theta)
        + (945 / 352) * np.cos(2 * phi + 2 * theta)
        + (2205 / 704) * np.cos(4 * phi - theta)
        + (2205 / 704) * np.cos(4 * phi + theta)
    )


def M_vv_36_6_0():
    def func(k):
        return (1 / 1225) / k**2

    return func


def N_vv_36_6_0(theta, phi):
    return (
        (3675 / 1408) * np.cos(2 * phi)
        + (1225 / 128) * np.cos(6 * phi)
        + (2625 / 1408) * np.cos(theta)
        + (875 / 4224) * np.cos(3 * theta)
        + (1225 / 1408) * np.cos(2 * phi - 2 * theta)
        + (1225 / 1408) * np.cos(2 * phi + 2 * theta)
        + (3675 / 1408) * np.cos(4 * phi - theta)
        + (3675 / 1408) * np.cos(4 * phi + theta)
    )


def M_vv_37_0_0():
    def func(k):
        return (1 / 3) * k**2

    return func


def N_vv_37_0_0(theta, phi):
    return 3 * np.cos(theta)


def M_vv_37_2_0():
    def func(k):
        return (1 / 3) * k**2

    return func


def N_vv_37_2_0(theta, phi):
    return (9 / 2) * np.cos(2 * phi) + (3 / 2) * np.cos(theta)


def M_vv_38_0_0():
    def func(k):
        return (2 / 3) * k**2

    return func


def N_vv_38_0_0(theta, phi):
    return 3 * np.cos(theta)


def M_vv_38_2_0():
    def func(k):
        return (2 / 3) * k**2

    return func


def N_vv_38_2_0(theta, phi):
    return (9 / 2) * np.cos(2 * phi) + (3 / 2) * np.cos(theta)


def M_vv_39_0_0():
    def func(k):
        return (2 / 3) * k**2

    return func


def N_vv_39_0_0(theta, phi):
    return 3 * np.cos(theta)


def M_vv_39_2_0():
    def func(k):
        return (2 / 3) * k**2

    return func


def N_vv_39_2_0(theta, phi):
    return (9 / 2) * np.cos(2 * phi) + (3 / 2) * np.cos(theta)


def M_vv_40_0_0():
    def func(k):
        return (2 / 3) * k**2

    return func


def N_vv_40_0_0(theta, phi):
    return 3 * np.cos(theta)


def M_vv_40_2_0():
    def func(k):
        return (2 / 3) * k**2

    return func


def N_vv_40_2_0(theta, phi):
    return (9 / 2) * np.cos(2 * phi) + (3 / 2) * np.cos(theta)


def M_vv_41_0_0():
    def func(k):
        return 2 * k**4

    return func


def N_vv_41_0_0(theta, phi):
    return 3 * np.cos(theta)


def M_vv_41_2_0():
    def func(k):
        return 2 * k**4

    return func


def N_vv_41_2_0(theta, phi):
    return (9 / 2) * np.cos(2 * phi) + (3 / 2) * np.cos(theta)


def M_vv_42_0_0():
    def func(k):
        return (1 / 6) * k**2

    return func


def N_vv_42_0_0(theta, phi):
    return 3 * np.cos(theta)


def M_vv_42_2_0():
    def func(k):
        return (1 / 6) * k**2

    return func


def N_vv_42_2_0(theta, phi):
    return (9 / 2) * np.cos(2 * phi) + (3 / 2) * np.cos(theta)


def M_vv_43_0_0():
    def func(k):
        return (1 / 6) * k**2

    return func


def N_vv_43_0_0(theta, phi):
    return 3 * np.cos(theta)


def M_vv_43_2_0():
    def func(k):
        return (1 / 6) * k**2

    return func


def N_vv_43_2_0(theta, phi):
    return (9 / 2) * np.cos(2 * phi) + (3 / 2) * np.cos(theta)


def M_vv_44_0_0():
    def func(k):
        return (1 / 3) * k**2

    return func


def N_vv_44_0_0(theta, phi):
    return 3 * np.cos(theta)


def M_vv_44_2_0():
    def func(k):
        return (1 / 3) * k**2

    return func


def N_vv_44_2_0(theta, phi):
    return (9 / 2) * np.cos(2 * phi) + (3 / 2) * np.cos(theta)


def M_vv_45_0_0():
    def func(k):
        return (2 / 3) * k**2

    return func


def N_vv_45_0_0(theta, phi):
    return 3 * np.cos(theta)


def M_vv_45_2_0():
    def func(k):
        return (2 / 3) * k**2

    return func


def N_vv_45_2_0(theta, phi):
    return (9 / 2) * np.cos(2 * phi) + (3 / 2) * np.cos(theta)


def M_vv_46_0_0():
    def func(k):
        return -2 / 9

    return func


def N_vv_46_0_0(theta, phi):
    return 3 * np.cos(theta)


def M_vv_46_2_0():
    def func(k):
        return -2 / 9

    return func


def N_vv_46_2_0(theta, phi):
    return (9 / 2) * np.cos(2 * phi) + (3 / 2) * np.cos(theta)


def M_vv_47_0_0():
    def func(k):
        return -2 / 15

    return func


def N_vv_47_0_0(theta, phi):
    return 3 * np.cos(theta)


def M_vv_47_2_0():
    def func(k):
        return -2 / 15

    return func


def N_vv_47_2_0(theta, phi):
    return (9 / 2) * np.cos(2 * phi) + (3 / 2) * np.cos(theta)


def M_vv_47_2_1():
    def func(k):
        return -2 / 105

    return func


def N_vv_47_2_1(theta, phi):
    return (
        (9 / 8) * np.cos(2 * phi)
        + (9 / 4) * np.cos(theta)
        + (45 / 8) * np.cos(2 * phi + 2 * theta)
    )


def M_vv_47_2_2():
    def func(k):
        return -2 / 105

    return func


def N_vv_47_2_2(theta, phi):
    return (
        (9 / 8) * np.cos(2 * phi)
        + (9 / 4) * np.cos(theta)
        + (45 / 8) * np.cos(2 * phi - 2 * theta)
    )


def M_vv_47_4_0():
    def func(k):
        return -2 / 105

    return func


def N_vv_47_4_0(theta, phi):
    return (
        (45 / 16) * np.cos(2 * phi)
        + (27 / 16) * np.cos(theta)
        + (15 / 16) * np.cos(2 * phi + 2 * theta)
        + (105 / 16) * np.cos(4 * phi + theta)
    )


def M_vv_47_4_1():
    def func(k):
        return -2 / 105

    return func


def N_vv_47_4_1(theta, phi):
    return (
        (45 / 16) * np.cos(2 * phi)
        + (27 / 16) * np.cos(theta)
        + (15 / 16) * np.cos(2 * phi - 2 * theta)
        + (105 / 16) * np.cos(4 * phi - theta)
    )


def M_vv_48_0_0():
    def func(k):
        return -4 / 9 * k**2

    return func


def N_vv_48_0_0(theta, phi):
    return 3 * np.cos(theta)


def M_vv_48_2_0():
    def func(k):
        return -4 / 9 * k**2

    return func


def N_vv_48_2_0(theta, phi):
    return (9 / 2) * np.cos(2 * phi) + (3 / 2) * np.cos(theta)


def M_vv_49_0_0():
    def func(k):
        return -4 / 15 * k**2

    return func


def N_vv_49_0_0(theta, phi):
    return 3 * np.cos(theta)


def M_vv_49_2_0():
    def func(k):
        return -4 / 15 * k**2

    return func


def N_vv_49_2_0(theta, phi):
    return (9 / 2) * np.cos(2 * phi) + (3 / 2) * np.cos(theta)


def M_vv_49_2_1():
    def func(k):
        return -4 / 105 * k**2

    return func


def N_vv_49_2_1(theta, phi):
    return (
        (9 / 8) * np.cos(2 * phi)
        + (9 / 4) * np.cos(theta)
        + (45 / 8) * np.cos(2 * phi + 2 * theta)
    )


def M_vv_49_2_2():
    def func(k):
        return -4 / 105 * k**2

    return func


def N_vv_49_2_2(theta, phi):
    return (
        (9 / 8) * np.cos(2 * phi)
        + (9 / 4) * np.cos(theta)
        + (45 / 8) * np.cos(2 * phi - 2 * theta)
    )


def M_vv_49_4_0():
    def func(k):
        return -4 / 105 * k**2

    return func


def N_vv_49_4_0(theta, phi):
    return (
        (45 / 16) * np.cos(2 * phi)
        + (27 / 16) * np.cos(theta)
        + (15 / 16) * np.cos(2 * phi + 2 * theta)
        + (105 / 16) * np.cos(4 * phi + theta)
    )


def M_vv_49_4_1():
    def func(k):
        return -4 / 105 * k**2

    return func


def N_vv_49_4_1(theta, phi):
    return (
        (45 / 16) * np.cos(2 * phi)
        + (27 / 16) * np.cos(theta)
        + (15 / 16) * np.cos(2 * phi - 2 * theta)
        + (105 / 16) * np.cos(4 * phi - theta)
    )


def M_vv_50_0_0():
    def func(k):
        return -1 / 9

    return func


def N_vv_50_0_0(theta, phi):
    return 3 * np.cos(theta)


def M_vv_50_2_0():
    def func(k):
        return -1 / 9

    return func


def N_vv_50_2_0(theta, phi):
    return (9 / 2) * np.cos(2 * phi) + (3 / 2) * np.cos(theta)


def M_vv_51_0_0():
    def func(k):
        return -1 / 9

    return func


def N_vv_51_0_0(theta, phi):
    return 3 * np.cos(theta)


def M_vv_51_2_0():
    def func(k):
        return -1 / 9

    return func


def N_vv_51_2_0(theta, phi):
    return (9 / 2) * np.cos(2 * phi) + (3 / 2) * np.cos(theta)


def M_vv_52_0_0():
    def func(k):
        return -1 / 15

    return func


def N_vv_52_0_0(theta, phi):
    return 3 * np.cos(theta)


def M_vv_52_2_0():
    def func(k):
        return -1 / 15

    return func


def N_vv_52_2_0(theta, phi):
    return (9 / 2) * np.cos(2 * phi) + (3 / 2) * np.cos(theta)


def M_vv_52_2_1():
    def func(k):
        return -1 / 105

    return func


def N_vv_52_2_1(theta, phi):
    return (
        (9 / 8) * np.cos(2 * phi)
        + (9 / 4) * np.cos(theta)
        + (45 / 8) * np.cos(2 * phi + 2 * theta)
    )


def M_vv_52_2_2():
    def func(k):
        return -1 / 105

    return func


def N_vv_52_2_2(theta, phi):
    return (
        (9 / 8) * np.cos(2 * phi)
        + (9 / 4) * np.cos(theta)
        + (45 / 8) * np.cos(2 * phi - 2 * theta)
    )


def M_vv_52_4_0():
    def func(k):
        return -1 / 105

    return func


def N_vv_52_4_0(theta, phi):
    return (
        (45 / 16) * np.cos(2 * phi)
        + (27 / 16) * np.cos(theta)
        + (15 / 16) * np.cos(2 * phi + 2 * theta)
        + (105 / 16) * np.cos(4 * phi + theta)
    )


def M_vv_52_4_1():
    def func(k):
        return -1 / 105

    return func


def N_vv_52_4_1(theta, phi):
    return (
        (45 / 16) * np.cos(2 * phi)
        + (27 / 16) * np.cos(theta)
        + (15 / 16) * np.cos(2 * phi - 2 * theta)
        + (105 / 16) * np.cos(4 * phi - theta)
    )


def M_vv_53_0_0():
    def func(k):
        return -1 / 15

    return func


def N_vv_53_0_0(theta, phi):
    return 3 * np.cos(theta)


def M_vv_53_2_0():
    def func(k):
        return -1 / 15

    return func


def N_vv_53_2_0(theta, phi):
    return (9 / 2) * np.cos(2 * phi) + (3 / 2) * np.cos(theta)


def M_vv_53_2_1():
    def func(k):
        return -1 / 105

    return func


def N_vv_53_2_1(theta, phi):
    return (
        (9 / 8) * np.cos(2 * phi)
        + (9 / 4) * np.cos(theta)
        + (45 / 8) * np.cos(2 * phi + 2 * theta)
    )


def M_vv_53_2_2():
    def func(k):
        return -1 / 105

    return func


def N_vv_53_2_2(theta, phi):
    return (
        (9 / 8) * np.cos(2 * phi)
        + (9 / 4) * np.cos(theta)
        + (45 / 8) * np.cos(2 * phi - 2 * theta)
    )


def M_vv_53_4_0():
    def func(k):
        return -1 / 105

    return func


def N_vv_53_4_0(theta, phi):
    return (
        (45 / 16) * np.cos(2 * phi)
        + (27 / 16) * np.cos(theta)
        + (15 / 16) * np.cos(2 * phi + 2 * theta)
        + (105 / 16) * np.cos(4 * phi + theta)
    )


def M_vv_53_4_1():
    def func(k):
        return -1 / 105

    return func


def N_vv_53_4_1(theta, phi):
    return (
        (45 / 16) * np.cos(2 * phi)
        + (27 / 16) * np.cos(theta)
        + (15 / 16) * np.cos(2 * phi - 2 * theta)
        + (105 / 16) * np.cos(4 * phi - theta)
    )


def M_vv_54_0_0():
    def func(k):
        return -(77 / 120) / k**2

    return func


def N_vv_54_0_0(theta, phi):
    return 3 * np.cos(theta)


def M_vv_54_2_0():
    def func(k):
        return -(77 / 120) / k**2

    return func


def N_vv_54_2_0(theta, phi):
    return (9 / 2) * np.cos(2 * phi) + (3 / 2) * np.cos(theta)


def M_vv_54_2_1():
    def func(k):
        return -(11 / 120) / k**2

    return func


def N_vv_54_2_1(theta, phi):
    return (
        (9 / 8) * np.cos(2 * phi)
        + (9 / 4) * np.cos(theta)
        + (45 / 8) * np.cos(2 * phi + 2 * theta)
    )


def M_vv_54_2_2():
    def func(k):
        return -(11 / 120) / k**2

    return func


def N_vv_54_2_2(theta, phi):
    return (
        (9 / 8) * np.cos(2 * phi)
        + (9 / 4) * np.cos(theta)
        + (45 / 8) * np.cos(2 * phi - 2 * theta)
    )


def M_vv_54_4_0():
    def func(k):
        return -(11 / 120) / k**2

    return func


def N_vv_54_4_0(theta, phi):
    return (
        (45 / 16) * np.cos(2 * phi)
        + (27 / 16) * np.cos(theta)
        + (15 / 16) * np.cos(2 * phi + 2 * theta)
        + (105 / 16) * np.cos(4 * phi + theta)
    )


def M_vv_54_4_1():
    def func(k):
        return -(11 / 120) / k**2

    return func


def N_vv_54_4_1(theta, phi):
    return (
        (45 / 16) * np.cos(2 * phi)
        + (27 / 16) * np.cos(theta)
        + (15 / 16) * np.cos(2 * phi - 2 * theta)
        + (105 / 16) * np.cos(4 * phi - theta)
    )


def M_vv_55_0_0():
    def func(k):
        return -(77 / 144) / k**2

    return func


def N_vv_55_0_0(theta, phi):
    return -2 * np.cos(theta)


def M_vv_55_0_1():
    def func(k):
        return -(77 / 900) / k**2

    return func


def N_vv_55_0_1(theta, phi):
    return -25 / 4 * np.cos(theta) - 15 / 4 * np.cos(3 * theta)


def M_vv_55_2_0():
    def func(k):
        return -(77 / 360) / k**2

    return func


def N_vv_55_2_0(theta, phi):
    return (
        -15 / 4 * np.cos(2 * phi)
        - 5 / 2 * np.cos(theta)
        - 15 / 4 * np.cos(2 * phi + 2 * theta)
    )


def M_vv_55_2_1():
    def func(k):
        return -(77 / 360) / k**2

    return func


def N_vv_55_2_1(theta, phi):
    return (
        -15 / 4 * np.cos(2 * phi)
        - 5 / 2 * np.cos(theta)
        - 15 / 4 * np.cos(2 * phi - 2 * theta)
    )


def M_vv_55_2_2():
    def func(k):
        return -(77 / 900) / k**2

    return func


def N_vv_55_2_2(theta, phi):
    return (
        -75 / 14 * np.cos(2 * phi)
        - 25 / 28 * np.cos(theta)
        - 75 / 28 * np.cos(3 * theta)
        - 75 / 28 * np.cos(2 * phi - 2 * theta)
        - 75 / 28 * np.cos(2 * phi + 2 * theta)
    )


def M_vv_55_4_0():
    def func(k):
        return -(77 / 900) / k**2

    return func


def N_vv_55_4_0(theta, phi):
    return (
        -225 / 56 * np.cos(2 * phi)
        - 675 / 224 * np.cos(theta)
        - 135 / 224 * np.cos(3 * theta)
        - 225 / 112 * np.cos(2 * phi - 2 * theta)
        - 225 / 112 * np.cos(2 * phi + 2 * theta)
        - 225 / 32 * np.cos(4 * phi - theta)
        - 225 / 32 * np.cos(4 * phi + theta)
    )


def M_vv_56_0_0():
    def func(k):
        return (7 / 20) / k**2

    return func


def N_vv_56_0_0(theta, phi):
    return 3 * np.cos(theta)


def M_vv_56_2_0():
    def func(k):
        return (7 / 20) / k**2

    return func


def N_vv_56_2_0(theta, phi):
    return (9 / 2) * np.cos(2 * phi) + (3 / 2) * np.cos(theta)


def M_vv_56_2_1():
    def func(k):
        return (1 / 20) / k**2

    return func


def N_vv_56_2_1(theta, phi):
    return (
        (9 / 8) * np.cos(2 * phi)
        + (9 / 4) * np.cos(theta)
        + (45 / 8) * np.cos(2 * phi + 2 * theta)
    )


def M_vv_56_2_2():
    def func(k):
        return (1 / 20) / k**2

    return func


def N_vv_56_2_2(theta, phi):
    return (
        (9 / 8) * np.cos(2 * phi)
        + (9 / 4) * np.cos(theta)
        + (45 / 8) * np.cos(2 * phi - 2 * theta)
    )


def M_vv_56_4_0():
    def func(k):
        return (1 / 20) / k**2

    return func


def N_vv_56_4_0(theta, phi):
    return (
        (45 / 16) * np.cos(2 * phi)
        + (27 / 16) * np.cos(theta)
        + (15 / 16) * np.cos(2 * phi + 2 * theta)
        + (105 / 16) * np.cos(4 * phi + theta)
    )


def M_vv_56_4_1():
    def func(k):
        return (1 / 20) / k**2

    return func


def N_vv_56_4_1(theta, phi):
    return (
        (45 / 16) * np.cos(2 * phi)
        + (27 / 16) * np.cos(theta)
        + (15 / 16) * np.cos(2 * phi - 2 * theta)
        + (105 / 16) * np.cos(4 * phi - theta)
    )


def M_vv_57_0_0():
    def func(k):
        return (7 / 24) / k**2

    return func


def N_vv_57_0_0(theta, phi):
    return -2 * np.cos(theta)


def M_vv_57_0_1():
    def func(k):
        return (7 / 150) / k**2

    return func


def N_vv_57_0_1(theta, phi):
    return -25 / 4 * np.cos(theta) - 15 / 4 * np.cos(3 * theta)


def M_vv_57_2_0():
    def func(k):
        return (7 / 60) / k**2

    return func


def N_vv_57_2_0(theta, phi):
    return (
        -15 / 4 * np.cos(2 * phi)
        - 5 / 2 * np.cos(theta)
        - 15 / 4 * np.cos(2 * phi + 2 * theta)
    )


def M_vv_57_2_1():
    def func(k):
        return (7 / 60) / k**2

    return func


def N_vv_57_2_1(theta, phi):
    return (
        -15 / 4 * np.cos(2 * phi)
        - 5 / 2 * np.cos(theta)
        - 15 / 4 * np.cos(2 * phi - 2 * theta)
    )


def M_vv_57_2_2():
    def func(k):
        return (7 / 150) / k**2

    return func


def N_vv_57_2_2(theta, phi):
    return (
        -75 / 14 * np.cos(2 * phi)
        - 25 / 28 * np.cos(theta)
        - 75 / 28 * np.cos(3 * theta)
        - 75 / 28 * np.cos(2 * phi - 2 * theta)
        - 75 / 28 * np.cos(2 * phi + 2 * theta)
    )


def M_vv_57_4_0():
    def func(k):
        return (7 / 150) / k**2

    return func


def N_vv_57_4_0(theta, phi):
    return (
        -225 / 56 * np.cos(2 * phi)
        - 675 / 224 * np.cos(theta)
        - 135 / 224 * np.cos(3 * theta)
        - 225 / 112 * np.cos(2 * phi - 2 * theta)
        - 225 / 112 * np.cos(2 * phi + 2 * theta)
        - 225 / 32 * np.cos(4 * phi - theta)
        - 225 / 32 * np.cos(4 * phi + theta)
    )


def M_vv_58_0_0():
    def func(k):
        return (7 / 24) / k**2

    return func


def N_vv_58_0_0(theta, phi):
    return 3 * np.cos(theta)


def M_vv_58_2_0():
    def func(k):
        return (7 / 24) / k**2

    return func


def N_vv_58_2_0(theta, phi):
    return (9 / 2) * np.cos(2 * phi) + (3 / 2) * np.cos(theta)


def M_vv_58_2_1():
    def func(k):
        return (1 / 24) / k**2

    return func


def N_vv_58_2_1(theta, phi):
    return (
        (9 / 8) * np.cos(2 * phi)
        + (9 / 4) * np.cos(theta)
        + (45 / 8) * np.cos(2 * phi + 2 * theta)
    )


def M_vv_58_2_2():
    def func(k):
        return (1 / 24) / k**2

    return func


def N_vv_58_2_2(theta, phi):
    return (
        (9 / 8) * np.cos(2 * phi)
        + (9 / 4) * np.cos(theta)
        + (45 / 8) * np.cos(2 * phi - 2 * theta)
    )


def M_vv_58_4_0():
    def func(k):
        return (1 / 24) / k**2

    return func


def N_vv_58_4_0(theta, phi):
    return (
        (45 / 16) * np.cos(2 * phi)
        + (27 / 16) * np.cos(theta)
        + (15 / 16) * np.cos(2 * phi + 2 * theta)
        + (105 / 16) * np.cos(4 * phi + theta)
    )


def M_vv_58_4_1():
    def func(k):
        return (1 / 24) / k**2

    return func


def N_vv_58_4_1(theta, phi):
    return (
        (45 / 16) * np.cos(2 * phi)
        + (27 / 16) * np.cos(theta)
        + (15 / 16) * np.cos(2 * phi - 2 * theta)
        + (105 / 16) * np.cos(4 * phi - theta)
    )


def M_vv_59_0_0():
    def func(k):
        return (35 / 144) / k**2

    return func


def N_vv_59_0_0(theta, phi):
    return -2 * np.cos(theta)


def M_vv_59_0_1():
    def func(k):
        return (7 / 180) / k**2

    return func


def N_vv_59_0_1(theta, phi):
    return -25 / 4 * np.cos(theta) - 15 / 4 * np.cos(3 * theta)


def M_vv_59_2_0():
    def func(k):
        return (7 / 72) / k**2

    return func


def N_vv_59_2_0(theta, phi):
    return (
        -15 / 4 * np.cos(2 * phi)
        - 5 / 2 * np.cos(theta)
        - 15 / 4 * np.cos(2 * phi + 2 * theta)
    )


def M_vv_59_2_1():
    def func(k):
        return (7 / 72) / k**2

    return func


def N_vv_59_2_1(theta, phi):
    return (
        -15 / 4 * np.cos(2 * phi)
        - 5 / 2 * np.cos(theta)
        - 15 / 4 * np.cos(2 * phi - 2 * theta)
    )


def M_vv_59_2_2():
    def func(k):
        return (7 / 180) / k**2

    return func


def N_vv_59_2_2(theta, phi):
    return (
        -75 / 14 * np.cos(2 * phi)
        - 25 / 28 * np.cos(theta)
        - 75 / 28 * np.cos(3 * theta)
        - 75 / 28 * np.cos(2 * phi - 2 * theta)
        - 75 / 28 * np.cos(2 * phi + 2 * theta)
    )


def M_vv_59_4_0():
    def func(k):
        return (7 / 180) / k**2

    return func


def N_vv_59_4_0(theta, phi):
    return (
        -225 / 56 * np.cos(2 * phi)
        - 675 / 224 * np.cos(theta)
        - 135 / 224 * np.cos(3 * theta)
        - 225 / 112 * np.cos(2 * phi - 2 * theta)
        - 225 / 112 * np.cos(2 * phi + 2 * theta)
        - 225 / 32 * np.cos(4 * phi - theta)
        - 225 / 32 * np.cos(4 * phi + theta)
    )


dictionary_terms = {
    "gg": [
        "0",
        "1",
        "2",
        "3",
        "4",
        "5",
        "6",
        "7",
        "8",
        "9",
        "10",
        "11",
        "12",
        "13",
        "14",
        "15",
        "16",
        "17",
        "18",
        "19",
        "20",
        "21",
        "22",
        "23",
        "24",
        "25",
        "26",
        "27",
        "28",
        "29",
        "30",
        "31",
        "32",
        "33",
        "34",
        "35",
        "36",
        "37",
        "38",
        "39",
        "40",
        "41",
        "42",
        "43",
        "44",
        "45",
        "46",
        "47",
        "48",
        "49",
        "50",
        "51",
        "52",
        "53",
        "54",
        "55",
        "56",
        "57",
        "58",
        "59",
        "60",
        "61",
        "62",
        "63",
        "64",
        "65",
        "66",
        "67",
        "68",
        "69",
        "70",
        "71",
        "72",
        "73",
        "74",
        "75",
        "76",
        "77",
        "78",
        "79",
        "80",
        "81",
        "82",
        "83",
        "84",
        "85",
        "86",
        "87",
        "88",
        "89",
        "90",
        "91",
        "92",
        "93",
        "94",
        "95",
        "96",
        "97",
        "98",
        "99",
        "100",
        "101",
        "102",
        "103",
        "104",
        "105",
        "106",
        "107",
        "108",
        "109",
        "110",
        "111",
        "112",
        "113",
        "114",
        "115",
        "116",
        "117",
        "118",
        "119",
        "120",
        "121",
        "122",
        "123",
        "124",
        "125",
        "126",
        "127",
    ],
    "gv": [
        "0",
        "1",
        "2",
        "3",
        "4",
        "5",
        "6",
        "7",
        "8",
        "9",
        "10",
        "11",
        "12",
        "13",
        "14",
        "15",
        "16",
        "17",
        "18",
        "19",
        "20",
        "21",
        "22",
        "23",
        "24",
        "25",
        "26",
        "27",
        "28",
        "29",
        "30",
        "31",
        "32",
        "33",
        "34",
        "35",
        "36",
        "37",
        "38",
        "39",
        "40",
        "41",
        "42",
        "43",
        "44",
        "45",
        "46",
        "47",
        "48",
        "49",
        "50",
        "51",
        "52",
        "53",
        "54",
        "55",
        "56",
        "57",
        "58",
        "59",
        "60",
        "61",
        "62",
        "63",
        "64",
        "65",
        "66",
        "67",
        "68",
        "69",
        "70",
        "71",
        "72",
        "73",
        "74",
        "75",
        "76",
        "77",
        "78",
        "79",
        "80",
        "81",
        "82",
        "83",
        "84",
        "85",
        "86",
        "87",
        "88",
        "89",
        "90",
        "91",
        "92",
        "93",
        "94",
        "95",
        "96",
        "97",
        "98",
        "99",
        "100",
        "101",
        "102",
        "103",
        "104",
        "105",
        "106",
        "107",
        "108",
        "109",
        "110",
        "111",
        "112",
        "113",
        "114",
        "115",
        "116",
        "117",
        "118",
    ],
    "vv": [
        "0",
        "1",
        "2",
        "3",
        "4",
        "5",
        "6",
        "7",
        "8",
        "9",
        "10",
        "11",
        "12",
        "13",
        "14",
        "15",
        "16",
        "17",
        "18",
        "19",
        "20",
        "21",
        "22",
        "23",
        "24",
        "25",
        "26",
        "27",
        "28",
        "29",
        "30",
        "31",
        "32",
        "33",
        "34",
        "35",
        "36",
        "37",
        "38",
        "39",
        "40",
        "41",
        "42",
        "43",
        "44",
        "45",
        "46",
        "47",
        "48",
        "49",
        "50",
        "51",
        "52",
        "53",
        "54",
        "55",
        "56",
        "57",
        "58",
        "59",
    ],
}
dictionary_lmax = {
    "gg": [
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        4,
        2,
        4,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        6,
        4,
        6,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        2,
        4,
        4,
        4,
        4,
        6,
        4,
        6,
        4,
        6,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        6,
        6,
        6,
        6,
        6,
        6,
        4,
        4,
        6,
        8,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        6,
        4,
        6,
        4,
        4,
        6,
        6,
        6,
        6,
        6,
        6,
        6,
        6,
    ],
    "gv": [
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        3,
        1,
        3,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        3,
        3,
        3,
        3,
        3,
        3,
        3,
        3,
        3,
        3,
        3,
        3,
        3,
        5,
        3,
        5,
        3,
        3,
        3,
        3,
        3,
        3,
        3,
        3,
        3,
        3,
        3,
        3,
        3,
        3,
        3,
        3,
        3,
        3,
        3,
        3,
        3,
        1,
        3,
        3,
        3,
        3,
        5,
        3,
        5,
        3,
        5,
        3,
        3,
        3,
        3,
        3,
        3,
        3,
        3,
        3,
        3,
        3,
        3,
        3,
        3,
        5,
        5,
        5,
        5,
        5,
        5,
        3,
        3,
        5,
        7,
        3,
        3,
        3,
        3,
        3,
        3,
        3,
        3,
        3,
        3,
        5,
        3,
        5,
        3,
        3,
        5,
        5,
        5,
        5,
        5,
        5,
        5,
        5,
    ],
    "vv": [
        2,
        2,
        2,
        0,
        2,
        2,
        2,
        2,
        4,
        2,
        4,
        2,
        4,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        4,
        4,
        4,
        4,
        4,
        4,
        2,
        2,
        4,
        6,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        4,
        2,
        4,
        2,
        2,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
    ],
}
dictionary_subterms = {
    "gg_0_0": 1,
    "gg_1_0": 1,
    "gg_2_0": 1,
    "gg_3_0": 1,
    "gg_4_0": 1,
    "gg_5_0": 1,
    "gg_6_0": 1,
    "gg_7_0": 1,
    "gg_8_0": 1,
    "gg_9_0": 1,
    "gg_9_1": 0,
    "gg_9_2": 2,
    "gg_10_0": 1,
    "gg_10_1": 0,
    "gg_10_2": 2,
    "gg_11_0": 1,
    "gg_11_1": 0,
    "gg_11_2": 2,
    "gg_12_0": 1,
    "gg_12_1": 0,
    "gg_12_2": 2,
    "gg_13_0": 1,
    "gg_13_1": 0,
    "gg_13_2": 2,
    "gg_14_0": 1,
    "gg_14_1": 0,
    "gg_14_2": 2,
    "gg_15_0": 1,
    "gg_15_1": 0,
    "gg_15_2": 2,
    "gg_16_0": 1,
    "gg_16_1": 0,
    "gg_16_2": 2,
    "gg_17_0": 1,
    "gg_17_1": 0,
    "gg_17_2": 2,
    "gg_18_0": 1,
    "gg_18_1": 0,
    "gg_18_2": 2,
    "gg_19_0": 1,
    "gg_19_1": 0,
    "gg_19_2": 2,
    "gg_20_0": 1,
    "gg_20_1": 0,
    "gg_20_2": 2,
    "gg_20_3": 0,
    "gg_20_4": 2,
    "gg_21_0": 1,
    "gg_21_1": 0,
    "gg_21_2": 2,
    "gg_22_0": 1,
    "gg_22_1": 0,
    "gg_22_2": 2,
    "gg_22_3": 0,
    "gg_22_4": 2,
    "gg_23_0": 1,
    "gg_23_1": 0,
    "gg_23_2": 2,
    "gg_24_0": 1,
    "gg_24_1": 0,
    "gg_24_2": 2,
    "gg_25_0": 1,
    "gg_25_1": 0,
    "gg_25_2": 2,
    "gg_26_0": 1,
    "gg_26_1": 0,
    "gg_26_2": 2,
    "gg_27_0": 1,
    "gg_27_1": 0,
    "gg_27_2": 2,
    "gg_28_0": 1,
    "gg_28_1": 0,
    "gg_28_2": 2,
    "gg_29_0": 1,
    "gg_29_1": 0,
    "gg_29_2": 2,
    "gg_30_0": 1,
    "gg_30_1": 0,
    "gg_30_2": 2,
    "gg_31_0": 1,
    "gg_31_1": 0,
    "gg_31_2": 2,
    "gg_32_0": 1,
    "gg_32_1": 0,
    "gg_32_2": 2,
    "gg_33_0": 1,
    "gg_33_1": 0,
    "gg_33_2": 2,
    "gg_34_0": 1,
    "gg_34_1": 0,
    "gg_34_2": 2,
    "gg_34_3": 0,
    "gg_34_4": 2,
    "gg_35_0": 1,
    "gg_35_1": 0,
    "gg_35_2": 2,
    "gg_35_3": 0,
    "gg_35_4": 2,
    "gg_36_0": 1,
    "gg_36_1": 0,
    "gg_36_2": 2,
    "gg_36_3": 0,
    "gg_36_4": 2,
    "gg_37_0": 1,
    "gg_37_1": 0,
    "gg_37_2": 2,
    "gg_37_3": 0,
    "gg_37_4": 2,
    "gg_38_0": 1,
    "gg_38_1": 0,
    "gg_38_2": 2,
    "gg_38_3": 0,
    "gg_38_4": 2,
    "gg_39_0": 1,
    "gg_39_1": 0,
    "gg_39_2": 2,
    "gg_39_3": 0,
    "gg_39_4": 2,
    "gg_40_0": 1,
    "gg_40_1": 0,
    "gg_40_2": 2,
    "gg_40_3": 0,
    "gg_40_4": 2,
    "gg_41_0": 1,
    "gg_41_1": 0,
    "gg_41_2": 2,
    "gg_41_3": 0,
    "gg_41_4": 2,
    "gg_42_0": 1,
    "gg_42_1": 0,
    "gg_42_2": 2,
    "gg_42_3": 0,
    "gg_42_4": 2,
    "gg_43_0": 1,
    "gg_43_1": 0,
    "gg_43_2": 2,
    "gg_43_3": 0,
    "gg_43_4": 2,
    "gg_44_0": 1,
    "gg_44_1": 0,
    "gg_44_2": 2,
    "gg_44_3": 0,
    "gg_44_4": 2,
    "gg_45_0": 1,
    "gg_45_1": 0,
    "gg_45_2": 2,
    "gg_45_3": 0,
    "gg_45_4": 2,
    "gg_46_0": 1,
    "gg_46_1": 0,
    "gg_46_2": 2,
    "gg_46_3": 0,
    "gg_46_4": 2,
    "gg_47_0": 1,
    "gg_47_1": 0,
    "gg_47_2": 2,
    "gg_47_3": 0,
    "gg_47_4": 2,
    "gg_47_5": 0,
    "gg_47_6": 2,
    "gg_48_0": 1,
    "gg_48_1": 0,
    "gg_48_2": 2,
    "gg_48_3": 0,
    "gg_48_4": 2,
    "gg_49_0": 1,
    "gg_49_1": 0,
    "gg_49_2": 2,
    "gg_49_3": 0,
    "gg_49_4": 2,
    "gg_49_5": 0,
    "gg_49_6": 2,
    "gg_50_0": 1,
    "gg_50_1": 0,
    "gg_50_2": 2,
    "gg_50_3": 0,
    "gg_50_4": 2,
    "gg_51_0": 1,
    "gg_51_1": 0,
    "gg_51_2": 2,
    "gg_51_3": 0,
    "gg_51_4": 2,
    "gg_52_0": 1,
    "gg_52_1": 0,
    "gg_52_2": 2,
    "gg_52_3": 0,
    "gg_52_4": 2,
    "gg_53_0": 1,
    "gg_53_1": 0,
    "gg_53_2": 2,
    "gg_53_3": 0,
    "gg_53_4": 2,
    "gg_54_0": 1,
    "gg_54_1": 0,
    "gg_54_2": 2,
    "gg_54_3": 0,
    "gg_54_4": 2,
    "gg_55_0": 1,
    "gg_55_1": 0,
    "gg_55_2": 2,
    "gg_55_3": 0,
    "gg_55_4": 2,
    "gg_56_0": 1,
    "gg_56_1": 0,
    "gg_56_2": 2,
    "gg_56_3": 0,
    "gg_56_4": 2,
    "gg_57_0": 1,
    "gg_57_1": 0,
    "gg_57_2": 2,
    "gg_57_3": 0,
    "gg_57_4": 2,
    "gg_58_0": 1,
    "gg_58_1": 0,
    "gg_58_2": 2,
    "gg_58_3": 0,
    "gg_58_4": 2,
    "gg_59_0": 1,
    "gg_59_1": 0,
    "gg_59_2": 2,
    "gg_59_3": 0,
    "gg_59_4": 2,
    "gg_60_0": 1,
    "gg_60_1": 0,
    "gg_60_2": 2,
    "gg_60_3": 0,
    "gg_60_4": 2,
    "gg_61_0": 1,
    "gg_61_1": 0,
    "gg_61_2": 2,
    "gg_61_3": 0,
    "gg_61_4": 2,
    "gg_62_0": 1,
    "gg_62_1": 0,
    "gg_62_2": 2,
    "gg_62_3": 0,
    "gg_62_4": 2,
    "gg_63_0": 1,
    "gg_63_1": 0,
    "gg_63_2": 2,
    "gg_63_3": 0,
    "gg_63_4": 2,
    "gg_64_0": 1,
    "gg_64_1": 0,
    "gg_64_2": 2,
    "gg_64_3": 0,
    "gg_64_4": 2,
    "gg_65_0": 1,
    "gg_65_1": 0,
    "gg_65_2": 2,
    "gg_65_3": 0,
    "gg_65_4": 2,
    "gg_66_0": 1,
    "gg_66_1": 0,
    "gg_66_2": 2,
    "gg_66_3": 0,
    "gg_66_4": 2,
    "gg_67_0": 1,
    "gg_67_1": 0,
    "gg_67_2": 2,
    "gg_67_3": 0,
    "gg_67_4": 2,
    "gg_68_0": 2,
    "gg_68_1": 0,
    "gg_68_2": 3,
    "gg_68_3": 0,
    "gg_68_4": 1,
    "gg_69_0": 2,
    "gg_69_1": 0,
    "gg_69_2": 3,
    "gg_69_3": 0,
    "gg_69_4": 1,
    "gg_70_0": 2,
    "gg_70_1": 0,
    "gg_70_2": 3,
    "gg_70_3": 0,
    "gg_70_4": 1,
    "gg_71_0": 1,
    "gg_71_1": 0,
    "gg_71_2": 1,
    "gg_72_0": 2,
    "gg_72_1": 0,
    "gg_72_2": 3,
    "gg_72_3": 0,
    "gg_72_4": 1,
    "gg_73_0": 2,
    "gg_73_1": 0,
    "gg_73_2": 3,
    "gg_73_3": 0,
    "gg_73_4": 1,
    "gg_74_0": 2,
    "gg_74_1": 0,
    "gg_74_2": 3,
    "gg_74_3": 0,
    "gg_74_4": 1,
    "gg_75_0": 2,
    "gg_75_1": 0,
    "gg_75_2": 3,
    "gg_75_3": 0,
    "gg_75_4": 1,
    "gg_76_0": 2,
    "gg_76_1": 0,
    "gg_76_2": 5,
    "gg_76_3": 0,
    "gg_76_4": 5,
    "gg_76_5": 0,
    "gg_76_6": 2,
    "gg_77_0": 2,
    "gg_77_1": 0,
    "gg_77_2": 3,
    "gg_77_3": 0,
    "gg_77_4": 1,
    "gg_78_0": 2,
    "gg_78_1": 0,
    "gg_78_2": 5,
    "gg_78_3": 0,
    "gg_78_4": 5,
    "gg_78_5": 0,
    "gg_78_6": 2,
    "gg_79_0": 2,
    "gg_79_1": 0,
    "gg_79_2": 3,
    "gg_79_3": 0,
    "gg_79_4": 1,
    "gg_80_0": 2,
    "gg_80_1": 0,
    "gg_80_2": 5,
    "gg_80_3": 0,
    "gg_80_4": 5,
    "gg_80_5": 0,
    "gg_80_6": 2,
    "gg_81_0": 2,
    "gg_81_1": 0,
    "gg_81_2": 3,
    "gg_81_3": 0,
    "gg_81_4": 1,
    "gg_82_0": 2,
    "gg_82_1": 0,
    "gg_82_2": 3,
    "gg_82_3": 0,
    "gg_82_4": 1,
    "gg_83_0": 2,
    "gg_83_1": 0,
    "gg_83_2": 3,
    "gg_83_3": 0,
    "gg_83_4": 1,
    "gg_84_0": 2,
    "gg_84_1": 0,
    "gg_84_2": 3,
    "gg_84_3": 0,
    "gg_84_4": 1,
    "gg_85_0": 2,
    "gg_85_1": 0,
    "gg_85_2": 3,
    "gg_85_3": 0,
    "gg_85_4": 1,
    "gg_86_0": 2,
    "gg_86_1": 0,
    "gg_86_2": 3,
    "gg_86_3": 0,
    "gg_86_4": 1,
    "gg_87_0": 2,
    "gg_87_1": 0,
    "gg_87_2": 3,
    "gg_87_3": 0,
    "gg_87_4": 1,
    "gg_88_0": 2,
    "gg_88_1": 0,
    "gg_88_2": 3,
    "gg_88_3": 0,
    "gg_88_4": 1,
    "gg_89_0": 2,
    "gg_89_1": 0,
    "gg_89_2": 3,
    "gg_89_3": 0,
    "gg_89_4": 1,
    "gg_90_0": 2,
    "gg_90_1": 0,
    "gg_90_2": 3,
    "gg_90_3": 0,
    "gg_90_4": 1,
    "gg_91_0": 2,
    "gg_91_1": 0,
    "gg_91_2": 3,
    "gg_91_3": 0,
    "gg_91_4": 1,
    "gg_92_0": 2,
    "gg_92_1": 0,
    "gg_92_2": 3,
    "gg_92_3": 0,
    "gg_92_4": 1,
    "gg_93_0": 2,
    "gg_93_1": 0,
    "gg_93_2": 3,
    "gg_93_3": 0,
    "gg_93_4": 1,
    "gg_94_0": 1,
    "gg_94_1": 0,
    "gg_94_2": 3,
    "gg_94_3": 0,
    "gg_94_4": 2,
    "gg_95_0": 2,
    "gg_95_1": 0,
    "gg_95_2": 5,
    "gg_95_3": 0,
    "gg_95_4": 5,
    "gg_95_5": 0,
    "gg_95_6": 2,
    "gg_96_0": 2,
    "gg_96_1": 0,
    "gg_96_2": 5,
    "gg_96_3": 0,
    "gg_96_4": 5,
    "gg_96_5": 0,
    "gg_96_6": 2,
    "gg_97_0": 2,
    "gg_97_1": 0,
    "gg_97_2": 5,
    "gg_97_3": 0,
    "gg_97_4": 5,
    "gg_97_5": 0,
    "gg_97_6": 2,
    "gg_98_0": 2,
    "gg_98_1": 0,
    "gg_98_2": 5,
    "gg_98_3": 0,
    "gg_98_4": 5,
    "gg_98_5": 0,
    "gg_98_6": 2,
    "gg_99_0": 2,
    "gg_99_1": 0,
    "gg_99_2": 5,
    "gg_99_3": 0,
    "gg_99_4": 5,
    "gg_99_5": 0,
    "gg_99_6": 2,
    "gg_100_0": 2,
    "gg_100_1": 0,
    "gg_100_2": 5,
    "gg_100_3": 0,
    "gg_100_4": 5,
    "gg_100_5": 0,
    "gg_100_6": 2,
    "gg_101_0": 1,
    "gg_101_1": 0,
    "gg_101_2": 3,
    "gg_101_3": 0,
    "gg_101_4": 2,
    "gg_102_0": 2,
    "gg_102_1": 0,
    "gg_102_2": 3,
    "gg_102_3": 0,
    "gg_102_4": 1,
    "gg_103_0": 2,
    "gg_103_1": 0,
    "gg_103_2": 4,
    "gg_103_3": 0,
    "gg_103_4": 3,
    "gg_103_5": 0,
    "gg_103_6": 1,
    "gg_104_0": 3,
    "gg_104_1": 0,
    "gg_104_2": 6,
    "gg_104_3": 0,
    "gg_104_4": 6,
    "gg_104_5": 0,
    "gg_104_6": 3,
    "gg_104_7": 0,
    "gg_104_8": 1,
    "gg_105_0": 2,
    "gg_105_1": 0,
    "gg_105_2": 3,
    "gg_105_3": 0,
    "gg_105_4": 1,
    "gg_106_0": 2,
    "gg_106_1": 0,
    "gg_106_2": 3,
    "gg_106_3": 0,
    "gg_106_4": 1,
    "gg_107_0": 2,
    "gg_107_1": 0,
    "gg_107_2": 3,
    "gg_107_3": 0,
    "gg_107_4": 1,
    "gg_108_0": 2,
    "gg_108_1": 0,
    "gg_108_2": 3,
    "gg_108_3": 0,
    "gg_108_4": 1,
    "gg_109_0": 2,
    "gg_109_1": 0,
    "gg_109_2": 3,
    "gg_109_3": 0,
    "gg_109_4": 1,
    "gg_110_0": 2,
    "gg_110_1": 0,
    "gg_110_2": 3,
    "gg_110_3": 0,
    "gg_110_4": 1,
    "gg_111_0": 2,
    "gg_111_1": 0,
    "gg_111_2": 3,
    "gg_111_3": 0,
    "gg_111_4": 1,
    "gg_112_0": 2,
    "gg_112_1": 0,
    "gg_112_2": 3,
    "gg_112_3": 0,
    "gg_112_4": 1,
    "gg_113_0": 2,
    "gg_113_1": 0,
    "gg_113_2": 3,
    "gg_113_3": 0,
    "gg_113_4": 1,
    "gg_114_0": 2,
    "gg_114_1": 0,
    "gg_114_2": 3,
    "gg_114_3": 0,
    "gg_114_4": 1,
    "gg_115_0": 2,
    "gg_115_1": 0,
    "gg_115_2": 5,
    "gg_115_3": 0,
    "gg_115_4": 5,
    "gg_115_5": 0,
    "gg_115_6": 2,
    "gg_116_0": 2,
    "gg_116_1": 0,
    "gg_116_2": 3,
    "gg_116_3": 0,
    "gg_116_4": 1,
    "gg_117_0": 2,
    "gg_117_1": 0,
    "gg_117_2": 5,
    "gg_117_3": 0,
    "gg_117_4": 5,
    "gg_117_5": 0,
    "gg_117_6": 2,
    "gg_118_0": 2,
    "gg_118_1": 0,
    "gg_118_2": 3,
    "gg_118_3": 0,
    "gg_118_4": 1,
    "gg_119_0": 2,
    "gg_119_1": 0,
    "gg_119_2": 3,
    "gg_119_3": 0,
    "gg_119_4": 1,
    "gg_120_0": 2,
    "gg_120_1": 0,
    "gg_120_2": 5,
    "gg_120_3": 0,
    "gg_120_4": 5,
    "gg_120_5": 0,
    "gg_120_6": 2,
    "gg_121_0": 2,
    "gg_121_1": 0,
    "gg_121_2": 5,
    "gg_121_3": 0,
    "gg_121_4": 5,
    "gg_121_5": 0,
    "gg_121_6": 2,
    "gg_122_0": 2,
    "gg_122_1": 0,
    "gg_122_2": 5,
    "gg_122_3": 0,
    "gg_122_4": 5,
    "gg_122_5": 0,
    "gg_122_6": 2,
    "gg_123_0": 2,
    "gg_123_1": 0,
    "gg_123_2": 4,
    "gg_123_3": 0,
    "gg_123_4": 3,
    "gg_123_5": 0,
    "gg_123_6": 1,
    "gg_124_0": 2,
    "gg_124_1": 0,
    "gg_124_2": 5,
    "gg_124_3": 0,
    "gg_124_4": 5,
    "gg_124_5": 0,
    "gg_124_6": 2,
    "gg_125_0": 2,
    "gg_125_1": 0,
    "gg_125_2": 4,
    "gg_125_3": 0,
    "gg_125_4": 3,
    "gg_125_5": 0,
    "gg_125_6": 1,
    "gg_126_0": 2,
    "gg_126_1": 0,
    "gg_126_2": 5,
    "gg_126_3": 0,
    "gg_126_4": 5,
    "gg_126_5": 0,
    "gg_126_6": 2,
    "gg_127_0": 2,
    "gg_127_1": 0,
    "gg_127_2": 4,
    "gg_127_3": 0,
    "gg_127_4": 3,
    "gg_127_5": 0,
    "gg_127_6": 1,
    "gv_0_0": 0,
    "gv_0_1": 1,
    "gv_1_0": 0,
    "gv_1_1": 1,
    "gv_2_0": 0,
    "gv_2_1": 1,
    "gv_3_0": 0,
    "gv_3_1": 1,
    "gv_4_0": 0,
    "gv_4_1": 1,
    "gv_5_0": 0,
    "gv_5_1": 1,
    "gv_6_0": 0,
    "gv_6_1": 1,
    "gv_7_0": 0,
    "gv_7_1": 1,
    "gv_8_0": 0,
    "gv_8_1": 1,
    "gv_9_0": 0,
    "gv_9_1": 1,
    "gv_10_0": 0,
    "gv_10_1": 1,
    "gv_11_0": 0,
    "gv_11_1": 1,
    "gv_11_2": 0,
    "gv_11_3": 1,
    "gv_12_0": 0,
    "gv_12_1": 1,
    "gv_13_0": 0,
    "gv_13_1": 1,
    "gv_13_2": 0,
    "gv_13_3": 1,
    "gv_14_0": 0,
    "gv_14_1": 1,
    "gv_15_0": 0,
    "gv_15_1": 1,
    "gv_16_0": 0,
    "gv_16_1": 1,
    "gv_17_0": 0,
    "gv_17_1": 1,
    "gv_18_0": 0,
    "gv_18_1": 1,
    "gv_19_0": 0,
    "gv_19_1": 1,
    "gv_20_0": 0,
    "gv_20_1": 1,
    "gv_21_0": 0,
    "gv_21_1": 1,
    "gv_22_0": 0,
    "gv_22_1": 1,
    "gv_23_0": 0,
    "gv_23_1": 1,
    "gv_24_0": 0,
    "gv_24_1": 1,
    "gv_25_0": 0,
    "gv_25_1": 1,
    "gv_25_2": 0,
    "gv_25_3": 1,
    "gv_26_0": 0,
    "gv_26_1": 1,
    "gv_26_2": 0,
    "gv_26_3": 1,
    "gv_27_0": 0,
    "gv_27_1": 1,
    "gv_27_2": 0,
    "gv_27_3": 1,
    "gv_28_0": 0,
    "gv_28_1": 1,
    "gv_28_2": 0,
    "gv_28_3": 1,
    "gv_29_0": 0,
    "gv_29_1": 1,
    "gv_29_2": 0,
    "gv_29_3": 1,
    "gv_30_0": 0,
    "gv_30_1": 1,
    "gv_30_2": 0,
    "gv_30_3": 1,
    "gv_31_0": 0,
    "gv_31_1": 1,
    "gv_31_2": 0,
    "gv_31_3": 1,
    "gv_32_0": 0,
    "gv_32_1": 1,
    "gv_32_2": 0,
    "gv_32_3": 1,
    "gv_33_0": 0,
    "gv_33_1": 1,
    "gv_33_2": 0,
    "gv_33_3": 1,
    "gv_34_0": 0,
    "gv_34_1": 1,
    "gv_34_2": 0,
    "gv_34_3": 1,
    "gv_35_0": 0,
    "gv_35_1": 1,
    "gv_35_2": 0,
    "gv_35_3": 1,
    "gv_36_0": 0,
    "gv_36_1": 1,
    "gv_36_2": 0,
    "gv_36_3": 1,
    "gv_37_0": 0,
    "gv_37_1": 1,
    "gv_37_2": 0,
    "gv_37_3": 1,
    "gv_38_0": 0,
    "gv_38_1": 1,
    "gv_38_2": 0,
    "gv_38_3": 1,
    "gv_38_4": 0,
    "gv_38_5": 1,
    "gv_39_0": 0,
    "gv_39_1": 1,
    "gv_39_2": 0,
    "gv_39_3": 1,
    "gv_40_0": 0,
    "gv_40_1": 1,
    "gv_40_2": 0,
    "gv_40_3": 1,
    "gv_40_4": 0,
    "gv_40_5": 1,
    "gv_41_0": 0,
    "gv_41_1": 1,
    "gv_41_2": 0,
    "gv_41_3": 1,
    "gv_42_0": 0,
    "gv_42_1": 1,
    "gv_42_2": 0,
    "gv_42_3": 1,
    "gv_43_0": 0,
    "gv_43_1": 1,
    "gv_43_2": 0,
    "gv_43_3": 1,
    "gv_44_0": 0,
    "gv_44_1": 1,
    "gv_44_2": 0,
    "gv_44_3": 1,
    "gv_45_0": 0,
    "gv_45_1": 1,
    "gv_45_2": 0,
    "gv_45_3": 1,
    "gv_46_0": 0,
    "gv_46_1": 1,
    "gv_46_2": 0,
    "gv_46_3": 1,
    "gv_47_0": 0,
    "gv_47_1": 1,
    "gv_47_2": 0,
    "gv_47_3": 1,
    "gv_48_0": 0,
    "gv_48_1": 1,
    "gv_48_2": 0,
    "gv_48_3": 1,
    "gv_49_0": 0,
    "gv_49_1": 1,
    "gv_49_2": 0,
    "gv_49_3": 1,
    "gv_50_0": 0,
    "gv_50_1": 1,
    "gv_50_2": 0,
    "gv_50_3": 1,
    "gv_51_0": 0,
    "gv_51_1": 1,
    "gv_51_2": 0,
    "gv_51_3": 1,
    "gv_52_0": 0,
    "gv_52_1": 1,
    "gv_52_2": 0,
    "gv_52_3": 1,
    "gv_53_0": 0,
    "gv_53_1": 1,
    "gv_53_2": 0,
    "gv_53_3": 1,
    "gv_54_0": 0,
    "gv_54_1": 1,
    "gv_54_2": 0,
    "gv_54_3": 1,
    "gv_55_0": 0,
    "gv_55_1": 1,
    "gv_55_2": 0,
    "gv_55_3": 1,
    "gv_56_0": 0,
    "gv_56_1": 1,
    "gv_56_2": 0,
    "gv_56_3": 1,
    "gv_57_0": 0,
    "gv_57_1": 1,
    "gv_57_2": 0,
    "gv_57_3": 1,
    "gv_58_0": 0,
    "gv_58_1": 1,
    "gv_58_2": 0,
    "gv_58_3": 1,
    "gv_59_0": 0,
    "gv_59_1": 2,
    "gv_59_2": 0,
    "gv_59_3": 1,
    "gv_60_0": 0,
    "gv_60_1": 2,
    "gv_60_2": 0,
    "gv_60_3": 1,
    "gv_61_0": 0,
    "gv_61_1": 2,
    "gv_61_2": 0,
    "gv_61_3": 1,
    "gv_62_0": 0,
    "gv_62_1": 1,
    "gv_63_0": 0,
    "gv_63_1": 2,
    "gv_63_2": 0,
    "gv_63_3": 1,
    "gv_64_0": 0,
    "gv_64_1": 2,
    "gv_64_2": 0,
    "gv_64_3": 1,
    "gv_65_0": 0,
    "gv_65_1": 2,
    "gv_65_2": 0,
    "gv_65_3": 1,
    "gv_66_0": 0,
    "gv_66_1": 2,
    "gv_66_2": 0,
    "gv_66_3": 1,
    "gv_67_0": 0,
    "gv_67_1": 3,
    "gv_67_2": 0,
    "gv_67_3": 4,
    "gv_67_4": 0,
    "gv_67_5": 2,
    "gv_68_0": 0,
    "gv_68_1": 2,
    "gv_68_2": 0,
    "gv_68_3": 1,
    "gv_69_0": 0,
    "gv_69_1": 3,
    "gv_69_2": 0,
    "gv_69_3": 4,
    "gv_69_4": 0,
    "gv_69_5": 2,
    "gv_70_0": 0,
    "gv_70_1": 2,
    "gv_70_2": 0,
    "gv_70_3": 1,
    "gv_71_0": 0,
    "gv_71_1": 3,
    "gv_71_2": 0,
    "gv_71_3": 4,
    "gv_71_4": 0,
    "gv_71_5": 2,
    "gv_72_0": 0,
    "gv_72_1": 2,
    "gv_72_2": 0,
    "gv_72_3": 1,
    "gv_73_0": 0,
    "gv_73_1": 2,
    "gv_73_2": 0,
    "gv_73_3": 1,
    "gv_74_0": 0,
    "gv_74_1": 2,
    "gv_74_2": 0,
    "gv_74_3": 1,
    "gv_75_0": 0,
    "gv_75_1": 2,
    "gv_75_2": 0,
    "gv_75_3": 1,
    "gv_76_0": 0,
    "gv_76_1": 2,
    "gv_76_2": 0,
    "gv_76_3": 1,
    "gv_77_0": 0,
    "gv_77_1": 2,
    "gv_77_2": 0,
    "gv_77_3": 1,
    "gv_78_0": 0,
    "gv_78_1": 2,
    "gv_78_2": 0,
    "gv_78_3": 1,
    "gv_79_0": 0,
    "gv_79_1": 2,
    "gv_79_2": 0,
    "gv_79_3": 1,
    "gv_80_0": 0,
    "gv_80_1": 2,
    "gv_80_2": 0,
    "gv_80_3": 1,
    "gv_81_0": 0,
    "gv_81_1": 2,
    "gv_81_2": 0,
    "gv_81_3": 1,
    "gv_82_0": 0,
    "gv_82_1": 2,
    "gv_82_2": 0,
    "gv_82_3": 1,
    "gv_83_0": 0,
    "gv_83_1": 2,
    "gv_83_2": 0,
    "gv_83_3": 1,
    "gv_84_0": 0,
    "gv_84_1": 2,
    "gv_84_2": 0,
    "gv_84_3": 1,
    "gv_85_0": 0,
    "gv_85_1": 2,
    "gv_85_2": 0,
    "gv_85_3": 2,
    "gv_86_0": 0,
    "gv_86_1": 3,
    "gv_86_2": 0,
    "gv_86_3": 4,
    "gv_86_4": 0,
    "gv_86_5": 2,
    "gv_87_0": 0,
    "gv_87_1": 3,
    "gv_87_2": 0,
    "gv_87_3": 4,
    "gv_87_4": 0,
    "gv_87_5": 2,
    "gv_88_0": 0,
    "gv_88_1": 3,
    "gv_88_2": 0,
    "gv_88_3": 4,
    "gv_88_4": 0,
    "gv_88_5": 2,
    "gv_89_0": 0,
    "gv_89_1": 3,
    "gv_89_2": 0,
    "gv_89_3": 4,
    "gv_89_4": 0,
    "gv_89_5": 2,
    "gv_90_0": 0,
    "gv_90_1": 3,
    "gv_90_2": 0,
    "gv_90_3": 4,
    "gv_90_4": 0,
    "gv_90_5": 2,
    "gv_91_0": 0,
    "gv_91_1": 3,
    "gv_91_2": 0,
    "gv_91_3": 4,
    "gv_91_4": 0,
    "gv_91_5": 2,
    "gv_92_0": 0,
    "gv_92_1": 2,
    "gv_92_2": 0,
    "gv_92_3": 2,
    "gv_93_0": 0,
    "gv_93_1": 2,
    "gv_93_2": 0,
    "gv_93_3": 1,
    "gv_94_0": 0,
    "gv_94_1": 3,
    "gv_94_2": 0,
    "gv_94_3": 3,
    "gv_94_4": 0,
    "gv_94_5": 1,
    "gv_95_0": 0,
    "gv_95_1": 4,
    "gv_95_2": 0,
    "gv_95_3": 5,
    "gv_95_4": 0,
    "gv_95_5": 3,
    "gv_95_6": 0,
    "gv_95_7": 1,
    "gv_96_0": 0,
    "gv_96_1": 2,
    "gv_96_2": 0,
    "gv_96_3": 1,
    "gv_97_0": 0,
    "gv_97_1": 2,
    "gv_97_2": 0,
    "gv_97_3": 1,
    "gv_98_0": 0,
    "gv_98_1": 2,
    "gv_98_2": 0,
    "gv_98_3": 1,
    "gv_99_0": 0,
    "gv_99_1": 2,
    "gv_99_2": 0,
    "gv_99_3": 1,
    "gv_100_0": 0,
    "gv_100_1": 2,
    "gv_100_2": 0,
    "gv_100_3": 1,
    "gv_101_0": 0,
    "gv_101_1": 2,
    "gv_101_2": 0,
    "gv_101_3": 1,
    "gv_102_0": 0,
    "gv_102_1": 2,
    "gv_102_2": 0,
    "gv_102_3": 1,
    "gv_103_0": 0,
    "gv_103_1": 2,
    "gv_103_2": 0,
    "gv_103_3": 1,
    "gv_104_0": 0,
    "gv_104_1": 2,
    "gv_104_2": 0,
    "gv_104_3": 1,
    "gv_105_0": 0,
    "gv_105_1": 2,
    "gv_105_2": 0,
    "gv_105_3": 1,
    "gv_106_0": 0,
    "gv_106_1": 3,
    "gv_106_2": 0,
    "gv_106_3": 4,
    "gv_106_4": 0,
    "gv_106_5": 2,
    "gv_107_0": 0,
    "gv_107_1": 2,
    "gv_107_2": 0,
    "gv_107_3": 1,
    "gv_108_0": 0,
    "gv_108_1": 3,
    "gv_108_2": 0,
    "gv_108_3": 4,
    "gv_108_4": 0,
    "gv_108_5": 2,
    "gv_109_0": 0,
    "gv_109_1": 2,
    "gv_109_2": 0,
    "gv_109_3": 1,
    "gv_110_0": 0,
    "gv_110_1": 2,
    "gv_110_2": 0,
    "gv_110_3": 1,
    "gv_111_0": 0,
    "gv_111_1": 3,
    "gv_111_2": 0,
    "gv_111_3": 4,
    "gv_111_4": 0,
    "gv_111_5": 2,
    "gv_112_0": 0,
    "gv_112_1": 3,
    "gv_112_2": 0,
    "gv_112_3": 4,
    "gv_112_4": 0,
    "gv_112_5": 2,
    "gv_113_0": 0,
    "gv_113_1": 3,
    "gv_113_2": 0,
    "gv_113_3": 4,
    "gv_113_4": 0,
    "gv_113_5": 2,
    "gv_114_0": 0,
    "gv_114_1": 3,
    "gv_114_2": 0,
    "gv_114_3": 3,
    "gv_114_4": 0,
    "gv_114_5": 1,
    "gv_115_0": 0,
    "gv_115_1": 3,
    "gv_115_2": 0,
    "gv_115_3": 4,
    "gv_115_4": 0,
    "gv_115_5": 2,
    "gv_116_0": 0,
    "gv_116_1": 3,
    "gv_116_2": 0,
    "gv_116_3": 3,
    "gv_116_4": 0,
    "gv_116_5": 1,
    "gv_117_0": 0,
    "gv_117_1": 3,
    "gv_117_2": 0,
    "gv_117_3": 4,
    "gv_117_4": 0,
    "gv_117_5": 2,
    "gv_118_0": 0,
    "gv_118_1": 3,
    "gv_118_2": 0,
    "gv_118_3": 3,
    "gv_118_4": 0,
    "gv_118_5": 1,
    "vv_0_0": 1,
    "vv_0_1": 0,
    "vv_0_2": 1,
    "vv_1_0": 1,
    "vv_1_1": 0,
    "vv_1_2": 1,
    "vv_2_0": 1,
    "vv_2_1": 0,
    "vv_2_2": 1,
    "vv_3_0": 1,
    "vv_4_0": 1,
    "vv_4_1": 0,
    "vv_4_2": 1,
    "vv_5_0": 1,
    "vv_5_1": 0,
    "vv_5_2": 1,
    "vv_6_0": 1,
    "vv_6_1": 0,
    "vv_6_2": 1,
    "vv_7_0": 1,
    "vv_7_1": 0,
    "vv_7_2": 1,
    "vv_8_0": 1,
    "vv_8_1": 0,
    "vv_8_2": 3,
    "vv_8_3": 0,
    "vv_8_4": 2,
    "vv_9_0": 1,
    "vv_9_1": 0,
    "vv_9_2": 1,
    "vv_10_0": 1,
    "vv_10_1": 0,
    "vv_10_2": 3,
    "vv_10_3": 0,
    "vv_10_4": 2,
    "vv_11_0": 1,
    "vv_11_1": 0,
    "vv_11_2": 1,
    "vv_12_0": 1,
    "vv_12_1": 0,
    "vv_12_2": 3,
    "vv_12_3": 0,
    "vv_12_4": 2,
    "vv_13_0": 1,
    "vv_13_1": 0,
    "vv_13_2": 1,
    "vv_14_0": 1,
    "vv_14_1": 0,
    "vv_14_2": 1,
    "vv_15_0": 1,
    "vv_15_1": 0,
    "vv_15_2": 1,
    "vv_16_0": 1,
    "vv_16_1": 0,
    "vv_16_2": 1,
    "vv_17_0": 1,
    "vv_17_1": 0,
    "vv_17_2": 1,
    "vv_18_0": 1,
    "vv_18_1": 0,
    "vv_18_2": 1,
    "vv_19_0": 1,
    "vv_19_1": 0,
    "vv_19_2": 1,
    "vv_20_0": 1,
    "vv_20_1": 0,
    "vv_20_2": 1,
    "vv_21_0": 1,
    "vv_21_1": 0,
    "vv_21_2": 1,
    "vv_22_0": 1,
    "vv_22_1": 0,
    "vv_22_2": 1,
    "vv_23_0": 1,
    "vv_23_1": 0,
    "vv_23_2": 1,
    "vv_24_0": 1,
    "vv_24_1": 0,
    "vv_24_2": 1,
    "vv_25_0": 1,
    "vv_25_1": 0,
    "vv_25_2": 1,
    "vv_26_0": 1,
    "vv_26_1": 0,
    "vv_26_2": 2,
    "vv_27_0": 1,
    "vv_27_1": 0,
    "vv_27_2": 3,
    "vv_27_3": 0,
    "vv_27_4": 2,
    "vv_28_0": 1,
    "vv_28_1": 0,
    "vv_28_2": 3,
    "vv_28_3": 0,
    "vv_28_4": 2,
    "vv_29_0": 1,
    "vv_29_1": 0,
    "vv_29_2": 3,
    "vv_29_3": 0,
    "vv_29_4": 2,
    "vv_30_0": 1,
    "vv_30_1": 0,
    "vv_30_2": 3,
    "vv_30_3": 0,
    "vv_30_4": 2,
    "vv_31_0": 1,
    "vv_31_1": 0,
    "vv_31_2": 3,
    "vv_31_3": 0,
    "vv_31_4": 2,
    "vv_32_0": 1,
    "vv_32_1": 0,
    "vv_32_2": 3,
    "vv_32_3": 0,
    "vv_32_4": 2,
    "vv_33_0": 1,
    "vv_33_1": 0,
    "vv_33_2": 2,
    "vv_34_0": 1,
    "vv_34_1": 0,
    "vv_34_2": 1,
    "vv_35_0": 2,
    "vv_35_1": 0,
    "vv_35_2": 3,
    "vv_35_3": 0,
    "vv_35_4": 1,
    "vv_36_0": 2,
    "vv_36_1": 0,
    "vv_36_2": 4,
    "vv_36_3": 0,
    "vv_36_4": 3,
    "vv_36_5": 0,
    "vv_36_6": 1,
    "vv_37_0": 1,
    "vv_37_1": 0,
    "vv_37_2": 1,
    "vv_38_0": 1,
    "vv_38_1": 0,
    "vv_38_2": 1,
    "vv_39_0": 1,
    "vv_39_1": 0,
    "vv_39_2": 1,
    "vv_40_0": 1,
    "vv_40_1": 0,
    "vv_40_2": 1,
    "vv_41_0": 1,
    "vv_41_1": 0,
    "vv_41_2": 1,
    "vv_42_0": 1,
    "vv_42_1": 0,
    "vv_42_2": 1,
    "vv_43_0": 1,
    "vv_43_1": 0,
    "vv_43_2": 1,
    "vv_44_0": 1,
    "vv_44_1": 0,
    "vv_44_2": 1,
    "vv_45_0": 1,
    "vv_45_1": 0,
    "vv_45_2": 1,
    "vv_46_0": 1,
    "vv_46_1": 0,
    "vv_46_2": 1,
    "vv_47_0": 1,
    "vv_47_1": 0,
    "vv_47_2": 3,
    "vv_47_3": 0,
    "vv_47_4": 2,
    "vv_48_0": 1,
    "vv_48_1": 0,
    "vv_48_2": 1,
    "vv_49_0": 1,
    "vv_49_1": 0,
    "vv_49_2": 3,
    "vv_49_3": 0,
    "vv_49_4": 2,
    "vv_50_0": 1,
    "vv_50_1": 0,
    "vv_50_2": 1,
    "vv_51_0": 1,
    "vv_51_1": 0,
    "vv_51_2": 1,
    "vv_52_0": 1,
    "vv_52_1": 0,
    "vv_52_2": 3,
    "vv_52_3": 0,
    "vv_52_4": 2,
    "vv_53_0": 1,
    "vv_53_1": 0,
    "vv_53_2": 3,
    "vv_53_3": 0,
    "vv_53_4": 2,
    "vv_54_0": 1,
    "vv_54_1": 0,
    "vv_54_2": 3,
    "vv_54_3": 0,
    "vv_54_4": 2,
    "vv_55_0": 2,
    "vv_55_1": 0,
    "vv_55_2": 3,
    "vv_55_3": 0,
    "vv_55_4": 1,
    "vv_56_0": 1,
    "vv_56_1": 0,
    "vv_56_2": 3,
    "vv_56_3": 0,
    "vv_56_4": 2,
    "vv_57_0": 2,
    "vv_57_1": 0,
    "vv_57_2": 3,
    "vv_57_3": 0,
    "vv_57_4": 1,
    "vv_58_0": 1,
    "vv_58_1": 0,
    "vv_58_2": 3,
    "vv_58_3": 0,
    "vv_58_4": 2,
    "vv_59_0": 2,
    "vv_59_1": 0,
    "vv_59_2": 3,
    "vv_59_3": 0,
    "vv_59_4": 1,
}
multi_index_model = False
regularize_M_terms = None
