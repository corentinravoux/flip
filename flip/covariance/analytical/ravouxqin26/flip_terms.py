import mpmath
import numpy
import scipy


def set_backend(module):
    global np, erf
    if module == "numpy":
        np = numpy
        erf = scipy.special.erf
    elif module == "mpmath":
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
        return 4 / 15

    return func


def N_gg_9_2_0(theta, phi):
    return (15 / 4) * np.cos(2 * phi) + 5 / 4


def M_gg_10_0_0():
    def func(k):
        return 4 / 3

    return func


def N_gg_10_0_0(theta, phi):
    return 1


def M_gg_10_2_0():
    def func(k):
        return 8 / 15

    return func


def N_gg_10_2_0(theta, phi):
    return (15 / 4) * np.cos(2 * phi) + 5 / 4


def M_gg_11_0_0():
    def func(k):
        return 4 / 3

    return func


def N_gg_11_0_0(theta, phi):
    return 1


def M_gg_11_2_0():
    def func(k):
        return 8 / 15

    return func


def N_gg_11_2_0(theta, phi):
    return (15 / 4) * np.cos(2 * phi) + 5 / 4


def M_gg_12_0_0():
    def func(k):
        return 4 * k**2

    return func


def N_gg_12_0_0(theta, phi):
    return 1


def M_gg_12_2_0():
    def func(k):
        return (8 / 5) * k**2

    return func


def N_gg_12_2_0(theta, phi):
    return (15 / 4) * np.cos(2 * phi) + 5 / 4


def M_gg_13_0_0():
    def func(k):
        return 4 * k**2

    return func


def N_gg_13_0_0(theta, phi):
    return 1


def M_gg_13_2_0():
    def func(k):
        return (8 / 5) * k**2

    return func


def N_gg_13_2_0(theta, phi):
    return (15 / 4) * np.cos(2 * phi) + 5 / 4


def M_gg_14_0_0():
    def func(k):
        return -2 / 3

    return func


def N_gg_14_0_0(theta, phi):
    return 1


def M_gg_14_2_0():
    def func(k):
        return -4 / 15

    return func


def N_gg_14_2_0(theta, phi):
    return (15 / 4) * np.cos(2 * phi) + 5 / 4


def M_gg_15_0_0():
    def func(k):
        return -2 / 3

    return func


def N_gg_15_0_0(theta, phi):
    return 1


def M_gg_15_2_0():
    def func(k):
        return -4 / 15

    return func


def N_gg_15_2_0(theta, phi):
    return (15 / 4) * np.cos(2 * phi) + 5 / 4


def M_gg_16_0_0():
    def func(k):
        return -2 / 3

    return func


def N_gg_16_0_0(theta, phi):
    return 1


def M_gg_16_2_0():
    def func(k):
        return -4 / 15

    return func


def N_gg_16_2_0(theta, phi):
    return (15 / 4) * np.cos(2 * phi) + 5 / 4


def M_gg_17_0_0():
    def func(k):
        return -2 / 3

    return func


def N_gg_17_0_0(theta, phi):
    return 1


def M_gg_17_2_0():
    def func(k):
        return -4 / 15

    return func


def N_gg_17_2_0(theta, phi):
    return (15 / 4) * np.cos(2 * phi) + 5 / 4


def M_gg_18_0_0():
    def func(k):
        return -2 / 3

    return func


def N_gg_18_0_0(theta, phi):
    return 1


def M_gg_18_2_0():
    def func(k):
        return -4 / 15

    return func


def N_gg_18_2_0(theta, phi):
    return (15 / 4) * np.cos(2 * phi) + 5 / 4


def M_gg_19_0_0():
    def func(k):
        return 1 / 3

    return func


def N_gg_19_0_0(theta, phi):
    return 1


def M_gg_19_2_0():
    def func(k):
        return 2 / 15

    return func


def N_gg_19_2_0(theta, phi):
    return (15 / 4) * np.cos(2 * phi) + 5 / 4


def M_gg_20_0_0():
    def func(k):
        return 1 / 5

    return func


def N_gg_20_0_0(theta, phi):
    return 1


def M_gg_20_2_0():
    def func(k):
        return 4 / 35

    return func


def N_gg_20_2_0(theta, phi):
    return (15 / 4) * np.cos(2 * phi) + 5 / 4


def M_gg_20_4_0():
    def func(k):
        return 8 / 315

    return func


def N_gg_20_4_0(theta, phi):
    return (45 / 16) * np.cos(2 * phi) + (315 / 64) * np.cos(4 * phi) + 81 / 64


def M_gg_21_0_0():
    def func(k):
        return (2 / 3) * k**2

    return func


def N_gg_21_0_0(theta, phi):
    return 1


def M_gg_21_2_0():
    def func(k):
        return (4 / 15) * k**2

    return func


def N_gg_21_2_0(theta, phi):
    return (15 / 4) * np.cos(2 * phi) + 5 / 4


def M_gg_22_0_0():
    def func(k):
        return (2 / 5) * k**2

    return func


def N_gg_22_0_0(theta, phi):
    return 1


def M_gg_22_2_0():
    def func(k):
        return (8 / 35) * k**2

    return func


def N_gg_22_2_0(theta, phi):
    return (15 / 4) * np.cos(2 * phi) + 5 / 4


def M_gg_22_4_0():
    def func(k):
        return (16 / 315) * k**2

    return func


def N_gg_22_4_0(theta, phi):
    return (45 / 16) * np.cos(2 * phi) + (315 / 64) * np.cos(4 * phi) + 81 / 64


def M_gg_23_0_0():
    def func(k):
        return -1 / 3 * k**2

    return func


def N_gg_23_0_0(theta, phi):
    return 1


def M_gg_23_2_0():
    def func(k):
        return -2 / 15 * k**2

    return func


def N_gg_23_2_0(theta, phi):
    return (15 / 4) * np.cos(2 * phi) + 5 / 4


def M_gg_24_0_0():
    def func(k):
        return -2 / 3 * k**2

    return func


def N_gg_24_0_0(theta, phi):
    return 1


def M_gg_24_2_0():
    def func(k):
        return -4 / 15 * k**2

    return func


def N_gg_24_2_0(theta, phi):
    return (15 / 4) * np.cos(2 * phi) + 5 / 4


def M_gg_25_0_0():
    def func(k):
        return -2 / 3 * k**2

    return func


def N_gg_25_0_0(theta, phi):
    return 1


def M_gg_25_2_0():
    def func(k):
        return -4 / 15 * k**2

    return func


def N_gg_25_2_0(theta, phi):
    return (15 / 4) * np.cos(2 * phi) + 5 / 4


def M_gg_26_0_0():
    def func(k):
        return -2 / 3 * k**2

    return func


def N_gg_26_0_0(theta, phi):
    return 1


def M_gg_26_2_0():
    def func(k):
        return -4 / 15 * k**2

    return func


def N_gg_26_2_0(theta, phi):
    return (15 / 4) * np.cos(2 * phi) + 5 / 4


def M_gg_27_0_0():
    def func(k):
        return -2 * k**4

    return func


def N_gg_27_0_0(theta, phi):
    return 1


def M_gg_27_2_0():
    def func(k):
        return -4 / 5 * k**4

    return func


def N_gg_27_2_0(theta, phi):
    return (15 / 4) * np.cos(2 * phi) + 5 / 4


def M_gg_28_0_0():
    def func(k):
        return -1 / 6 * k**2

    return func


def N_gg_28_0_0(theta, phi):
    return 1


def M_gg_28_2_0():
    def func(k):
        return -1 / 15 * k**2

    return func


def N_gg_28_2_0(theta, phi):
    return (15 / 4) * np.cos(2 * phi) + 5 / 4


def M_gg_29_0_0():
    def func(k):
        return -1 / 6 * k**2

    return func


def N_gg_29_0_0(theta, phi):
    return 1


def M_gg_29_2_0():
    def func(k):
        return -1 / 15 * k**2

    return func


def N_gg_29_2_0(theta, phi):
    return (15 / 4) * np.cos(2 * phi) + 5 / 4


def M_gg_30_0_0():
    def func(k):
        return -1 / 3 * k**2

    return func


def N_gg_30_0_0(theta, phi):
    return 1


def M_gg_30_2_0():
    def func(k):
        return -2 / 15 * k**2

    return func


def N_gg_30_2_0(theta, phi):
    return (15 / 4) * np.cos(2 * phi) + 5 / 4


def M_gg_31_0_0():
    def func(k):
        return -2 / 3 * k**2

    return func


def N_gg_31_0_0(theta, phi):
    return 1


def M_gg_31_2_0():
    def func(k):
        return -4 / 15 * k**2

    return func


def N_gg_31_2_0(theta, phi):
    return (15 / 4) * np.cos(2 * phi) + 5 / 4


def M_gg_32_0_0():
    def func(k):
        return 1 / 3

    return func


def N_gg_32_0_0(theta, phi):
    return 1


def M_gg_32_2_0():
    def func(k):
        return 2 / 15

    return func


def N_gg_32_2_0(theta, phi):
    return (15 / 4) * np.cos(2 * phi) + 5 / 4


def M_gg_33_0_0():
    def func(k):
        return 1 / 3

    return func


def N_gg_33_0_0(theta, phi):
    return 1


def M_gg_33_2_0():
    def func(k):
        return 2 / 15

    return func


def N_gg_33_2_0(theta, phi):
    return (15 / 4) * np.cos(2 * phi) + 5 / 4


def M_gg_34_0_0():
    def func(k):
        return 1 / 5

    return func


def N_gg_34_0_0(theta, phi):
    return 1


def M_gg_34_2_0():
    def func(k):
        return 4 / 35

    return func


def N_gg_34_2_0(theta, phi):
    return (15 / 4) * np.cos(2 * phi) + 5 / 4


def M_gg_34_4_0():
    def func(k):
        return 8 / 315

    return func


def N_gg_34_4_0(theta, phi):
    return (45 / 16) * np.cos(2 * phi) + (315 / 64) * np.cos(4 * phi) + 81 / 64


def M_gg_35_0_0():
    def func(k):
        return 1 / 5

    return func


def N_gg_35_0_0(theta, phi):
    return 1


def M_gg_35_2_0():
    def func(k):
        return 4 / 35

    return func


def N_gg_35_2_0(theta, phi):
    return (15 / 4) * np.cos(2 * phi) + 5 / 4


def M_gg_35_4_0():
    def func(k):
        return 8 / 315

    return func


def N_gg_35_4_0(theta, phi):
    return (45 / 16) * np.cos(2 * phi) + (315 / 64) * np.cos(4 * phi) + 81 / 64


def M_gg_36_0_0():
    def func(k):
        return 1 / 5

    return func


def N_gg_36_0_0(theta, phi):
    return 1


def M_gg_36_2_0():
    def func(k):
        return 4 / 35

    return func


def N_gg_36_2_0(theta, phi):
    return (15 / 4) * np.cos(2 * phi) + 5 / 4


def M_gg_36_4_0():
    def func(k):
        return 8 / 315

    return func


def N_gg_36_4_0(theta, phi):
    return (45 / 16) * np.cos(2 * phi) + (315 / 64) * np.cos(4 * phi) + 81 / 64


def M_gg_37_0_0():
    def func(k):
        return 2 / 5

    return func


def N_gg_37_0_0(theta, phi):
    return 1


def M_gg_37_2_0():
    def func(k):
        return 8 / 35

    return func


def N_gg_37_2_0(theta, phi):
    return (15 / 4) * np.cos(2 * phi) + 5 / 4


def M_gg_37_4_0():
    def func(k):
        return 16 / 315

    return func


def N_gg_37_4_0(theta, phi):
    return (45 / 16) * np.cos(2 * phi) + (315 / 64) * np.cos(4 * phi) + 81 / 64


def M_gg_38_0_0():
    def func(k):
        return 4 / 5

    return func


def N_gg_38_0_0(theta, phi):
    return 1


def M_gg_38_2_0():
    def func(k):
        return 16 / 35

    return func


def N_gg_38_2_0(theta, phi):
    return (15 / 4) * np.cos(2 * phi) + 5 / 4


def M_gg_38_4_0():
    def func(k):
        return 32 / 315

    return func


def N_gg_38_4_0(theta, phi):
    return (45 / 16) * np.cos(2 * phi) + (315 / 64) * np.cos(4 * phi) + 81 / 64


def M_gg_39_0_0():
    def func(k):
        return 1 / 3

    return func


def N_gg_39_0_0(theta, phi):
    return 1


def M_gg_39_2_0():
    def func(k):
        return 2 / 15

    return func


def N_gg_39_2_0(theta, phi):
    return (15 / 4) * np.cos(2 * phi) + 5 / 4


def M_gg_40_0_0():
    def func(k):
        return 1 / 5

    return func


def N_gg_40_0_0(theta, phi):
    return 1


def M_gg_40_2_0():
    def func(k):
        return 4 / 35

    return func


def N_gg_40_2_0(theta, phi):
    return (15 / 4) * np.cos(2 * phi) + 5 / 4


def M_gg_40_4_0():
    def func(k):
        return 8 / 315

    return func


def N_gg_40_4_0(theta, phi):
    return (45 / 16) * np.cos(2 * phi) + (315 / 64) * np.cos(4 * phi) + 81 / 64


def M_gg_41_0_0():
    def func(k):
        return (12 / 5) * k**2

    return func


def N_gg_41_0_0(theta, phi):
    return 1


def M_gg_41_2_0():
    def func(k):
        return (48 / 35) * k**2

    return func


def N_gg_41_2_0(theta, phi):
    return (15 / 4) * np.cos(2 * phi) + 5 / 4


def M_gg_41_4_0():
    def func(k):
        return (32 / 105) * k**2

    return func


def N_gg_41_4_0(theta, phi):
    return (45 / 16) * np.cos(2 * phi) + (315 / 64) * np.cos(4 * phi) + 81 / 64


def M_gg_42_0_0():
    def func(k):
        return (6 / 5) * k**2

    return func


def N_gg_42_0_0(theta, phi):
    return 1


def M_gg_42_2_0():
    def func(k):
        return (24 / 35) * k**2

    return func


def N_gg_42_2_0(theta, phi):
    return (15 / 4) * np.cos(2 * phi) + 5 / 4


def M_gg_42_4_0():
    def func(k):
        return (16 / 105) * k**2

    return func


def N_gg_42_4_0(theta, phi):
    return (45 / 16) * np.cos(2 * phi) + (315 / 64) * np.cos(4 * phi) + 81 / 64


def M_gg_43_0_0():
    def func(k):
        return -1 / 5 * k**2

    return func


def N_gg_43_0_0(theta, phi):
    return 1


def M_gg_43_2_0():
    def func(k):
        return -4 / 35 * k**2

    return func


def N_gg_43_2_0(theta, phi):
    return (15 / 4) * np.cos(2 * phi) + 5 / 4


def M_gg_43_4_0():
    def func(k):
        return -8 / 315 * k**2

    return func


def N_gg_43_4_0(theta, phi):
    return (45 / 16) * np.cos(2 * phi) + (315 / 64) * np.cos(4 * phi) + 81 / 64


def M_gg_44_0_0():
    def func(k):
        return -2 / 5 * k**2

    return func


def N_gg_44_0_0(theta, phi):
    return 1


def M_gg_44_2_0():
    def func(k):
        return -8 / 35 * k**2

    return func


def N_gg_44_2_0(theta, phi):
    return (15 / 4) * np.cos(2 * phi) + 5 / 4


def M_gg_44_4_0():
    def func(k):
        return -16 / 315 * k**2

    return func


def N_gg_44_4_0(theta, phi):
    return (45 / 16) * np.cos(2 * phi) + (315 / 64) * np.cos(4 * phi) + 81 / 64


def M_gg_45_0_0():
    def func(k):
        return -2 / 5 * k**2

    return func


def N_gg_45_0_0(theta, phi):
    return 1


def M_gg_45_2_0():
    def func(k):
        return -8 / 35 * k**2

    return func


def N_gg_45_2_0(theta, phi):
    return (15 / 4) * np.cos(2 * phi) + 5 / 4


def M_gg_45_4_0():
    def func(k):
        return -16 / 315 * k**2

    return func


def N_gg_45_4_0(theta, phi):
    return (45 / 16) * np.cos(2 * phi) + (315 / 64) * np.cos(4 * phi) + 81 / 64


def M_gg_46_0_0():
    def func(k):
        return -6 / 5 * k**4

    return func


def N_gg_46_0_0(theta, phi):
    return 1


def M_gg_46_2_0():
    def func(k):
        return -24 / 35 * k**4

    return func


def N_gg_46_2_0(theta, phi):
    return (15 / 4) * np.cos(2 * phi) + 5 / 4


def M_gg_46_4_0():
    def func(k):
        return -16 / 105 * k**4

    return func


def N_gg_46_4_0(theta, phi):
    return (45 / 16) * np.cos(2 * phi) + (315 / 64) * np.cos(4 * phi) + 81 / 64


def M_gg_47_0_0():
    def func(k):
        return -6 / 5 * k**4

    return func


def N_gg_47_0_0(theta, phi):
    return 1


def M_gg_47_2_0():
    def func(k):
        return -24 / 35 * k**4

    return func


def N_gg_47_2_0(theta, phi):
    return (15 / 4) * np.cos(2 * phi) + 5 / 4


def M_gg_47_4_0():
    def func(k):
        return -16 / 105 * k**4

    return func


def N_gg_47_4_0(theta, phi):
    return (45 / 16) * np.cos(2 * phi) + (315 / 64) * np.cos(4 * phi) + 81 / 64


def M_gg_48_0_0():
    def func(k):
        return (1 / 5) * k**2

    return func


def N_gg_48_0_0(theta, phi):
    return 1


def M_gg_48_2_0():
    def func(k):
        return (4 / 35) * k**2

    return func


def N_gg_48_2_0(theta, phi):
    return (15 / 4) * np.cos(2 * phi) + 5 / 4


def M_gg_48_4_0():
    def func(k):
        return (8 / 315) * k**2

    return func


def N_gg_48_4_0(theta, phi):
    return (45 / 16) * np.cos(2 * phi) + (315 / 64) * np.cos(4 * phi) + 81 / 64


def M_gg_49_0_0():
    def func(k):
        return (1 / 5) * k**2

    return func


def N_gg_49_0_0(theta, phi):
    return 1


def M_gg_49_2_0():
    def func(k):
        return (4 / 35) * k**2

    return func


def N_gg_49_2_0(theta, phi):
    return (15 / 4) * np.cos(2 * phi) + 5 / 4


def M_gg_49_4_0():
    def func(k):
        return (8 / 315) * k**2

    return func


def N_gg_49_4_0(theta, phi):
    return (45 / 16) * np.cos(2 * phi) + (315 / 64) * np.cos(4 * phi) + 81 / 64


def M_gg_50_0_0():
    def func(k):
        return (1 / 5) * k**2

    return func


def N_gg_50_0_0(theta, phi):
    return 1


def M_gg_50_2_0():
    def func(k):
        return (4 / 35) * k**2

    return func


def N_gg_50_2_0(theta, phi):
    return (15 / 4) * np.cos(2 * phi) + 5 / 4


def M_gg_50_4_0():
    def func(k):
        return (8 / 315) * k**2

    return func


def N_gg_50_4_0(theta, phi):
    return (45 / 16) * np.cos(2 * phi) + (315 / 64) * np.cos(4 * phi) + 81 / 64


def M_gg_51_0_0():
    def func(k):
        return (1 / 5) * k**2

    return func


def N_gg_51_0_0(theta, phi):
    return 1


def M_gg_51_2_0():
    def func(k):
        return (4 / 35) * k**2

    return func


def N_gg_51_2_0(theta, phi):
    return (15 / 4) * np.cos(2 * phi) + 5 / 4


def M_gg_51_4_0():
    def func(k):
        return (8 / 315) * k**2

    return func


def N_gg_51_4_0(theta, phi):
    return (45 / 16) * np.cos(2 * phi) + (315 / 64) * np.cos(4 * phi) + 81 / 64


def M_gg_52_0_0():
    def func(k):
        return (1 / 5) * k**2

    return func


def N_gg_52_0_0(theta, phi):
    return 1


def M_gg_52_2_0():
    def func(k):
        return (4 / 35) * k**2

    return func


def N_gg_52_2_0(theta, phi):
    return (15 / 4) * np.cos(2 * phi) + 5 / 4


def M_gg_52_4_0():
    def func(k):
        return (8 / 315) * k**2

    return func


def N_gg_52_4_0(theta, phi):
    return (45 / 16) * np.cos(2 * phi) + (315 / 64) * np.cos(4 * phi) + 81 / 64


def M_gg_53_0_0():
    def func(k):
        return -1 / 10 * k**2

    return func


def N_gg_53_0_0(theta, phi):
    return 1


def M_gg_53_2_0():
    def func(k):
        return -2 / 35 * k**2

    return func


def N_gg_53_2_0(theta, phi):
    return (15 / 4) * np.cos(2 * phi) + 5 / 4


def M_gg_53_4_0():
    def func(k):
        return -4 / 315 * k**2

    return func


def N_gg_53_4_0(theta, phi):
    return (45 / 16) * np.cos(2 * phi) + (315 / 64) * np.cos(4 * phi) + 81 / 64


def M_gg_54_0_0():
    def func(k):
        return -1 / 14 * k**2

    return func


def N_gg_54_0_0(theta, phi):
    return 1


def M_gg_54_2_0():
    def func(k):
        return -1 / 21 * k**2

    return func


def N_gg_54_2_0(theta, phi):
    return (15 / 4) * np.cos(2 * phi) + 5 / 4


def M_gg_54_4_0():
    def func(k):
        return -4 / 231 * k**2

    return func


def N_gg_54_4_0(theta, phi):
    return (45 / 16) * np.cos(2 * phi) + (315 / 64) * np.cos(4 * phi) + 81 / 64


def M_gg_54_6_0():
    def func(k):
        return -8 / 3003 * k**2

    return func


def N_gg_54_6_0(theta, phi):
    return (
        (1365 / 512) * np.cos(2 * phi)
        + (819 / 256) * np.cos(4 * phi)
        + (3003 / 512) * np.cos(6 * phi)
        + 325 / 256
    )


def M_gg_55_0_0():
    def func(k):
        return -1 / 5 * k**4

    return func


def N_gg_55_0_0(theta, phi):
    return 1


def M_gg_55_2_0():
    def func(k):
        return -4 / 35 * k**4

    return func


def N_gg_55_2_0(theta, phi):
    return (15 / 4) * np.cos(2 * phi) + 5 / 4


def M_gg_55_4_0():
    def func(k):
        return -8 / 315 * k**4

    return func


def N_gg_55_4_0(theta, phi):
    return (45 / 16) * np.cos(2 * phi) + (315 / 64) * np.cos(4 * phi) + 81 / 64


def M_gg_56_0_0():
    def func(k):
        return -1 / 7 * k**4

    return func


def N_gg_56_0_0(theta, phi):
    return 1


def M_gg_56_2_0():
    def func(k):
        return -2 / 21 * k**4

    return func


def N_gg_56_2_0(theta, phi):
    return (15 / 4) * np.cos(2 * phi) + 5 / 4


def M_gg_56_4_0():
    def func(k):
        return -8 / 231 * k**4

    return func


def N_gg_56_4_0(theta, phi):
    return (45 / 16) * np.cos(2 * phi) + (315 / 64) * np.cos(4 * phi) + 81 / 64


def M_gg_56_6_0():
    def func(k):
        return -16 / 3003 * k**4

    return func


def N_gg_56_6_0(theta, phi):
    return (
        (1365 / 512) * np.cos(2 * phi)
        + (819 / 256) * np.cos(4 * phi)
        + (3003 / 512) * np.cos(6 * phi)
        + 325 / 256
    )


def M_gg_57_0_0():
    def func(k):
        return (1 / 20) * k**4

    return func


def N_gg_57_0_0(theta, phi):
    return 1


def M_gg_57_2_0():
    def func(k):
        return (1 / 35) * k**4

    return func


def N_gg_57_2_0(theta, phi):
    return (15 / 4) * np.cos(2 * phi) + 5 / 4


def M_gg_57_4_0():
    def func(k):
        return (2 / 315) * k**4

    return func


def N_gg_57_4_0(theta, phi):
    return (45 / 16) * np.cos(2 * phi) + (315 / 64) * np.cos(4 * phi) + 81 / 64


def M_gg_58_0_0():
    def func(k):
        return (1 / 10) * k**4

    return func


def N_gg_58_0_0(theta, phi):
    return 1


def M_gg_58_2_0():
    def func(k):
        return (2 / 35) * k**4

    return func


def N_gg_58_2_0(theta, phi):
    return (15 / 4) * np.cos(2 * phi) + 5 / 4


def M_gg_58_4_0():
    def func(k):
        return (4 / 315) * k**4

    return func


def N_gg_58_4_0(theta, phi):
    return (45 / 16) * np.cos(2 * phi) + (315 / 64) * np.cos(4 * phi) + 81 / 64


def M_gg_59_0_0():
    def func(k):
        return (1 / 10) * k**4

    return func


def N_gg_59_0_0(theta, phi):
    return 1


def M_gg_59_2_0():
    def func(k):
        return (2 / 35) * k**4

    return func


def N_gg_59_2_0(theta, phi):
    return (15 / 4) * np.cos(2 * phi) + 5 / 4


def M_gg_59_4_0():
    def func(k):
        return (4 / 315) * k**4

    return func


def N_gg_59_4_0(theta, phi):
    return (45 / 16) * np.cos(2 * phi) + (315 / 64) * np.cos(4 * phi) + 81 / 64


def M_gg_60_0_0():
    def func(k):
        return (1 / 10) * k**4

    return func


def N_gg_60_0_0(theta, phi):
    return 1


def M_gg_60_2_0():
    def func(k):
        return (2 / 35) * k**4

    return func


def N_gg_60_2_0(theta, phi):
    return (15 / 4) * np.cos(2 * phi) + 5 / 4


def M_gg_60_4_0():
    def func(k):
        return (4 / 315) * k**4

    return func


def N_gg_60_4_0(theta, phi):
    return (45 / 16) * np.cos(2 * phi) + (315 / 64) * np.cos(4 * phi) + 81 / 64


def M_gg_61_0_0():
    def func(k):
        return (3 / 10) * k**6

    return func


def N_gg_61_0_0(theta, phi):
    return 1


def M_gg_61_2_0():
    def func(k):
        return (6 / 35) * k**6

    return func


def N_gg_61_2_0(theta, phi):
    return (15 / 4) * np.cos(2 * phi) + 5 / 4


def M_gg_61_4_0():
    def func(k):
        return (4 / 105) * k**6

    return func


def N_gg_61_4_0(theta, phi):
    return (45 / 16) * np.cos(2 * phi) + (315 / 64) * np.cos(4 * phi) + 81 / 64


def M_gg_62_0_0():
    def func(k):
        return (1 / 40) * k**4

    return func


def N_gg_62_0_0(theta, phi):
    return 1


def M_gg_62_2_0():
    def func(k):
        return (1 / 70) * k**4

    return func


def N_gg_62_2_0(theta, phi):
    return (15 / 4) * np.cos(2 * phi) + 5 / 4


def M_gg_62_4_0():
    def func(k):
        return (1 / 315) * k**4

    return func


def N_gg_62_4_0(theta, phi):
    return (45 / 16) * np.cos(2 * phi) + (315 / 64) * np.cos(4 * phi) + 81 / 64


def M_gg_63_0_0():
    def func(k):
        return (1 / 40) * k**4

    return func


def N_gg_63_0_0(theta, phi):
    return 1


def M_gg_63_2_0():
    def func(k):
        return (1 / 70) * k**4

    return func


def N_gg_63_2_0(theta, phi):
    return (15 / 4) * np.cos(2 * phi) + 5 / 4


def M_gg_63_4_0():
    def func(k):
        return (1 / 315) * k**4

    return func


def N_gg_63_4_0(theta, phi):
    return (45 / 16) * np.cos(2 * phi) + (315 / 64) * np.cos(4 * phi) + 81 / 64


def M_gg_64_0_0():
    def func(k):
        return (1 / 20) * k**4

    return func


def N_gg_64_0_0(theta, phi):
    return 1


def M_gg_64_2_0():
    def func(k):
        return (1 / 35) * k**4

    return func


def N_gg_64_2_0(theta, phi):
    return (15 / 4) * np.cos(2 * phi) + 5 / 4


def M_gg_64_4_0():
    def func(k):
        return (2 / 315) * k**4

    return func


def N_gg_64_4_0(theta, phi):
    return (45 / 16) * np.cos(2 * phi) + (315 / 64) * np.cos(4 * phi) + 81 / 64


def M_gg_65_0_0():
    def func(k):
        return (1 / 10) * k**4

    return func


def N_gg_65_0_0(theta, phi):
    return 1


def M_gg_65_2_0():
    def func(k):
        return (2 / 35) * k**4

    return func


def N_gg_65_2_0(theta, phi):
    return (15 / 4) * np.cos(2 * phi) + 5 / 4


def M_gg_65_4_0():
    def func(k):
        return (4 / 315) * k**4

    return func


def N_gg_65_4_0(theta, phi):
    return (45 / 16) * np.cos(2 * phi) + (315 / 64) * np.cos(4 * phi) + 81 / 64


def M_gg_66_0_0():
    def func(k):
        return (1 / 20) * k**4

    return func


def N_gg_66_0_0(theta, phi):
    return 1


def M_gg_66_2_0():
    def func(k):
        return (1 / 35) * k**4

    return func


def N_gg_66_2_0(theta, phi):
    return (15 / 4) * np.cos(2 * phi) + 5 / 4


def M_gg_66_4_0():
    def func(k):
        return (2 / 315) * k**4

    return func


def N_gg_66_4_0(theta, phi):
    return (45 / 16) * np.cos(2 * phi) + (315 / 64) * np.cos(4 * phi) + 81 / 64


def M_gg_67_0_0():
    def func(k):
        return (1 / 10) * k**4

    return func


def N_gg_67_0_0(theta, phi):
    return 1


def M_gg_67_2_0():
    def func(k):
        return (2 / 35) * k**4

    return func


def N_gg_67_2_0(theta, phi):
    return (15 / 4) * np.cos(2 * phi) + 5 / 4


def M_gg_67_4_0():
    def func(k):
        return (4 / 315) * k**4

    return func


def N_gg_67_4_0(theta, phi):
    return (45 / 16) * np.cos(2 * phi) + (315 / 64) * np.cos(4 * phi) + 81 / 64


def M_gg_68_0_0():
    def func(k):
        return (1 / 10) * k**4

    return func


def N_gg_68_0_0(theta, phi):
    return 1


def M_gg_68_2_0():
    def func(k):
        return (2 / 35) * k**4

    return func


def N_gg_68_2_0(theta, phi):
    return (15 / 4) * np.cos(2 * phi) + 5 / 4


def M_gg_68_4_0():
    def func(k):
        return (4 / 315) * k**4

    return func


def N_gg_68_4_0(theta, phi):
    return (45 / 16) * np.cos(2 * phi) + (315 / 64) * np.cos(4 * phi) + 81 / 64


def M_gg_69_0_0():
    def func(k):
        return (1 / 10) * k**4

    return func


def N_gg_69_0_0(theta, phi):
    return 1


def M_gg_69_2_0():
    def func(k):
        return (2 / 35) * k**4

    return func


def N_gg_69_2_0(theta, phi):
    return (15 / 4) * np.cos(2 * phi) + 5 / 4


def M_gg_69_4_0():
    def func(k):
        return (4 / 315) * k**4

    return func


def N_gg_69_4_0(theta, phi):
    return (45 / 16) * np.cos(2 * phi) + (315 / 64) * np.cos(4 * phi) + 81 / 64


def M_gg_70_0_0():
    def func(k):
        return (3 / 10) * k**6

    return func


def N_gg_70_0_0(theta, phi):
    return 1


def M_gg_70_2_0():
    def func(k):
        return (6 / 35) * k**6

    return func


def N_gg_70_2_0(theta, phi):
    return (15 / 4) * np.cos(2 * phi) + 5 / 4


def M_gg_70_4_0():
    def func(k):
        return (4 / 105) * k**6

    return func


def N_gg_70_4_0(theta, phi):
    return (45 / 16) * np.cos(2 * phi) + (315 / 64) * np.cos(4 * phi) + 81 / 64


def M_gg_71_0_0():
    def func(k):
        return (1 / 40) * k**4

    return func


def N_gg_71_0_0(theta, phi):
    return 1


def M_gg_71_2_0():
    def func(k):
        return (1 / 70) * k**4

    return func


def N_gg_71_2_0(theta, phi):
    return (15 / 4) * np.cos(2 * phi) + 5 / 4


def M_gg_71_4_0():
    def func(k):
        return (1 / 315) * k**4

    return func


def N_gg_71_4_0(theta, phi):
    return (45 / 16) * np.cos(2 * phi) + (315 / 64) * np.cos(4 * phi) + 81 / 64


def M_gg_72_0_0():
    def func(k):
        return (1 / 40) * k**4

    return func


def N_gg_72_0_0(theta, phi):
    return 1


def M_gg_72_2_0():
    def func(k):
        return (1 / 70) * k**4

    return func


def N_gg_72_2_0(theta, phi):
    return (15 / 4) * np.cos(2 * phi) + 5 / 4


def M_gg_72_4_0():
    def func(k):
        return (1 / 315) * k**4

    return func


def N_gg_72_4_0(theta, phi):
    return (45 / 16) * np.cos(2 * phi) + (315 / 64) * np.cos(4 * phi) + 81 / 64


def M_gg_73_0_0():
    def func(k):
        return (1 / 20) * k**4

    return func


def N_gg_73_0_0(theta, phi):
    return 1


def M_gg_73_2_0():
    def func(k):
        return (1 / 35) * k**4

    return func


def N_gg_73_2_0(theta, phi):
    return (15 / 4) * np.cos(2 * phi) + 5 / 4


def M_gg_73_4_0():
    def func(k):
        return (2 / 315) * k**4

    return func


def N_gg_73_4_0(theta, phi):
    return (45 / 16) * np.cos(2 * phi) + (315 / 64) * np.cos(4 * phi) + 81 / 64


def M_gg_74_0_0():
    def func(k):
        return (1 / 10) * k**4

    return func


def N_gg_74_0_0(theta, phi):
    return 1


def M_gg_74_2_0():
    def func(k):
        return (2 / 35) * k**4

    return func


def N_gg_74_2_0(theta, phi):
    return (15 / 4) * np.cos(2 * phi) + 5 / 4


def M_gg_74_4_0():
    def func(k):
        return (4 / 315) * k**4

    return func


def N_gg_74_4_0(theta, phi):
    return (45 / 16) * np.cos(2 * phi) + (315 / 64) * np.cos(4 * phi) + 81 / 64


def M_gg_75_0_0():
    def func(k):
        return 1 / 5

    return func


def N_gg_75_0_0(theta, phi):
    return 1


def M_gg_75_2_0():
    def func(k):
        return 4 / 35

    return func


def N_gg_75_2_0(theta, phi):
    return (15 / 4) * np.cos(2 * phi) + 5 / 4


def M_gg_75_4_0():
    def func(k):
        return 8 / 315

    return func


def N_gg_75_4_0(theta, phi):
    return (45 / 16) * np.cos(2 * phi) + (315 / 64) * np.cos(4 * phi) + 81 / 64


def M_gg_76_0_0():
    def func(k):
        return 1 / 7

    return func


def N_gg_76_0_0(theta, phi):
    return 1


def M_gg_76_2_0():
    def func(k):
        return 2 / 21

    return func


def N_gg_76_2_0(theta, phi):
    return (15 / 4) * np.cos(2 * phi) + 5 / 4


def M_gg_76_4_0():
    def func(k):
        return 8 / 231

    return func


def N_gg_76_4_0(theta, phi):
    return (45 / 16) * np.cos(2 * phi) + (315 / 64) * np.cos(4 * phi) + 81 / 64


def M_gg_76_6_0():
    def func(k):
        return 16 / 3003

    return func


def N_gg_76_6_0(theta, phi):
    return (
        (1365 / 512) * np.cos(2 * phi)
        + (819 / 256) * np.cos(4 * phi)
        + (3003 / 512) * np.cos(6 * phi)
        + 325 / 256
    )


def M_gg_77_0_0():
    def func(k):
        return -1 / 5

    return func


def N_gg_77_0_0(theta, phi):
    return 1


def M_gg_77_2_0():
    def func(k):
        return -4 / 35

    return func


def N_gg_77_2_0(theta, phi):
    return (15 / 4) * np.cos(2 * phi) + 5 / 4


def M_gg_77_4_0():
    def func(k):
        return -8 / 315

    return func


def N_gg_77_4_0(theta, phi):
    return (45 / 16) * np.cos(2 * phi) + (315 / 64) * np.cos(4 * phi) + 81 / 64


def M_gg_78_0_0():
    def func(k):
        return -1 / 7

    return func


def N_gg_78_0_0(theta, phi):
    return 1


def M_gg_78_2_0():
    def func(k):
        return -2 / 21

    return func


def N_gg_78_2_0(theta, phi):
    return (15 / 4) * np.cos(2 * phi) + 5 / 4


def M_gg_78_4_0():
    def func(k):
        return -8 / 231

    return func


def N_gg_78_4_0(theta, phi):
    return (45 / 16) * np.cos(2 * phi) + (315 / 64) * np.cos(4 * phi) + 81 / 64


def M_gg_78_6_0():
    def func(k):
        return -16 / 3003

    return func


def N_gg_78_6_0(theta, phi):
    return (
        (1365 / 512) * np.cos(2 * phi)
        + (819 / 256) * np.cos(4 * phi)
        + (3003 / 512) * np.cos(6 * phi)
        + 325 / 256
    )


def M_gg_79_0_0():
    def func(k):
        return (2 / 5) * k**2

    return func


def N_gg_79_0_0(theta, phi):
    return 1


def M_gg_79_2_0():
    def func(k):
        return (8 / 35) * k**2

    return func


def N_gg_79_2_0(theta, phi):
    return (15 / 4) * np.cos(2 * phi) + 5 / 4


def M_gg_79_4_0():
    def func(k):
        return (16 / 315) * k**2

    return func


def N_gg_79_4_0(theta, phi):
    return (45 / 16) * np.cos(2 * phi) + (315 / 64) * np.cos(4 * phi) + 81 / 64


def M_gg_80_0_0():
    def func(k):
        return (2 / 7) * k**2

    return func


def N_gg_80_0_0(theta, phi):
    return 1


def M_gg_80_2_0():
    def func(k):
        return (4 / 21) * k**2

    return func


def N_gg_80_2_0(theta, phi):
    return (15 / 4) * np.cos(2 * phi) + 5 / 4


def M_gg_80_4_0():
    def func(k):
        return (16 / 231) * k**2

    return func


def N_gg_80_4_0(theta, phi):
    return (45 / 16) * np.cos(2 * phi) + (315 / 64) * np.cos(4 * phi) + 81 / 64


def M_gg_80_6_0():
    def func(k):
        return (32 / 3003) * k**2

    return func


def N_gg_80_6_0(theta, phi):
    return (
        (1365 / 512) * np.cos(2 * phi)
        + (819 / 256) * np.cos(4 * phi)
        + (3003 / 512) * np.cos(6 * phi)
        + 325 / 256
    )


def M_gg_81_0_0():
    def func(k):
        return -1 / 5 * k**2

    return func


def N_gg_81_0_0(theta, phi):
    return 1


def M_gg_81_2_0():
    def func(k):
        return -4 / 35 * k**2

    return func


def N_gg_81_2_0(theta, phi):
    return (15 / 4) * np.cos(2 * phi) + 5 / 4


def M_gg_81_4_0():
    def func(k):
        return -8 / 315 * k**2

    return func


def N_gg_81_4_0(theta, phi):
    return (45 / 16) * np.cos(2 * phi) + (315 / 64) * np.cos(4 * phi) + 81 / 64


def M_gg_82_0_0():
    def func(k):
        return -2 / 5 * k**2

    return func


def N_gg_82_0_0(theta, phi):
    return 1


def M_gg_82_2_0():
    def func(k):
        return -8 / 35 * k**2

    return func


def N_gg_82_2_0(theta, phi):
    return (15 / 4) * np.cos(2 * phi) + 5 / 4


def M_gg_82_4_0():
    def func(k):
        return -16 / 315 * k**2

    return func


def N_gg_82_4_0(theta, phi):
    return (45 / 16) * np.cos(2 * phi) + (315 / 64) * np.cos(4 * phi) + 81 / 64


def M_gg_83_0_0():
    def func(k):
        return -2 / 5 * k**2

    return func


def N_gg_83_0_0(theta, phi):
    return 1


def M_gg_83_2_0():
    def func(k):
        return -8 / 35 * k**2

    return func


def N_gg_83_2_0(theta, phi):
    return (15 / 4) * np.cos(2 * phi) + 5 / 4


def M_gg_83_4_0():
    def func(k):
        return -16 / 315 * k**2

    return func


def N_gg_83_4_0(theta, phi):
    return (45 / 16) * np.cos(2 * phi) + (315 / 64) * np.cos(4 * phi) + 81 / 64


def M_gg_84_0_0():
    def func(k):
        return -6 / 5 * k**4

    return func


def N_gg_84_0_0(theta, phi):
    return 1


def M_gg_84_2_0():
    def func(k):
        return -24 / 35 * k**4

    return func


def N_gg_84_2_0(theta, phi):
    return (15 / 4) * np.cos(2 * phi) + 5 / 4


def M_gg_84_4_0():
    def func(k):
        return -16 / 105 * k**4

    return func


def N_gg_84_4_0(theta, phi):
    return (45 / 16) * np.cos(2 * phi) + (315 / 64) * np.cos(4 * phi) + 81 / 64


def M_gg_85_0_0():
    def func(k):
        return -6 / 5 * k**4

    return func


def N_gg_85_0_0(theta, phi):
    return 1


def M_gg_85_2_0():
    def func(k):
        return -24 / 35 * k**4

    return func


def N_gg_85_2_0(theta, phi):
    return (15 / 4) * np.cos(2 * phi) + 5 / 4


def M_gg_85_4_0():
    def func(k):
        return -16 / 105 * k**4

    return func


def N_gg_85_4_0(theta, phi):
    return (45 / 16) * np.cos(2 * phi) + (315 / 64) * np.cos(4 * phi) + 81 / 64


def M_gg_86_0_0():
    def func(k):
        return (1 / 5) * k**2

    return func


def N_gg_86_0_0(theta, phi):
    return 1


def M_gg_86_2_0():
    def func(k):
        return (4 / 35) * k**2

    return func


def N_gg_86_2_0(theta, phi):
    return (15 / 4) * np.cos(2 * phi) + 5 / 4


def M_gg_86_4_0():
    def func(k):
        return (8 / 315) * k**2

    return func


def N_gg_86_4_0(theta, phi):
    return (45 / 16) * np.cos(2 * phi) + (315 / 64) * np.cos(4 * phi) + 81 / 64


def M_gg_87_0_0():
    def func(k):
        return (1 / 5) * k**2

    return func


def N_gg_87_0_0(theta, phi):
    return 1


def M_gg_87_2_0():
    def func(k):
        return (4 / 35) * k**2

    return func


def N_gg_87_2_0(theta, phi):
    return (15 / 4) * np.cos(2 * phi) + 5 / 4


def M_gg_87_4_0():
    def func(k):
        return (8 / 315) * k**2

    return func


def N_gg_87_4_0(theta, phi):
    return (45 / 16) * np.cos(2 * phi) + (315 / 64) * np.cos(4 * phi) + 81 / 64


def M_gg_88_0_0():
    def func(k):
        return (1 / 5) * k**2

    return func


def N_gg_88_0_0(theta, phi):
    return 1


def M_gg_88_2_0():
    def func(k):
        return (4 / 35) * k**2

    return func


def N_gg_88_2_0(theta, phi):
    return (15 / 4) * np.cos(2 * phi) + 5 / 4


def M_gg_88_4_0():
    def func(k):
        return (8 / 315) * k**2

    return func


def N_gg_88_4_0(theta, phi):
    return (45 / 16) * np.cos(2 * phi) + (315 / 64) * np.cos(4 * phi) + 81 / 64


def M_gg_89_0_0():
    def func(k):
        return (1 / 5) * k**2

    return func


def N_gg_89_0_0(theta, phi):
    return 1


def M_gg_89_2_0():
    def func(k):
        return (4 / 35) * k**2

    return func


def N_gg_89_2_0(theta, phi):
    return (15 / 4) * np.cos(2 * phi) + 5 / 4


def M_gg_89_4_0():
    def func(k):
        return (8 / 315) * k**2

    return func


def N_gg_89_4_0(theta, phi):
    return (45 / 16) * np.cos(2 * phi) + (315 / 64) * np.cos(4 * phi) + 81 / 64


def M_gg_90_0_0():
    def func(k):
        return (1 / 5) * k**2

    return func


def N_gg_90_0_0(theta, phi):
    return 1


def M_gg_90_2_0():
    def func(k):
        return (4 / 35) * k**2

    return func


def N_gg_90_2_0(theta, phi):
    return (15 / 4) * np.cos(2 * phi) + 5 / 4


def M_gg_90_4_0():
    def func(k):
        return (8 / 315) * k**2

    return func


def N_gg_90_4_0(theta, phi):
    return (45 / 16) * np.cos(2 * phi) + (315 / 64) * np.cos(4 * phi) + 81 / 64


def M_gg_91_0_0():
    def func(k):
        return (2 / 5) * k**2

    return func


def N_gg_91_0_0(theta, phi):
    return 1


def M_gg_91_2_0():
    def func(k):
        return (8 / 35) * k**2

    return func


def N_gg_91_2_0(theta, phi):
    return (15 / 4) * np.cos(2 * phi) + 5 / 4


def M_gg_91_4_0():
    def func(k):
        return (16 / 315) * k**2

    return func


def N_gg_91_4_0(theta, phi):
    return (45 / 16) * np.cos(2 * phi) + (315 / 64) * np.cos(4 * phi) + 81 / 64


def M_gg_92_0_0():
    def func(k):
        return (6 / 5) * k**4

    return func


def N_gg_92_0_0(theta, phi):
    return 1


def M_gg_92_2_0():
    def func(k):
        return (24 / 35) * k**4

    return func


def N_gg_92_2_0(theta, phi):
    return (15 / 4) * np.cos(2 * phi) + 5 / 4


def M_gg_92_4_0():
    def func(k):
        return (16 / 105) * k**4

    return func


def N_gg_92_4_0(theta, phi):
    return (45 / 16) * np.cos(2 * phi) + (315 / 64) * np.cos(4 * phi) + 81 / 64


def M_gg_93_0_0():
    def func(k):
        return -1 / 7 * k**2

    return func


def N_gg_93_0_0(theta, phi):
    return 1


def M_gg_93_2_0():
    def func(k):
        return -2 / 21 * k**2

    return func


def N_gg_93_2_0(theta, phi):
    return (15 / 4) * np.cos(2 * phi) + 5 / 4


def M_gg_93_4_0():
    def func(k):
        return -8 / 231 * k**2

    return func


def N_gg_93_4_0(theta, phi):
    return (45 / 16) * np.cos(2 * phi) + (315 / 64) * np.cos(4 * phi) + 81 / 64


def M_gg_93_6_0():
    def func(k):
        return -16 / 3003 * k**2

    return func


def N_gg_93_6_0(theta, phi):
    return (
        (1365 / 512) * np.cos(2 * phi)
        + (819 / 256) * np.cos(4 * phi)
        + (3003 / 512) * np.cos(6 * phi)
        + 325 / 256
    )


def M_gg_94_0_0():
    def func(k):
        return -2 / 7 * k**2

    return func


def N_gg_94_0_0(theta, phi):
    return 1


def M_gg_94_2_0():
    def func(k):
        return -4 / 21 * k**2

    return func


def N_gg_94_2_0(theta, phi):
    return (15 / 4) * np.cos(2 * phi) + 5 / 4


def M_gg_94_4_0():
    def func(k):
        return -16 / 231 * k**2

    return func


def N_gg_94_4_0(theta, phi):
    return (45 / 16) * np.cos(2 * phi) + (315 / 64) * np.cos(4 * phi) + 81 / 64


def M_gg_94_6_0():
    def func(k):
        return -32 / 3003 * k**2

    return func


def N_gg_94_6_0(theta, phi):
    return (
        (1365 / 512) * np.cos(2 * phi)
        + (819 / 256) * np.cos(4 * phi)
        + (3003 / 512) * np.cos(6 * phi)
        + 325 / 256
    )


def M_gg_95_0_0():
    def func(k):
        return -4 / 7 * k**2

    return func


def N_gg_95_0_0(theta, phi):
    return 1


def M_gg_95_2_0():
    def func(k):
        return -8 / 21 * k**2

    return func


def N_gg_95_2_0(theta, phi):
    return (15 / 4) * np.cos(2 * phi) + 5 / 4


def M_gg_95_4_0():
    def func(k):
        return -32 / 231 * k**2

    return func


def N_gg_95_4_0(theta, phi):
    return (45 / 16) * np.cos(2 * phi) + (315 / 64) * np.cos(4 * phi) + 81 / 64


def M_gg_95_6_0():
    def func(k):
        return -64 / 3003 * k**2

    return func


def N_gg_95_6_0(theta, phi):
    return (
        (1365 / 512) * np.cos(2 * phi)
        + (819 / 256) * np.cos(4 * phi)
        + (3003 / 512) * np.cos(6 * phi)
        + 325 / 256
    )


def M_gg_96_0_0():
    def func(k):
        return -6 / 7 * k**4

    return func


def N_gg_96_0_0(theta, phi):
    return 1


def M_gg_96_2_0():
    def func(k):
        return -4 / 7 * k**4

    return func


def N_gg_96_2_0(theta, phi):
    return (15 / 4) * np.cos(2 * phi) + 5 / 4


def M_gg_96_4_0():
    def func(k):
        return -16 / 77 * k**4

    return func


def N_gg_96_4_0(theta, phi):
    return (45 / 16) * np.cos(2 * phi) + (315 / 64) * np.cos(4 * phi) + 81 / 64


def M_gg_96_6_0():
    def func(k):
        return -32 / 1001 * k**4

    return func


def N_gg_96_6_0(theta, phi):
    return (
        (1365 / 512) * np.cos(2 * phi)
        + (819 / 256) * np.cos(4 * phi)
        + (3003 / 512) * np.cos(6 * phi)
        + 325 / 256
    )


def M_gg_97_0_0():
    def func(k):
        return -12 / 7 * k**4

    return func


def N_gg_97_0_0(theta, phi):
    return 1


def M_gg_97_2_0():
    def func(k):
        return -8 / 7 * k**4

    return func


def N_gg_97_2_0(theta, phi):
    return (15 / 4) * np.cos(2 * phi) + 5 / 4


def M_gg_97_4_0():
    def func(k):
        return -32 / 77 * k**4

    return func


def N_gg_97_4_0(theta, phi):
    return (45 / 16) * np.cos(2 * phi) + (315 / 64) * np.cos(4 * phi) + 81 / 64


def M_gg_97_6_0():
    def func(k):
        return -64 / 1001 * k**4

    return func


def N_gg_97_6_0(theta, phi):
    return (
        (1365 / 512) * np.cos(2 * phi)
        + (819 / 256) * np.cos(4 * phi)
        + (3003 / 512) * np.cos(6 * phi)
        + 325 / 256
    )


def M_gg_98_0_0():
    def func(k):
        return -1 / 7 * k**2

    return func


def N_gg_98_0_0(theta, phi):
    return 1


def M_gg_98_2_0():
    def func(k):
        return -2 / 21 * k**2

    return func


def N_gg_98_2_0(theta, phi):
    return (15 / 4) * np.cos(2 * phi) + 5 / 4


def M_gg_98_4_0():
    def func(k):
        return -8 / 231 * k**2

    return func


def N_gg_98_4_0(theta, phi):
    return (45 / 16) * np.cos(2 * phi) + (315 / 64) * np.cos(4 * phi) + 81 / 64


def M_gg_98_6_0():
    def func(k):
        return -16 / 3003 * k**2

    return func


def N_gg_98_6_0(theta, phi):
    return (
        (1365 / 512) * np.cos(2 * phi)
        + (819 / 256) * np.cos(4 * phi)
        + (3003 / 512) * np.cos(6 * phi)
        + 325 / 256
    )


def M_gg_99_0_0():
    def func(k):
        return -1 / 5 * k**2

    return func


def N_gg_99_0_0(theta, phi):
    return 1


def M_gg_99_2_0():
    def func(k):
        return -4 / 35 * k**2

    return func


def N_gg_99_2_0(theta, phi):
    return (15 / 4) * np.cos(2 * phi) + 5 / 4


def M_gg_99_4_0():
    def func(k):
        return -8 / 315 * k**2

    return func


def N_gg_99_4_0(theta, phi):
    return (45 / 16) * np.cos(2 * phi) + (315 / 64) * np.cos(4 * phi) + 81 / 64


def M_gg_100_0_0():
    def func(k):
        return 1 / 80

    return func


def N_gg_100_0_0(theta, phi):
    return 1


def M_gg_100_2_0():
    def func(k):
        return 1 / 140

    return func


def N_gg_100_2_0(theta, phi):
    return (15 / 4) * np.cos(2 * phi) + 5 / 4


def M_gg_100_4_0():
    def func(k):
        return 1 / 630

    return func


def N_gg_100_4_0(theta, phi):
    return (45 / 16) * np.cos(2 * phi) + (315 / 64) * np.cos(4 * phi) + 81 / 64


def M_gg_101_0_0():
    def func(k):
        return 1 / 56

    return func


def N_gg_101_0_0(theta, phi):
    return 1


def M_gg_101_2_0():
    def func(k):
        return 1 / 84

    return func


def N_gg_101_2_0(theta, phi):
    return (15 / 4) * np.cos(2 * phi) + 5 / 4


def M_gg_101_4_0():
    def func(k):
        return 1 / 231

    return func


def N_gg_101_4_0(theta, phi):
    return (45 / 16) * np.cos(2 * phi) + (315 / 64) * np.cos(4 * phi) + 81 / 64


def M_gg_101_6_0():
    def func(k):
        return 2 / 3003

    return func


def N_gg_101_6_0(theta, phi):
    return (
        (1365 / 512) * np.cos(2 * phi)
        + (819 / 256) * np.cos(4 * phi)
        + (3003 / 512) * np.cos(6 * phi)
        + 325 / 256
    )


def M_gg_102_0_0():
    def func(k):
        return 1 / 144

    return func


def N_gg_102_0_0(theta, phi):
    return 1


def M_gg_102_2_0():
    def func(k):
        return 1 / 198

    return func


def N_gg_102_2_0(theta, phi):
    return (15 / 4) * np.cos(2 * phi) + 5 / 4


def M_gg_102_4_0():
    def func(k):
        return 1 / 429

    return func


def N_gg_102_4_0(theta, phi):
    return (45 / 16) * np.cos(2 * phi) + (315 / 64) * np.cos(4 * phi) + 81 / 64


def M_gg_102_6_0():
    def func(k):
        return 4 / 6435

    return func


def N_gg_102_6_0(theta, phi):
    return (
        (1365 / 512) * np.cos(2 * phi)
        + (819 / 256) * np.cos(4 * phi)
        + (3003 / 512) * np.cos(6 * phi)
        + 325 / 256
    )


def M_gg_102_8_0():
    def func(k):
        return 8 / 109395

    return func


def N_gg_102_8_0(theta, phi):
    return (
        (5355 / 2048) * np.cos(2 * phi)
        + (11781 / 4096) * np.cos(4 * phi)
        + (7293 / 2048) * np.cos(6 * phi)
        + (109395 / 16384) * np.cos(8 * phi)
        + 20825 / 16384
    )


def M_gg_103_0_0():
    def func(k):
        return (3 / 20) * k**4

    return func


def N_gg_103_0_0(theta, phi):
    return 1


def M_gg_103_2_0():
    def func(k):
        return (3 / 35) * k**4

    return func


def N_gg_103_2_0(theta, phi):
    return (15 / 4) * np.cos(2 * phi) + 5 / 4


def M_gg_103_4_0():
    def func(k):
        return (2 / 105) * k**4

    return func


def N_gg_103_4_0(theta, phi):
    return (45 / 16) * np.cos(2 * phi) + (315 / 64) * np.cos(4 * phi) + 81 / 64


def M_gg_104_0_0():
    def func(k):
        return (3 / 10) * k**4

    return func


def N_gg_104_0_0(theta, phi):
    return 1


def M_gg_104_2_0():
    def func(k):
        return (6 / 35) * k**4

    return func


def N_gg_104_2_0(theta, phi):
    return (15 / 4) * np.cos(2 * phi) + 5 / 4


def M_gg_104_4_0():
    def func(k):
        return (4 / 105) * k**4

    return func


def N_gg_104_4_0(theta, phi):
    return (45 / 16) * np.cos(2 * phi) + (315 / 64) * np.cos(4 * phi) + 81 / 64


def M_gg_105_0_0():
    def func(k):
        return (3 / 10) * k**4

    return func


def N_gg_105_0_0(theta, phi):
    return 1


def M_gg_105_2_0():
    def func(k):
        return (6 / 35) * k**4

    return func


def N_gg_105_2_0(theta, phi):
    return (15 / 4) * np.cos(2 * phi) + 5 / 4


def M_gg_105_4_0():
    def func(k):
        return (4 / 105) * k**4

    return func


def N_gg_105_4_0(theta, phi):
    return (45 / 16) * np.cos(2 * phi) + (315 / 64) * np.cos(4 * phi) + 81 / 64


def M_gg_106_0_0():
    def func(k):
        return (3 / 10) * k**4

    return func


def N_gg_106_0_0(theta, phi):
    return 1


def M_gg_106_2_0():
    def func(k):
        return (6 / 35) * k**4

    return func


def N_gg_106_2_0(theta, phi):
    return (15 / 4) * np.cos(2 * phi) + 5 / 4


def M_gg_106_4_0():
    def func(k):
        return (4 / 105) * k**4

    return func


def N_gg_106_4_0(theta, phi):
    return (45 / 16) * np.cos(2 * phi) + (315 / 64) * np.cos(4 * phi) + 81 / 64


def M_gg_107_0_0():
    def func(k):
        return (9 / 10) * k**6

    return func


def N_gg_107_0_0(theta, phi):
    return 1


def M_gg_107_2_0():
    def func(k):
        return (18 / 35) * k**6

    return func


def N_gg_107_2_0(theta, phi):
    return (15 / 4) * np.cos(2 * phi) + 5 / 4


def M_gg_107_4_0():
    def func(k):
        return (4 / 35) * k**6

    return func


def N_gg_107_4_0(theta, phi):
    return (45 / 16) * np.cos(2 * phi) + (315 / 64) * np.cos(4 * phi) + 81 / 64


def M_gg_108_0_0():
    def func(k):
        return (3 / 40) * k**4

    return func


def N_gg_108_0_0(theta, phi):
    return 1


def M_gg_108_2_0():
    def func(k):
        return (3 / 70) * k**4

    return func


def N_gg_108_2_0(theta, phi):
    return (15 / 4) * np.cos(2 * phi) + 5 / 4


def M_gg_108_4_0():
    def func(k):
        return (1 / 105) * k**4

    return func


def N_gg_108_4_0(theta, phi):
    return (45 / 16) * np.cos(2 * phi) + (315 / 64) * np.cos(4 * phi) + 81 / 64


def M_gg_109_0_0():
    def func(k):
        return (3 / 40) * k**4

    return func


def N_gg_109_0_0(theta, phi):
    return 1


def M_gg_109_2_0():
    def func(k):
        return (3 / 70) * k**4

    return func


def N_gg_109_2_0(theta, phi):
    return (15 / 4) * np.cos(2 * phi) + 5 / 4


def M_gg_109_4_0():
    def func(k):
        return (1 / 105) * k**4

    return func


def N_gg_109_4_0(theta, phi):
    return (45 / 16) * np.cos(2 * phi) + (315 / 64) * np.cos(4 * phi) + 81 / 64


def M_gg_110_0_0():
    def func(k):
        return (3 / 20) * k**4

    return func


def N_gg_110_0_0(theta, phi):
    return 1


def M_gg_110_2_0():
    def func(k):
        return (3 / 35) * k**4

    return func


def N_gg_110_2_0(theta, phi):
    return (15 / 4) * np.cos(2 * phi) + 5 / 4


def M_gg_110_4_0():
    def func(k):
        return (2 / 105) * k**4

    return func


def N_gg_110_4_0(theta, phi):
    return (45 / 16) * np.cos(2 * phi) + (315 / 64) * np.cos(4 * phi) + 81 / 64


def M_gg_111_0_0():
    def func(k):
        return (3 / 10) * k**4

    return func


def N_gg_111_0_0(theta, phi):
    return 1


def M_gg_111_2_0():
    def func(k):
        return (6 / 35) * k**4

    return func


def N_gg_111_2_0(theta, phi):
    return (15 / 4) * np.cos(2 * phi) + 5 / 4


def M_gg_111_4_0():
    def func(k):
        return (4 / 105) * k**4

    return func


def N_gg_111_4_0(theta, phi):
    return (45 / 16) * np.cos(2 * phi) + (315 / 64) * np.cos(4 * phi) + 81 / 64


def M_gg_112_0_0():
    def func(k):
        return -1 / 10 * k**2

    return func


def N_gg_112_0_0(theta, phi):
    return 1


def M_gg_112_2_0():
    def func(k):
        return -2 / 35 * k**2

    return func


def N_gg_112_2_0(theta, phi):
    return (15 / 4) * np.cos(2 * phi) + 5 / 4


def M_gg_112_4_0():
    def func(k):
        return -4 / 315 * k**2

    return func


def N_gg_112_4_0(theta, phi):
    return (45 / 16) * np.cos(2 * phi) + (315 / 64) * np.cos(4 * phi) + 81 / 64


def M_gg_113_0_0():
    def func(k):
        return -1 / 14 * k**2

    return func


def N_gg_113_0_0(theta, phi):
    return 1


def M_gg_113_2_0():
    def func(k):
        return -1 / 21 * k**2

    return func


def N_gg_113_2_0(theta, phi):
    return (15 / 4) * np.cos(2 * phi) + 5 / 4


def M_gg_113_4_0():
    def func(k):
        return -4 / 231 * k**2

    return func


def N_gg_113_4_0(theta, phi):
    return (45 / 16) * np.cos(2 * phi) + (315 / 64) * np.cos(4 * phi) + 81 / 64


def M_gg_113_6_0():
    def func(k):
        return -8 / 3003 * k**2

    return func


def N_gg_113_6_0(theta, phi):
    return (
        (1365 / 512) * np.cos(2 * phi)
        + (819 / 256) * np.cos(4 * phi)
        + (3003 / 512) * np.cos(6 * phi)
        + 325 / 256
    )


def M_gg_114_0_0():
    def func(k):
        return -1 / 5 * k**4

    return func


def N_gg_114_0_0(theta, phi):
    return 1


def M_gg_114_2_0():
    def func(k):
        return -4 / 35 * k**4

    return func


def N_gg_114_2_0(theta, phi):
    return (15 / 4) * np.cos(2 * phi) + 5 / 4


def M_gg_114_4_0():
    def func(k):
        return -8 / 315 * k**4

    return func


def N_gg_114_4_0(theta, phi):
    return (45 / 16) * np.cos(2 * phi) + (315 / 64) * np.cos(4 * phi) + 81 / 64


def M_gg_115_0_0():
    def func(k):
        return -1 / 7 * k**4

    return func


def N_gg_115_0_0(theta, phi):
    return 1


def M_gg_115_2_0():
    def func(k):
        return -2 / 21 * k**4

    return func


def N_gg_115_2_0(theta, phi):
    return (15 / 4) * np.cos(2 * phi) + 5 / 4


def M_gg_115_4_0():
    def func(k):
        return -8 / 231 * k**4

    return func


def N_gg_115_4_0(theta, phi):
    return (45 / 16) * np.cos(2 * phi) + (315 / 64) * np.cos(4 * phi) + 81 / 64


def M_gg_115_6_0():
    def func(k):
        return -16 / 3003 * k**4

    return func


def N_gg_115_6_0(theta, phi):
    return (
        (1365 / 512) * np.cos(2 * phi)
        + (819 / 256) * np.cos(4 * phi)
        + (3003 / 512) * np.cos(6 * phi)
        + 325 / 256
    )


def M_gg_116_0_0():
    def func(k):
        return -1 / 10 * k**2

    return func


def N_gg_116_0_0(theta, phi):
    return 1


def M_gg_116_2_0():
    def func(k):
        return -2 / 35 * k**2

    return func


def N_gg_116_2_0(theta, phi):
    return (15 / 4) * np.cos(2 * phi) + 5 / 4


def M_gg_116_4_0():
    def func(k):
        return -4 / 315 * k**2

    return func


def N_gg_116_4_0(theta, phi):
    return (45 / 16) * np.cos(2 * phi) + (315 / 64) * np.cos(4 * phi) + 81 / 64


def M_gg_117_0_0():
    def func(k):
        return -1 / 10 * k**2

    return func


def N_gg_117_0_0(theta, phi):
    return 1


def M_gg_117_2_0():
    def func(k):
        return -2 / 35 * k**2

    return func


def N_gg_117_2_0(theta, phi):
    return (15 / 4) * np.cos(2 * phi) + 5 / 4


def M_gg_117_4_0():
    def func(k):
        return -4 / 315 * k**2

    return func


def N_gg_117_4_0(theta, phi):
    return (45 / 16) * np.cos(2 * phi) + (315 / 64) * np.cos(4 * phi) + 81 / 64


def M_gg_118_0_0():
    def func(k):
        return -1 / 14 * k**2

    return func


def N_gg_118_0_0(theta, phi):
    return 1


def M_gg_118_2_0():
    def func(k):
        return -1 / 21 * k**2

    return func


def N_gg_118_2_0(theta, phi):
    return (15 / 4) * np.cos(2 * phi) + 5 / 4


def M_gg_118_4_0():
    def func(k):
        return -4 / 231 * k**2

    return func


def N_gg_118_4_0(theta, phi):
    return (45 / 16) * np.cos(2 * phi) + (315 / 64) * np.cos(4 * phi) + 81 / 64


def M_gg_118_6_0():
    def func(k):
        return -8 / 3003 * k**2

    return func


def N_gg_118_6_0(theta, phi):
    return (
        (1365 / 512) * np.cos(2 * phi)
        + (819 / 256) * np.cos(4 * phi)
        + (3003 / 512) * np.cos(6 * phi)
        + 325 / 256
    )


def M_gg_119_0_0():
    def func(k):
        return -1 / 14 * k**2

    return func


def N_gg_119_0_0(theta, phi):
    return 1


def M_gg_119_2_0():
    def func(k):
        return -1 / 21 * k**2

    return func


def N_gg_119_2_0(theta, phi):
    return (15 / 4) * np.cos(2 * phi) + 5 / 4


def M_gg_119_4_0():
    def func(k):
        return -4 / 231 * k**2

    return func


def N_gg_119_4_0(theta, phi):
    return (45 / 16) * np.cos(2 * phi) + (315 / 64) * np.cos(4 * phi) + 81 / 64


def M_gg_119_6_0():
    def func(k):
        return -8 / 3003 * k**2

    return func


def N_gg_119_6_0(theta, phi):
    return (
        (1365 / 512) * np.cos(2 * phi)
        + (819 / 256) * np.cos(4 * phi)
        + (3003 / 512) * np.cos(6 * phi)
        + 325 / 256
    )


def M_gg_120_0_0():
    def func(k):
        return (1 / 20) * k**2

    return func


def N_gg_120_0_0(theta, phi):
    return 1


def M_gg_120_2_0():
    def func(k):
        return (1 / 35) * k**2

    return func


def N_gg_120_2_0(theta, phi):
    return (15 / 4) * np.cos(2 * phi) + 5 / 4


def M_gg_120_4_0():
    def func(k):
        return (2 / 315) * k**2

    return func


def N_gg_120_4_0(theta, phi):
    return (45 / 16) * np.cos(2 * phi) + (315 / 64) * np.cos(4 * phi) + 81 / 64


def M_gg_121_0_0():
    def func(k):
        return (1 / 20) * k**2

    return func


def N_gg_121_0_0(theta, phi):
    return 1


def M_gg_121_2_0():
    def func(k):
        return (1 / 35) * k**2

    return func


def N_gg_121_2_0(theta, phi):
    return (15 / 4) * np.cos(2 * phi) + 5 / 4


def M_gg_121_4_0():
    def func(k):
        return (2 / 315) * k**2

    return func


def N_gg_121_4_0(theta, phi):
    return (45 / 16) * np.cos(2 * phi) + (315 / 64) * np.cos(4 * phi) + 81 / 64


def M_gg_122_0_0():
    def func(k):
        return (1 / 28) * k**2

    return func


def N_gg_122_0_0(theta, phi):
    return 1


def M_gg_122_2_0():
    def func(k):
        return (1 / 42) * k**2

    return func


def N_gg_122_2_0(theta, phi):
    return (15 / 4) * np.cos(2 * phi) + 5 / 4


def M_gg_122_4_0():
    def func(k):
        return (2 / 231) * k**2

    return func


def N_gg_122_4_0(theta, phi):
    return (45 / 16) * np.cos(2 * phi) + (315 / 64) * np.cos(4 * phi) + 81 / 64


def M_gg_122_6_0():
    def func(k):
        return (4 / 3003) * k**2

    return func


def N_gg_122_6_0(theta, phi):
    return (
        (1365 / 512) * np.cos(2 * phi)
        + (819 / 256) * np.cos(4 * phi)
        + (3003 / 512) * np.cos(6 * phi)
        + 325 / 256
    )


def M_gg_123_0_0():
    def func(k):
        return (1 / 28) * k**2

    return func


def N_gg_123_0_0(theta, phi):
    return 1


def M_gg_123_2_0():
    def func(k):
        return (1 / 42) * k**2

    return func


def N_gg_123_2_0(theta, phi):
    return (15 / 4) * np.cos(2 * phi) + 5 / 4


def M_gg_123_4_0():
    def func(k):
        return (2 / 231) * k**2

    return func


def N_gg_123_4_0(theta, phi):
    return (45 / 16) * np.cos(2 * phi) + (315 / 64) * np.cos(4 * phi) + 81 / 64


def M_gg_123_6_0():
    def func(k):
        return (4 / 3003) * k**2

    return func


def N_gg_123_6_0(theta, phi):
    return (
        (1365 / 512) * np.cos(2 * phi)
        + (819 / 256) * np.cos(4 * phi)
        + (3003 / 512) * np.cos(6 * phi)
        + 325 / 256
    )


def M_gv_0_1_0():
    def func(k):
        return (1 / 3) / k

    return func


def N_gv_0_1_0(theta, phi):
    return -3 * np.cos(phi)


def M_gv_1_1_0():
    def func(k):
        return (2 / 3) / k

    return func


def N_gv_1_1_0(theta, phi):
    return -3 * np.cos(phi)


def M_gv_2_1_0():
    def func(k):
        return (2 / 3) / k

    return func


def N_gv_2_1_0(theta, phi):
    return -3 * np.cos(phi)


def M_gv_3_1_0():
    def func(k):
        return 2 * k

    return func


def N_gv_3_1_0(theta, phi):
    return -3 * np.cos(phi)


def M_gv_4_1_0():
    def func(k):
        return 2 * k

    return func


def N_gv_4_1_0(theta, phi):
    return -3 * np.cos(phi)


def M_gv_5_1_0():
    def func(k):
        return -(1 / 3) / k

    return func


def N_gv_5_1_0(theta, phi):
    return -3 * np.cos(phi)


def M_gv_6_1_0():
    def func(k):
        return -(1 / 3) / k

    return func


def N_gv_6_1_0(theta, phi):
    return -3 * np.cos(phi)


def M_gv_7_1_0():
    def func(k):
        return -(1 / 3) / k

    return func


def N_gv_7_1_0(theta, phi):
    return -3 * np.cos(phi)


def M_gv_8_1_0():
    def func(k):
        return -(1 / 3) / k

    return func


def N_gv_8_1_0(theta, phi):
    return -3 * np.cos(phi)


def M_gv_9_1_0():
    def func(k):
        return -(1 / 3) / k

    return func


def N_gv_9_1_0(theta, phi):
    return -3 * np.cos(phi)


def M_gv_10_1_0():
    def func(k):
        return (1 / 3) / k

    return func


def N_gv_10_1_0(theta, phi):
    return -3 * np.cos(phi)


def M_gv_11_1_0():
    def func(k):
        return (1 / 5) / k

    return func


def N_gv_11_1_0(theta, phi):
    return -3 * np.cos(phi)


def M_gv_11_3_0():
    def func(k):
        return (2 / 35) / k

    return func


def N_gv_11_3_0(theta, phi):
    return -21 / 8 * np.cos(phi) - 35 / 8 * np.cos(3 * phi)


def M_gv_12_1_0():
    def func(k):
        return (2 / 3) * k

    return func


def N_gv_12_1_0(theta, phi):
    return -3 * np.cos(phi)


def M_gv_13_1_0():
    def func(k):
        return (2 / 5) * k

    return func


def N_gv_13_1_0(theta, phi):
    return -3 * np.cos(phi)


def M_gv_13_3_0():
    def func(k):
        return (4 / 35) * k

    return func


def N_gv_13_3_0(theta, phi):
    return -21 / 8 * np.cos(phi) - 35 / 8 * np.cos(3 * phi)


def M_gv_14_1_0():
    def func(k):
        return -1 / 3 * k

    return func


def N_gv_14_1_0(theta, phi):
    return -3 * np.cos(phi)


def M_gv_15_1_0():
    def func(k):
        return -1 / 3 * k

    return func


def N_gv_15_1_0(theta, phi):
    return -3 * np.cos(phi)


def M_gv_16_1_0():
    def func(k):
        return -1 / 3 * k

    return func


def N_gv_16_1_0(theta, phi):
    return -3 * np.cos(phi)


def M_gv_17_1_0():
    def func(k):
        return -2 / 3 * k

    return func


def N_gv_17_1_0(theta, phi):
    return -3 * np.cos(phi)


def M_gv_18_1_0():
    def func(k):
        return -2 * k**3

    return func


def N_gv_18_1_0(theta, phi):
    return -3 * np.cos(phi)


def M_gv_19_1_0():
    def func(k):
        return -1 / 6 * k

    return func


def N_gv_19_1_0(theta, phi):
    return -3 * np.cos(phi)


def M_gv_20_1_0():
    def func(k):
        return -1 / 6 * k

    return func


def N_gv_20_1_0(theta, phi):
    return -3 * np.cos(phi)


def M_gv_21_1_0():
    def func(k):
        return -1 / 6 * k

    return func


def N_gv_21_1_0(theta, phi):
    return -3 * np.cos(phi)


def M_gv_22_1_0():
    def func(k):
        return -2 / 3 * k

    return func


def N_gv_22_1_0(theta, phi):
    return -3 * np.cos(phi)


def M_gv_23_1_0():
    def func(k):
        return (1 / 3) / k

    return func


def N_gv_23_1_0(theta, phi):
    return -3 * np.cos(phi)


def M_gv_24_1_0():
    def func(k):
        return (1 / 3) / k

    return func


def N_gv_24_1_0(theta, phi):
    return -3 * np.cos(phi)


def M_gv_25_1_0():
    def func(k):
        return (1 / 5) / k

    return func


def N_gv_25_1_0(theta, phi):
    return -3 * np.cos(phi)


def M_gv_25_3_0():
    def func(k):
        return (2 / 35) / k

    return func


def N_gv_25_3_0(theta, phi):
    return -21 / 8 * np.cos(phi) - 35 / 8 * np.cos(3 * phi)


def M_gv_26_1_0():
    def func(k):
        return (1 / 5) / k

    return func


def N_gv_26_1_0(theta, phi):
    return -3 * np.cos(phi)


def M_gv_26_3_0():
    def func(k):
        return (2 / 35) / k

    return func


def N_gv_26_3_0(theta, phi):
    return -21 / 8 * np.cos(phi) - 35 / 8 * np.cos(3 * phi)


def M_gv_27_1_0():
    def func(k):
        return (1 / 5) / k

    return func


def N_gv_27_1_0(theta, phi):
    return -3 * np.cos(phi)


def M_gv_27_3_0():
    def func(k):
        return (2 / 35) / k

    return func


def N_gv_27_3_0(theta, phi):
    return -21 / 8 * np.cos(phi) - 35 / 8 * np.cos(3 * phi)


def M_gv_28_1_0():
    def func(k):
        return (2 / 5) / k

    return func


def N_gv_28_1_0(theta, phi):
    return -3 * np.cos(phi)


def M_gv_28_3_0():
    def func(k):
        return (4 / 35) / k

    return func


def N_gv_28_3_0(theta, phi):
    return -21 / 8 * np.cos(phi) - 35 / 8 * np.cos(3 * phi)


def M_gv_29_1_0():
    def func(k):
        return (2 / 5) / k

    return func


def N_gv_29_1_0(theta, phi):
    return -3 * np.cos(phi)


def M_gv_29_3_0():
    def func(k):
        return (4 / 35) / k

    return func


def N_gv_29_3_0(theta, phi):
    return -21 / 8 * np.cos(phi) - 35 / 8 * np.cos(3 * phi)


def M_gv_30_1_0():
    def func(k):
        return (1 / 3) / k

    return func


def N_gv_30_1_0(theta, phi):
    return -3 * np.cos(phi)


def M_gv_31_1_0():
    def func(k):
        return (1 / 5) / k

    return func


def N_gv_31_1_0(theta, phi):
    return -3 * np.cos(phi)


def M_gv_31_3_0():
    def func(k):
        return (2 / 35) / k

    return func


def N_gv_31_3_0(theta, phi):
    return -21 / 8 * np.cos(phi) - 35 / 8 * np.cos(3 * phi)


def M_gv_32_1_0():
    def func(k):
        return (6 / 5) * k

    return func


def N_gv_32_1_0(theta, phi):
    return -3 * np.cos(phi)


def M_gv_32_3_0():
    def func(k):
        return (12 / 35) * k

    return func


def N_gv_32_3_0(theta, phi):
    return -21 / 8 * np.cos(phi) - 35 / 8 * np.cos(3 * phi)


def M_gv_33_1_0():
    def func(k):
        return (6 / 5) * k

    return func


def N_gv_33_1_0(theta, phi):
    return -3 * np.cos(phi)


def M_gv_33_3_0():
    def func(k):
        return (12 / 35) * k

    return func


def N_gv_33_3_0(theta, phi):
    return -21 / 8 * np.cos(phi) - 35 / 8 * np.cos(3 * phi)


def M_gv_34_1_0():
    def func(k):
        return -3 / 10 * k

    return func


def N_gv_34_1_0(theta, phi):
    return -3 * np.cos(phi)


def M_gv_34_3_0():
    def func(k):
        return -3 / 35 * k

    return func


def N_gv_34_3_0(theta, phi):
    return -21 / 8 * np.cos(phi) - 35 / 8 * np.cos(3 * phi)


def M_gv_35_1_0():
    def func(k):
        return -3 / 5 * k

    return func


def N_gv_35_1_0(theta, phi):
    return -3 * np.cos(phi)


def M_gv_35_3_0():
    def func(k):
        return -6 / 35 * k

    return func


def N_gv_35_3_0(theta, phi):
    return -21 / 8 * np.cos(phi) - 35 / 8 * np.cos(3 * phi)


def M_gv_36_1_0():
    def func(k):
        return -3 / 5 * k

    return func


def N_gv_36_1_0(theta, phi):
    return -3 * np.cos(phi)


def M_gv_36_3_0():
    def func(k):
        return -6 / 35 * k

    return func


def N_gv_36_3_0(theta, phi):
    return -21 / 8 * np.cos(phi) - 35 / 8 * np.cos(3 * phi)


def M_gv_37_1_0():
    def func(k):
        return -9 / 5 * k**3

    return func


def N_gv_37_1_0(theta, phi):
    return -3 * np.cos(phi)


def M_gv_37_3_0():
    def func(k):
        return -18 / 35 * k**3

    return func


def N_gv_37_3_0(theta, phi):
    return -21 / 8 * np.cos(phi) - 35 / 8 * np.cos(3 * phi)


def M_gv_38_1_0():
    def func(k):
        return -9 / 5 * k**3

    return func


def N_gv_38_1_0(theta, phi):
    return -3 * np.cos(phi)


def M_gv_38_3_0():
    def func(k):
        return -18 / 35 * k**3

    return func


def N_gv_38_3_0(theta, phi):
    return -21 / 8 * np.cos(phi) - 35 / 8 * np.cos(3 * phi)


def M_gv_39_1_0():
    def func(k):
        return (3 / 10) * k

    return func


def N_gv_39_1_0(theta, phi):
    return -3 * np.cos(phi)


def M_gv_39_3_0():
    def func(k):
        return (3 / 35) * k

    return func


def N_gv_39_3_0(theta, phi):
    return -21 / 8 * np.cos(phi) - 35 / 8 * np.cos(3 * phi)


def M_gv_40_1_0():
    def func(k):
        return (3 / 10) * k

    return func


def N_gv_40_1_0(theta, phi):
    return -3 * np.cos(phi)


def M_gv_40_3_0():
    def func(k):
        return (3 / 35) * k

    return func


def N_gv_40_3_0(theta, phi):
    return -21 / 8 * np.cos(phi) - 35 / 8 * np.cos(3 * phi)


def M_gv_41_1_0():
    def func(k):
        return (3 / 10) * k

    return func


def N_gv_41_1_0(theta, phi):
    return -3 * np.cos(phi)


def M_gv_41_3_0():
    def func(k):
        return (3 / 35) * k

    return func


def N_gv_41_3_0(theta, phi):
    return -21 / 8 * np.cos(phi) - 35 / 8 * np.cos(3 * phi)


def M_gv_42_1_0():
    def func(k):
        return (3 / 10) * k

    return func


def N_gv_42_1_0(theta, phi):
    return -3 * np.cos(phi)


def M_gv_42_3_0():
    def func(k):
        return (3 / 35) * k

    return func


def N_gv_42_3_0(theta, phi):
    return -21 / 8 * np.cos(phi) - 35 / 8 * np.cos(3 * phi)


def M_gv_43_1_0():
    def func(k):
        return (3 / 10) * k

    return func


def N_gv_43_1_0(theta, phi):
    return -3 * np.cos(phi)


def M_gv_43_3_0():
    def func(k):
        return (3 / 35) * k

    return func


def N_gv_43_3_0(theta, phi):
    return -21 / 8 * np.cos(phi) - 35 / 8 * np.cos(3 * phi)


def M_gv_44_1_0():
    def func(k):
        return -1 / 5 * k

    return func


def N_gv_44_1_0(theta, phi):
    return -3 * np.cos(phi)


def M_gv_44_3_0():
    def func(k):
        return -2 / 35 * k

    return func


def N_gv_44_3_0(theta, phi):
    return -21 / 8 * np.cos(phi) - 35 / 8 * np.cos(3 * phi)


def M_gv_45_1_0():
    def func(k):
        return -1 / 7 * k

    return func


def N_gv_45_1_0(theta, phi):
    return -3 * np.cos(phi)


def M_gv_45_3_0():
    def func(k):
        return -4 / 63 * k

    return func


def N_gv_45_3_0(theta, phi):
    return -21 / 8 * np.cos(phi) - 35 / 8 * np.cos(3 * phi)


def M_gv_45_5_0():
    def func(k):
        return -8 / 693 * k

    return func


def N_gv_45_5_0(theta, phi):
    return (
        -165 / 64 * np.cos(phi)
        - 385 / 128 * np.cos(3 * phi)
        - 693 / 128 * np.cos(5 * phi)
    )


def M_gv_46_1_0():
    def func(k):
        return -2 / 5 * k**3

    return func


def N_gv_46_1_0(theta, phi):
    return -3 * np.cos(phi)


def M_gv_46_3_0():
    def func(k):
        return -4 / 35 * k**3

    return func


def N_gv_46_3_0(theta, phi):
    return -21 / 8 * np.cos(phi) - 35 / 8 * np.cos(3 * phi)


def M_gv_47_1_0():
    def func(k):
        return -2 / 7 * k**3

    return func


def N_gv_47_1_0(theta, phi):
    return -3 * np.cos(phi)


def M_gv_47_3_0():
    def func(k):
        return -8 / 63 * k**3

    return func


def N_gv_47_3_0(theta, phi):
    return -21 / 8 * np.cos(phi) - 35 / 8 * np.cos(3 * phi)


def M_gv_47_5_0():
    def func(k):
        return -16 / 693 * k**3

    return func


def N_gv_47_5_0(theta, phi):
    return (
        -165 / 64 * np.cos(phi)
        - 385 / 128 * np.cos(3 * phi)
        - 693 / 128 * np.cos(5 * phi)
    )


def M_gv_48_1_0():
    def func(k):
        return (1 / 10) * k**3

    return func


def N_gv_48_1_0(theta, phi):
    return -3 * np.cos(phi)


def M_gv_48_3_0():
    def func(k):
        return (1 / 35) * k**3

    return func


def N_gv_48_3_0(theta, phi):
    return -21 / 8 * np.cos(phi) - 35 / 8 * np.cos(3 * phi)


def M_gv_49_1_0():
    def func(k):
        return (1 / 10) * k**3

    return func


def N_gv_49_1_0(theta, phi):
    return -3 * np.cos(phi)


def M_gv_49_3_0():
    def func(k):
        return (1 / 35) * k**3

    return func


def N_gv_49_3_0(theta, phi):
    return -21 / 8 * np.cos(phi) - 35 / 8 * np.cos(3 * phi)


def M_gv_50_1_0():
    def func(k):
        return (1 / 10) * k**3

    return func


def N_gv_50_1_0(theta, phi):
    return -3 * np.cos(phi)


def M_gv_50_3_0():
    def func(k):
        return (1 / 35) * k**3

    return func


def N_gv_50_3_0(theta, phi):
    return -21 / 8 * np.cos(phi) - 35 / 8 * np.cos(3 * phi)


def M_gv_51_1_0():
    def func(k):
        return (1 / 5) * k**3

    return func


def N_gv_51_1_0(theta, phi):
    return -3 * np.cos(phi)


def M_gv_51_3_0():
    def func(k):
        return (2 / 35) * k**3

    return func


def N_gv_51_3_0(theta, phi):
    return -21 / 8 * np.cos(phi) - 35 / 8 * np.cos(3 * phi)


def M_gv_52_1_0():
    def func(k):
        return (3 / 5) * k**5

    return func


def N_gv_52_1_0(theta, phi):
    return -3 * np.cos(phi)


def M_gv_52_3_0():
    def func(k):
        return (6 / 35) * k**5

    return func


def N_gv_52_3_0(theta, phi):
    return -21 / 8 * np.cos(phi) - 35 / 8 * np.cos(3 * phi)


def M_gv_53_1_0():
    def func(k):
        return (1 / 20) * k**3

    return func


def N_gv_53_1_0(theta, phi):
    return -3 * np.cos(phi)


def M_gv_53_3_0():
    def func(k):
        return (1 / 70) * k**3

    return func


def N_gv_53_3_0(theta, phi):
    return -21 / 8 * np.cos(phi) - 35 / 8 * np.cos(3 * phi)


def M_gv_54_1_0():
    def func(k):
        return (1 / 20) * k**3

    return func


def N_gv_54_1_0(theta, phi):
    return -3 * np.cos(phi)


def M_gv_54_3_0():
    def func(k):
        return (1 / 70) * k**3

    return func


def N_gv_54_3_0(theta, phi):
    return -21 / 8 * np.cos(phi) - 35 / 8 * np.cos(3 * phi)


def M_gv_55_1_0():
    def func(k):
        return (1 / 20) * k**3

    return func


def N_gv_55_1_0(theta, phi):
    return -3 * np.cos(phi)


def M_gv_55_3_0():
    def func(k):
        return (1 / 70) * k**3

    return func


def N_gv_55_3_0(theta, phi):
    return -21 / 8 * np.cos(phi) - 35 / 8 * np.cos(3 * phi)


def M_gv_56_1_0():
    def func(k):
        return (1 / 5) * k**3

    return func


def N_gv_56_1_0(theta, phi):
    return -3 * np.cos(phi)


def M_gv_56_3_0():
    def func(k):
        return (2 / 35) * k**3

    return func


def N_gv_56_3_0(theta, phi):
    return -21 / 8 * np.cos(phi) - 35 / 8 * np.cos(3 * phi)


def M_gv_57_1_0():
    def func(k):
        return (1 / 10) * k**3

    return func


def N_gv_57_1_0(theta, phi):
    return -3 * np.cos(phi)


def M_gv_57_3_0():
    def func(k):
        return (1 / 35) * k**3

    return func


def N_gv_57_3_0(theta, phi):
    return -21 / 8 * np.cos(phi) - 35 / 8 * np.cos(3 * phi)


def M_gv_58_1_0():
    def func(k):
        return (1 / 10) * k**3

    return func


def N_gv_58_1_0(theta, phi):
    return -3 * np.cos(phi)


def M_gv_58_3_0():
    def func(k):
        return (1 / 35) * k**3

    return func


def N_gv_58_3_0(theta, phi):
    return -21 / 8 * np.cos(phi) - 35 / 8 * np.cos(3 * phi)


def M_gv_59_1_0():
    def func(k):
        return (1 / 10) * k**3

    return func


def N_gv_59_1_0(theta, phi):
    return -3 * np.cos(phi)


def M_gv_59_3_0():
    def func(k):
        return (1 / 35) * k**3

    return func


def N_gv_59_3_0(theta, phi):
    return -21 / 8 * np.cos(phi) - 35 / 8 * np.cos(3 * phi)


def M_gv_60_1_0():
    def func(k):
        return (1 / 5) * k**3

    return func


def N_gv_60_1_0(theta, phi):
    return -3 * np.cos(phi)


def M_gv_60_3_0():
    def func(k):
        return (2 / 35) * k**3

    return func


def N_gv_60_3_0(theta, phi):
    return -21 / 8 * np.cos(phi) - 35 / 8 * np.cos(3 * phi)


def M_gv_61_1_0():
    def func(k):
        return (3 / 5) * k**5

    return func


def N_gv_61_1_0(theta, phi):
    return -3 * np.cos(phi)


def M_gv_61_3_0():
    def func(k):
        return (6 / 35) * k**5

    return func


def N_gv_61_3_0(theta, phi):
    return -21 / 8 * np.cos(phi) - 35 / 8 * np.cos(3 * phi)


def M_gv_62_1_0():
    def func(k):
        return (1 / 20) * k**3

    return func


def N_gv_62_1_0(theta, phi):
    return -3 * np.cos(phi)


def M_gv_62_3_0():
    def func(k):
        return (1 / 70) * k**3

    return func


def N_gv_62_3_0(theta, phi):
    return -21 / 8 * np.cos(phi) - 35 / 8 * np.cos(3 * phi)


def M_gv_63_1_0():
    def func(k):
        return (1 / 20) * k**3

    return func


def N_gv_63_1_0(theta, phi):
    return -3 * np.cos(phi)


def M_gv_63_3_0():
    def func(k):
        return (1 / 70) * k**3

    return func


def N_gv_63_3_0(theta, phi):
    return -21 / 8 * np.cos(phi) - 35 / 8 * np.cos(3 * phi)


def M_gv_64_1_0():
    def func(k):
        return (1 / 20) * k**3

    return func


def N_gv_64_1_0(theta, phi):
    return -3 * np.cos(phi)


def M_gv_64_3_0():
    def func(k):
        return (1 / 70) * k**3

    return func


def N_gv_64_3_0(theta, phi):
    return -21 / 8 * np.cos(phi) - 35 / 8 * np.cos(3 * phi)


def M_gv_65_1_0():
    def func(k):
        return (1 / 5) * k**3

    return func


def N_gv_65_1_0(theta, phi):
    return -3 * np.cos(phi)


def M_gv_65_3_0():
    def func(k):
        return (2 / 35) * k**3

    return func


def N_gv_65_3_0(theta, phi):
    return -21 / 8 * np.cos(phi) - 35 / 8 * np.cos(3 * phi)


def M_gv_66_1_0():
    def func(k):
        return (3 / 10) / k

    return func


def N_gv_66_1_0(theta, phi):
    return -3 * np.cos(phi)


def M_gv_66_3_0():
    def func(k):
        return (3 / 35) / k

    return func


def N_gv_66_3_0(theta, phi):
    return -21 / 8 * np.cos(phi) - 35 / 8 * np.cos(3 * phi)


def M_gv_67_1_0():
    def func(k):
        return (3 / 14) / k

    return func


def N_gv_67_1_0(theta, phi):
    return -3 * np.cos(phi)


def M_gv_67_3_0():
    def func(k):
        return (2 / 21) / k

    return func


def N_gv_67_3_0(theta, phi):
    return -21 / 8 * np.cos(phi) - 35 / 8 * np.cos(3 * phi)


def M_gv_67_5_0():
    def func(k):
        return (4 / 231) / k

    return func


def N_gv_67_5_0(theta, phi):
    return (
        -165 / 64 * np.cos(phi)
        - 385 / 128 * np.cos(3 * phi)
        - 693 / 128 * np.cos(5 * phi)
    )


def M_gv_68_1_0():
    def func(k):
        return -(3 / 10) / k

    return func


def N_gv_68_1_0(theta, phi):
    return -3 * np.cos(phi)


def M_gv_68_3_0():
    def func(k):
        return -(3 / 35) / k

    return func


def N_gv_68_3_0(theta, phi):
    return -21 / 8 * np.cos(phi) - 35 / 8 * np.cos(3 * phi)


def M_gv_69_1_0():
    def func(k):
        return -(3 / 14) / k

    return func


def N_gv_69_1_0(theta, phi):
    return -3 * np.cos(phi)


def M_gv_69_3_0():
    def func(k):
        return -(2 / 21) / k

    return func


def N_gv_69_3_0(theta, phi):
    return -21 / 8 * np.cos(phi) - 35 / 8 * np.cos(3 * phi)


def M_gv_69_5_0():
    def func(k):
        return -(4 / 231) / k

    return func


def N_gv_69_5_0(theta, phi):
    return (
        -165 / 64 * np.cos(phi)
        - 385 / 128 * np.cos(3 * phi)
        - 693 / 128 * np.cos(5 * phi)
    )


def M_gv_70_1_0():
    def func(k):
        return (3 / 5) * k

    return func


def N_gv_70_1_0(theta, phi):
    return -3 * np.cos(phi)


def M_gv_70_3_0():
    def func(k):
        return (6 / 35) * k

    return func


def N_gv_70_3_0(theta, phi):
    return -21 / 8 * np.cos(phi) - 35 / 8 * np.cos(3 * phi)


def M_gv_71_1_0():
    def func(k):
        return (3 / 7) * k

    return func


def N_gv_71_1_0(theta, phi):
    return -3 * np.cos(phi)


def M_gv_71_3_0():
    def func(k):
        return (4 / 21) * k

    return func


def N_gv_71_3_0(theta, phi):
    return -21 / 8 * np.cos(phi) - 35 / 8 * np.cos(3 * phi)


def M_gv_71_5_0():
    def func(k):
        return (8 / 231) * k

    return func


def N_gv_71_5_0(theta, phi):
    return (
        -165 / 64 * np.cos(phi)
        - 385 / 128 * np.cos(3 * phi)
        - 693 / 128 * np.cos(5 * phi)
    )


def M_gv_72_1_0():
    def func(k):
        return -3 / 10 * k

    return func


def N_gv_72_1_0(theta, phi):
    return -3 * np.cos(phi)


def M_gv_72_3_0():
    def func(k):
        return -3 / 35 * k

    return func


def N_gv_72_3_0(theta, phi):
    return -21 / 8 * np.cos(phi) - 35 / 8 * np.cos(3 * phi)


def M_gv_73_1_0():
    def func(k):
        return -3 / 5 * k

    return func


def N_gv_73_1_0(theta, phi):
    return -3 * np.cos(phi)


def M_gv_73_3_0():
    def func(k):
        return -6 / 35 * k

    return func


def N_gv_73_3_0(theta, phi):
    return -21 / 8 * np.cos(phi) - 35 / 8 * np.cos(3 * phi)


def M_gv_74_1_0():
    def func(k):
        return -3 / 5 * k

    return func


def N_gv_74_1_0(theta, phi):
    return -3 * np.cos(phi)


def M_gv_74_3_0():
    def func(k):
        return -6 / 35 * k

    return func


def N_gv_74_3_0(theta, phi):
    return -21 / 8 * np.cos(phi) - 35 / 8 * np.cos(3 * phi)


def M_gv_75_1_0():
    def func(k):
        return -9 / 5 * k**3

    return func


def N_gv_75_1_0(theta, phi):
    return -3 * np.cos(phi)


def M_gv_75_3_0():
    def func(k):
        return -18 / 35 * k**3

    return func


def N_gv_75_3_0(theta, phi):
    return -21 / 8 * np.cos(phi) - 35 / 8 * np.cos(3 * phi)


def M_gv_76_1_0():
    def func(k):
        return -9 / 5 * k**3

    return func


def N_gv_76_1_0(theta, phi):
    return -3 * np.cos(phi)


def M_gv_76_3_0():
    def func(k):
        return -18 / 35 * k**3

    return func


def N_gv_76_3_0(theta, phi):
    return -21 / 8 * np.cos(phi) - 35 / 8 * np.cos(3 * phi)


def M_gv_77_1_0():
    def func(k):
        return (3 / 10) * k

    return func


def N_gv_77_1_0(theta, phi):
    return -3 * np.cos(phi)


def M_gv_77_3_0():
    def func(k):
        return (3 / 35) * k

    return func


def N_gv_77_3_0(theta, phi):
    return -21 / 8 * np.cos(phi) - 35 / 8 * np.cos(3 * phi)


def M_gv_78_1_0():
    def func(k):
        return (3 / 10) * k

    return func


def N_gv_78_1_0(theta, phi):
    return -3 * np.cos(phi)


def M_gv_78_3_0():
    def func(k):
        return (3 / 35) * k

    return func


def N_gv_78_3_0(theta, phi):
    return -21 / 8 * np.cos(phi) - 35 / 8 * np.cos(3 * phi)


def M_gv_79_1_0():
    def func(k):
        return (3 / 10) * k

    return func


def N_gv_79_1_0(theta, phi):
    return -3 * np.cos(phi)


def M_gv_79_3_0():
    def func(k):
        return (3 / 35) * k

    return func


def N_gv_79_3_0(theta, phi):
    return -21 / 8 * np.cos(phi) - 35 / 8 * np.cos(3 * phi)


def M_gv_80_1_0():
    def func(k):
        return (3 / 10) * k

    return func


def N_gv_80_1_0(theta, phi):
    return -3 * np.cos(phi)


def M_gv_80_3_0():
    def func(k):
        return (3 / 35) * k

    return func


def N_gv_80_3_0(theta, phi):
    return -21 / 8 * np.cos(phi) - 35 / 8 * np.cos(3 * phi)


def M_gv_81_1_0():
    def func(k):
        return (3 / 10) * k

    return func


def N_gv_81_1_0(theta, phi):
    return -3 * np.cos(phi)


def M_gv_81_3_0():
    def func(k):
        return (3 / 35) * k

    return func


def N_gv_81_3_0(theta, phi):
    return -21 / 8 * np.cos(phi) - 35 / 8 * np.cos(3 * phi)


def M_gv_82_1_0():
    def func(k):
        return (3 / 5) * k

    return func


def N_gv_82_1_0(theta, phi):
    return -3 * np.cos(phi)


def M_gv_82_3_0():
    def func(k):
        return (6 / 35) * k

    return func


def N_gv_82_3_0(theta, phi):
    return -21 / 8 * np.cos(phi) - 35 / 8 * np.cos(3 * phi)


def M_gv_83_1_0():
    def func(k):
        return (9 / 5) * k**3

    return func


def N_gv_83_1_0(theta, phi):
    return -3 * np.cos(phi)


def M_gv_83_3_0():
    def func(k):
        return (18 / 35) * k**3

    return func


def N_gv_83_3_0(theta, phi):
    return -21 / 8 * np.cos(phi) - 35 / 8 * np.cos(3 * phi)


def M_gv_84_1_0():
    def func(k):
        return -2 / 7 * k

    return func


def N_gv_84_1_0(theta, phi):
    return -3 * np.cos(phi)


def M_gv_84_3_0():
    def func(k):
        return -8 / 63 * k

    return func


def N_gv_84_3_0(theta, phi):
    return -21 / 8 * np.cos(phi) - 35 / 8 * np.cos(3 * phi)


def M_gv_84_5_0():
    def func(k):
        return -16 / 693 * k

    return func


def N_gv_84_5_0(theta, phi):
    return (
        -165 / 64 * np.cos(phi)
        - 385 / 128 * np.cos(3 * phi)
        - 693 / 128 * np.cos(5 * phi)
    )


def M_gv_85_1_0():
    def func(k):
        return -4 / 7 * k

    return func


def N_gv_85_1_0(theta, phi):
    return -3 * np.cos(phi)


def M_gv_85_3_0():
    def func(k):
        return -16 / 63 * k

    return func


def N_gv_85_3_0(theta, phi):
    return -21 / 8 * np.cos(phi) - 35 / 8 * np.cos(3 * phi)


def M_gv_85_5_0():
    def func(k):
        return -32 / 693 * k

    return func


def N_gv_85_5_0(theta, phi):
    return (
        -165 / 64 * np.cos(phi)
        - 385 / 128 * np.cos(3 * phi)
        - 693 / 128 * np.cos(5 * phi)
    )


def M_gv_86_1_0():
    def func(k):
        return -4 / 7 * k

    return func


def N_gv_86_1_0(theta, phi):
    return -3 * np.cos(phi)


def M_gv_86_3_0():
    def func(k):
        return -16 / 63 * k

    return func


def N_gv_86_3_0(theta, phi):
    return -21 / 8 * np.cos(phi) - 35 / 8 * np.cos(3 * phi)


def M_gv_86_5_0():
    def func(k):
        return -32 / 693 * k

    return func


def N_gv_86_5_0(theta, phi):
    return (
        -165 / 64 * np.cos(phi)
        - 385 / 128 * np.cos(3 * phi)
        - 693 / 128 * np.cos(5 * phi)
    )


def M_gv_87_1_0():
    def func(k):
        return -12 / 7 * k**3

    return func


def N_gv_87_1_0(theta, phi):
    return -3 * np.cos(phi)


def M_gv_87_3_0():
    def func(k):
        return -16 / 21 * k**3

    return func


def N_gv_87_3_0(theta, phi):
    return -21 / 8 * np.cos(phi) - 35 / 8 * np.cos(3 * phi)


def M_gv_87_5_0():
    def func(k):
        return -32 / 231 * k**3

    return func


def N_gv_87_5_0(theta, phi):
    return (
        -165 / 64 * np.cos(phi)
        - 385 / 128 * np.cos(3 * phi)
        - 693 / 128 * np.cos(5 * phi)
    )


def M_gv_88_1_0():
    def func(k):
        return -12 / 7 * k**3

    return func


def N_gv_88_1_0(theta, phi):
    return -3 * np.cos(phi)


def M_gv_88_3_0():
    def func(k):
        return -16 / 21 * k**3

    return func


def N_gv_88_3_0(theta, phi):
    return -21 / 8 * np.cos(phi) - 35 / 8 * np.cos(3 * phi)


def M_gv_88_5_0():
    def func(k):
        return -32 / 231 * k**3

    return func


def N_gv_88_5_0(theta, phi):
    return (
        -165 / 64 * np.cos(phi)
        - 385 / 128 * np.cos(3 * phi)
        - 693 / 128 * np.cos(5 * phi)
    )


def M_gv_89_1_0():
    def func(k):
        return -2 / 7 * k

    return func


def N_gv_89_1_0(theta, phi):
    return -3 * np.cos(phi)


def M_gv_89_3_0():
    def func(k):
        return -8 / 63 * k

    return func


def N_gv_89_3_0(theta, phi):
    return -21 / 8 * np.cos(phi) - 35 / 8 * np.cos(3 * phi)


def M_gv_89_5_0():
    def func(k):
        return -16 / 693 * k

    return func


def N_gv_89_5_0(theta, phi):
    return (
        -165 / 64 * np.cos(phi)
        - 385 / 128 * np.cos(3 * phi)
        - 693 / 128 * np.cos(5 * phi)
    )


def M_gv_90_1_0():
    def func(k):
        return -2 / 5 * k

    return func


def N_gv_90_1_0(theta, phi):
    return -3 * np.cos(phi)


def M_gv_90_3_0():
    def func(k):
        return -4 / 35 * k

    return func


def N_gv_90_3_0(theta, phi):
    return -21 / 8 * np.cos(phi) - 35 / 8 * np.cos(3 * phi)


def M_gv_91_1_0():
    def func(k):
        return (1 / 40) / k

    return func


def N_gv_91_1_0(theta, phi):
    return -3 * np.cos(phi)


def M_gv_91_3_0():
    def func(k):
        return (1 / 140) / k

    return func


def N_gv_91_3_0(theta, phi):
    return -21 / 8 * np.cos(phi) - 35 / 8 * np.cos(3 * phi)


def M_gv_92_1_0():
    def func(k):
        return (1 / 28) / k

    return func


def N_gv_92_1_0(theta, phi):
    return -3 * np.cos(phi)


def M_gv_92_3_0():
    def func(k):
        return (1 / 63) / k

    return func


def N_gv_92_3_0(theta, phi):
    return -21 / 8 * np.cos(phi) - 35 / 8 * np.cos(3 * phi)


def M_gv_92_5_0():
    def func(k):
        return (2 / 693) / k

    return func


def N_gv_92_5_0(theta, phi):
    return (
        -165 / 64 * np.cos(phi)
        - 385 / 128 * np.cos(3 * phi)
        - 693 / 128 * np.cos(5 * phi)
    )


def M_gv_93_1_0():
    def func(k):
        return (1 / 72) / k

    return func


def N_gv_93_1_0(theta, phi):
    return -3 * np.cos(phi)


def M_gv_93_3_0():
    def func(k):
        return (1 / 132) / k

    return func


def N_gv_93_3_0(theta, phi):
    return -21 / 8 * np.cos(phi) - 35 / 8 * np.cos(3 * phi)


def M_gv_93_5_0():
    def func(k):
        return (1 / 429) / k

    return func


def N_gv_93_5_0(theta, phi):
    return (
        -165 / 64 * np.cos(phi)
        - 385 / 128 * np.cos(3 * phi)
        - 693 / 128 * np.cos(5 * phi)
    )


def M_gv_93_7_0():
    def func(k):
        return (2 / 6435) / k

    return func


def N_gv_93_7_0(theta, phi):
    return (
        -2625 / 1024 * np.cos(phi)
        - 2835 / 1024 * np.cos(3 * phi)
        - 3465 / 1024 * np.cos(5 * phi)
        - 6435 / 1024 * np.cos(7 * phi)
    )


def M_gv_94_1_0():
    def func(k):
        return (3 / 10) * k**3

    return func


def N_gv_94_1_0(theta, phi):
    return -3 * np.cos(phi)


def M_gv_94_3_0():
    def func(k):
        return (3 / 35) * k**3

    return func


def N_gv_94_3_0(theta, phi):
    return -21 / 8 * np.cos(phi) - 35 / 8 * np.cos(3 * phi)


def M_gv_95_1_0():
    def func(k):
        return (3 / 10) * k**3

    return func


def N_gv_95_1_0(theta, phi):
    return -3 * np.cos(phi)


def M_gv_95_3_0():
    def func(k):
        return (3 / 35) * k**3

    return func


def N_gv_95_3_0(theta, phi):
    return -21 / 8 * np.cos(phi) - 35 / 8 * np.cos(3 * phi)


def M_gv_96_1_0():
    def func(k):
        return (3 / 10) * k**3

    return func


def N_gv_96_1_0(theta, phi):
    return -3 * np.cos(phi)


def M_gv_96_3_0():
    def func(k):
        return (3 / 35) * k**3

    return func


def N_gv_96_3_0(theta, phi):
    return -21 / 8 * np.cos(phi) - 35 / 8 * np.cos(3 * phi)


def M_gv_97_1_0():
    def func(k):
        return (3 / 5) * k**3

    return func


def N_gv_97_1_0(theta, phi):
    return -3 * np.cos(phi)


def M_gv_97_3_0():
    def func(k):
        return (6 / 35) * k**3

    return func


def N_gv_97_3_0(theta, phi):
    return -21 / 8 * np.cos(phi) - 35 / 8 * np.cos(3 * phi)


def M_gv_98_1_0():
    def func(k):
        return (9 / 5) * k**5

    return func


def N_gv_98_1_0(theta, phi):
    return -3 * np.cos(phi)


def M_gv_98_3_0():
    def func(k):
        return (18 / 35) * k**5

    return func


def N_gv_98_3_0(theta, phi):
    return -21 / 8 * np.cos(phi) - 35 / 8 * np.cos(3 * phi)


def M_gv_99_1_0():
    def func(k):
        return (3 / 20) * k**3

    return func


def N_gv_99_1_0(theta, phi):
    return -3 * np.cos(phi)


def M_gv_99_3_0():
    def func(k):
        return (3 / 70) * k**3

    return func


def N_gv_99_3_0(theta, phi):
    return -21 / 8 * np.cos(phi) - 35 / 8 * np.cos(3 * phi)


def M_gv_100_1_0():
    def func(k):
        return (3 / 20) * k**3

    return func


def N_gv_100_1_0(theta, phi):
    return -3 * np.cos(phi)


def M_gv_100_3_0():
    def func(k):
        return (3 / 70) * k**3

    return func


def N_gv_100_3_0(theta, phi):
    return -21 / 8 * np.cos(phi) - 35 / 8 * np.cos(3 * phi)


def M_gv_101_1_0():
    def func(k):
        return (3 / 20) * k**3

    return func


def N_gv_101_1_0(theta, phi):
    return -3 * np.cos(phi)


def M_gv_101_3_0():
    def func(k):
        return (3 / 70) * k**3

    return func


def N_gv_101_3_0(theta, phi):
    return -21 / 8 * np.cos(phi) - 35 / 8 * np.cos(3 * phi)


def M_gv_102_1_0():
    def func(k):
        return (3 / 5) * k**3

    return func


def N_gv_102_1_0(theta, phi):
    return -3 * np.cos(phi)


def M_gv_102_3_0():
    def func(k):
        return (6 / 35) * k**3

    return func


def N_gv_102_3_0(theta, phi):
    return -21 / 8 * np.cos(phi) - 35 / 8 * np.cos(3 * phi)


def M_gv_103_1_0():
    def func(k):
        return -1 / 5 * k

    return func


def N_gv_103_1_0(theta, phi):
    return -3 * np.cos(phi)


def M_gv_103_3_0():
    def func(k):
        return -2 / 35 * k

    return func


def N_gv_103_3_0(theta, phi):
    return -21 / 8 * np.cos(phi) - 35 / 8 * np.cos(3 * phi)


def M_gv_104_1_0():
    def func(k):
        return -1 / 7 * k

    return func


def N_gv_104_1_0(theta, phi):
    return -3 * np.cos(phi)


def M_gv_104_3_0():
    def func(k):
        return -4 / 63 * k

    return func


def N_gv_104_3_0(theta, phi):
    return -21 / 8 * np.cos(phi) - 35 / 8 * np.cos(3 * phi)


def M_gv_104_5_0():
    def func(k):
        return -8 / 693 * k

    return func


def N_gv_104_5_0(theta, phi):
    return (
        -165 / 64 * np.cos(phi)
        - 385 / 128 * np.cos(3 * phi)
        - 693 / 128 * np.cos(5 * phi)
    )


def M_gv_105_1_0():
    def func(k):
        return -2 / 5 * k**3

    return func


def N_gv_105_1_0(theta, phi):
    return -3 * np.cos(phi)


def M_gv_105_3_0():
    def func(k):
        return -4 / 35 * k**3

    return func


def N_gv_105_3_0(theta, phi):
    return -21 / 8 * np.cos(phi) - 35 / 8 * np.cos(3 * phi)


def M_gv_106_1_0():
    def func(k):
        return -2 / 7 * k**3

    return func


def N_gv_106_1_0(theta, phi):
    return -3 * np.cos(phi)


def M_gv_106_3_0():
    def func(k):
        return -8 / 63 * k**3

    return func


def N_gv_106_3_0(theta, phi):
    return -21 / 8 * np.cos(phi) - 35 / 8 * np.cos(3 * phi)


def M_gv_106_5_0():
    def func(k):
        return -16 / 693 * k**3

    return func


def N_gv_106_5_0(theta, phi):
    return (
        -165 / 64 * np.cos(phi)
        - 385 / 128 * np.cos(3 * phi)
        - 693 / 128 * np.cos(5 * phi)
    )


def M_gv_107_1_0():
    def func(k):
        return -1 / 5 * k

    return func


def N_gv_107_1_0(theta, phi):
    return -3 * np.cos(phi)


def M_gv_107_3_0():
    def func(k):
        return -2 / 35 * k

    return func


def N_gv_107_3_0(theta, phi):
    return -21 / 8 * np.cos(phi) - 35 / 8 * np.cos(3 * phi)


def M_gv_108_1_0():
    def func(k):
        return -1 / 5 * k

    return func


def N_gv_108_1_0(theta, phi):
    return -3 * np.cos(phi)


def M_gv_108_3_0():
    def func(k):
        return -2 / 35 * k

    return func


def N_gv_108_3_0(theta, phi):
    return -21 / 8 * np.cos(phi) - 35 / 8 * np.cos(3 * phi)


def M_gv_109_1_0():
    def func(k):
        return -1 / 7 * k

    return func


def N_gv_109_1_0(theta, phi):
    return -3 * np.cos(phi)


def M_gv_109_3_0():
    def func(k):
        return -4 / 63 * k

    return func


def N_gv_109_3_0(theta, phi):
    return -21 / 8 * np.cos(phi) - 35 / 8 * np.cos(3 * phi)


def M_gv_109_5_0():
    def func(k):
        return -8 / 693 * k

    return func


def N_gv_109_5_0(theta, phi):
    return (
        -165 / 64 * np.cos(phi)
        - 385 / 128 * np.cos(3 * phi)
        - 693 / 128 * np.cos(5 * phi)
    )


def M_gv_110_1_0():
    def func(k):
        return -1 / 7 * k

    return func


def N_gv_110_1_0(theta, phi):
    return -3 * np.cos(phi)


def M_gv_110_3_0():
    def func(k):
        return -4 / 63 * k

    return func


def N_gv_110_3_0(theta, phi):
    return -21 / 8 * np.cos(phi) - 35 / 8 * np.cos(3 * phi)


def M_gv_110_5_0():
    def func(k):
        return -8 / 693 * k

    return func


def N_gv_110_5_0(theta, phi):
    return (
        -165 / 64 * np.cos(phi)
        - 385 / 128 * np.cos(3 * phi)
        - 693 / 128 * np.cos(5 * phi)
    )


def M_gv_111_1_0():
    def func(k):
        return (1 / 10) * k

    return func


def N_gv_111_1_0(theta, phi):
    return -3 * np.cos(phi)


def M_gv_111_3_0():
    def func(k):
        return (1 / 35) * k

    return func


def N_gv_111_3_0(theta, phi):
    return -21 / 8 * np.cos(phi) - 35 / 8 * np.cos(3 * phi)


def M_gv_112_1_0():
    def func(k):
        return (1 / 10) * k

    return func


def N_gv_112_1_0(theta, phi):
    return -3 * np.cos(phi)


def M_gv_112_3_0():
    def func(k):
        return (1 / 35) * k

    return func


def N_gv_112_3_0(theta, phi):
    return -21 / 8 * np.cos(phi) - 35 / 8 * np.cos(3 * phi)


def M_gv_113_1_0():
    def func(k):
        return (1 / 14) * k

    return func


def N_gv_113_1_0(theta, phi):
    return -3 * np.cos(phi)


def M_gv_113_3_0():
    def func(k):
        return (2 / 63) * k

    return func


def N_gv_113_3_0(theta, phi):
    return -21 / 8 * np.cos(phi) - 35 / 8 * np.cos(3 * phi)


def M_gv_113_5_0():
    def func(k):
        return (4 / 693) * k

    return func


def N_gv_113_5_0(theta, phi):
    return (
        -165 / 64 * np.cos(phi)
        - 385 / 128 * np.cos(3 * phi)
        - 693 / 128 * np.cos(5 * phi)
    )


def M_gv_114_1_0():
    def func(k):
        return (1 / 14) * k

    return func


def N_gv_114_1_0(theta, phi):
    return -3 * np.cos(phi)


def M_gv_114_3_0():
    def func(k):
        return (2 / 63) * k

    return func


def N_gv_114_3_0(theta, phi):
    return -21 / 8 * np.cos(phi) - 35 / 8 * np.cos(3 * phi)


def M_gv_114_5_0():
    def func(k):
        return (4 / 693) * k

    return func


def N_gv_114_5_0(theta, phi):
    return (
        -165 / 64 * np.cos(phi)
        - 385 / 128 * np.cos(3 * phi)
        - 693 / 128 * np.cos(5 * phi)
    )


def M_vv_0_0_0():
    def func(k):
        return (1 / 3) / k**2

    return func


def N_vv_0_0_0(theta, phi):
    return 1


def M_vv_0_2_0():
    def func(k):
        return (2 / 15) / k**2

    return func


def N_vv_0_2_0(theta, phi):
    return (15 / 4) * np.cos(2 * phi) + 5 / 4


def M_vv_1_0_0():
    def func(k):
        return (2 / 3) / k**2

    return func


def N_vv_1_0_0(theta, phi):
    return 1


def M_vv_1_2_0():
    def func(k):
        return (4 / 15) / k**2

    return func


def N_vv_1_2_0(theta, phi):
    return (15 / 4) * np.cos(2 * phi) + 5 / 4


def M_vv_2_0_0():
    def func(k):
        return (4 / 3) / k**2

    return func


def N_vv_2_0_0(theta, phi):
    return 1


def M_vv_2_2_0():
    def func(k):
        return (8 / 15) / k**2

    return func


def N_vv_2_2_0(theta, phi):
    return (15 / 4) * np.cos(2 * phi) + 5 / 4


def M_vv_3_0_0():
    def func(k):
        return k ** (-2)

    return func


def N_vv_3_0_0(theta, phi):
    return 1


def M_vv_4_0_0():
    def func(k):
        return (1 / 3) / k**2

    return func


def N_vv_4_0_0(theta, phi):
    return 1


def M_vv_4_2_0():
    def func(k):
        return (2 / 15) / k**2

    return func


def N_vv_4_2_0(theta, phi):
    return (15 / 4) * np.cos(2 * phi) + 5 / 4


def M_vv_5_0_0():
    def func(k):
        return 4

    return func


def N_vv_5_0_0(theta, phi):
    return 1


def M_vv_5_2_0():
    def func(k):
        return 8 / 5

    return func


def N_vv_5_2_0(theta, phi):
    return (15 / 4) * np.cos(2 * phi) + 5 / 4


def M_vv_6_0_0():
    def func(k):
        return 2

    return func


def N_vv_6_0_0(theta, phi):
    return 1


def M_vv_6_2_0():
    def func(k):
        return 4 / 5

    return func


def N_vv_6_2_0(theta, phi):
    return (15 / 4) * np.cos(2 * phi) + 5 / 4


def M_vv_7_0_0():
    def func(k):
        return (2 / 3) / k**2

    return func


def N_vv_7_0_0(theta, phi):
    return 1


def M_vv_7_2_0():
    def func(k):
        return (4 / 15) / k**2

    return func


def N_vv_7_2_0(theta, phi):
    return (15 / 4) * np.cos(2 * phi) + 5 / 4


def M_vv_8_0_0():
    def func(k):
        return (2 / 5) / k**2

    return func


def N_vv_8_0_0(theta, phi):
    return 1


def M_vv_8_2_0():
    def func(k):
        return (8 / 35) / k**2

    return func


def N_vv_8_2_0(theta, phi):
    return (15 / 4) * np.cos(2 * phi) + 5 / 4


def M_vv_8_4_0():
    def func(k):
        return (16 / 315) / k**2

    return func


def N_vv_8_4_0(theta, phi):
    return (45 / 16) * np.cos(2 * phi) + (315 / 64) * np.cos(4 * phi) + 81 / 64


def M_vv_9_0_0():
    def func(k):
        return -(2 / 3) / k**2

    return func


def N_vv_9_0_0(theta, phi):
    return 1


def M_vv_9_2_0():
    def func(k):
        return -(4 / 15) / k**2

    return func


def N_vv_9_2_0(theta, phi):
    return (15 / 4) * np.cos(2 * phi) + 5 / 4


def M_vv_10_0_0():
    def func(k):
        return -(2 / 5) / k**2

    return func


def N_vv_10_0_0(theta, phi):
    return 1


def M_vv_10_2_0():
    def func(k):
        return -(8 / 35) / k**2

    return func


def N_vv_10_2_0(theta, phi):
    return (15 / 4) * np.cos(2 * phi) + 5 / 4


def M_vv_10_4_0():
    def func(k):
        return -(16 / 315) / k**2

    return func


def N_vv_10_4_0(theta, phi):
    return (45 / 16) * np.cos(2 * phi) + (315 / 64) * np.cos(4 * phi) + 81 / 64


def M_vv_11_0_0():
    def func(k):
        return 4 / 3

    return func


def N_vv_11_0_0(theta, phi):
    return 1


def M_vv_11_2_0():
    def func(k):
        return 8 / 15

    return func


def N_vv_11_2_0(theta, phi):
    return (15 / 4) * np.cos(2 * phi) + 5 / 4


def M_vv_12_0_0():
    def func(k):
        return 4 / 5

    return func


def N_vv_12_0_0(theta, phi):
    return 1


def M_vv_12_2_0():
    def func(k):
        return 16 / 35

    return func


def N_vv_12_2_0(theta, phi):
    return (15 / 4) * np.cos(2 * phi) + 5 / 4


def M_vv_12_4_0():
    def func(k):
        return 32 / 315

    return func


def N_vv_12_4_0(theta, phi):
    return (45 / 16) * np.cos(2 * phi) + (315 / 64) * np.cos(4 * phi) + 81 / 64


def M_vv_13_0_0():
    def func(k):
        return -2 / 3

    return func


def N_vv_13_0_0(theta, phi):
    return 1


def M_vv_13_2_0():
    def func(k):
        return -4 / 15

    return func


def N_vv_13_2_0(theta, phi):
    return (15 / 4) * np.cos(2 * phi) + 5 / 4


def M_vv_14_0_0():
    def func(k):
        return -4 / 3

    return func


def N_vv_14_0_0(theta, phi):
    return 1


def M_vv_14_2_0():
    def func(k):
        return -8 / 15

    return func


def N_vv_14_2_0(theta, phi):
    return (15 / 4) * np.cos(2 * phi) + 5 / 4


def M_vv_15_0_0():
    def func(k):
        return -4 / 3

    return func


def N_vv_15_0_0(theta, phi):
    return 1


def M_vv_15_2_0():
    def func(k):
        return -8 / 15

    return func


def N_vv_15_2_0(theta, phi):
    return (15 / 4) * np.cos(2 * phi) + 5 / 4


def M_vv_16_0_0():
    def func(k):
        return -4 * k**2

    return func


def N_vv_16_0_0(theta, phi):
    return 1


def M_vv_16_2_0():
    def func(k):
        return -8 / 5 * k**2

    return func


def N_vv_16_2_0(theta, phi):
    return (15 / 4) * np.cos(2 * phi) + 5 / 4


def M_vv_17_0_0():
    def func(k):
        return -4 * k**2

    return func


def N_vv_17_0_0(theta, phi):
    return 1


def M_vv_17_2_0():
    def func(k):
        return -8 / 5 * k**2

    return func


def N_vv_17_2_0(theta, phi):
    return (15 / 4) * np.cos(2 * phi) + 5 / 4


def M_vv_18_0_0():
    def func(k):
        return 2 / 3

    return func


def N_vv_18_0_0(theta, phi):
    return 1


def M_vv_18_2_0():
    def func(k):
        return 4 / 15

    return func


def N_vv_18_2_0(theta, phi):
    return (15 / 4) * np.cos(2 * phi) + 5 / 4


def M_vv_19_0_0():
    def func(k):
        return 2 / 3

    return func


def N_vv_19_0_0(theta, phi):
    return 1


def M_vv_19_2_0():
    def func(k):
        return 4 / 15

    return func


def N_vv_19_2_0(theta, phi):
    return (15 / 4) * np.cos(2 * phi) + 5 / 4


def M_vv_20_0_0():
    def func(k):
        return 2 / 3

    return func


def N_vv_20_0_0(theta, phi):
    return 1


def M_vv_20_2_0():
    def func(k):
        return 4 / 15

    return func


def N_vv_20_2_0(theta, phi):
    return (15 / 4) * np.cos(2 * phi) + 5 / 4


def M_vv_21_0_0():
    def func(k):
        return 2 / 3

    return func


def N_vv_21_0_0(theta, phi):
    return 1


def M_vv_21_2_0():
    def func(k):
        return 4 / 15

    return func


def N_vv_21_2_0(theta, phi):
    return (15 / 4) * np.cos(2 * phi) + 5 / 4


def M_vv_22_0_0():
    def func(k):
        return 2 / 3

    return func


def N_vv_22_0_0(theta, phi):
    return 1


def M_vv_22_2_0():
    def func(k):
        return 4 / 15

    return func


def N_vv_22_2_0(theta, phi):
    return (15 / 4) * np.cos(2 * phi) + 5 / 4


def M_vv_23_0_0():
    def func(k):
        return 4 / 3

    return func


def N_vv_23_0_0(theta, phi):
    return 1


def M_vv_23_2_0():
    def func(k):
        return 8 / 15

    return func


def N_vv_23_2_0(theta, phi):
    return (15 / 4) * np.cos(2 * phi) + 5 / 4


def M_vv_24_0_0():
    def func(k):
        return 4 * k**2

    return func


def N_vv_24_0_0(theta, phi):
    return 1


def M_vv_24_2_0():
    def func(k):
        return (8 / 5) * k**2

    return func


def N_vv_24_2_0(theta, phi):
    return (15 / 4) * np.cos(2 * phi) + 5 / 4


def M_vv_25_0_0():
    def func(k):
        return -3 / 5

    return func


def N_vv_25_0_0(theta, phi):
    return 1


def M_vv_25_2_0():
    def func(k):
        return -12 / 35

    return func


def N_vv_25_2_0(theta, phi):
    return (15 / 4) * np.cos(2 * phi) + 5 / 4


def M_vv_25_4_0():
    def func(k):
        return -8 / 105

    return func


def N_vv_25_4_0(theta, phi):
    return (45 / 16) * np.cos(2 * phi) + (315 / 64) * np.cos(4 * phi) + 81 / 64


def M_vv_26_0_0():
    def func(k):
        return -6 / 5

    return func


def N_vv_26_0_0(theta, phi):
    return 1


def M_vv_26_2_0():
    def func(k):
        return -24 / 35

    return func


def N_vv_26_2_0(theta, phi):
    return (15 / 4) * np.cos(2 * phi) + 5 / 4


def M_vv_26_4_0():
    def func(k):
        return -16 / 105

    return func


def N_vv_26_4_0(theta, phi):
    return (45 / 16) * np.cos(2 * phi) + (315 / 64) * np.cos(4 * phi) + 81 / 64


def M_vv_27_0_0():
    def func(k):
        return -12 / 5

    return func


def N_vv_27_0_0(theta, phi):
    return 1


def M_vv_27_2_0():
    def func(k):
        return -48 / 35

    return func


def N_vv_27_2_0(theta, phi):
    return (15 / 4) * np.cos(2 * phi) + 5 / 4


def M_vv_27_4_0():
    def func(k):
        return -32 / 105

    return func


def N_vv_27_4_0(theta, phi):
    return (45 / 16) * np.cos(2 * phi) + (315 / 64) * np.cos(4 * phi) + 81 / 64


def M_vv_28_0_0():
    def func(k):
        return -18 / 5 * k**2

    return func


def N_vv_28_0_0(theta, phi):
    return 1


def M_vv_28_2_0():
    def func(k):
        return -72 / 35 * k**2

    return func


def N_vv_28_2_0(theta, phi):
    return (15 / 4) * np.cos(2 * phi) + 5 / 4


def M_vv_28_4_0():
    def func(k):
        return -16 / 35 * k**2

    return func


def N_vv_28_4_0(theta, phi):
    return (45 / 16) * np.cos(2 * phi) + (315 / 64) * np.cos(4 * phi) + 81 / 64


def M_vv_29_0_0():
    def func(k):
        return -36 / 5 * k**2

    return func


def N_vv_29_0_0(theta, phi):
    return 1


def M_vv_29_2_0():
    def func(k):
        return -144 / 35 * k**2

    return func


def N_vv_29_2_0(theta, phi):
    return (15 / 4) * np.cos(2 * phi) + 5 / 4


def M_vv_29_4_0():
    def func(k):
        return -32 / 35 * k**2

    return func


def N_vv_29_4_0(theta, phi):
    return (45 / 16) * np.cos(2 * phi) + (315 / 64) * np.cos(4 * phi) + 81 / 64


def M_vv_30_0_0():
    def func(k):
        return -3 / 5

    return func


def N_vv_30_0_0(theta, phi):
    return 1


def M_vv_30_2_0():
    def func(k):
        return -12 / 35

    return func


def N_vv_30_2_0(theta, phi):
    return (15 / 4) * np.cos(2 * phi) + 5 / 4


def M_vv_30_4_0():
    def func(k):
        return -8 / 105

    return func


def N_vv_30_4_0(theta, phi):
    return (45 / 16) * np.cos(2 * phi) + (315 / 64) * np.cos(4 * phi) + 81 / 64


def M_vv_31_0_0():
    def func(k):
        return -1

    return func


def N_vv_31_0_0(theta, phi):
    return 1


def M_vv_31_2_0():
    def func(k):
        return -2 / 5

    return func


def N_vv_31_2_0(theta, phi):
    return (15 / 4) * np.cos(2 * phi) + 5 / 4


def M_vv_32_0_0():
    def func(k):
        return (1 / 12) / k**2

    return func


def N_vv_32_0_0(theta, phi):
    return 1


def M_vv_32_2_0():
    def func(k):
        return (1 / 30) / k**2

    return func


def N_vv_32_2_0(theta, phi):
    return (15 / 4) * np.cos(2 * phi) + 5 / 4


def M_vv_33_0_0():
    def func(k):
        return (1 / 10) / k**2

    return func


def N_vv_33_0_0(theta, phi):
    return 1


def M_vv_33_2_0():
    def func(k):
        return (2 / 35) / k**2

    return func


def N_vv_33_2_0(theta, phi):
    return (15 / 4) * np.cos(2 * phi) + 5 / 4


def M_vv_33_4_0():
    def func(k):
        return (4 / 315) / k**2

    return func


def N_vv_33_4_0(theta, phi):
    return (45 / 16) * np.cos(2 * phi) + (315 / 64) * np.cos(4 * phi) + 81 / 64


def M_vv_34_0_0():
    def func(k):
        return (1 / 28) / k**2

    return func


def N_vv_34_0_0(theta, phi):
    return 1


def M_vv_34_2_0():
    def func(k):
        return (1 / 42) / k**2

    return func


def N_vv_34_2_0(theta, phi):
    return (15 / 4) * np.cos(2 * phi) + 5 / 4


def M_vv_34_4_0():
    def func(k):
        return (2 / 231) / k**2

    return func


def N_vv_34_4_0(theta, phi):
    return (45 / 16) * np.cos(2 * phi) + (315 / 64) * np.cos(4 * phi) + 81 / 64


def M_vv_34_6_0():
    def func(k):
        return (4 / 3003) / k**2

    return func


def N_vv_34_6_0(theta, phi):
    return (
        (1365 / 512) * np.cos(2 * phi)
        + (819 / 256) * np.cos(4 * phi)
        + (3003 / 512) * np.cos(6 * phi)
        + 325 / 256
    )


def M_vv_35_0_0():
    def func(k):
        return k**2

    return func


def N_vv_35_0_0(theta, phi):
    return 1


def M_vv_35_2_0():
    def func(k):
        return (2 / 5) * k**2

    return func


def N_vv_35_2_0(theta, phi):
    return (15 / 4) * np.cos(2 * phi) + 5 / 4


def M_vv_36_0_0():
    def func(k):
        return 2 * k**2

    return func


def N_vv_36_0_0(theta, phi):
    return 1


def M_vv_36_2_0():
    def func(k):
        return (4 / 5) * k**2

    return func


def N_vv_36_2_0(theta, phi):
    return (15 / 4) * np.cos(2 * phi) + 5 / 4


def M_vv_37_0_0():
    def func(k):
        return 2 * k**2

    return func


def N_vv_37_0_0(theta, phi):
    return 1


def M_vv_37_2_0():
    def func(k):
        return (4 / 5) * k**2

    return func


def N_vv_37_2_0(theta, phi):
    return (15 / 4) * np.cos(2 * phi) + 5 / 4


def M_vv_38_0_0():
    def func(k):
        return 2 * k**2

    return func


def N_vv_38_0_0(theta, phi):
    return 1


def M_vv_38_2_0():
    def func(k):
        return (4 / 5) * k**2

    return func


def N_vv_38_2_0(theta, phi):
    return (15 / 4) * np.cos(2 * phi) + 5 / 4


def M_vv_39_0_0():
    def func(k):
        return 6 * k**4

    return func


def N_vv_39_0_0(theta, phi):
    return 1


def M_vv_39_2_0():
    def func(k):
        return (12 / 5) * k**4

    return func


def N_vv_39_2_0(theta, phi):
    return (15 / 4) * np.cos(2 * phi) + 5 / 4


def M_vv_40_0_0():
    def func(k):
        return (1 / 2) * k**2

    return func


def N_vv_40_0_0(theta, phi):
    return 1


def M_vv_40_2_0():
    def func(k):
        return (1 / 5) * k**2

    return func


def N_vv_40_2_0(theta, phi):
    return (15 / 4) * np.cos(2 * phi) + 5 / 4


def M_vv_41_0_0():
    def func(k):
        return (1 / 2) * k**2

    return func


def N_vv_41_0_0(theta, phi):
    return 1


def M_vv_41_2_0():
    def func(k):
        return (1 / 5) * k**2

    return func


def N_vv_41_2_0(theta, phi):
    return (15 / 4) * np.cos(2 * phi) + 5 / 4


def M_vv_42_0_0():
    def func(k):
        return k**2

    return func


def N_vv_42_0_0(theta, phi):
    return 1


def M_vv_42_2_0():
    def func(k):
        return (2 / 5) * k**2

    return func


def N_vv_42_2_0(theta, phi):
    return (15 / 4) * np.cos(2 * phi) + 5 / 4


def M_vv_43_0_0():
    def func(k):
        return 2 * k**2

    return func


def N_vv_43_0_0(theta, phi):
    return 1


def M_vv_43_2_0():
    def func(k):
        return (4 / 5) * k**2

    return func


def N_vv_43_2_0(theta, phi):
    return (15 / 4) * np.cos(2 * phi) + 5 / 4


def M_vv_44_0_0():
    def func(k):
        return -2 / 3

    return func


def N_vv_44_0_0(theta, phi):
    return 1


def M_vv_44_2_0():
    def func(k):
        return -4 / 15

    return func


def N_vv_44_2_0(theta, phi):
    return (15 / 4) * np.cos(2 * phi) + 5 / 4


def M_vv_45_0_0():
    def func(k):
        return -2 / 5

    return func


def N_vv_45_0_0(theta, phi):
    return 1


def M_vv_45_2_0():
    def func(k):
        return -8 / 35

    return func


def N_vv_45_2_0(theta, phi):
    return (15 / 4) * np.cos(2 * phi) + 5 / 4


def M_vv_45_4_0():
    def func(k):
        return -16 / 315

    return func


def N_vv_45_4_0(theta, phi):
    return (45 / 16) * np.cos(2 * phi) + (315 / 64) * np.cos(4 * phi) + 81 / 64


def M_vv_46_0_0():
    def func(k):
        return -4 / 3 * k**2

    return func


def N_vv_46_0_0(theta, phi):
    return 1


def M_vv_46_2_0():
    def func(k):
        return -8 / 15 * k**2

    return func


def N_vv_46_2_0(theta, phi):
    return (15 / 4) * np.cos(2 * phi) + 5 / 4


def M_vv_47_0_0():
    def func(k):
        return -4 / 5 * k**2

    return func


def N_vv_47_0_0(theta, phi):
    return 1


def M_vv_47_2_0():
    def func(k):
        return -16 / 35 * k**2

    return func


def N_vv_47_2_0(theta, phi):
    return (15 / 4) * np.cos(2 * phi) + 5 / 4


def M_vv_47_4_0():
    def func(k):
        return -32 / 315 * k**2

    return func


def N_vv_47_4_0(theta, phi):
    return (45 / 16) * np.cos(2 * phi) + (315 / 64) * np.cos(4 * phi) + 81 / 64


def M_vv_48_0_0():
    def func(k):
        return -2 / 3

    return func


def N_vv_48_0_0(theta, phi):
    return 1


def M_vv_48_2_0():
    def func(k):
        return -4 / 15

    return func


def N_vv_48_2_0(theta, phi):
    return (15 / 4) * np.cos(2 * phi) + 5 / 4


def M_vv_49_0_0():
    def func(k):
        return -2 / 3

    return func


def N_vv_49_0_0(theta, phi):
    return 1


def M_vv_49_2_0():
    def func(k):
        return -4 / 15

    return func


def N_vv_49_2_0(theta, phi):
    return (15 / 4) * np.cos(2 * phi) + 5 / 4


def M_vv_50_0_0():
    def func(k):
        return -2 / 5

    return func


def N_vv_50_0_0(theta, phi):
    return 1


def M_vv_50_2_0():
    def func(k):
        return -8 / 35

    return func


def N_vv_50_2_0(theta, phi):
    return (15 / 4) * np.cos(2 * phi) + 5 / 4


def M_vv_50_4_0():
    def func(k):
        return -16 / 315

    return func


def N_vv_50_4_0(theta, phi):
    return (45 / 16) * np.cos(2 * phi) + (315 / 64) * np.cos(4 * phi) + 81 / 64


def M_vv_51_0_0():
    def func(k):
        return -2 / 5

    return func


def N_vv_51_0_0(theta, phi):
    return 1


def M_vv_51_2_0():
    def func(k):
        return -8 / 35

    return func


def N_vv_51_2_0(theta, phi):
    return (15 / 4) * np.cos(2 * phi) + 5 / 4


def M_vv_51_4_0():
    def func(k):
        return -16 / 315

    return func


def N_vv_51_4_0(theta, phi):
    return (45 / 16) * np.cos(2 * phi) + (315 / 64) * np.cos(4 * phi) + 81 / 64


def M_vv_52_0_0():
    def func(k):
        return 1 / 3

    return func


def N_vv_52_0_0(theta, phi):
    return 1


def M_vv_52_2_0():
    def func(k):
        return 2 / 15

    return func


def N_vv_52_2_0(theta, phi):
    return (15 / 4) * np.cos(2 * phi) + 5 / 4


def M_vv_53_0_0():
    def func(k):
        return 1 / 3

    return func


def N_vv_53_0_0(theta, phi):
    return 1


def M_vv_53_2_0():
    def func(k):
        return 2 / 15

    return func


def N_vv_53_2_0(theta, phi):
    return (15 / 4) * np.cos(2 * phi) + 5 / 4


def M_vv_54_0_0():
    def func(k):
        return 1 / 5

    return func


def N_vv_54_0_0(theta, phi):
    return 1


def M_vv_54_2_0():
    def func(k):
        return 4 / 35

    return func


def N_vv_54_2_0(theta, phi):
    return (15 / 4) * np.cos(2 * phi) + 5 / 4


def M_vv_54_4_0():
    def func(k):
        return 8 / 315

    return func


def N_vv_54_4_0(theta, phi):
    return (45 / 16) * np.cos(2 * phi) + (315 / 64) * np.cos(4 * phi) + 81 / 64


def M_vv_55_0_0():
    def func(k):
        return 1 / 5

    return func


def N_vv_55_0_0(theta, phi):
    return 1


def M_vv_55_2_0():
    def func(k):
        return 4 / 35

    return func


def N_vv_55_2_0(theta, phi):
    return (15 / 4) * np.cos(2 * phi) + 5 / 4


def M_vv_55_4_0():
    def func(k):
        return 8 / 315

    return func


def N_vv_55_4_0(theta, phi):
    return (45 / 16) * np.cos(2 * phi) + (315 / 64) * np.cos(4 * phi) + 81 / 64


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
        4,
        4,
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
        3,
        3,
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
        2,
        2,
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
    "gg_9_2": 1,
    "gg_10_0": 1,
    "gg_10_1": 0,
    "gg_10_2": 1,
    "gg_11_0": 1,
    "gg_11_1": 0,
    "gg_11_2": 1,
    "gg_12_0": 1,
    "gg_12_1": 0,
    "gg_12_2": 1,
    "gg_13_0": 1,
    "gg_13_1": 0,
    "gg_13_2": 1,
    "gg_14_0": 1,
    "gg_14_1": 0,
    "gg_14_2": 1,
    "gg_15_0": 1,
    "gg_15_1": 0,
    "gg_15_2": 1,
    "gg_16_0": 1,
    "gg_16_1": 0,
    "gg_16_2": 1,
    "gg_17_0": 1,
    "gg_17_1": 0,
    "gg_17_2": 1,
    "gg_18_0": 1,
    "gg_18_1": 0,
    "gg_18_2": 1,
    "gg_19_0": 1,
    "gg_19_1": 0,
    "gg_19_2": 1,
    "gg_20_0": 1,
    "gg_20_1": 0,
    "gg_20_2": 1,
    "gg_20_3": 0,
    "gg_20_4": 1,
    "gg_21_0": 1,
    "gg_21_1": 0,
    "gg_21_2": 1,
    "gg_22_0": 1,
    "gg_22_1": 0,
    "gg_22_2": 1,
    "gg_22_3": 0,
    "gg_22_4": 1,
    "gg_23_0": 1,
    "gg_23_1": 0,
    "gg_23_2": 1,
    "gg_24_0": 1,
    "gg_24_1": 0,
    "gg_24_2": 1,
    "gg_25_0": 1,
    "gg_25_1": 0,
    "gg_25_2": 1,
    "gg_26_0": 1,
    "gg_26_1": 0,
    "gg_26_2": 1,
    "gg_27_0": 1,
    "gg_27_1": 0,
    "gg_27_2": 1,
    "gg_28_0": 1,
    "gg_28_1": 0,
    "gg_28_2": 1,
    "gg_29_0": 1,
    "gg_29_1": 0,
    "gg_29_2": 1,
    "gg_30_0": 1,
    "gg_30_1": 0,
    "gg_30_2": 1,
    "gg_31_0": 1,
    "gg_31_1": 0,
    "gg_31_2": 1,
    "gg_32_0": 1,
    "gg_32_1": 0,
    "gg_32_2": 1,
    "gg_33_0": 1,
    "gg_33_1": 0,
    "gg_33_2": 1,
    "gg_34_0": 1,
    "gg_34_1": 0,
    "gg_34_2": 1,
    "gg_34_3": 0,
    "gg_34_4": 1,
    "gg_35_0": 1,
    "gg_35_1": 0,
    "gg_35_2": 1,
    "gg_35_3": 0,
    "gg_35_4": 1,
    "gg_36_0": 1,
    "gg_36_1": 0,
    "gg_36_2": 1,
    "gg_36_3": 0,
    "gg_36_4": 1,
    "gg_37_0": 1,
    "gg_37_1": 0,
    "gg_37_2": 1,
    "gg_37_3": 0,
    "gg_37_4": 1,
    "gg_38_0": 1,
    "gg_38_1": 0,
    "gg_38_2": 1,
    "gg_38_3": 0,
    "gg_38_4": 1,
    "gg_39_0": 1,
    "gg_39_1": 0,
    "gg_39_2": 1,
    "gg_40_0": 1,
    "gg_40_1": 0,
    "gg_40_2": 1,
    "gg_40_3": 0,
    "gg_40_4": 1,
    "gg_41_0": 1,
    "gg_41_1": 0,
    "gg_41_2": 1,
    "gg_41_3": 0,
    "gg_41_4": 1,
    "gg_42_0": 1,
    "gg_42_1": 0,
    "gg_42_2": 1,
    "gg_42_3": 0,
    "gg_42_4": 1,
    "gg_43_0": 1,
    "gg_43_1": 0,
    "gg_43_2": 1,
    "gg_43_3": 0,
    "gg_43_4": 1,
    "gg_44_0": 1,
    "gg_44_1": 0,
    "gg_44_2": 1,
    "gg_44_3": 0,
    "gg_44_4": 1,
    "gg_45_0": 1,
    "gg_45_1": 0,
    "gg_45_2": 1,
    "gg_45_3": 0,
    "gg_45_4": 1,
    "gg_46_0": 1,
    "gg_46_1": 0,
    "gg_46_2": 1,
    "gg_46_3": 0,
    "gg_46_4": 1,
    "gg_47_0": 1,
    "gg_47_1": 0,
    "gg_47_2": 1,
    "gg_47_3": 0,
    "gg_47_4": 1,
    "gg_48_0": 1,
    "gg_48_1": 0,
    "gg_48_2": 1,
    "gg_48_3": 0,
    "gg_48_4": 1,
    "gg_49_0": 1,
    "gg_49_1": 0,
    "gg_49_2": 1,
    "gg_49_3": 0,
    "gg_49_4": 1,
    "gg_50_0": 1,
    "gg_50_1": 0,
    "gg_50_2": 1,
    "gg_50_3": 0,
    "gg_50_4": 1,
    "gg_51_0": 1,
    "gg_51_1": 0,
    "gg_51_2": 1,
    "gg_51_3": 0,
    "gg_51_4": 1,
    "gg_52_0": 1,
    "gg_52_1": 0,
    "gg_52_2": 1,
    "gg_52_3": 0,
    "gg_52_4": 1,
    "gg_53_0": 1,
    "gg_53_1": 0,
    "gg_53_2": 1,
    "gg_53_3": 0,
    "gg_53_4": 1,
    "gg_54_0": 1,
    "gg_54_1": 0,
    "gg_54_2": 1,
    "gg_54_3": 0,
    "gg_54_4": 1,
    "gg_54_5": 0,
    "gg_54_6": 1,
    "gg_55_0": 1,
    "gg_55_1": 0,
    "gg_55_2": 1,
    "gg_55_3": 0,
    "gg_55_4": 1,
    "gg_56_0": 1,
    "gg_56_1": 0,
    "gg_56_2": 1,
    "gg_56_3": 0,
    "gg_56_4": 1,
    "gg_56_5": 0,
    "gg_56_6": 1,
    "gg_57_0": 1,
    "gg_57_1": 0,
    "gg_57_2": 1,
    "gg_57_3": 0,
    "gg_57_4": 1,
    "gg_58_0": 1,
    "gg_58_1": 0,
    "gg_58_2": 1,
    "gg_58_3": 0,
    "gg_58_4": 1,
    "gg_59_0": 1,
    "gg_59_1": 0,
    "gg_59_2": 1,
    "gg_59_3": 0,
    "gg_59_4": 1,
    "gg_60_0": 1,
    "gg_60_1": 0,
    "gg_60_2": 1,
    "gg_60_3": 0,
    "gg_60_4": 1,
    "gg_61_0": 1,
    "gg_61_1": 0,
    "gg_61_2": 1,
    "gg_61_3": 0,
    "gg_61_4": 1,
    "gg_62_0": 1,
    "gg_62_1": 0,
    "gg_62_2": 1,
    "gg_62_3": 0,
    "gg_62_4": 1,
    "gg_63_0": 1,
    "gg_63_1": 0,
    "gg_63_2": 1,
    "gg_63_3": 0,
    "gg_63_4": 1,
    "gg_64_0": 1,
    "gg_64_1": 0,
    "gg_64_2": 1,
    "gg_64_3": 0,
    "gg_64_4": 1,
    "gg_65_0": 1,
    "gg_65_1": 0,
    "gg_65_2": 1,
    "gg_65_3": 0,
    "gg_65_4": 1,
    "gg_66_0": 1,
    "gg_66_1": 0,
    "gg_66_2": 1,
    "gg_66_3": 0,
    "gg_66_4": 1,
    "gg_67_0": 1,
    "gg_67_1": 0,
    "gg_67_2": 1,
    "gg_67_3": 0,
    "gg_67_4": 1,
    "gg_68_0": 1,
    "gg_68_1": 0,
    "gg_68_2": 1,
    "gg_68_3": 0,
    "gg_68_4": 1,
    "gg_69_0": 1,
    "gg_69_1": 0,
    "gg_69_2": 1,
    "gg_69_3": 0,
    "gg_69_4": 1,
    "gg_70_0": 1,
    "gg_70_1": 0,
    "gg_70_2": 1,
    "gg_70_3": 0,
    "gg_70_4": 1,
    "gg_71_0": 1,
    "gg_71_1": 0,
    "gg_71_2": 1,
    "gg_71_3": 0,
    "gg_71_4": 1,
    "gg_72_0": 1,
    "gg_72_1": 0,
    "gg_72_2": 1,
    "gg_72_3": 0,
    "gg_72_4": 1,
    "gg_73_0": 1,
    "gg_73_1": 0,
    "gg_73_2": 1,
    "gg_73_3": 0,
    "gg_73_4": 1,
    "gg_74_0": 1,
    "gg_74_1": 0,
    "gg_74_2": 1,
    "gg_74_3": 0,
    "gg_74_4": 1,
    "gg_75_0": 1,
    "gg_75_1": 0,
    "gg_75_2": 1,
    "gg_75_3": 0,
    "gg_75_4": 1,
    "gg_76_0": 1,
    "gg_76_1": 0,
    "gg_76_2": 1,
    "gg_76_3": 0,
    "gg_76_4": 1,
    "gg_76_5": 0,
    "gg_76_6": 1,
    "gg_77_0": 1,
    "gg_77_1": 0,
    "gg_77_2": 1,
    "gg_77_3": 0,
    "gg_77_4": 1,
    "gg_78_0": 1,
    "gg_78_1": 0,
    "gg_78_2": 1,
    "gg_78_3": 0,
    "gg_78_4": 1,
    "gg_78_5": 0,
    "gg_78_6": 1,
    "gg_79_0": 1,
    "gg_79_1": 0,
    "gg_79_2": 1,
    "gg_79_3": 0,
    "gg_79_4": 1,
    "gg_80_0": 1,
    "gg_80_1": 0,
    "gg_80_2": 1,
    "gg_80_3": 0,
    "gg_80_4": 1,
    "gg_80_5": 0,
    "gg_80_6": 1,
    "gg_81_0": 1,
    "gg_81_1": 0,
    "gg_81_2": 1,
    "gg_81_3": 0,
    "gg_81_4": 1,
    "gg_82_0": 1,
    "gg_82_1": 0,
    "gg_82_2": 1,
    "gg_82_3": 0,
    "gg_82_4": 1,
    "gg_83_0": 1,
    "gg_83_1": 0,
    "gg_83_2": 1,
    "gg_83_3": 0,
    "gg_83_4": 1,
    "gg_84_0": 1,
    "gg_84_1": 0,
    "gg_84_2": 1,
    "gg_84_3": 0,
    "gg_84_4": 1,
    "gg_85_0": 1,
    "gg_85_1": 0,
    "gg_85_2": 1,
    "gg_85_3": 0,
    "gg_85_4": 1,
    "gg_86_0": 1,
    "gg_86_1": 0,
    "gg_86_2": 1,
    "gg_86_3": 0,
    "gg_86_4": 1,
    "gg_87_0": 1,
    "gg_87_1": 0,
    "gg_87_2": 1,
    "gg_87_3": 0,
    "gg_87_4": 1,
    "gg_88_0": 1,
    "gg_88_1": 0,
    "gg_88_2": 1,
    "gg_88_3": 0,
    "gg_88_4": 1,
    "gg_89_0": 1,
    "gg_89_1": 0,
    "gg_89_2": 1,
    "gg_89_3": 0,
    "gg_89_4": 1,
    "gg_90_0": 1,
    "gg_90_1": 0,
    "gg_90_2": 1,
    "gg_90_3": 0,
    "gg_90_4": 1,
    "gg_91_0": 1,
    "gg_91_1": 0,
    "gg_91_2": 1,
    "gg_91_3": 0,
    "gg_91_4": 1,
    "gg_92_0": 1,
    "gg_92_1": 0,
    "gg_92_2": 1,
    "gg_92_3": 0,
    "gg_92_4": 1,
    "gg_93_0": 1,
    "gg_93_1": 0,
    "gg_93_2": 1,
    "gg_93_3": 0,
    "gg_93_4": 1,
    "gg_93_5": 0,
    "gg_93_6": 1,
    "gg_94_0": 1,
    "gg_94_1": 0,
    "gg_94_2": 1,
    "gg_94_3": 0,
    "gg_94_4": 1,
    "gg_94_5": 0,
    "gg_94_6": 1,
    "gg_95_0": 1,
    "gg_95_1": 0,
    "gg_95_2": 1,
    "gg_95_3": 0,
    "gg_95_4": 1,
    "gg_95_5": 0,
    "gg_95_6": 1,
    "gg_96_0": 1,
    "gg_96_1": 0,
    "gg_96_2": 1,
    "gg_96_3": 0,
    "gg_96_4": 1,
    "gg_96_5": 0,
    "gg_96_6": 1,
    "gg_97_0": 1,
    "gg_97_1": 0,
    "gg_97_2": 1,
    "gg_97_3": 0,
    "gg_97_4": 1,
    "gg_97_5": 0,
    "gg_97_6": 1,
    "gg_98_0": 1,
    "gg_98_1": 0,
    "gg_98_2": 1,
    "gg_98_3": 0,
    "gg_98_4": 1,
    "gg_98_5": 0,
    "gg_98_6": 1,
    "gg_99_0": 1,
    "gg_99_1": 0,
    "gg_99_2": 1,
    "gg_99_3": 0,
    "gg_99_4": 1,
    "gg_100_0": 1,
    "gg_100_1": 0,
    "gg_100_2": 1,
    "gg_100_3": 0,
    "gg_100_4": 1,
    "gg_101_0": 1,
    "gg_101_1": 0,
    "gg_101_2": 1,
    "gg_101_3": 0,
    "gg_101_4": 1,
    "gg_101_5": 0,
    "gg_101_6": 1,
    "gg_102_0": 1,
    "gg_102_1": 0,
    "gg_102_2": 1,
    "gg_102_3": 0,
    "gg_102_4": 1,
    "gg_102_5": 0,
    "gg_102_6": 1,
    "gg_102_7": 0,
    "gg_102_8": 1,
    "gg_103_0": 1,
    "gg_103_1": 0,
    "gg_103_2": 1,
    "gg_103_3": 0,
    "gg_103_4": 1,
    "gg_104_0": 1,
    "gg_104_1": 0,
    "gg_104_2": 1,
    "gg_104_3": 0,
    "gg_104_4": 1,
    "gg_105_0": 1,
    "gg_105_1": 0,
    "gg_105_2": 1,
    "gg_105_3": 0,
    "gg_105_4": 1,
    "gg_106_0": 1,
    "gg_106_1": 0,
    "gg_106_2": 1,
    "gg_106_3": 0,
    "gg_106_4": 1,
    "gg_107_0": 1,
    "gg_107_1": 0,
    "gg_107_2": 1,
    "gg_107_3": 0,
    "gg_107_4": 1,
    "gg_108_0": 1,
    "gg_108_1": 0,
    "gg_108_2": 1,
    "gg_108_3": 0,
    "gg_108_4": 1,
    "gg_109_0": 1,
    "gg_109_1": 0,
    "gg_109_2": 1,
    "gg_109_3": 0,
    "gg_109_4": 1,
    "gg_110_0": 1,
    "gg_110_1": 0,
    "gg_110_2": 1,
    "gg_110_3": 0,
    "gg_110_4": 1,
    "gg_111_0": 1,
    "gg_111_1": 0,
    "gg_111_2": 1,
    "gg_111_3": 0,
    "gg_111_4": 1,
    "gg_112_0": 1,
    "gg_112_1": 0,
    "gg_112_2": 1,
    "gg_112_3": 0,
    "gg_112_4": 1,
    "gg_113_0": 1,
    "gg_113_1": 0,
    "gg_113_2": 1,
    "gg_113_3": 0,
    "gg_113_4": 1,
    "gg_113_5": 0,
    "gg_113_6": 1,
    "gg_114_0": 1,
    "gg_114_1": 0,
    "gg_114_2": 1,
    "gg_114_3": 0,
    "gg_114_4": 1,
    "gg_115_0": 1,
    "gg_115_1": 0,
    "gg_115_2": 1,
    "gg_115_3": 0,
    "gg_115_4": 1,
    "gg_115_5": 0,
    "gg_115_6": 1,
    "gg_116_0": 1,
    "gg_116_1": 0,
    "gg_116_2": 1,
    "gg_116_3": 0,
    "gg_116_4": 1,
    "gg_117_0": 1,
    "gg_117_1": 0,
    "gg_117_2": 1,
    "gg_117_3": 0,
    "gg_117_4": 1,
    "gg_118_0": 1,
    "gg_118_1": 0,
    "gg_118_2": 1,
    "gg_118_3": 0,
    "gg_118_4": 1,
    "gg_118_5": 0,
    "gg_118_6": 1,
    "gg_119_0": 1,
    "gg_119_1": 0,
    "gg_119_2": 1,
    "gg_119_3": 0,
    "gg_119_4": 1,
    "gg_119_5": 0,
    "gg_119_6": 1,
    "gg_120_0": 1,
    "gg_120_1": 0,
    "gg_120_2": 1,
    "gg_120_3": 0,
    "gg_120_4": 1,
    "gg_121_0": 1,
    "gg_121_1": 0,
    "gg_121_2": 1,
    "gg_121_3": 0,
    "gg_121_4": 1,
    "gg_122_0": 1,
    "gg_122_1": 0,
    "gg_122_2": 1,
    "gg_122_3": 0,
    "gg_122_4": 1,
    "gg_122_5": 0,
    "gg_122_6": 1,
    "gg_123_0": 1,
    "gg_123_1": 0,
    "gg_123_2": 1,
    "gg_123_3": 0,
    "gg_123_4": 1,
    "gg_123_5": 0,
    "gg_123_6": 1,
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
    "gv_39_0": 0,
    "gv_39_1": 1,
    "gv_39_2": 0,
    "gv_39_3": 1,
    "gv_40_0": 0,
    "gv_40_1": 1,
    "gv_40_2": 0,
    "gv_40_3": 1,
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
    "gv_45_4": 0,
    "gv_45_5": 1,
    "gv_46_0": 0,
    "gv_46_1": 1,
    "gv_46_2": 0,
    "gv_46_3": 1,
    "gv_47_0": 0,
    "gv_47_1": 1,
    "gv_47_2": 0,
    "gv_47_3": 1,
    "gv_47_4": 0,
    "gv_47_5": 1,
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
    "gv_59_1": 1,
    "gv_59_2": 0,
    "gv_59_3": 1,
    "gv_60_0": 0,
    "gv_60_1": 1,
    "gv_60_2": 0,
    "gv_60_3": 1,
    "gv_61_0": 0,
    "gv_61_1": 1,
    "gv_61_2": 0,
    "gv_61_3": 1,
    "gv_62_0": 0,
    "gv_62_1": 1,
    "gv_62_2": 0,
    "gv_62_3": 1,
    "gv_63_0": 0,
    "gv_63_1": 1,
    "gv_63_2": 0,
    "gv_63_3": 1,
    "gv_64_0": 0,
    "gv_64_1": 1,
    "gv_64_2": 0,
    "gv_64_3": 1,
    "gv_65_0": 0,
    "gv_65_1": 1,
    "gv_65_2": 0,
    "gv_65_3": 1,
    "gv_66_0": 0,
    "gv_66_1": 1,
    "gv_66_2": 0,
    "gv_66_3": 1,
    "gv_67_0": 0,
    "gv_67_1": 1,
    "gv_67_2": 0,
    "gv_67_3": 1,
    "gv_67_4": 0,
    "gv_67_5": 1,
    "gv_68_0": 0,
    "gv_68_1": 1,
    "gv_68_2": 0,
    "gv_68_3": 1,
    "gv_69_0": 0,
    "gv_69_1": 1,
    "gv_69_2": 0,
    "gv_69_3": 1,
    "gv_69_4": 0,
    "gv_69_5": 1,
    "gv_70_0": 0,
    "gv_70_1": 1,
    "gv_70_2": 0,
    "gv_70_3": 1,
    "gv_71_0": 0,
    "gv_71_1": 1,
    "gv_71_2": 0,
    "gv_71_3": 1,
    "gv_71_4": 0,
    "gv_71_5": 1,
    "gv_72_0": 0,
    "gv_72_1": 1,
    "gv_72_2": 0,
    "gv_72_3": 1,
    "gv_73_0": 0,
    "gv_73_1": 1,
    "gv_73_2": 0,
    "gv_73_3": 1,
    "gv_74_0": 0,
    "gv_74_1": 1,
    "gv_74_2": 0,
    "gv_74_3": 1,
    "gv_75_0": 0,
    "gv_75_1": 1,
    "gv_75_2": 0,
    "gv_75_3": 1,
    "gv_76_0": 0,
    "gv_76_1": 1,
    "gv_76_2": 0,
    "gv_76_3": 1,
    "gv_77_0": 0,
    "gv_77_1": 1,
    "gv_77_2": 0,
    "gv_77_3": 1,
    "gv_78_0": 0,
    "gv_78_1": 1,
    "gv_78_2": 0,
    "gv_78_3": 1,
    "gv_79_0": 0,
    "gv_79_1": 1,
    "gv_79_2": 0,
    "gv_79_3": 1,
    "gv_80_0": 0,
    "gv_80_1": 1,
    "gv_80_2": 0,
    "gv_80_3": 1,
    "gv_81_0": 0,
    "gv_81_1": 1,
    "gv_81_2": 0,
    "gv_81_3": 1,
    "gv_82_0": 0,
    "gv_82_1": 1,
    "gv_82_2": 0,
    "gv_82_3": 1,
    "gv_83_0": 0,
    "gv_83_1": 1,
    "gv_83_2": 0,
    "gv_83_3": 1,
    "gv_84_0": 0,
    "gv_84_1": 1,
    "gv_84_2": 0,
    "gv_84_3": 1,
    "gv_84_4": 0,
    "gv_84_5": 1,
    "gv_85_0": 0,
    "gv_85_1": 1,
    "gv_85_2": 0,
    "gv_85_3": 1,
    "gv_85_4": 0,
    "gv_85_5": 1,
    "gv_86_0": 0,
    "gv_86_1": 1,
    "gv_86_2": 0,
    "gv_86_3": 1,
    "gv_86_4": 0,
    "gv_86_5": 1,
    "gv_87_0": 0,
    "gv_87_1": 1,
    "gv_87_2": 0,
    "gv_87_3": 1,
    "gv_87_4": 0,
    "gv_87_5": 1,
    "gv_88_0": 0,
    "gv_88_1": 1,
    "gv_88_2": 0,
    "gv_88_3": 1,
    "gv_88_4": 0,
    "gv_88_5": 1,
    "gv_89_0": 0,
    "gv_89_1": 1,
    "gv_89_2": 0,
    "gv_89_3": 1,
    "gv_89_4": 0,
    "gv_89_5": 1,
    "gv_90_0": 0,
    "gv_90_1": 1,
    "gv_90_2": 0,
    "gv_90_3": 1,
    "gv_91_0": 0,
    "gv_91_1": 1,
    "gv_91_2": 0,
    "gv_91_3": 1,
    "gv_92_0": 0,
    "gv_92_1": 1,
    "gv_92_2": 0,
    "gv_92_3": 1,
    "gv_92_4": 0,
    "gv_92_5": 1,
    "gv_93_0": 0,
    "gv_93_1": 1,
    "gv_93_2": 0,
    "gv_93_3": 1,
    "gv_93_4": 0,
    "gv_93_5": 1,
    "gv_93_6": 0,
    "gv_93_7": 1,
    "gv_94_0": 0,
    "gv_94_1": 1,
    "gv_94_2": 0,
    "gv_94_3": 1,
    "gv_95_0": 0,
    "gv_95_1": 1,
    "gv_95_2": 0,
    "gv_95_3": 1,
    "gv_96_0": 0,
    "gv_96_1": 1,
    "gv_96_2": 0,
    "gv_96_3": 1,
    "gv_97_0": 0,
    "gv_97_1": 1,
    "gv_97_2": 0,
    "gv_97_3": 1,
    "gv_98_0": 0,
    "gv_98_1": 1,
    "gv_98_2": 0,
    "gv_98_3": 1,
    "gv_99_0": 0,
    "gv_99_1": 1,
    "gv_99_2": 0,
    "gv_99_3": 1,
    "gv_100_0": 0,
    "gv_100_1": 1,
    "gv_100_2": 0,
    "gv_100_3": 1,
    "gv_101_0": 0,
    "gv_101_1": 1,
    "gv_101_2": 0,
    "gv_101_3": 1,
    "gv_102_0": 0,
    "gv_102_1": 1,
    "gv_102_2": 0,
    "gv_102_3": 1,
    "gv_103_0": 0,
    "gv_103_1": 1,
    "gv_103_2": 0,
    "gv_103_3": 1,
    "gv_104_0": 0,
    "gv_104_1": 1,
    "gv_104_2": 0,
    "gv_104_3": 1,
    "gv_104_4": 0,
    "gv_104_5": 1,
    "gv_105_0": 0,
    "gv_105_1": 1,
    "gv_105_2": 0,
    "gv_105_3": 1,
    "gv_106_0": 0,
    "gv_106_1": 1,
    "gv_106_2": 0,
    "gv_106_3": 1,
    "gv_106_4": 0,
    "gv_106_5": 1,
    "gv_107_0": 0,
    "gv_107_1": 1,
    "gv_107_2": 0,
    "gv_107_3": 1,
    "gv_108_0": 0,
    "gv_108_1": 1,
    "gv_108_2": 0,
    "gv_108_3": 1,
    "gv_109_0": 0,
    "gv_109_1": 1,
    "gv_109_2": 0,
    "gv_109_3": 1,
    "gv_109_4": 0,
    "gv_109_5": 1,
    "gv_110_0": 0,
    "gv_110_1": 1,
    "gv_110_2": 0,
    "gv_110_3": 1,
    "gv_110_4": 0,
    "gv_110_5": 1,
    "gv_111_0": 0,
    "gv_111_1": 1,
    "gv_111_2": 0,
    "gv_111_3": 1,
    "gv_112_0": 0,
    "gv_112_1": 1,
    "gv_112_2": 0,
    "gv_112_3": 1,
    "gv_113_0": 0,
    "gv_113_1": 1,
    "gv_113_2": 0,
    "gv_113_3": 1,
    "gv_113_4": 0,
    "gv_113_5": 1,
    "gv_114_0": 0,
    "gv_114_1": 1,
    "gv_114_2": 0,
    "gv_114_3": 1,
    "gv_114_4": 0,
    "gv_114_5": 1,
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
    "vv_8_2": 1,
    "vv_8_3": 0,
    "vv_8_4": 1,
    "vv_9_0": 1,
    "vv_9_1": 0,
    "vv_9_2": 1,
    "vv_10_0": 1,
    "vv_10_1": 0,
    "vv_10_2": 1,
    "vv_10_3": 0,
    "vv_10_4": 1,
    "vv_11_0": 1,
    "vv_11_1": 0,
    "vv_11_2": 1,
    "vv_12_0": 1,
    "vv_12_1": 0,
    "vv_12_2": 1,
    "vv_12_3": 0,
    "vv_12_4": 1,
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
    "vv_25_3": 0,
    "vv_25_4": 1,
    "vv_26_0": 1,
    "vv_26_1": 0,
    "vv_26_2": 1,
    "vv_26_3": 0,
    "vv_26_4": 1,
    "vv_27_0": 1,
    "vv_27_1": 0,
    "vv_27_2": 1,
    "vv_27_3": 0,
    "vv_27_4": 1,
    "vv_28_0": 1,
    "vv_28_1": 0,
    "vv_28_2": 1,
    "vv_28_3": 0,
    "vv_28_4": 1,
    "vv_29_0": 1,
    "vv_29_1": 0,
    "vv_29_2": 1,
    "vv_29_3": 0,
    "vv_29_4": 1,
    "vv_30_0": 1,
    "vv_30_1": 0,
    "vv_30_2": 1,
    "vv_30_3": 0,
    "vv_30_4": 1,
    "vv_31_0": 1,
    "vv_31_1": 0,
    "vv_31_2": 1,
    "vv_32_0": 1,
    "vv_32_1": 0,
    "vv_32_2": 1,
    "vv_33_0": 1,
    "vv_33_1": 0,
    "vv_33_2": 1,
    "vv_33_3": 0,
    "vv_33_4": 1,
    "vv_34_0": 1,
    "vv_34_1": 0,
    "vv_34_2": 1,
    "vv_34_3": 0,
    "vv_34_4": 1,
    "vv_34_5": 0,
    "vv_34_6": 1,
    "vv_35_0": 1,
    "vv_35_1": 0,
    "vv_35_2": 1,
    "vv_36_0": 1,
    "vv_36_1": 0,
    "vv_36_2": 1,
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
    "vv_45_3": 0,
    "vv_45_4": 1,
    "vv_46_0": 1,
    "vv_46_1": 0,
    "vv_46_2": 1,
    "vv_47_0": 1,
    "vv_47_1": 0,
    "vv_47_2": 1,
    "vv_47_3": 0,
    "vv_47_4": 1,
    "vv_48_0": 1,
    "vv_48_1": 0,
    "vv_48_2": 1,
    "vv_49_0": 1,
    "vv_49_1": 0,
    "vv_49_2": 1,
    "vv_50_0": 1,
    "vv_50_1": 0,
    "vv_50_2": 1,
    "vv_50_3": 0,
    "vv_50_4": 1,
    "vv_51_0": 1,
    "vv_51_1": 0,
    "vv_51_2": 1,
    "vv_51_3": 0,
    "vv_51_4": 1,
    "vv_52_0": 1,
    "vv_52_1": 0,
    "vv_52_2": 1,
    "vv_53_0": 1,
    "vv_53_1": 0,
    "vv_53_2": 1,
    "vv_54_0": 1,
    "vv_54_1": 0,
    "vv_54_2": 1,
    "vv_54_3": 0,
    "vv_54_4": 1,
    "vv_55_0": 1,
    "vv_55_1": 0,
    "vv_55_2": 1,
    "vv_55_3": 0,
    "vv_55_4": 1,
}
multi_index_model = False
regularize_M_terms = None
