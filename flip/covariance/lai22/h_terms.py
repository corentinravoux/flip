import numpy as np

# Density-Density


def H_gg_l0_p0_q0(theta, phi):
    return 1


def H_gg_l2_p0_q0(theta, phi):
    return 0


def H_gg_l0_p0_q1(theta, phi):
    return 1 / 3


def H_gg_l2_p0_q1(theta, phi):
    return (1 / 2) * np.cos(2 * phi + theta) + 1 / 6


def H_gg_l4_p0_q1(theta, phi):
    return 0


def H_gg_l0_p0_q2(theta, phi):
    return 1 / 5


def H_gg_l2_p0_q2(theta, phi):
    return (3 / 7) * np.cos(2 * phi + theta) + 1 / 7


def H_gg_l4_p0_q2(theta, phi):
    return (
        (1 / 14) * np.cos(2 * phi + theta)
        + (1 / 8) * np.cos(4 * phi + 2 * theta)
        + 9 / 280
    )


def H_gg_l6_p0_q2(theta, phi):
    return 0


def H_gg_l0_p0_q3(theta, phi):
    return 1 / 7


def H_gg_l2_p0_q3(theta, phi):
    return (5 / 14) * np.cos(2 * phi + theta) + 5 / 42


def H_gg_l4_p0_q3(theta, phi):
    return (
        (15 / 154) * np.cos(2 * phi + theta)
        + (15 / 88) * np.cos(4 * phi + 2 * theta)
        + 27 / 616
    )


def H_gg_l6_p0_q3(theta, phi):
    return (
        (5 / 352) * np.cos(2 * phi + theta)
        + (3 / 176) * np.cos(4 * phi + 2 * theta)
        + (1 / 32) * np.cos(6 * phi + 3 * theta)
        + 25 / 3696
    )


def H_gg_l8_p0_q3(theta, phi):
    return 0


def H_gg_l0_p0_q4(theta, phi):
    return 1 / 9


def H_gg_l2_p0_q4(theta, phi):
    return (10 / 33) * np.cos(2 * phi + theta) + 10 / 99


def H_gg_l4_p0_q4(theta, phi):
    return (
        (15 / 143) * np.cos(2 * phi + theta)
        + (105 / 572) * np.cos(4 * phi + 2 * theta)
        + 27 / 572
    )


def H_gg_l6_p0_q4(theta, phi):
    return (
        (7 / 264) * np.cos(2 * phi + theta)
        + (7 / 220) * np.cos(4 * phi + 2 * theta)
        + (7 / 120) * np.cos(6 * phi + 3 * theta)
        + 5 / 396
    )


def H_gg_l8_p0_q4(theta, phi):
    return (
        (7 / 2288) * np.cos(2 * phi + theta)
        + (7 / 2080) * np.cos(4 * phi + 2 * theta)
        + (1 / 240) * np.cos(6 * phi + 3 * theta)
        + (1 / 128) * np.cos(8 * phi + 4 * theta)
        + 245 / 164736
    )


def H_gg_l10_p0_q4(theta, phi):
    return 0


def H_gg_l0_p1_q0(theta, phi):
    return 1 / 3


def H_gg_l2_p1_q0(theta, phi):
    return (1 / 2) * np.cos(2 * phi - theta) + 1 / 6


def H_gg_l4_p1_q0(theta, phi):
    return 0


def H_gg_l0_p1_q1(theta, phi):
    return (1 / 15) * np.cos(2 * theta) + 2 / 15


def H_gg_l2_p1_q1(theta, phi):
    return (
        (1 / 21) * np.cos(2 * theta)
        + (3 / 14) * np.cos(2 * phi - theta)
        + (3 / 14) * np.cos(2 * phi + theta)
        + 2 / 21
    )


def H_gg_l4_p1_q1(theta, phi):
    return (
        (1 / 8) * np.cos(4 * phi)
        + (3 / 280) * np.cos(2 * theta)
        + (1 / 28) * np.cos(2 * phi - theta)
        + (1 / 28) * np.cos(2 * phi + theta)
        + 3 / 140
    )


def H_gg_l6_p1_q1(theta, phi):
    return 0


def H_gg_l0_p1_q2(theta, phi):
    return (2 / 35) * np.cos(2 * theta) + 3 / 35


def H_gg_l2_p1_q2(theta, phi):
    return (
        (1 / 21) * np.cos(2 * theta)
        + (1 / 7) * np.cos(2 * phi - theta)
        + (4 / 21) * np.cos(2 * phi + theta)
        + (1 / 42) * np.cos(2 * phi + 3 * theta)
        + 1 / 14
    )


def H_gg_l4_p1_q2(theta, phi):
    return (
        (5 / 44) * np.cos(4 * phi)
        + (27 / 1540) * np.cos(2 * theta)
        + (3 / 77) * np.cos(2 * phi - theta)
        + (4 / 77) * np.cos(2 * phi + theta)
        + (1 / 154) * np.cos(2 * phi + 3 * theta)
        + (5 / 88) * np.cos(4 * phi + 2 * theta)
        + 81 / 3080
    )


def H_gg_l6_p1_q2(theta, phi):
    return (
        (1 / 88) * np.cos(4 * phi)
        + (5 / 1848) * np.cos(2 * theta)
        + (1 / 176) * np.cos(2 * phi - theta)
        + (1 / 132) * np.cos(2 * phi + theta)
        + (1 / 1056) * np.cos(2 * phi + 3 * theta)
        + (1 / 176) * np.cos(4 * phi + 2 * theta)
        + (1 / 32) * np.cos(6 * phi + theta)
        + 5 / 1232
    )


def H_gg_l8_p1_q2(theta, phi):
    return 0


def H_gg_l0_p1_q3(theta, phi):
    return (1 / 21) * np.cos(2 * theta) + 4 / 63


def H_gg_l2_p1_q3(theta, phi):
    return (
        (10 / 231) * np.cos(2 * theta)
        + (25 / 231) * np.cos(2 * phi - theta)
        + (25 / 154) * np.cos(2 * phi + theta)
        + (5 / 154) * np.cos(2 * phi + 3 * theta)
        + 40 / 693
    )


def H_gg_l4_p1_q3(theta, phi):
    return (
        (225 / 2288) * np.cos(4 * phi)
        + (81 / 4004) * np.cos(2 * theta)
        + (75 / 2002) * np.cos(2 * phi - theta)
        + (225 / 4004) * np.cos(2 * phi + theta)
        + (45 / 4004) * np.cos(2 * phi + 3 * theta)
        + (45 / 572) * np.cos(4 * phi + 2 * theta)
        + (15 / 2288) * np.cos(4 * phi + 4 * theta)
        + 27 / 1001
    )


def H_gg_l6_p1_q3(theta, phi):
    return (
        (3 / 176) * np.cos(4 * phi)
        + (5 / 924) * np.cos(2 * theta)
        + (5 / 528) * np.cos(2 * phi - theta)
        + (5 / 352) * np.cos(2 * phi + theta)
        + (1 / 352) * np.cos(2 * phi + 3 * theta)
        + (3 / 220) * np.cos(4 * phi + 2 * theta)
        + (1 / 880) * np.cos(4 * phi + 4 * theta)
        + (7 / 160) * np.cos(6 * phi + theta)
        + (7 / 480) * np.cos(6 * phi + 3 * theta)
        + 5 / 693
    )


def H_gg_l8_p1_q3(theta, phi):
    return (
        (3 / 1664) * np.cos(4 * phi)
        + (35 / 54912) * np.cos(2 * theta)
        + (5 / 4576) * np.cos(2 * phi - theta)
        + (15 / 9152) * np.cos(2 * phi + theta)
        + (3 / 9152) * np.cos(2 * phi + 3 * theta)
        + (3 / 2080) * np.cos(4 * phi + 2 * theta)
        + (1 / 8320) * np.cos(4 * phi + 4 * theta)
        + (1 / 320) * np.cos(6 * phi + theta)
        + (1 / 960) * np.cos(6 * phi + 3 * theta)
        + (1 / 128) * np.cos(8 * phi + 2 * theta)
        + 35 / 41184
    )


def H_gg_l10_p1_q3(theta, phi):
    return 0


def H_gg_l0_p1_q4(theta, phi):
    return (4 / 99) * np.cos(2 * theta) + 5 / 99


def H_gg_l2_p1_q4(theta, phi):
    return (
        (50 / 1287) * np.cos(2 * theta)
        + (25 / 286) * np.cos(2 * phi - theta)
        + (20 / 143) * np.cos(2 * phi + theta)
        + (5 / 143) * np.cos(2 * phi + 3 * theta)
        + 125 / 2574
    )


def H_gg_l4_p1_q4(theta, phi):
    return (
        (49 / 572) * np.cos(4 * phi)
        + (3 / 143) * np.cos(2 * theta)
        + (5 / 143) * np.cos(2 * phi - theta)
        + (8 / 143) * np.cos(2 * phi + theta)
        + (2 / 143) * np.cos(2 * phi + 3 * theta)
        + (49 / 572) * np.cos(4 * phi + 2 * theta)
        + (7 / 572) * np.cos(4 * phi + 4 * theta)
        + 15 / 572
    )


def H_gg_l6_p1_q4(theta, phi):
    return (
        (147 / 7480) * np.cos(4 * phi)
        + (25 / 3366) * np.cos(2 * theta)
        + (35 / 2992) * np.cos(2 * phi - theta)
        + (7 / 374) * np.cos(2 * phi + theta)
        + (7 / 1496) * np.cos(2 * phi + 3 * theta)
        + (147 / 7480) * np.cos(4 * phi + 2 * theta)
        + (21 / 7480) * np.cos(4 * phi + 4 * theta)
        + (49 / 1020) * np.cos(6 * phi + theta)
        + (7 / 255) * np.cos(6 * phi + 3 * theta)
        + (7 / 4080) * np.cos(6 * phi + 5 * theta)
        + 125 / 13464
    )


def H_gg_l8_p1_q4(theta, phi):
    return (
        (147 / 39520) * np.cos(4 * phi)
        + (1225 / 782496) * np.cos(2 * theta)
        + (105 / 43472) * np.cos(2 * phi - theta)
        + (21 / 5434) * np.cos(2 * phi + theta)
        + (21 / 21736) * np.cos(2 * phi + 3 * theta)
        + (147 / 39520) * np.cos(4 * phi + 2 * theta)
        + (21 / 39520) * np.cos(4 * phi + 4 * theta)
        + (7 / 1140) * np.cos(6 * phi + theta)
        + (1 / 285) * np.cos(6 * phi + 3 * theta)
        + (1 / 4560) * np.cos(6 * phi + 5 * theta)
        + (9 / 608) * np.cos(8 * phi + 2 * theta)
        + (9 / 2432) * np.cos(8 * phi + 4 * theta)
        + 6125 / 3129984
    )


def H_gg_l10_p1_q4(theta, phi):
    return (
        (7 / 20672) * np.cos(4 * phi)
        + (441 / 2956096) * np.cos(2 * theta)
        + (245 / 1074944) * np.cos(2 * phi - theta)
        + (49 / 134368) * np.cos(2 * phi + theta)
        + (49 / 537472) * np.cos(2 * phi + 3 * theta)
        + (7 / 20672) * np.cos(4 * phi + 2 * theta)
        + (1 / 20672) * np.cos(4 * phi + 4 * theta)
        + (21 / 41344) * np.cos(6 * phi + theta)
        + (3 / 10336) * np.cos(6 * phi + 3 * theta)
        + (3 / 165376) * np.cos(6 * phi + 5 * theta)
        + (1 / 1216) * np.cos(8 * phi + 2 * theta)
        + (1 / 4864) * np.cos(8 * phi + 4 * theta)
        + (1 / 512) * np.cos(10 * phi + 3 * theta)
        + 2205 / 11824384
    )


def H_gg_l12_p1_q4(theta, phi):
    return 0


def H_gg_l0_p2_q0(theta, phi):
    return 1 / 5


def H_gg_l2_p2_q0(theta, phi):
    return (3 / 7) * np.cos(2 * phi - theta) + 1 / 7


def H_gg_l4_p2_q0(theta, phi):
    return (
        (1 / 14) * np.cos(2 * phi - theta)
        + (1 / 8) * np.cos(4 * phi - 2 * theta)
        + 9 / 280
    )


def H_gg_l6_p2_q0(theta, phi):
    return 0


def H_gg_l0_p2_q1(theta, phi):
    return (2 / 35) * np.cos(2 * theta) + 3 / 35


def H_gg_l2_p2_q1(theta, phi):
    return (
        (1 / 21) * np.cos(2 * theta)
        + (1 / 42) * np.cos(2 * phi - 3 * theta)
        + (4 / 21) * np.cos(2 * phi - theta)
        + (1 / 7) * np.cos(2 * phi + theta)
        + 1 / 14
    )


def H_gg_l4_p2_q1(theta, phi):
    return (
        (5 / 44) * np.cos(4 * phi)
        + (27 / 1540) * np.cos(2 * theta)
        + (1 / 154) * np.cos(2 * phi - 3 * theta)
        + (4 / 77) * np.cos(2 * phi - theta)
        + (3 / 77) * np.cos(2 * phi + theta)
        + (5 / 88) * np.cos(4 * phi - 2 * theta)
        + 81 / 3080
    )


def H_gg_l6_p2_q1(theta, phi):
    return (
        (1 / 88) * np.cos(4 * phi)
        + (5 / 1848) * np.cos(2 * theta)
        + (1 / 1056) * np.cos(2 * phi - 3 * theta)
        + (1 / 132) * np.cos(2 * phi - theta)
        + (1 / 176) * np.cos(2 * phi + theta)
        + (1 / 176) * np.cos(4 * phi - 2 * theta)
        + (1 / 32) * np.cos(6 * phi - theta)
        + 5 / 1232
    )


def H_gg_l8_p2_q1(theta, phi):
    return 0


def H_gg_l0_p2_q2(theta, phi):
    return (16 / 315) * np.cos(2 * theta) + (1 / 315) * np.cos(4 * theta) + 2 / 35


def H_gg_l2_p2_q2(theta, phi):
    return (
        (32 / 693) * np.cos(2 * theta)
        + (2 / 693) * np.cos(4 * theta)
        + (5 / 231) * np.cos(2 * phi - 3 * theta)
        + (10 / 77) * np.cos(2 * phi - theta)
        + (10 / 77) * np.cos(2 * phi + theta)
        + (5 / 231) * np.cos(2 * phi + 3 * theta)
        + 4 / 77
    )


def H_gg_l4_p2_q2(theta, phi):
    return (
        (15 / 143) * np.cos(4 * phi)
        + (108 / 5005) * np.cos(2 * theta)
        + (27 / 20020) * np.cos(4 * theta)
        + (15 / 2002) * np.cos(2 * phi - 3 * theta)
        + (45 / 1001) * np.cos(2 * phi - theta)
        + (45 / 1001) * np.cos(2 * phi + theta)
        + (15 / 2002) * np.cos(2 * phi + 3 * theta)
        + (45 / 1144) * np.cos(4 * phi - 2 * theta)
        + (45 / 1144) * np.cos(4 * phi + 2 * theta)
        + 243 / 10010
    )


def H_gg_l6_p2_q2(theta, phi):
    return (
        (1 / 55) * np.cos(4 * phi)
        + (4 / 693) * np.cos(2 * theta)
        + (1 / 2772) * np.cos(4 * theta)
        + (1 / 528) * np.cos(2 * phi - 3 * theta)
        + (1 / 88) * np.cos(2 * phi - theta)
        + (1 / 88) * np.cos(2 * phi + theta)
        + (1 / 528) * np.cos(2 * phi + 3 * theta)
        + (3 / 440) * np.cos(4 * phi - 2 * theta)
        + (3 / 440) * np.cos(4 * phi + 2 * theta)
        + (7 / 240) * np.cos(6 * phi - theta)
        + (7 / 240) * np.cos(6 * phi + theta)
        + 1 / 154
    )


def H_gg_l8_p2_q2(theta, phi):
    return (
        (1 / 520) * np.cos(4 * phi)
        + (1 / 128) * np.cos(8 * phi)
        + (7 / 10296) * np.cos(2 * theta)
        + (7 / 164736) * np.cos(4 * theta)
        + (1 / 4576) * np.cos(2 * phi - 3 * theta)
        + (3 / 2288) * np.cos(2 * phi - theta)
        + (3 / 2288) * np.cos(2 * phi + theta)
        + (1 / 4576) * np.cos(2 * phi + 3 * theta)
        + (3 / 4160) * np.cos(4 * phi - 2 * theta)
        + (3 / 4160) * np.cos(4 * phi + 2 * theta)
        + (1 / 480) * np.cos(6 * phi - theta)
        + (1 / 480) * np.cos(6 * phi + theta)
        + 7 / 9152
    )


def H_gg_l10_p2_q2(theta, phi):
    return 0


def H_gg_l0_p2_q3(theta, phi):
    return (10 / 231) * np.cos(2 * theta) + (1 / 231) * np.cos(4 * theta) + 10 / 231


def H_gg_l2_p2_q3(theta, phi):
    return (
        (125 / 3003) * np.cos(2 * theta)
        + (25 / 6006) * np.cos(4 * theta)
        + (75 / 4004) * np.cos(2 * phi - 3 * theta)
        + (100 / 1001) * np.cos(2 * phi - theta)
        + (225 / 2002) * np.cos(2 * phi + theta)
        + (30 / 1001) * np.cos(2 * phi + 3 * theta)
        + (5 / 4004) * np.cos(2 * phi + 5 * theta)
        + 125 / 3003
    )


def H_gg_l4_p2_q3(theta, phi):
    return (
        (105 / 1144) * np.cos(4 * phi)
        + (45 / 2002) * np.cos(2 * theta)
        + (9 / 4004) * np.cos(4 * theta)
        + (15 / 2002) * np.cos(2 * phi - 3 * theta)
        + (40 / 1001) * np.cos(2 * phi - theta)
        + (45 / 1001) * np.cos(2 * phi + theta)
        + (12 / 1001) * np.cos(2 * phi + 3 * theta)
        + (1 / 2002) * np.cos(2 * phi + 5 * theta)
        + (35 / 1144) * np.cos(4 * phi - 2 * theta)
        + (63 / 1144) * np.cos(4 * phi + 2 * theta)
        + (7 / 1144) * np.cos(4 * phi + 4 * theta)
        + 45 / 2002
    )


def H_gg_l6_p2_q3(theta, phi):
    return (
        (63 / 2992) * np.cos(4 * phi)
        + (125 / 15708) * np.cos(2 * theta)
        + (25 / 31416) * np.cos(4 * theta)
        + (15 / 5984) * np.cos(2 * phi - 3 * theta)
        + (5 / 374) * np.cos(2 * phi - theta)
        + (45 / 2992) * np.cos(2 * phi + theta)
        + (3 / 748) * np.cos(2 * phi + 3 * theta)
        + (1 / 5984) * np.cos(2 * phi + 5 * theta)
        + (21 / 2992) * np.cos(4 * phi - 2 * theta)
        + (189 / 14960) * np.cos(4 * phi + 2 * theta)
        + (21 / 14960) * np.cos(4 * phi + 4 * theta)
        + (7 / 272) * np.cos(6 * phi - theta)
        + (7 / 170) * np.cos(6 * phi + theta)
        + (7 / 680) * np.cos(6 * phi + 3 * theta)
        + 125 / 15708
    )


def H_gg_l8_p2_q3(theta, phi):
    return (
        (63 / 15808) * np.cos(4 * phi)
        + (27 / 2432) * np.cos(8 * phi)
        + (875 / 521664) * np.cos(2 * theta)
        + (175 / 1043328) * np.cos(4 * theta)
        + (45 / 86944) * np.cos(2 * phi - 3 * theta)
        + (15 / 5434) * np.cos(2 * phi - theta)
        + (135 / 43472) * np.cos(2 * phi + theta)
        + (9 / 10868) * np.cos(2 * phi + 3 * theta)
        + (3 / 86944) * np.cos(2 * phi + 5 * theta)
        + (21 / 15808) * np.cos(4 * phi - 2 * theta)
        + (189 / 79040) * np.cos(4 * phi + 2 * theta)
        + (21 / 79040) * np.cos(4 * phi + 4 * theta)
        + (1 / 304) * np.cos(6 * phi - theta)
        + (1 / 190) * np.cos(6 * phi + theta)
        + (1 / 760) * np.cos(6 * phi + 3 * theta)
        + (9 / 1216) * np.cos(8 * phi + 2 * theta)
        + 875 / 521664
    )


def H_gg_l10_p2_q3(theta, phi):
    return (
        (15 / 41344) * np.cos(4 * phi)
        + (3 / 4864) * np.cos(8 * phi)
        + (945 / 5912192) * np.cos(2 * theta)
        + (189 / 11824384) * np.cos(4 * theta)
        + (105 / 2149888) * np.cos(2 * phi - 3 * theta)
        + (35 / 134368) * np.cos(2 * phi - theta)
        + (315 / 1074944) * np.cos(2 * phi + theta)
        + (21 / 268736) * np.cos(2 * phi + 3 * theta)
        + (7 / 2149888) * np.cos(2 * phi + 5 * theta)
        + (5 / 41344) * np.cos(4 * phi - 2 * theta)
        + (9 / 41344) * np.cos(4 * phi + 2 * theta)
        + (1 / 41344) * np.cos(4 * phi + 4 * theta)
        + (45 / 165376) * np.cos(6 * phi - theta)
        + (9 / 20672) * np.cos(6 * phi + theta)
        + (9 / 82688) * np.cos(6 * phi + 3 * theta)
        + (1 / 2432) * np.cos(8 * phi + 2 * theta)
        + (1 / 512) * np.cos(10 * phi + theta)
        + 945 / 5912192
    )


def H_gg_l12_p2_q3(theta, phi):
    return 0


def H_gg_l0_p2_q4(theta, phi):
    return (16 / 429) * np.cos(2 * theta) + (2 / 429) * np.cos(4 * theta) + 5 / 143


def H_gg_l2_p2_q4(theta, phi):
    return (
        (16 / 429) * np.cos(2 * theta)
        + (2 / 429) * np.cos(4 * theta)
        + (7 / 429) * np.cos(2 * phi - 3 * theta)
        + (35 / 429) * np.cos(2 * phi - theta)
        + (14 / 143) * np.cos(2 * phi + theta)
        + (14 / 429) * np.cos(2 * phi + 3 * theta)
        + (1 / 429) * np.cos(2 * phi + 5 * theta)
        + 5 / 143
    )


def H_gg_l4_p2_q4(theta, phi):
    return (
        (196 / 2431) * np.cos(4 * phi)
        + (54 / 2431) * np.cos(2 * theta)
        + (27 / 9724) * np.cos(4 * theta)
        + (35 / 4862) * np.cos(2 * phi - 3 * theta)
        + (175 / 4862) * np.cos(2 * phi - theta)
        + (105 / 2431) * np.cos(2 * phi + theta)
        + (35 / 2431) * np.cos(2 * phi + 3 * theta)
        + (5 / 4862) * np.cos(2 * phi + 5 * theta)
        + (245 / 9724) * np.cos(4 * phi - 2 * theta)
        + (147 / 2431) * np.cos(4 * phi + 2 * theta)
        + (28 / 2431) * np.cos(4 * phi + 4 * theta)
        + (7 / 19448) * np.cos(4 * phi + 6 * theta)
        + 405 / 19448
    )


def H_gg_l6_p2_q4(theta, phi):
    return (
        (392 / 17765) * np.cos(4 * phi)
        + (100 / 10659) * np.cos(2 * theta)
        + (25 / 21318) * np.cos(4 * theta)
        + (245 / 85272) * np.cos(2 * phi - 3 * theta)
        + (1225 / 85272) * np.cos(2 * phi - theta)
        + (245 / 14212) * np.cos(2 * phi + theta)
        + (245 / 42636) * np.cos(2 * phi + 3 * theta)
        + (35 / 85272) * np.cos(2 * phi + 5 * theta)
        + (49 / 7106) * np.cos(4 * phi - 2 * theta)
        + (294 / 17765) * np.cos(4 * phi + 2 * theta)
        + (56 / 17765) * np.cos(4 * phi + 4 * theta)
        + (7 / 71060) * np.cos(4 * phi + 6 * theta)
        + (147 / 6460) * np.cos(6 * phi - theta)
        + (147 / 3230) * np.cos(6 * phi + theta)
        + (63 / 3230) * np.cos(6 * phi + 3 * theta)
        + (21 / 12920) * np.cos(6 * phi + 5 * theta)
        + 125 / 14212
    )


def H_gg_l8_p2_q4(theta, phi):
    return (
        (7 / 1235) * np.cos(4 * phi)
        + (15 / 1216) * np.cos(8 * phi)
        + (175 / 65208) * np.cos(2 * theta)
        + (175 / 521664) * np.cos(4 * theta)
        + (35 / 43472) * np.cos(2 * phi - 3 * theta)
        + (175 / 43472) * np.cos(2 * phi - theta)
        + (105 / 21736) * np.cos(2 * phi + theta)
        + (35 / 21736) * np.cos(2 * phi + 3 * theta)
        + (5 / 43472) * np.cos(2 * phi + 5 * theta)
        + (7 / 3952) * np.cos(4 * phi - 2 * theta)
        + (21 / 4940) * np.cos(4 * phi + 2 * theta)
        + (1 / 1235) * np.cos(4 * phi + 4 * theta)
        + (1 / 39520) * np.cos(4 * phi + 6 * theta)
        + (3 / 760) * np.cos(6 * phi - theta)
        + (3 / 380) * np.cos(6 * phi + theta)
        + (9 / 2660) * np.cos(6 * phi + 3 * theta)
        + (3 / 10640) * np.cos(6 * phi + 5 * theta)
        + (15 / 1064) * np.cos(8 * phi + 2 * theta)
        + (45 / 17024) * np.cos(8 * phi + 4 * theta)
        + 875 / 347776
    )


def H_gg_l10_p2_q4(theta, phi):
    return (
        (7 / 7429) * np.cos(4 * phi)
        + (35 / 27968) * np.cos(8 * phi)
        + (3969 / 8498776) * np.cos(2 * theta)
        + (3969 / 67990208) * np.cos(4 * theta)
        + (1715 / 12361856) * np.cos(2 * phi - 3 * theta)
        + (8575 / 12361856) * np.cos(2 * phi - theta)
        + (5145 / 6180928) * np.cos(2 * phi + theta)
        + (1715 / 6180928) * np.cos(2 * phi + 3 * theta)
        + (245 / 12361856) * np.cos(2 * phi + 5 * theta)
        + (35 / 118864) * np.cos(4 * phi - 2 * theta)
        + (21 / 29716) * np.cos(4 * phi + 2 * theta)
        + (1 / 7429) * np.cos(4 * phi + 4 * theta)
        + (1 / 237728) * np.cos(4 * phi + 6 * theta)
        + (567 / 950912) * np.cos(6 * phi - theta)
        + (567 / 475456) * np.cos(6 * phi + theta)
        + (243 / 475456) * np.cos(6 * phi + 3 * theta)
        + (81 / 1901824) * np.cos(6 * phi + 5 * theta)
        + (5 / 3496) * np.cos(8 * phi + 2 * theta)
        + (15 / 55936) * np.cos(8 * phi + 4 * theta)
        + (11 / 2944) * np.cos(10 * phi + theta)
        + (11 / 5888) * np.cos(10 * phi + 3 * theta)
        + 59535 / 135980416
    )


def H_gg_l12_p2_q4(theta, phi):
    return (
        (35 / 475456) * np.cos(4 * phi)
        + (1 / 11776) * np.cos(8 * phi)
        + (231 / 6180928) * np.cos(2 * theta)
        + (231 / 49447424) * np.cos(4 * theta)
        + (21 / 1901824) * np.cos(2 * phi - 3 * theta)
        + (105 / 1901824) * np.cos(2 * phi - theta)
        + (63 / 950912) * np.cos(2 * phi + theta)
        + (21 / 950912) * np.cos(2 * phi + 3 * theta)
        + (3 / 1901824) * np.cos(2 * phi + 5 * theta)
        + (175 / 7607296) * np.cos(4 * phi - 2 * theta)
        + (105 / 1901824) * np.cos(4 * phi + 2 * theta)
        + (5 / 475456) * np.cos(4 * phi + 4 * theta)
        + (5 / 15214592) * np.cos(4 * phi + 6 * theta)
        + (5 / 111872) * np.cos(6 * phi - theta)
        + (5 / 55936) * np.cos(6 * phi + theta)
        + (15 / 391552) * np.cos(6 * phi + 3 * theta)
        + (5 / 1566208) * np.cos(6 * phi + 5 * theta)
        + (1 / 10304) * np.cos(8 * phi + 2 * theta)
        + (3 / 164864) * np.cos(8 * phi + 4 * theta)
        + (1 / 5888) * np.cos(10 * phi + theta)
        + (1 / 11776) * np.cos(10 * phi + 3 * theta)
        + (1 / 2048) * np.cos(12 * phi + 2 * theta)
        + 3465 / 98894848
    )


def H_gg_l14_p2_q4(theta, phi):
    return 0


def H_gg_l0_p3_q0(theta, phi):
    return 1 / 7


def H_gg_l2_p3_q0(theta, phi):
    return (5 / 14) * np.cos(2 * phi - theta) + 5 / 42


def H_gg_l4_p3_q0(theta, phi):
    return (
        (15 / 154) * np.cos(2 * phi - theta)
        + (15 / 88) * np.cos(4 * phi - 2 * theta)
        + 27 / 616
    )


def H_gg_l6_p3_q0(theta, phi):
    return (
        (5 / 352) * np.cos(2 * phi - theta)
        + (3 / 176) * np.cos(4 * phi - 2 * theta)
        + (1 / 32) * np.cos(6 * phi - 3 * theta)
        + 25 / 3696
    )


def H_gg_l8_p3_q0(theta, phi):
    return 0


def H_gg_l0_p3_q1(theta, phi):
    return (1 / 21) * np.cos(2 * theta) + 4 / 63


def H_gg_l2_p3_q1(theta, phi):
    return (
        (10 / 231) * np.cos(2 * theta)
        + (5 / 154) * np.cos(2 * phi - 3 * theta)
        + (25 / 154) * np.cos(2 * phi - theta)
        + (25 / 231) * np.cos(2 * phi + theta)
        + 40 / 693
    )


def H_gg_l4_p3_q1(theta, phi):
    return (
        (225 / 2288) * np.cos(4 * phi)
        + (81 / 4004) * np.cos(2 * theta)
        + (45 / 4004) * np.cos(2 * phi - 3 * theta)
        + (225 / 4004) * np.cos(2 * phi - theta)
        + (75 / 2002) * np.cos(2 * phi + theta)
        + (15 / 2288) * np.cos(4 * phi - 4 * theta)
        + (45 / 572) * np.cos(4 * phi - 2 * theta)
        + 27 / 1001
    )


def H_gg_l6_p3_q1(theta, phi):
    return (
        (3 / 176) * np.cos(4 * phi)
        + (5 / 924) * np.cos(2 * theta)
        + (1 / 352) * np.cos(2 * phi - 3 * theta)
        + (5 / 352) * np.cos(2 * phi - theta)
        + (5 / 528) * np.cos(2 * phi + theta)
        + (1 / 880) * np.cos(4 * phi - 4 * theta)
        + (3 / 220) * np.cos(4 * phi - 2 * theta)
        + (7 / 480) * np.cos(6 * phi - 3 * theta)
        + (7 / 160) * np.cos(6 * phi - theta)
        + 5 / 693
    )


def H_gg_l8_p3_q1(theta, phi):
    return (
        (3 / 1664) * np.cos(4 * phi)
        + (35 / 54912) * np.cos(2 * theta)
        + (3 / 9152) * np.cos(2 * phi - 3 * theta)
        + (15 / 9152) * np.cos(2 * phi - theta)
        + (5 / 4576) * np.cos(2 * phi + theta)
        + (1 / 8320) * np.cos(4 * phi - 4 * theta)
        + (3 / 2080) * np.cos(4 * phi - 2 * theta)
        + (1 / 960) * np.cos(6 * phi - 3 * theta)
        + (1 / 320) * np.cos(6 * phi - theta)
        + (1 / 128) * np.cos(8 * phi - 2 * theta)
        + 35 / 41184
    )


def H_gg_l10_p3_q1(theta, phi):
    return 0


def H_gg_l0_p3_q2(theta, phi):
    return (10 / 231) * np.cos(2 * theta) + (1 / 231) * np.cos(4 * theta) + 10 / 231


def H_gg_l2_p3_q2(theta, phi):
    return (
        (125 / 3003) * np.cos(2 * theta)
        + (25 / 6006) * np.cos(4 * theta)
        + (5 / 4004) * np.cos(2 * phi - 5 * theta)
        + (30 / 1001) * np.cos(2 * phi - 3 * theta)
        + (225 / 2002) * np.cos(2 * phi - theta)
        + (100 / 1001) * np.cos(2 * phi + theta)
        + (75 / 4004) * np.cos(2 * phi + 3 * theta)
        + 125 / 3003
    )


def H_gg_l4_p3_q2(theta, phi):
    return (
        (105 / 1144) * np.cos(4 * phi)
        + (45 / 2002) * np.cos(2 * theta)
        + (9 / 4004) * np.cos(4 * theta)
        + (1 / 2002) * np.cos(2 * phi - 5 * theta)
        + (12 / 1001) * np.cos(2 * phi - 3 * theta)
        + (45 / 1001) * np.cos(2 * phi - theta)
        + (40 / 1001) * np.cos(2 * phi + theta)
        + (15 / 2002) * np.cos(2 * phi + 3 * theta)
        + (7 / 1144) * np.cos(4 * phi - 4 * theta)
        + (63 / 1144) * np.cos(4 * phi - 2 * theta)
        + (35 / 1144) * np.cos(4 * phi + 2 * theta)
        + 45 / 2002
    )


def H_gg_l6_p3_q2(theta, phi):
    return (
        (63 / 2992) * np.cos(4 * phi)
        + (125 / 15708) * np.cos(2 * theta)
        + (25 / 31416) * np.cos(4 * theta)
        + (1 / 5984) * np.cos(2 * phi - 5 * theta)
        + (3 / 748) * np.cos(2 * phi - 3 * theta)
        + (45 / 2992) * np.cos(2 * phi - theta)
        + (5 / 374) * np.cos(2 * phi + theta)
        + (15 / 5984) * np.cos(2 * phi + 3 * theta)
        + (21 / 14960) * np.cos(4 * phi - 4 * theta)
        + (189 / 14960) * np.cos(4 * phi - 2 * theta)
        + (21 / 2992) * np.cos(4 * phi + 2 * theta)
        + (7 / 680) * np.cos(6 * phi - 3 * theta)
        + (7 / 170) * np.cos(6 * phi - theta)
        + (7 / 272) * np.cos(6 * phi + theta)
        + 125 / 15708
    )


def H_gg_l8_p3_q2(theta, phi):
    return (
        (63 / 15808) * np.cos(4 * phi)
        + (27 / 2432) * np.cos(8 * phi)
        + (875 / 521664) * np.cos(2 * theta)
        + (175 / 1043328) * np.cos(4 * theta)
        + (3 / 86944) * np.cos(2 * phi - 5 * theta)
        + (9 / 10868) * np.cos(2 * phi - 3 * theta)
        + (135 / 43472) * np.cos(2 * phi - theta)
        + (15 / 5434) * np.cos(2 * phi + theta)
        + (45 / 86944) * np.cos(2 * phi + 3 * theta)
        + (21 / 79040) * np.cos(4 * phi - 4 * theta)
        + (189 / 79040) * np.cos(4 * phi - 2 * theta)
        + (21 / 15808) * np.cos(4 * phi + 2 * theta)
        + (1 / 760) * np.cos(6 * phi - 3 * theta)
        + (1 / 190) * np.cos(6 * phi - theta)
        + (1 / 304) * np.cos(6 * phi + theta)
        + (9 / 1216) * np.cos(8 * phi - 2 * theta)
        + 875 / 521664
    )


def H_gg_l10_p3_q2(theta, phi):
    return (
        (15 / 41344) * np.cos(4 * phi)
        + (3 / 4864) * np.cos(8 * phi)
        + (945 / 5912192) * np.cos(2 * theta)
        + (189 / 11824384) * np.cos(4 * theta)
        + (7 / 2149888) * np.cos(2 * phi - 5 * theta)
        + (21 / 268736) * np.cos(2 * phi - 3 * theta)
        + (315 / 1074944) * np.cos(2 * phi - theta)
        + (35 / 134368) * np.cos(2 * phi + theta)
        + (105 / 2149888) * np.cos(2 * phi + 3 * theta)
        + (1 / 41344) * np.cos(4 * phi - 4 * theta)
        + (9 / 41344) * np.cos(4 * phi - 2 * theta)
        + (5 / 41344) * np.cos(4 * phi + 2 * theta)
        + (9 / 82688) * np.cos(6 * phi - 3 * theta)
        + (9 / 20672) * np.cos(6 * phi - theta)
        + (45 / 165376) * np.cos(6 * phi + theta)
        + (1 / 2432) * np.cos(8 * phi - 2 * theta)
        + (1 / 512) * np.cos(10 * phi - theta)
        + 945 / 5912192
    )


def H_gg_l12_p3_q2(theta, phi):
    return 0


def H_gg_l0_p3_q3(theta, phi):
    return (
        (75 / 2002) * np.cos(2 * theta)
        + (6 / 1001) * np.cos(4 * theta)
        + (1 / 6006) * np.cos(6 * theta)
        + 100 / 3003
    )


def H_gg_l2_p3_q3(theta, phi):
    return (
        (75 / 2002) * np.cos(2 * theta)
        + (6 / 1001) * np.cos(4 * theta)
        + (1 / 6006) * np.cos(6 * theta)
        + (1 / 572) * np.cos(2 * phi - 5 * theta)
        + (15 / 572) * np.cos(2 * phi - 3 * theta)
        + (25 / 286) * np.cos(2 * phi - theta)
        + (25 / 286) * np.cos(2 * phi + theta)
        + (15 / 572) * np.cos(2 * phi + 3 * theta)
        + (1 / 572) * np.cos(2 * phi + 5 * theta)
        + 100 / 3003
    )


def H_gg_l4_p3_q3(theta, phi):
    return (
        (1575 / 19448) * np.cos(4 * phi)
        + (6075 / 272272) * np.cos(2 * theta)
        + (243 / 68068) * np.cos(4 * theta)
        + (27 / 272272) * np.cos(6 * theta)
        + (15 / 19448) * np.cos(2 * phi - 5 * theta)
        + (225 / 19448) * np.cos(2 * phi - 3 * theta)
        + (375 / 9724) * np.cos(2 * phi - theta)
        + (375 / 9724) * np.cos(2 * phi + theta)
        + (225 / 19448) * np.cos(2 * phi + 3 * theta)
        + (15 / 19448) * np.cos(2 * phi + 5 * theta)
        + (105 / 19448) * np.cos(4 * phi - 4 * theta)
        + (105 / 2431) * np.cos(4 * phi - 2 * theta)
        + (105 / 2431) * np.cos(4 * phi + 2 * theta)
        + (105 / 19448) * np.cos(4 * phi + 4 * theta)
        + 675 / 34034
    )


def H_gg_l6_p3_q3(theta, phi):
    return (
        (315 / 14212) * np.cos(4 * phi)
        + (1875 / 198968) * np.cos(2 * theta)
        + (75 / 49742) * np.cos(4 * theta)
        + (25 / 596904) * np.cos(6 * theta)
        + (35 / 113696) * np.cos(2 * phi - 5 * theta)
        + (525 / 113696) * np.cos(2 * phi - 3 * theta)
        + (875 / 56848) * np.cos(2 * phi - theta)
        + (875 / 56848) * np.cos(2 * phi + theta)
        + (525 / 113696) * np.cos(2 * phi + 3 * theta)
        + (35 / 113696) * np.cos(2 * phi + 5 * theta)
        + (21 / 14212) * np.cos(4 * phi - 4 * theta)
        + (42 / 3553) * np.cos(4 * phi - 2 * theta)
        + (42 / 3553) * np.cos(4 * phi + 2 * theta)
        + (21 / 14212) * np.cos(4 * phi + 4 * theta)
        + (21 / 2584) * np.cos(6 * phi - 3 * theta)
        + (189 / 5168) * np.cos(6 * phi - theta)
        + (189 / 5168) * np.cos(6 * phi + theta)
        + (21 / 2584) * np.cos(6 * phi + 3 * theta)
        + 625 / 74613
    )


def H_gg_l8_p3_q3(theta, phi):
    return (
        (45 / 7904) * np.cos(4 * phi)
        + (135 / 8512) * np.cos(8 * phi)
        + (1875 / 695552) * np.cos(2 * theta)
        + (75 / 173888) * np.cos(4 * theta)
        + (25 / 2086656) * np.cos(6 * theta)
        + (15 / 173888) * np.cos(2 * phi - 5 * theta)
        + (225 / 173888) * np.cos(2 * phi - 3 * theta)
        + (375 / 86944) * np.cos(2 * phi - theta)
        + (375 / 86944) * np.cos(2 * phi + theta)
        + (225 / 173888) * np.cos(2 * phi + 3 * theta)
        + (15 / 173888) * np.cos(2 * phi + 5 * theta)
        + (3 / 7904) * np.cos(4 * phi - 4 * theta)
        + (3 / 988) * np.cos(4 * phi - 2 * theta)
        + (3 / 988) * np.cos(4 * phi + 2 * theta)
        + (3 / 7904) * np.cos(4 * phi + 4 * theta)
        + (3 / 2128) * np.cos(6 * phi - 3 * theta)
        + (27 / 4256) * np.cos(6 * phi - theta)
        + (27 / 4256) * np.cos(6 * phi + theta)
        + (3 / 2128) * np.cos(6 * phi + 3 * theta)
        + (225 / 34048) * np.cos(8 * phi - 2 * theta)
        + (225 / 34048) * np.cos(8 * phi + 2 * theta)
        + 625 / 260832
    )


def H_gg_l10_p3_q3(theta, phi):
    return (
        (225 / 237728) * np.cos(4 * phi)
        + (45 / 27968) * np.cos(8 * phi)
        + (127575 / 271960832) * np.cos(2 * theta)
        + (5103 / 67990208) * np.cos(4 * theta)
        + (567 / 271960832) * np.cos(6 * theta)
        + (735 / 49447424) * np.cos(2 * phi - 5 * theta)
        + (11025 / 49447424) * np.cos(2 * phi - 3 * theta)
        + (18375 / 24723712) * np.cos(2 * phi - theta)
        + (18375 / 24723712) * np.cos(2 * phi + theta)
        + (11025 / 49447424) * np.cos(2 * phi + 3 * theta)
        + (735 / 49447424) * np.cos(2 * phi + 5 * theta)
        + (15 / 237728) * np.cos(4 * phi - 4 * theta)
        + (15 / 29716) * np.cos(4 * phi - 2 * theta)
        + (15 / 29716) * np.cos(4 * phi + 2 * theta)
        + (15 / 237728) * np.cos(4 * phi + 4 * theta)
        + (405 / 1901824) * np.cos(6 * phi - 3 * theta)
        + (3645 / 3803648) * np.cos(6 * phi - theta)
        + (3645 / 3803648) * np.cos(6 * phi + theta)
        + (405 / 1901824) * np.cos(6 * phi + 3 * theta)
        + (75 / 111872) * np.cos(8 * phi - 2 * theta)
        + (75 / 111872) * np.cos(8 * phi + 2 * theta)
        + (33 / 11776) * np.cos(10 * phi - theta)
        + (33 / 11776) * np.cos(10 * phi + theta)
        + 14175 / 33995104
    )


def H_gg_l12_p3_q3(theta, phi):
    return (
        (1125 / 15214592) * np.cos(4 * phi)
        + (9 / 82432) * np.cos(8 * phi)
        + (1 / 2048) * np.cos(12 * phi)
        + (7425 / 197789696) * np.cos(2 * theta)
        + (297 / 49447424) * np.cos(4 * theta)
        + (33 / 197789696) * np.cos(6 * theta)
        + (9 / 7607296) * np.cos(2 * phi - 5 * theta)
        + (135 / 7607296) * np.cos(2 * phi - 3 * theta)
        + (225 / 3803648) * np.cos(2 * phi - theta)
        + (225 / 3803648) * np.cos(2 * phi + theta)
        + (135 / 7607296) * np.cos(2 * phi + 3 * theta)
        + (9 / 7607296) * np.cos(2 * phi + 5 * theta)
        + (75 / 15214592) * np.cos(4 * phi - 4 * theta)
        + (75 / 1901824) * np.cos(4 * phi - 2 * theta)
        + (75 / 1901824) * np.cos(4 * phi + 2 * theta)
        + (75 / 15214592) * np.cos(4 * phi + 4 * theta)
        + (25 / 1566208) * np.cos(6 * phi - 3 * theta)
        + (225 / 3132416) * np.cos(6 * phi - theta)
        + (225 / 3132416) * np.cos(6 * phi + theta)
        + (25 / 1566208) * np.cos(6 * phi + 3 * theta)
        + (15 / 329728) * np.cos(8 * phi - 2 * theta)
        + (15 / 329728) * np.cos(8 * phi + 2 * theta)
        + (3 / 23552) * np.cos(10 * phi - theta)
        + (3 / 23552) * np.cos(10 * phi + theta)
        + 825 / 24723712
    )


def H_gg_l14_p3_q3(theta, phi):
    return 0


def H_gg_l0_p3_q4(theta, phi):
    return (
        (14 / 429) * np.cos(2 * theta)
        + (14 / 2145) * np.cos(4 * theta)
        + (2 / 6435) * np.cos(6 * theta)
        + 35 / 1287
    )


def H_gg_l2_p3_q4(theta, phi):
    return (
        (245 / 7293) * np.cos(2 * theta)
        + (49 / 7293) * np.cos(4 * theta)
        + (7 / 21879) * np.cos(6 * theta)
        + (14 / 7293) * np.cos(2 * phi - 5 * theta)
        + (56 / 2431) * np.cos(2 * phi - 3 * theta)
        + (175 / 2431) * np.cos(2 * phi - theta)
        + (560 / 7293) * np.cos(2 * phi + theta)
        + (70 / 2431) * np.cos(2 * phi + 3 * theta)
        + (8 / 2431) * np.cos(2 * phi + 5 * theta)
        + (1 / 14586) * np.cos(2 * phi + 7 * theta)
        + 1225 / 43758
    )


def H_gg_l4_p3_q4(theta, phi):
    return (
        (6615 / 92378) * np.cos(4 * phi)
        + (3969 / 184756) * np.cos(2 * theta)
        + (3969 / 923780) * np.cos(4 * theta)
        + (189 / 923780) * np.cos(6 * theta)
        + (42 / 46189) * np.cos(2 * phi - 5 * theta)
        + (504 / 46189) * np.cos(2 * phi - 3 * theta)
        + (1575 / 46189) * np.cos(2 * phi - theta)
        + (1680 / 46189) * np.cos(2 * phi + theta)
        + (630 / 46189) * np.cos(2 * phi + 3 * theta)
        + (72 / 46189) * np.cos(2 * phi + 5 * theta)
        + (3 / 92378) * np.cos(2 * phi + 7 * theta)
        + (441 / 92378) * np.cos(4 * phi - 4 * theta)
        + (6615 / 184756) * np.cos(4 * phi - 2 * theta)
        + (2205 / 46189) * np.cos(4 * phi + 2 * theta)
        + (945 / 92378) * np.cos(4 * phi + 4 * theta)
        + (189 / 369512) * np.cos(4 * phi + 6 * theta)
        + 6615 / 369512
    )


def H_gg_l6_p3_q4(theta, phi):
    return (
        (315 / 14212) * np.cos(4 * phi)
        + (875 / 85272) * np.cos(2 * theta)
        + (175 / 85272) * np.cos(4 * theta)
        + (25 / 255816) * np.cos(6 * theta)
        + (35 / 85272) * np.cos(2 * phi - 5 * theta)
        + (35 / 7106) * np.cos(2 * phi - 3 * theta)
        + (875 / 56848) * np.cos(2 * phi - theta)
        + (175 / 10659) * np.cos(2 * phi + theta)
        + (175 / 28424) * np.cos(2 * phi + 3 * theta)
        + (5 / 7106) * np.cos(2 * phi + 5 * theta)
        + (5 / 341088) * np.cos(2 * phi + 7 * theta)
        + (21 / 14212) * np.cos(4 * phi - 4 * theta)
        + (315 / 28424) * np.cos(4 * phi - 2 * theta)
        + (105 / 7106) * np.cos(4 * phi + 2 * theta)
        + (45 / 14212) * np.cos(4 * phi + 4 * theta)
        + (9 / 56848) * np.cos(4 * phi + 6 * theta)
        + (35 / 5168) * np.cos(6 * phi - 3 * theta)
        + (21 / 646) * np.cos(6 * phi - theta)
        + (105 / 2584) * np.cos(6 * phi + theta)
        + (5 / 323) * np.cos(6 * phi + 3 * theta)
        + (15 / 10336) * np.cos(6 * phi + 5 * theta)
        + 4375 / 511632
    )


def H_gg_l8_p3_q4(theta, phi):
    return (
        (315 / 45448) * np.cos(4 * phi)
        + (495 / 27968) * np.cos(8 * phi)
        + (42875 / 11998272) * np.cos(2 * theta)
        + (8575 / 11998272) * np.cos(4 * theta)
        + (1225 / 35994816) * np.cos(6 * theta)
        + (35 / 249964) * np.cos(2 * phi - 5 * theta)
        + (105 / 62491) * np.cos(2 * phi - 3 * theta)
        + (2625 / 499928) * np.cos(2 * phi - theta)
        + (350 / 62491) * np.cos(2 * phi + theta)
        + (525 / 249964) * np.cos(2 * phi + 3 * theta)
        + (15 / 62491) * np.cos(2 * phi + 5 * theta)
        + (5 / 999856) * np.cos(2 * phi + 7 * theta)
        + (21 / 45448) * np.cos(4 * phi - 4 * theta)
        + (315 / 90896) * np.cos(4 * phi - 2 * theta)
        + (105 / 22724) * np.cos(4 * phi + 2 * theta)
        + (45 / 45448) * np.cos(4 * phi + 4 * theta)
        + (9 / 181792) * np.cos(4 * phi + 6 * theta)
        + (5 / 3496) * np.cos(6 * phi - 3 * theta)
        + (3 / 437) * np.cos(6 * phi - theta)
        + (15 / 1748) * np.cos(6 * phi + theta)
        + (10 / 3059) * np.cos(6 * phi + 3 * theta)
        + (15 / 48944) * np.cos(6 * phi + 5 * theta)
        + (165 / 27968) * np.cos(8 * phi - 2 * theta)
        + (2475 / 195776) * np.cos(8 * phi + 2 * theta)
        + (825 / 391552) * np.cos(8 * phi + 4 * theta)
        + 214375 / 71989632
    )


def H_gg_l10_p3_q4(theta, phi):
    return (
        (189 / 118864) * np.cos(4 * phi)
        + (693 / 279680) * np.cos(8 * phi)
        + (583443 / 679902080) * np.cos(2 * theta)
        + (583443 / 3399510400) * np.cos(4 * theta)
        + (27783 / 3399510400) * np.cos(6 * theta)
        + (1029 / 30904640) * np.cos(2 * phi - 5 * theta)
        + (3087 / 7726160) * np.cos(2 * phi - 3 * theta)
        + (15435 / 12361856) * np.cos(2 * phi - theta)
        + (1029 / 772616) * np.cos(2 * phi + theta)
        + (3087 / 6180928) * np.cos(2 * phi + 3 * theta)
        + (441 / 7726160) * np.cos(2 * phi + 5 * theta)
        + (147 / 123618560) * np.cos(2 * phi + 7 * theta)
        + (63 / 594320) * np.cos(4 * phi - 4 * theta)
        + (189 / 237728) * np.cos(4 * phi - 2 * theta)
        + (63 / 59432) * np.cos(4 * phi + 2 * theta)
        + (27 / 118864) * np.cos(4 * phi + 4 * theta)
        + (27 / 2377280) * np.cos(4 * phi + 6 * theta)
        + (567 / 1901824) * np.cos(6 * phi - 3 * theta)
        + (1701 / 1188640) * np.cos(6 * phi - theta)
        + (1701 / 950912) * np.cos(6 * phi + theta)
        + (81 / 118864) * np.cos(6 * phi + 3 * theta)
        + (243 / 3803648) * np.cos(6 * phi + 5 * theta)
        + (231 / 279680) * np.cos(8 * phi - 2 * theta)
        + (99 / 55936) * np.cos(8 * phi + 2 * theta)
        + (33 / 111872) * np.cos(8 * phi + 4 * theta)
        + (231 / 73600) * np.cos(10 * phi - theta)
        + (99 / 18400) * np.cos(10 * phi + theta)
        + (99 / 58880) * np.cos(10 * phi + 3 * theta)
        + 194481 / 271960832
    )


def H_gg_l12_p3_q4(theta, phi):
    return (
        (875 / 3803648) * np.cos(4 * phi)
        + (11 / 35328) * np.cos(8 * phi)
        + (13 / 13824) * np.cos(12 * phi)
        + (18865 / 148342272) * np.cos(2 * theta)
        + (3773 / 148342272) * np.cos(4 * theta)
        + (539 / 445026816) * np.cos(6 * theta)
        + (7 / 1426368) * np.cos(2 * phi - 5 * theta)
        + (7 / 118864) * np.cos(2 * phi - 3 * theta)
        + (175 / 950912) * np.cos(2 * phi - theta)
        + (35 / 178296) * np.cos(2 * phi + theta)
        + (35 / 475456) * np.cos(2 * phi + 3 * theta)
        + (1 / 118864) * np.cos(2 * phi + 5 * theta)
        + (1 / 5705472) * np.cos(2 * phi + 7 * theta)
        + (175 / 11410944) * np.cos(4 * phi - 4 * theta)
        + (875 / 7607296) * np.cos(4 * phi - 2 * theta)
        + (875 / 5705472) * np.cos(4 * phi + 2 * theta)
        + (125 / 3803648) * np.cos(4 * phi + 4 * theta)
        + (25 / 15214592) * np.cos(4 * phi + 6 * theta)
        + (125 / 3020544) * np.cos(6 * phi - 3 * theta)
        + (25 / 125856) * np.cos(6 * phi - theta)
        + (125 / 503424) * np.cos(6 * phi + theta)
        + (125 / 1321488) * np.cos(6 * phi + 3 * theta)
        + (125 / 14095872) * np.cos(6 * phi + 5 * theta)
        + (11 / 105984) * np.cos(8 * phi - 2 * theta)
        + (55 / 247296) * np.cos(8 * phi + 2 * theta)
        + (55 / 1483776) * np.cos(8 * phi + 4 * theta)
        + (7 / 26496) * np.cos(10 * phi - theta)
        + (1 / 2208) * np.cos(10 * phi + theta)
        + (5 / 35328) * np.cos(10 * phi + 3 * theta)
        + (13 / 18432) * np.cos(12 * phi + 2 * theta)
        + 94325 / 890053632
    )


def H_gg_l14_p3_q4(theta, phi):
    return (
        (7 / 447488) * np.cos(4 * phi)
        + (7 / 353280) * np.cos(8 * phi)
        + (1 / 27648) * np.cos(12 * phi)
        + (1001 / 114109440) * np.cos(2 * theta)
        + (1001 / 570547200) * np.cos(4 * theta)
        + (143 / 1711641600) * np.cos(6 * theta)
        + (77 / 228218880) * np.cos(2 * phi - 5 * theta)
        + (77 / 19018240) * np.cos(2 * phi - 3 * theta)
        + (385 / 30429184) * np.cos(2 * phi - theta)
        + (77 / 5705472) * np.cos(2 * phi + theta)
        + (77 / 15214592) * np.cos(2 * phi + 3 * theta)
        + (11 / 19018240) * np.cos(2 * phi + 5 * theta)
        + (11 / 912875520) * np.cos(2 * phi + 7 * theta)
        + (7 / 6712320) * np.cos(4 * phi - 4 * theta)
        + (7 / 894976) * np.cos(4 * phi - 2 * theta)
        + (7 / 671232) * np.cos(4 * phi + 2 * theta)
        + (1 / 447488) * np.cos(4 * phi + 4 * theta)
        + (1 / 8949760) * np.cos(4 * phi + 6 * theta)
        + (7 / 2543616) * np.cos(6 * phi - 3 * theta)
        + (7 / 529920) * np.cos(6 * phi - theta)
        + (7 / 423936) * np.cos(6 * phi + theta)
        + (1 / 158976) * np.cos(6 * phi + 3 * theta)
        + (1 / 1695744) * np.cos(6 * phi + 5 * theta)
        + (7 / 1059840) * np.cos(8 * phi - 2 * theta)
        + (1 / 70656) * np.cos(8 * phi + 2 * theta)
        + (1 / 423936) * np.cos(8 * phi + 4 * theta)
        + (7 / 460800) * np.cos(10 * phi - theta)
        + (1 / 38400) * np.cos(10 * phi + theta)
        + (1 / 122880) * np.cos(10 * phi + 3 * theta)
        + (1 / 36864) * np.cos(12 * phi + 2 * theta)
        + (1 / 8192) * np.cos(14 * phi + theta)
        + 1001 / 136931328
    )


def H_gg_l16_p3_q4(theta, phi):
    return 0


def H_gg_l0_p4_q0(theta, phi):
    return 1 / 9


def H_gg_l2_p4_q0(theta, phi):
    return (10 / 33) * np.cos(2 * phi - theta) + 10 / 99


def H_gg_l4_p4_q0(theta, phi):
    return (
        (15 / 143) * np.cos(2 * phi - theta)
        + (105 / 572) * np.cos(4 * phi - 2 * theta)
        + 27 / 572
    )


def H_gg_l6_p4_q0(theta, phi):
    return (
        (7 / 264) * np.cos(2 * phi - theta)
        + (7 / 220) * np.cos(4 * phi - 2 * theta)
        + (7 / 120) * np.cos(6 * phi - 3 * theta)
        + 5 / 396
    )


def H_gg_l8_p4_q0(theta, phi):
    return (
        (7 / 2288) * np.cos(2 * phi - theta)
        + (7 / 2080) * np.cos(4 * phi - 2 * theta)
        + (1 / 240) * np.cos(6 * phi - 3 * theta)
        + (1 / 128) * np.cos(8 * phi - 4 * theta)
        + 245 / 164736
    )


def H_gg_l10_p4_q0(theta, phi):
    return 0


def H_gg_l0_p4_q1(theta, phi):
    return (4 / 99) * np.cos(2 * theta) + 5 / 99


def H_gg_l2_p4_q1(theta, phi):
    return (
        (50 / 1287) * np.cos(2 * theta)
        + (5 / 143) * np.cos(2 * phi - 3 * theta)
        + (20 / 143) * np.cos(2 * phi - theta)
        + (25 / 286) * np.cos(2 * phi + theta)
        + 125 / 2574
    )


def H_gg_l4_p4_q1(theta, phi):
    return (
        (49 / 572) * np.cos(4 * phi)
        + (3 / 143) * np.cos(2 * theta)
        + (2 / 143) * np.cos(2 * phi - 3 * theta)
        + (8 / 143) * np.cos(2 * phi - theta)
        + (5 / 143) * np.cos(2 * phi + theta)
        + (7 / 572) * np.cos(4 * phi - 4 * theta)
        + (49 / 572) * np.cos(4 * phi - 2 * theta)
        + 15 / 572
    )


def H_gg_l6_p4_q1(theta, phi):
    return (
        (147 / 7480) * np.cos(4 * phi)
        + (25 / 3366) * np.cos(2 * theta)
        + (7 / 1496) * np.cos(2 * phi - 3 * theta)
        + (7 / 374) * np.cos(2 * phi - theta)
        + (35 / 2992) * np.cos(2 * phi + theta)
        + (21 / 7480) * np.cos(4 * phi - 4 * theta)
        + (147 / 7480) * np.cos(4 * phi - 2 * theta)
        + (7 / 4080) * np.cos(6 * phi - 5 * theta)
        + (7 / 255) * np.cos(6 * phi - 3 * theta)
        + (49 / 1020) * np.cos(6 * phi - theta)
        + 125 / 13464
    )


def H_gg_l8_p4_q1(theta, phi):
    return (
        (147 / 39520) * np.cos(4 * phi)
        + (1225 / 782496) * np.cos(2 * theta)
        + (21 / 21736) * np.cos(2 * phi - 3 * theta)
        + (21 / 5434) * np.cos(2 * phi - theta)
        + (105 / 43472) * np.cos(2 * phi + theta)
        + (21 / 39520) * np.cos(4 * phi - 4 * theta)
        + (147 / 39520) * np.cos(4 * phi - 2 * theta)
        + (1 / 4560) * np.cos(6 * phi - 5 * theta)
        + (1 / 285) * np.cos(6 * phi - 3 * theta)
        + (7 / 1140) * np.cos(6 * phi - theta)
        + (9 / 2432) * np.cos(8 * phi - 4 * theta)
        + (9 / 608) * np.cos(8 * phi - 2 * theta)
        + 6125 / 3129984
    )


def H_gg_l10_p4_q1(theta, phi):
    return (
        (7 / 20672) * np.cos(4 * phi)
        + (441 / 2956096) * np.cos(2 * theta)
        + (49 / 537472) * np.cos(2 * phi - 3 * theta)
        + (49 / 134368) * np.cos(2 * phi - theta)
        + (245 / 1074944) * np.cos(2 * phi + theta)
        + (1 / 20672) * np.cos(4 * phi - 4 * theta)
        + (7 / 20672) * np.cos(4 * phi - 2 * theta)
        + (3 / 165376) * np.cos(6 * phi - 5 * theta)
        + (3 / 10336) * np.cos(6 * phi - 3 * theta)
        + (21 / 41344) * np.cos(6 * phi - theta)
        + (1 / 4864) * np.cos(8 * phi - 4 * theta)
        + (1 / 1216) * np.cos(8 * phi - 2 * theta)
        + (1 / 512) * np.cos(10 * phi - 3 * theta)
        + 2205 / 11824384
    )


def H_gg_l12_p4_q1(theta, phi):
    return 0


def H_gg_l0_p4_q2(theta, phi):
    return (16 / 429) * np.cos(2 * theta) + (2 / 429) * np.cos(4 * theta) + 5 / 143


def H_gg_l2_p4_q2(theta, phi):
    return (
        (16 / 429) * np.cos(2 * theta)
        + (2 / 429) * np.cos(4 * theta)
        + (1 / 429) * np.cos(2 * phi - 5 * theta)
        + (14 / 429) * np.cos(2 * phi - 3 * theta)
        + (14 / 143) * np.cos(2 * phi - theta)
        + (35 / 429) * np.cos(2 * phi + theta)
        + (7 / 429) * np.cos(2 * phi + 3 * theta)
        + 5 / 143
    )


def H_gg_l4_p4_q2(theta, phi):
    return (
        (196 / 2431) * np.cos(4 * phi)
        + (54 / 2431) * np.cos(2 * theta)
        + (27 / 9724) * np.cos(4 * theta)
        + (5 / 4862) * np.cos(2 * phi - 5 * theta)
        + (35 / 2431) * np.cos(2 * phi - 3 * theta)
        + (105 / 2431) * np.cos(2 * phi - theta)
        + (175 / 4862) * np.cos(2 * phi + theta)
        + (35 / 4862) * np.cos(2 * phi + 3 * theta)
        + (7 / 19448) * np.cos(4 * phi - 6 * theta)
        + (28 / 2431) * np.cos(4 * phi - 4 * theta)
        + (147 / 2431) * np.cos(4 * phi - 2 * theta)
        + (245 / 9724) * np.cos(4 * phi + 2 * theta)
        + 405 / 19448
    )


def H_gg_l6_p4_q2(theta, phi):
    return (
        (392 / 17765) * np.cos(4 * phi)
        + (100 / 10659) * np.cos(2 * theta)
        + (25 / 21318) * np.cos(4 * theta)
        + (35 / 85272) * np.cos(2 * phi - 5 * theta)
        + (245 / 42636) * np.cos(2 * phi - 3 * theta)
        + (245 / 14212) * np.cos(2 * phi - theta)
        + (1225 / 85272) * np.cos(2 * phi + theta)
        + (245 / 85272) * np.cos(2 * phi + 3 * theta)
        + (7 / 71060) * np.cos(4 * phi - 6 * theta)
        + (56 / 17765) * np.cos(4 * phi - 4 * theta)
        + (294 / 17765) * np.cos(4 * phi - 2 * theta)
        + (49 / 7106) * np.cos(4 * phi + 2 * theta)
        + (21 / 12920) * np.cos(6 * phi - 5 * theta)
        + (63 / 3230) * np.cos(6 * phi - 3 * theta)
        + (147 / 3230) * np.cos(6 * phi - theta)
        + (147 / 6460) * np.cos(6 * phi + theta)
        + 125 / 14212
    )


def H_gg_l8_p4_q2(theta, phi):
    return (
        (7 / 1235) * np.cos(4 * phi)
        + (15 / 1216) * np.cos(8 * phi)
        + (175 / 65208) * np.cos(2 * theta)
        + (175 / 521664) * np.cos(4 * theta)
        + (5 / 43472) * np.cos(2 * phi - 5 * theta)
        + (35 / 21736) * np.cos(2 * phi - 3 * theta)
        + (105 / 21736) * np.cos(2 * phi - theta)
        + (175 / 43472) * np.cos(2 * phi + theta)
        + (35 / 43472) * np.cos(2 * phi + 3 * theta)
        + (1 / 39520) * np.cos(4 * phi - 6 * theta)
        + (1 / 1235) * np.cos(4 * phi - 4 * theta)
        + (21 / 4940) * np.cos(4 * phi - 2 * theta)
        + (7 / 3952) * np.cos(4 * phi + 2 * theta)
        + (3 / 10640) * np.cos(6 * phi - 5 * theta)
        + (9 / 2660) * np.cos(6 * phi - 3 * theta)
        + (3 / 380) * np.cos(6 * phi - theta)
        + (3 / 760) * np.cos(6 * phi + theta)
        + (45 / 17024) * np.cos(8 * phi - 4 * theta)
        + (15 / 1064) * np.cos(8 * phi - 2 * theta)
        + 875 / 347776
    )


def H_gg_l10_p4_q2(theta, phi):
    return (
        (7 / 7429) * np.cos(4 * phi)
        + (35 / 27968) * np.cos(8 * phi)
        + (3969 / 8498776) * np.cos(2 * theta)
        + (3969 / 67990208) * np.cos(4 * theta)
        + (245 / 12361856) * np.cos(2 * phi - 5 * theta)
        + (1715 / 6180928) * np.cos(2 * phi - 3 * theta)
        + (5145 / 6180928) * np.cos(2 * phi - theta)
        + (8575 / 12361856) * np.cos(2 * phi + theta)
        + (1715 / 12361856) * np.cos(2 * phi + 3 * theta)
        + (1 / 237728) * np.cos(4 * phi - 6 * theta)
        + (1 / 7429) * np.cos(4 * phi - 4 * theta)
        + (21 / 29716) * np.cos(4 * phi - 2 * theta)
        + (35 / 118864) * np.cos(4 * phi + 2 * theta)
        + (81 / 1901824) * np.cos(6 * phi - 5 * theta)
        + (243 / 475456) * np.cos(6 * phi - 3 * theta)
        + (567 / 475456) * np.cos(6 * phi - theta)
        + (567 / 950912) * np.cos(6 * phi + theta)
        + (15 / 55936) * np.cos(8 * phi - 4 * theta)
        + (5 / 3496) * np.cos(8 * phi - 2 * theta)
        + (11 / 5888) * np.cos(10 * phi - 3 * theta)
        + (11 / 2944) * np.cos(10 * phi - theta)
        + 59535 / 135980416
    )


def H_gg_l12_p4_q2(theta, phi):
    return (
        (35 / 475456) * np.cos(4 * phi)
        + (1 / 11776) * np.cos(8 * phi)
        + (231 / 6180928) * np.cos(2 * theta)
        + (231 / 49447424) * np.cos(4 * theta)
        + (3 / 1901824) * np.cos(2 * phi - 5 * theta)
        + (21 / 950912) * np.cos(2 * phi - 3 * theta)
        + (63 / 950912) * np.cos(2 * phi - theta)
        + (105 / 1901824) * np.cos(2 * phi + theta)
        + (21 / 1901824) * np.cos(2 * phi + 3 * theta)
        + (5 / 15214592) * np.cos(4 * phi - 6 * theta)
        + (5 / 475456) * np.cos(4 * phi - 4 * theta)
        + (105 / 1901824) * np.cos(4 * phi - 2 * theta)
        + (175 / 7607296) * np.cos(4 * phi + 2 * theta)
        + (5 / 1566208) * np.cos(6 * phi - 5 * theta)
        + (15 / 391552) * np.cos(6 * phi - 3 * theta)
        + (5 / 55936) * np.cos(6 * phi - theta)
        + (5 / 111872) * np.cos(6 * phi + theta)
        + (3 / 164864) * np.cos(8 * phi - 4 * theta)
        + (1 / 10304) * np.cos(8 * phi - 2 * theta)
        + (1 / 11776) * np.cos(10 * phi - 3 * theta)
        + (1 / 5888) * np.cos(10 * phi - theta)
        + (1 / 2048) * np.cos(12 * phi - 2 * theta)
        + 3465 / 98894848
    )


def H_gg_l14_p4_q2(theta, phi):
    return 0


def H_gg_l0_p4_q3(theta, phi):
    return (
        (14 / 429) * np.cos(2 * theta)
        + (14 / 2145) * np.cos(4 * theta)
        + (2 / 6435) * np.cos(6 * theta)
        + 35 / 1287
    )


def H_gg_l2_p4_q3(theta, phi):
    return (
        (245 / 7293) * np.cos(2 * theta)
        + (49 / 7293) * np.cos(4 * theta)
        + (7 / 21879) * np.cos(6 * theta)
        + (1 / 14586) * np.cos(2 * phi - 7 * theta)
        + (8 / 2431) * np.cos(2 * phi - 5 * theta)
        + (70 / 2431) * np.cos(2 * phi - 3 * theta)
        + (560 / 7293) * np.cos(2 * phi - theta)
        + (175 / 2431) * np.cos(2 * phi + theta)
        + (56 / 2431) * np.cos(2 * phi + 3 * theta)
        + (14 / 7293) * np.cos(2 * phi + 5 * theta)
        + 1225 / 43758
    )


def H_gg_l4_p4_q3(theta, phi):
    return (
        (6615 / 92378) * np.cos(4 * phi)
        + (3969 / 184756) * np.cos(2 * theta)
        + (3969 / 923780) * np.cos(4 * theta)
        + (189 / 923780) * np.cos(6 * theta)
        + (3 / 92378) * np.cos(2 * phi - 7 * theta)
        + (72 / 46189) * np.cos(2 * phi - 5 * theta)
        + (630 / 46189) * np.cos(2 * phi - 3 * theta)
        + (1680 / 46189) * np.cos(2 * phi - theta)
        + (1575 / 46189) * np.cos(2 * phi + theta)
        + (504 / 46189) * np.cos(2 * phi + 3 * theta)
        + (42 / 46189) * np.cos(2 * phi + 5 * theta)
        + (189 / 369512) * np.cos(4 * phi - 6 * theta)
        + (945 / 92378) * np.cos(4 * phi - 4 * theta)
        + (2205 / 46189) * np.cos(4 * phi - 2 * theta)
        + (6615 / 184756) * np.cos(4 * phi + 2 * theta)
        + (441 / 92378) * np.cos(4 * phi + 4 * theta)
        + 6615 / 369512
    )


def H_gg_l6_p4_q3(theta, phi):
    return (
        (315 / 14212) * np.cos(4 * phi)
        + (875 / 85272) * np.cos(2 * theta)
        + (175 / 85272) * np.cos(4 * theta)
        + (25 / 255816) * np.cos(6 * theta)
        + (5 / 341088) * np.cos(2 * phi - 7 * theta)
        + (5 / 7106) * np.cos(2 * phi - 5 * theta)
        + (175 / 28424) * np.cos(2 * phi - 3 * theta)
        + (175 / 10659) * np.cos(2 * phi - theta)
        + (875 / 56848) * np.cos(2 * phi + theta)
        + (35 / 7106) * np.cos(2 * phi + 3 * theta)
        + (35 / 85272) * np.cos(2 * phi + 5 * theta)
        + (9 / 56848) * np.cos(4 * phi - 6 * theta)
        + (45 / 14212) * np.cos(4 * phi - 4 * theta)
        + (105 / 7106) * np.cos(4 * phi - 2 * theta)
        + (315 / 28424) * np.cos(4 * phi + 2 * theta)
        + (21 / 14212) * np.cos(4 * phi + 4 * theta)
        + (15 / 10336) * np.cos(6 * phi - 5 * theta)
        + (5 / 323) * np.cos(6 * phi - 3 * theta)
        + (105 / 2584) * np.cos(6 * phi - theta)
        + (21 / 646) * np.cos(6 * phi + theta)
        + (35 / 5168) * np.cos(6 * phi + 3 * theta)
        + 4375 / 511632
    )


def H_gg_l8_p4_q3(theta, phi):
    return (
        (315 / 45448) * np.cos(4 * phi)
        + (495 / 27968) * np.cos(8 * phi)
        + (42875 / 11998272) * np.cos(2 * theta)
        + (8575 / 11998272) * np.cos(4 * theta)
        + (1225 / 35994816) * np.cos(6 * theta)
        + (5 / 999856) * np.cos(2 * phi - 7 * theta)
        + (15 / 62491) * np.cos(2 * phi - 5 * theta)
        + (525 / 249964) * np.cos(2 * phi - 3 * theta)
        + (350 / 62491) * np.cos(2 * phi - theta)
        + (2625 / 499928) * np.cos(2 * phi + theta)
        + (105 / 62491) * np.cos(2 * phi + 3 * theta)
        + (35 / 249964) * np.cos(2 * phi + 5 * theta)
        + (9 / 181792) * np.cos(4 * phi - 6 * theta)
        + (45 / 45448) * np.cos(4 * phi - 4 * theta)
        + (105 / 22724) * np.cos(4 * phi - 2 * theta)
        + (315 / 90896) * np.cos(4 * phi + 2 * theta)
        + (21 / 45448) * np.cos(4 * phi + 4 * theta)
        + (15 / 48944) * np.cos(6 * phi - 5 * theta)
        + (10 / 3059) * np.cos(6 * phi - 3 * theta)
        + (15 / 1748) * np.cos(6 * phi - theta)
        + (3 / 437) * np.cos(6 * phi + theta)
        + (5 / 3496) * np.cos(6 * phi + 3 * theta)
        + (825 / 391552) * np.cos(8 * phi - 4 * theta)
        + (2475 / 195776) * np.cos(8 * phi - 2 * theta)
        + (165 / 27968) * np.cos(8 * phi + 2 * theta)
        + 214375 / 71989632
    )


def H_gg_l10_p4_q3(theta, phi):
    return (
        (189 / 118864) * np.cos(4 * phi)
        + (693 / 279680) * np.cos(8 * phi)
        + (583443 / 679902080) * np.cos(2 * theta)
        + (583443 / 3399510400) * np.cos(4 * theta)
        + (27783 / 3399510400) * np.cos(6 * theta)
        + (147 / 123618560) * np.cos(2 * phi - 7 * theta)
        + (441 / 7726160) * np.cos(2 * phi - 5 * theta)
        + (3087 / 6180928) * np.cos(2 * phi - 3 * theta)
        + (1029 / 772616) * np.cos(2 * phi - theta)
        + (15435 / 12361856) * np.cos(2 * phi + theta)
        + (3087 / 7726160) * np.cos(2 * phi + 3 * theta)
        + (1029 / 30904640) * np.cos(2 * phi + 5 * theta)
        + (27 / 2377280) * np.cos(4 * phi - 6 * theta)
        + (27 / 118864) * np.cos(4 * phi - 4 * theta)
        + (63 / 59432) * np.cos(4 * phi - 2 * theta)
        + (189 / 237728) * np.cos(4 * phi + 2 * theta)
        + (63 / 594320) * np.cos(4 * phi + 4 * theta)
        + (243 / 3803648) * np.cos(6 * phi - 5 * theta)
        + (81 / 118864) * np.cos(6 * phi - 3 * theta)
        + (1701 / 950912) * np.cos(6 * phi - theta)
        + (1701 / 1188640) * np.cos(6 * phi + theta)
        + (567 / 1901824) * np.cos(6 * phi + 3 * theta)
        + (33 / 111872) * np.cos(8 * phi - 4 * theta)
        + (99 / 55936) * np.cos(8 * phi - 2 * theta)
        + (231 / 279680) * np.cos(8 * phi + 2 * theta)
        + (99 / 58880) * np.cos(10 * phi - 3 * theta)
        + (99 / 18400) * np.cos(10 * phi - theta)
        + (231 / 73600) * np.cos(10 * phi + theta)
        + 194481 / 271960832
    )


def H_gg_l12_p4_q3(theta, phi):
    return (
        (875 / 3803648) * np.cos(4 * phi)
        + (11 / 35328) * np.cos(8 * phi)
        + (13 / 13824) * np.cos(12 * phi)
        + (18865 / 148342272) * np.cos(2 * theta)
        + (3773 / 148342272) * np.cos(4 * theta)
        + (539 / 445026816) * np.cos(6 * theta)
        + (1 / 5705472) * np.cos(2 * phi - 7 * theta)
        + (1 / 118864) * np.cos(2 * phi - 5 * theta)
        + (35 / 475456) * np.cos(2 * phi - 3 * theta)
        + (35 / 178296) * np.cos(2 * phi - theta)
        + (175 / 950912) * np.cos(2 * phi + theta)
        + (7 / 118864) * np.cos(2 * phi + 3 * theta)
        + (7 / 1426368) * np.cos(2 * phi + 5 * theta)
        + (25 / 15214592) * np.cos(4 * phi - 6 * theta)
        + (125 / 3803648) * np.cos(4 * phi - 4 * theta)
        + (875 / 5705472) * np.cos(4 * phi - 2 * theta)
        + (875 / 7607296) * np.cos(4 * phi + 2 * theta)
        + (175 / 11410944) * np.cos(4 * phi + 4 * theta)
        + (125 / 14095872) * np.cos(6 * phi - 5 * theta)
        + (125 / 1321488) * np.cos(6 * phi - 3 * theta)
        + (125 / 503424) * np.cos(6 * phi - theta)
        + (25 / 125856) * np.cos(6 * phi + theta)
        + (125 / 3020544) * np.cos(6 * phi + 3 * theta)
        + (55 / 1483776) * np.cos(8 * phi - 4 * theta)
        + (55 / 247296) * np.cos(8 * phi - 2 * theta)
        + (11 / 105984) * np.cos(8 * phi + 2 * theta)
        + (5 / 35328) * np.cos(10 * phi - 3 * theta)
        + (1 / 2208) * np.cos(10 * phi - theta)
        + (7 / 26496) * np.cos(10 * phi + theta)
        + (13 / 18432) * np.cos(12 * phi - 2 * theta)
        + 94325 / 890053632
    )


def H_gg_l14_p4_q3(theta, phi):
    return (
        (7 / 447488) * np.cos(4 * phi)
        + (7 / 353280) * np.cos(8 * phi)
        + (1 / 27648) * np.cos(12 * phi)
        + (1001 / 114109440) * np.cos(2 * theta)
        + (1001 / 570547200) * np.cos(4 * theta)
        + (143 / 1711641600) * np.cos(6 * theta)
        + (11 / 912875520) * np.cos(2 * phi - 7 * theta)
        + (11 / 19018240) * np.cos(2 * phi - 5 * theta)
        + (77 / 15214592) * np.cos(2 * phi - 3 * theta)
        + (77 / 5705472) * np.cos(2 * phi - theta)
        + (385 / 30429184) * np.cos(2 * phi + theta)
        + (77 / 19018240) * np.cos(2 * phi + 3 * theta)
        + (77 / 228218880) * np.cos(2 * phi + 5 * theta)
        + (1 / 8949760) * np.cos(4 * phi - 6 * theta)
        + (1 / 447488) * np.cos(4 * phi - 4 * theta)
        + (7 / 671232) * np.cos(4 * phi - 2 * theta)
        + (7 / 894976) * np.cos(4 * phi + 2 * theta)
        + (7 / 6712320) * np.cos(4 * phi + 4 * theta)
        + (1 / 1695744) * np.cos(6 * phi - 5 * theta)
        + (1 / 158976) * np.cos(6 * phi - 3 * theta)
        + (7 / 423936) * np.cos(6 * phi - theta)
        + (7 / 529920) * np.cos(6 * phi + theta)
        + (7 / 2543616) * np.cos(6 * phi + 3 * theta)
        + (1 / 423936) * np.cos(8 * phi - 4 * theta)
        + (1 / 70656) * np.cos(8 * phi - 2 * theta)
        + (7 / 1059840) * np.cos(8 * phi + 2 * theta)
        + (1 / 122880) * np.cos(10 * phi - 3 * theta)
        + (1 / 38400) * np.cos(10 * phi - theta)
        + (7 / 460800) * np.cos(10 * phi + theta)
        + (1 / 36864) * np.cos(12 * phi - 2 * theta)
        + (1 / 8192) * np.cos(14 * phi - theta)
        + 1001 / 136931328
    )


def H_gg_l16_p4_q3(theta, phi):
    return 0


def H_gg_l0_p4_q4(theta, phi):
    return (
        (3136 / 109395) * np.cos(2 * theta)
        + (784 / 109395) * np.cos(4 * theta)
        + (64 / 109395) * np.cos(6 * theta)
        + (1 / 109395) * np.cos(8 * theta)
        + 490 / 21879
    )


def H_gg_l2_p4_q4(theta, phi):
    return (
        (12544 / 415701) * np.cos(2 * theta)
        + (3136 / 415701) * np.cos(4 * theta)
        + (256 / 415701) * np.cos(6 * theta)
        + (4 / 415701) * np.cos(8 * theta)
        + (6 / 46189) * np.cos(2 * phi - 7 * theta)
        + (168 / 46189) * np.cos(2 * phi - 5 * theta)
        + (1176 / 46189) * np.cos(2 * phi - 3 * theta)
        + (2940 / 46189) * np.cos(2 * phi - theta)
        + (2940 / 46189) * np.cos(2 * phi + theta)
        + (1176 / 46189) * np.cos(2 * phi + 3 * theta)
        + (168 / 46189) * np.cos(2 * phi + 5 * theta)
        + (6 / 46189) * np.cos(2 * phi + 7 * theta)
        + 9800 / 415701
    )


def H_gg_l4_p4_q4(theta, phi):
    return (
        (2940 / 46189) * np.cos(4 * phi)
        + (4704 / 230945) * np.cos(2 * theta)
        + (1176 / 230945) * np.cos(4 * theta)
        + (96 / 230945) * np.cos(6 * theta)
        + (3 / 461890) * np.cos(8 * theta)
        + (3 / 46189) * np.cos(2 * phi - 7 * theta)
        + (84 / 46189) * np.cos(2 * phi - 5 * theta)
        + (588 / 46189) * np.cos(2 * phi - 3 * theta)
        + (1470 / 46189) * np.cos(2 * phi - theta)
        + (1470 / 46189) * np.cos(2 * phi + theta)
        + (588 / 46189) * np.cos(2 * phi + 3 * theta)
        + (84 / 46189) * np.cos(2 * phi + 5 * theta)
        + (3 / 46189) * np.cos(2 * phi + 7 * theta)
        + (105 / 184756) * np.cos(4 * phi - 6 * theta)
        + (420 / 46189) * np.cos(4 * phi - 4 * theta)
        + (3675 / 92378) * np.cos(4 * phi - 2 * theta)
        + (3675 / 92378) * np.cos(4 * phi + 2 * theta)
        + (420 / 46189) * np.cos(4 * phi + 4 * theta)
        + (105 / 184756) * np.cos(4 * phi + 6 * theta)
        + 735 / 46189
    )


def H_gg_l6_p4_q4(theta, phi):
    return (
        (1764 / 81719) * np.cos(4 * phi)
        + (7840 / 735471) * np.cos(2 * theta)
        + (1960 / 735471) * np.cos(4 * theta)
        + (160 / 735471) * np.cos(6 * theta)
        + (5 / 1470942) * np.cos(8 * theta)
        + (21 / 653752) * np.cos(2 * phi - 7 * theta)
        + (147 / 163438) * np.cos(2 * phi - 5 * theta)
        + (1029 / 163438) * np.cos(2 * phi - 3 * theta)
        + (5145 / 326876) * np.cos(2 * phi - theta)
        + (5145 / 326876) * np.cos(2 * phi + theta)
        + (1029 / 163438) * np.cos(2 * phi + 3 * theta)
        + (147 / 163438) * np.cos(2 * phi + 5 * theta)
        + (21 / 653752) * np.cos(2 * phi + 7 * theta)
        + (63 / 326876) * np.cos(4 * phi - 6 * theta)
        + (252 / 81719) * np.cos(4 * phi - 4 * theta)
        + (2205 / 163438) * np.cos(4 * phi - 2 * theta)
        + (2205 / 163438) * np.cos(4 * phi + 2 * theta)
        + (252 / 81719) * np.cos(4 * phi + 4 * theta)
        + (63 / 326876) * np.cos(4 * phi + 6 * theta)
        + (77 / 59432) * np.cos(6 * phi - 5 * theta)
        + (385 / 29716) * np.cos(6 * phi - 3 * theta)
        + (539 / 14858) * np.cos(6 * phi - theta)
        + (539 / 14858) * np.cos(6 * phi + theta)
        + (385 / 29716) * np.cos(6 * phi + 3 * theta)
        + (77 / 59432) * np.cos(6 * phi + 5 * theta)
        + 6125 / 735471
    )


def H_gg_l8_p4_q4(theta, phi):
    return (
        (441 / 56810) * np.cos(4 * phi)
        + (693 / 34960) * np.cos(8 * phi)
        + (2401 / 562419) * np.cos(2 * theta)
        + (2401 / 2249676) * np.cos(4 * theta)
        + (49 / 562419) * np.cos(6 * theta)
        + (49 / 35994816) * np.cos(8 * theta)
        + (63 / 4999280) * np.cos(2 * phi - 7 * theta)
        + (441 / 1249820) * np.cos(2 * phi - 5 * theta)
        + (3087 / 1249820) * np.cos(2 * phi - 3 * theta)
        + (3087 / 499928) * np.cos(2 * phi - theta)
        + (3087 / 499928) * np.cos(2 * phi + theta)
        + (3087 / 1249820) * np.cos(2 * phi + 3 * theta)
        + (441 / 1249820) * np.cos(2 * phi + 5 * theta)
        + (63 / 4999280) * np.cos(2 * phi + 7 * theta)
        + (63 / 908960) * np.cos(4 * phi - 6 * theta)
        + (63 / 56810) * np.cos(4 * phi - 4 * theta)
        + (441 / 90896) * np.cos(4 * phi - 2 * theta)
        + (441 / 90896) * np.cos(4 * phi + 2 * theta)
        + (63 / 56810) * np.cos(4 * phi + 4 * theta)
        + (63 / 908960) * np.cos(4 * phi + 6 * theta)
        + (11 / 34960) * np.cos(6 * phi - 5 * theta)
        + (11 / 3496) * np.cos(6 * phi - 3 * theta)
        + (77 / 8740) * np.cos(6 * phi - theta)
        + (77 / 8740) * np.cos(6 * phi + theta)
        + (11 / 3496) * np.cos(6 * phi + 3 * theta)
        + (11 / 34960) * np.cos(6 * phi + 5 * theta)
        + (99 / 55936) * np.cos(8 * phi - 4 * theta)
        + (99 / 8740) * np.cos(8 * phi - 2 * theta)
        + (99 / 8740) * np.cos(8 * phi + 2 * theta)
        + (99 / 55936) * np.cos(8 * phi + 4 * theta)
        + 60025 / 17997408
    )


def H_gg_l10_p4_q4(theta, phi):
    return (
        (49 / 22287) * np.cos(4 * phi)
        + (539 / 157320) * np.cos(8 * phi)
        + (33614 / 26558675) * np.cos(2 * theta)
        + (16807 / 53117350) * np.cos(4 * theta)
        + (686 / 26558675) * np.cos(6 * theta)
        + (343 / 849877600) * np.cos(8 * theta)
        + (343 / 92713920) * np.cos(2 * phi - 7 * theta)
        + (2401 / 23178480) * np.cos(2 * phi - 5 * theta)
        + (16807 / 23178480) * np.cos(2 * phi - 3 * theta)
        + (16807 / 9271392) * np.cos(2 * phi - theta)
        + (16807 / 9271392) * np.cos(2 * phi + theta)
        + (16807 / 23178480) * np.cos(2 * phi + 3 * theta)
        + (2401 / 23178480) * np.cos(2 * phi + 5 * theta)
        + (343 / 92713920) * np.cos(2 * phi + 7 * theta)
        + (7 / 356592) * np.cos(4 * phi - 6 * theta)
        + (7 / 22287) * np.cos(4 * phi - 4 * theta)
        + (245 / 178296) * np.cos(4 * phi - 2 * theta)
        + (245 / 178296) * np.cos(4 * phi + 2 * theta)
        + (7 / 22287) * np.cos(4 * phi + 4 * theta)
        + (7 / 356592) * np.cos(4 * phi + 6 * theta)
        + (77 / 950912) * np.cos(6 * phi - 5 * theta)
        + (385 / 475456) * np.cos(6 * phi - 3 * theta)
        + (539 / 237728) * np.cos(6 * phi - theta)
        + (539 / 237728) * np.cos(6 * phi + theta)
        + (385 / 475456) * np.cos(6 * phi + 3 * theta)
        + (77 / 950912) * np.cos(6 * phi + 5 * theta)
        + (77 / 251712) * np.cos(8 * phi - 4 * theta)
        + (77 / 39330) * np.cos(8 * phi - 2 * theta)
        + (77 / 39330) * np.cos(8 * phi + 2 * theta)
        + (77 / 251712) * np.cos(8 * phi + 4 * theta)
        + (1001 / 662400) * np.cos(10 * phi - 3 * theta)
        + (1001 / 165600) * np.cos(10 * phi - theta)
        + (1001 / 165600) * np.cos(10 * phi + theta)
        + (1001 / 662400) * np.cos(10 * phi + 3 * theta)
        + 16807 / 16997552
    )


def H_gg_l12_p4_q4(theta, phi):
    return (
        (6125 / 13788224) * np.cos(4 * phi)
        + (77 / 128064) * np.cos(8 * phi)
        + (91 / 50112) * np.cos(12 * phi)
        + (26411 / 100826388) * np.cos(2 * theta)
        + (26411 / 403305552) * np.cos(4 * theta)
        + (539 / 100826388) * np.cos(6 * theta)
        + (539 / 6452888832) * np.cos(8 * theta)
        + (21 / 27576448) * np.cos(2 * phi - 7 * theta)
        + (147 / 6894112) * np.cos(2 * phi - 5 * theta)
        + (1029 / 6894112) * np.cos(2 * phi - 3 * theta)
        + (5145 / 13788224) * np.cos(2 * phi - theta)
        + (5145 / 13788224) * np.cos(2 * phi + theta)
        + (1029 / 6894112) * np.cos(2 * phi + 3 * theta)
        + (147 / 6894112) * np.cos(2 * phi + 5 * theta)
        + (21 / 27576448) * np.cos(2 * phi + 7 * theta)
        + (875 / 220611584) * np.cos(4 * phi - 6 * theta)
        + (875 / 13788224) * np.cos(4 * phi - 4 * theta)
        + (30625 / 110305792) * np.cos(4 * phi - 2 * theta)
        + (30625 / 110305792) * np.cos(4 * phi + 2 * theta)
        + (875 / 13788224) * np.cos(4 * phi + 4 * theta)
        + (875 / 220611584) * np.cos(4 * phi + 6 * theta)
        + (1375 / 87595776) * np.cos(6 * phi - 5 * theta)
        + (6875 / 43797888) * np.cos(6 * phi - 3 * theta)
        + (9625 / 21898944) * np.cos(6 * phi - theta)
        + (9625 / 21898944) * np.cos(6 * phi + theta)
        + (6875 / 43797888) * np.cos(6 * phi + 3 * theta)
        + (1375 / 87595776) * np.cos(6 * phi + 5 * theta)
        + (55 / 1024512) * np.cos(8 * phi - 4 * theta)
        + (11 / 32016) * np.cos(8 * phi - 2 * theta)
        + (11 / 32016) * np.cos(8 * phi + 2 * theta)
        + (55 / 1024512) * np.cos(8 * phi + 4 * theta)
        + (91 / 512256) * np.cos(10 * phi - 3 * theta)
        + (91 / 128064) * np.cos(10 * phi - theta)
        + (91 / 128064) * np.cos(10 * phi + theta)
        + (91 / 512256) * np.cos(10 * phi + 3 * theta)
        + (637 / 801792) * np.cos(12 * phi - 2 * theta)
        + (637 / 801792) * np.cos(12 * phi + 2 * theta)
        + 660275 / 3226444416
    )


def H_gg_l14_p4_q4(theta, phi):
    return (
        (49 / 867008) * np.cos(4 * phi)
        + (49 / 684480) * np.cos(8 * phi)
        + (7 / 53568) * np.cos(12 * phi)
        + (7007 / 207269100) * np.cos(2 * theta)
        + (7007 / 829076400) * np.cos(4 * theta)
        + (143 / 207269100) * np.cos(6 * theta)
        + (143 / 13265222400) * np.cos(8 * theta)
        + (231 / 2358261760) * np.cos(2 * phi - 7 * theta)
        + (1617 / 589565440) * np.cos(2 * phi - 5 * theta)
        + (11319 / 589565440) * np.cos(2 * phi - 3 * theta)
        + (11319 / 235826176) * np.cos(2 * phi - theta)
        + (11319 / 235826176) * np.cos(2 * phi + theta)
        + (11319 / 589565440) * np.cos(2 * phi + 3 * theta)
        + (1617 / 589565440) * np.cos(2 * phi + 5 * theta)
        + (231 / 2358261760) * np.cos(2 * phi + 7 * theta)
        + (7 / 13872128) * np.cos(4 * phi - 6 * theta)
        + (7 / 867008) * np.cos(4 * phi - 4 * theta)
        + (245 / 6936064) * np.cos(4 * phi - 2 * theta)
        + (245 / 6936064) * np.cos(4 * phi + 2 * theta)
        + (7 / 867008) * np.cos(4 * phi + 4 * theta)
        + (7 / 13872128) * np.cos(4 * phi + 6 * theta)
        + (77 / 39426048) * np.cos(6 * phi - 5 * theta)
        + (385 / 19713024) * np.cos(6 * phi - 3 * theta)
        + (539 / 9856512) * np.cos(6 * phi - theta)
        + (539 / 9856512) * np.cos(6 * phi + theta)
        + (385 / 19713024) * np.cos(6 * phi + 3 * theta)
        + (77 / 39426048) * np.cos(6 * phi + 5 * theta)
        + (7 / 1095168) * np.cos(8 * phi - 4 * theta)
        + (7 / 171120) * np.cos(8 * phi - 2 * theta)
        + (7 / 171120) * np.cos(8 * phi + 2 * theta)
        + (7 / 1095168) * np.cos(8 * phi + 4 * theta)
        + (91 / 4761600) * np.cos(10 * phi - 3 * theta)
        + (91 / 1190400) * np.cos(10 * phi - theta)
        + (91 / 1190400) * np.cos(10 * phi + theta)
        + (91 / 4761600) * np.cos(10 * phi + 3 * theta)
        + (49 / 857088) * np.cos(12 * phi - 2 * theta)
        + (49 / 857088) * np.cos(12 * phi + 2 * theta)
        + (15 / 63488) * np.cos(14 * phi - theta)
        + (15 / 63488) * np.cos(14 * phi + theta)
        + 7007 / 265304448
    )


def H_gg_l16_p4_q4(theta, phi):
    return (
        (539 / 158799360) * np.cos(4 * phi)
        + (343 / 82851840) * np.cos(8 * phi)
        + (3 / 460288) * np.cos(12 * phi)
        + (1 / 32768) * np.cos(16 * phi)
        + (7007 / 3419479552) * np.cos(2 * theta)
        + (7007 / 13677918208) * np.cos(4 * theta)
        + (143 / 3419479552) * np.cos(6 * theta)
        + (143 / 218846691328) * np.cos(8 * theta)
        + (143 / 24137502720) * np.cos(2 * phi - 7 * theta)
        + (1001 / 6034375680) * np.cos(2 * phi - 5 * theta)
        + (7007 / 6034375680) * np.cos(2 * phi - 3 * theta)
        + (7007 / 2413750272) * np.cos(2 * phi - theta)
        + (7007 / 2413750272) * np.cos(2 * phi + theta)
        + (7007 / 6034375680) * np.cos(2 * phi + 3 * theta)
        + (1001 / 6034375680) * np.cos(2 * phi + 5 * theta)
        + (143 / 24137502720) * np.cos(2 * phi + 7 * theta)
        + (77 / 2540789760) * np.cos(4 * phi - 6 * theta)
        + (77 / 158799360) * np.cos(4 * phi - 4 * theta)
        + (539 / 254078976) * np.cos(4 * phi - 2 * theta)
        + (539 / 254078976) * np.cos(4 * phi + 2 * theta)
        + (77 / 158799360) * np.cos(4 * phi + 4 * theta)
        + (77 / 2540789760) * np.cos(4 * phi + 6 * theta)
        + (49 / 423464960) * np.cos(6 * phi - 5 * theta)
        + (49 / 42346496) * np.cos(6 * phi - 3 * theta)
        + (343 / 105866240) * np.cos(6 * phi - theta)
        + (343 / 105866240) * np.cos(6 * phi + theta)
        + (49 / 42346496) * np.cos(6 * phi + 3 * theta)
        + (49 / 423464960) * np.cos(6 * phi + 5 * theta)
        + (49 / 132562944) * np.cos(8 * phi - 4 * theta)
        + (49 / 20712960) * np.cos(8 * phi - 2 * theta)
        + (49 / 20712960) * np.cos(8 * phi + 2 * theta)
        + (49 / 132562944) * np.cos(8 * phi + 4 * theta)
        + (35 / 33140736) * np.cos(10 * phi - 3 * theta)
        + (35 / 8285184) * np.cos(10 * phi - theta)
        + (35 / 8285184) * np.cos(10 * phi + theta)
        + (35 / 33140736) * np.cos(10 * phi + 3 * theta)
        + (21 / 7364608) * np.cos(12 * phi - 2 * theta)
        + (21 / 7364608) * np.cos(12 * phi + 2 * theta)
        + (1 / 126976) * np.cos(14 * phi - theta)
        + (1 / 126976) * np.cos(14 * phi + theta)
        + 175175 / 109423345664
    )


def H_gg_l18_p4_q4(theta, phi):
    return 0


# Density-Velocity
def H_gv_l1_p0(theta, phi):
    return -np.cos(phi + (1 / 2) * theta)


def H_gv_l1_p1(theta, phi):
    return -1 / 5 * np.cos(phi - 3 / 2 * theta) - 2 / 5 * np.cos(phi + (1 / 2) * theta)


def H_gv_l3_p1(theta, phi):
    return (
        -1 / 20 * np.cos(phi - 3 / 2 * theta)
        - 1 / 10 * np.cos(phi + (1 / 2) * theta)
        - 1 / 4 * np.cos(3 * phi - 1 / 2 * theta)
    )


def H_gv_l1_p2(theta, phi):
    return -6 / 35 * np.cos(phi - 3 / 2 * theta) - 9 / 35 * np.cos(
        phi + (1 / 2) * theta
    )


def H_gv_l3_p2(theta, phi):
    return (
        -1 / 15 * np.cos(phi - 3 / 2 * theta)
        - 1 / 10 * np.cos(phi + (1 / 2) * theta)
        - 1 / 18 * np.cos(3 * phi - 5 / 2 * theta)
        - 2 / 9 * np.cos(3 * phi - 1 / 2 * theta)
    )


def H_gv_l5_p2(theta, phi):
    return (
        -1 / 84 * np.cos(phi - 3 / 2 * theta)
        - 1 / 56 * np.cos(phi + (1 / 2) * theta)
        - 1 / 144 * np.cos(3 * phi - 5 / 2 * theta)
        - 1 / 36 * np.cos(3 * phi - 1 / 2 * theta)
        - 1 / 16 * np.cos(5 * phi - 3 / 2 * theta)
    )


def H_gv_l1_p3(theta, phi):
    return -1 / 7 * np.cos(phi - 3 / 2 * theta) - 4 / 21 * np.cos(phi + (1 / 2) * theta)


def H_gv_l3_p3(theta, phi):
    return (
        -3 / 44 * np.cos(phi - 3 / 2 * theta)
        - 1 / 11 * np.cos(phi + (1 / 2) * theta)
        - 5 / 66 * np.cos(3 * phi - 5 / 2 * theta)
        - 25 / 132 * np.cos(3 * phi - 1 / 2 * theta)
    )


def H_gv_l5_p3(theta, phi):
    return (
        -15 / 728 * np.cos(phi - 3 / 2 * theta)
        - 5 / 182 * np.cos(phi + (1 / 2) * theta)
        - 5 / 312 * np.cos(3 * phi - 5 / 2 * theta)
        - 25 / 624 * np.cos(3 * phi - 1 / 2 * theta)
        - 3 / 208 * np.cos(5 * phi - 7 / 2 * theta)
        - 9 / 104 * np.cos(5 * phi - 3 / 2 * theta)
    )


def H_gv_l7_p3(theta, phi):
    return (
        -25 / 9152 * np.cos(phi - 3 / 2 * theta)
        - 25 / 6864 * np.cos(phi + (1 / 2) * theta)
        - 9 / 4576 * np.cos(3 * phi - 5 / 2 * theta)
        - 45 / 9152 * np.cos(3 * phi - 1 / 2 * theta)
        - 1 / 832 * np.cos(5 * phi - 7 / 2 * theta)
        - 3 / 416 * np.cos(5 * phi - 3 / 2 * theta)
        - 1 / 64 * np.cos(7 * phi - 5 / 2 * theta)
    )


def H_gv_l1_p4(theta, phi):
    return -4 / 33 * np.cos(phi - 3 / 2 * theta) - 5 / 33 * np.cos(
        phi + (1 / 2) * theta
    )


def H_gv_l3_p4(theta, phi):
    return (
        -28 / 429 * np.cos(phi - 3 / 2 * theta)
        - 35 / 429 * np.cos(phi + (1 / 2) * theta)
        - 35 / 429 * np.cos(3 * phi - 5 / 2 * theta)
        - 70 / 429 * np.cos(3 * phi - 1 / 2 * theta)
    )


def H_gv_l5_p4(theta, phi):
    return (
        -1 / 39 * np.cos(phi - 3 / 2 * theta)
        - 5 / 156 * np.cos(phi + (1 / 2) * theta)
        - 7 / 312 * np.cos(3 * phi - 5 / 2 * theta)
        - 7 / 156 * np.cos(3 * phi - 1 / 2 * theta)
        - 7 / 260 * np.cos(5 * phi - 7 / 2 * theta)
        - 49 / 520 * np.cos(5 * phi - 3 / 2 * theta)
    )


def H_gv_l7_p4(theta, phi):
    return (
        -175 / 29172 * np.cos(phi - 3 / 2 * theta)
        - 875 / 116688 * np.cos(phi + (1 / 2) * theta)
        - 189 / 38896 * np.cos(3 * phi - 5 / 2 * theta)
        - 189 / 19448 * np.cos(3 * phi - 1 / 2 * theta)
        - 7 / 1768 * np.cos(5 * phi - 7 / 2 * theta)
        - 49 / 3536 * np.cos(5 * phi - 3 / 2 * theta)
        - 1 / 272 * np.cos(7 * phi - 9 / 2 * theta)
        - 1 / 34 * np.cos(7 * phi - 5 / 2 * theta)
    )


def H_gv_l9_p4(theta, phi):
    return (
        -49 / 77792 * np.cos(phi - 3 / 2 * theta)
        - 245 / 311168 * np.cos(phi + (1 / 2) * theta)
        - 7 / 14144 * np.cos(3 * phi - 5 / 2 * theta)
        - 7 / 7072 * np.cos(3 * phi - 1 / 2 * theta)
        - 1 / 2720 * np.cos(5 * phi - 7 / 2 * theta)
        - 7 / 5440 * np.cos(5 * phi - 3 / 2 * theta)
        - 1 / 4352 * np.cos(7 * phi - 9 / 2 * theta)
        - 1 / 544 * np.cos(7 * phi - 5 / 2 * theta)
        - 1 / 256 * np.cos(9 * phi - 7 / 2 * theta)
    )


# Velocity-Velocity
def H_vv_l0(theta, phi):
    return (1 / 3) * np.cos(theta)


def H_vv_l2(theta, phi):
    return (1 / 2) * np.cos(2 * phi) + (1 / 6) * np.cos(theta)
