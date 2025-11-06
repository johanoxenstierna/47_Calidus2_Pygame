import numpy as np

import P


def mercury_gi_():

    gi = {}
    gi['r'] = 50
    gi['phi'] = 0  # 0.2 * 2 * np.pi
    gi['period_days'] = 300
    gi['y_squeeze'] = 0.15
    gi['tilt'] = 0  # 0.04 * np.pi
    gi['scale'] = 0.2  # 0.2

    if P.REAL_SCALE == 1:
        gi['r'] = 0
        gi['period_days'] = 0
        gi['scale'] = 0.

    # if P.REAL_SCALE == 1:
    #     gi['r'] = 58
    #     gi['scale'] = 0.08
    #     gi['speed_gi'] = 166

    return gi