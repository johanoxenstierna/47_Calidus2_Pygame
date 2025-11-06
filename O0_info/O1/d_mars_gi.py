

import numpy as np

import P

def mars_gi_():  # 20

    gi = {}
    gi['r'] = 300
    gi['phi'] = 0  # 0.2 * 2 * np.pi
    gi['period_days'] = 600
    gi['y_squeeze'] = 0.15
    gi['tilt'] = 0  # 0.08 * 2 * np.pi
    gi['scale'] = 0.2  # 0.2

    if P.REAL_SCALE == 1:
        gi['r'] = 0
        gi['period_days'] = 0
        gi['scale'] = 0.

    # if P.REAL_SCALE == 1:
    #     gi['r'] = 234
    #     gi['scale'] = 0.05
    #     gi['speed_gi'] = 21

    return gi
