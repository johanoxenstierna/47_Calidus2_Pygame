import numpy as np

import P

def venus_gi_():

    gi = {}
    gi['r'] = 100
    gi['phi'] = 0  # 0.2 * 2 * np.pi
    gi['period_days'] = 200
    gi['y_squeeze'] = 0.15
    gi['tilt'] = 0  # 0.04 * 2 * np.pi
    gi['scale'] = 0.2  # 0.2

    if P.REAL_SCALE == 1:
        gi['r'] = 0
        gi['period_days'] = 0
        gi['scale'] = 0.


    # if P.REAL_SCALE == 1:
    #     gi['r'] = 110
    #     gi['scale'] = 0.05
    #     gi['speed_gi'] = 65

    return gi