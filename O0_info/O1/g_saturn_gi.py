



import numpy as np

import P

def saturn_gi_():  # 20

    gi = {}
    gi['r'] = 1100
    gi['phi'] = 0.8 * 2 * np.pi
    gi['speed_gi'] = 0.7  # 4
    gi['tilt'] = 0.07 * 2 * np.pi
    gi['scale'] = 0.5
    gi['centroid_mult'] = 4

    if P.REAL_SCALE == 1:
        gi['r'] = 1500
        gi['phi'] = 0.05 * 2 * np.pi
        gi['scale'] = 0.05
        gi['speed_gi'] = 0.3

    return gi