



import numpy as np

import P

def uranus_gi_():  # 20

    gi = {}
    gi['r'] = 1200
    gi['phi'] = 0.85 * 2 * np.pi
    gi['speed_gi'] = 0.4  # 4
    gi['tilt'] = 0.07 * 2 * np.pi
    gi['scale'] = 0.4
    gi['centroid_mult'] = 4

    if P.REAL_SCALE == 1:
        gi['r'] = 3000
        gi['phi'] = 0.0 * 2 * np.pi
        gi['scale'] = 0.03
        gi['speed_gi'] = 0.1

    return gi