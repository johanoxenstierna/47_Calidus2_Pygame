import numpy as np

import P

def astro0_gi_():  # 20

    gi = {}
    gi['r'] = 400
    gi['phi'] = 0
    gi['speed_gi'] = 1.7
    gi['tilt'] = 0  #0.1 * 2 * np.pi  # 0.15     0.12 * 2 * np.pi  is 45 deg
    gi['zorder'] = 2000
    gi['scale'] = 1

    if P.REAL_SCALE:
        gi['r'] = 317
        gi['scale'] = 1
        gi['speed_gi'] = 4

    return gi


def astro0b_gi_():  # 20

    gi = {}
    gi['r'] = 660
    gi['phi'] = 0.0 * 2 * np.pi
    # gi['speed_gi'] = 2
    gi['period_days'] = 4000
    gi['y_squeeze'] = 0.18
    gi['tilt'] = 0.05 * 2 * np.pi  # 0.15     0.12 * 2 * np.pi  is 45 deg
    gi['zorder'] = 2000
    gi['scale'] = 0.25
    # gi['centroid_mult'] = 8

    return gi