
import numpy as np
import P

def jupiter_gi_():

    """This one also includes all descriptive comments """

    gi = {}
    gi['r'] = 400  #  # doesn't affect bank
    gi['phi'] = 0.95 * 2 * np.pi   # 0.2
    gi['period_days'] = 4332  # OBS if P.SPEED_MULTIPLIER = 1, then period_days here == number of frames per orbit
    gi['y_squeeze'] = 0.99  # 0.2
    gi['tilt'] = 0 * np.pi  # NO EFFECT IF Y_SQUEEZE==1  0.25pi is 45 deg
    gi['scale'] = 0.3  # 0.3

    if P.REAL_SCALE == 1:
        gi['r'] = 0.  # 800
        gi['period_days'] = 0
        gi['scale'] = 0.

    return gi


def europa_gi_():
    gi = {}
    gi['r'] = 40
    gi['phi'] = 0.25 * 2 * np.pi
    gi['period_days'] = 300
    gi['y_squeeze'] = 1
    gi['tilt'] = 0  # 0.08 * 2 * np.pi
    gi['scale'] = 0.4  # 0.2

    if P.REAL_SCALE == 1:
        gi['r'] = 0
        gi['period_days'] = 0
        gi['scale'] = 0.

    return gi


def ganymede_gi_():
    gi = {}
    gi['r'] = 50
    gi['phi'] = 0  # 0.2 * 2 * np.pi
    gi['period_days'] = 300
    gi['y_squeeze'] = 1
    gi['tilt'] = 0  # 0.08 * 2 * np.pi
    gi['scale'] = 0.4  # 0.2

    if P.REAL_SCALE == 1:
        gi['r'] = 0
        gi['period_days'] = 0
        gi['scale'] = 0.

    return gi


def io_gi_():

    gi = {}
    gi['r'] = 50
    gi['phi'] = 0.0 * 2 * np.pi
    gi['period_days'] = 500
    gi['y_squeeze'] = 0.5
    gi['tilt'] = 0.0  # 0.08 * 2 * np.pi  # USE TILT CLOSE TO PARENT: It won't be a big issue when y_squeeze = 0.1
    gi['scale'] = 0.03  # 0.2

    if P.REAL_SCALE == 1:
        gi['r'] = 0
        gi['period_days'] = 0
        gi['scale'] = 0.

    return gi





