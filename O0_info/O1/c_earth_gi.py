
import numpy as np

import P

def earth_gi_():  # 20

    gi = {}
    gi['r'] = 200
    gi['phi'] = 0.0 * 2 * np.pi
    gi['period_days'] = 365
    gi['y_squeeze'] = 1 #0.15
    gi['tilt'] = 0 * np.pi  # 0.15     0.25pi is 45 deg
    gi['scale'] = 0.1  # 0.1

    return gi


def gss_gi_():  # 21
    gi = {}

    gi['r'] = 25
    gi['phi'] = 0.0 * 2 * np.pi
    gi['period_days'] = 100
    gi['y_squeeze'] = 0.4
    gi['tilt'] = 0.0 * 2 * np.pi
    gi['scale'] = 0.2

    return gi


def moon_gi_():

    gi = {}
    gi['r'] = 30
    gi['phi'] = 0.7 * 2 * np.pi
    gi['period_days'] = 200
    gi['y_squeeze'] = 0.4
    gi['tilt'] = 0.1 * 2 * np.pi  # USE TILT CLOSE TO PARENT: It won't be a big issue when y_squeeze = 0.1
    gi['scale'] = 0.1

    return gi


def nea_gi_():

    gi = {}
    gi['r'] = 40
    gi['phi'] = 0.0 * 2 * np.pi
    gi['period_days'] = 300
    gi['y_squeeze'] = 0.4
    gi['tilt'] = 0.1 * 2 * np.pi  # USE TILT CLOSE TO PARENT: It won't be a big issue when y_squeeze = 0.1
    gi['scale'] = 0.2

    return gi



