import P


def sunplaceholder_gi_():

    gi = {}
    gi['r'] = 0
    gi['phi'] = 0
    gi['speed_gi'] = 0

    return gi


def black_gi_():  # 20

    gi = {}
    gi['scale'] = 0.22
    gi['alpha'] = 0.99
    gi['zorder'] = 2000

    if P.REAL_SCALE:
        gi['scale'] = 0.03

    return gi


def sunshining_gi_():  # 20

    gi = {}
    gi['scale'] = 0.34
    gi['alpha'] = 0.99
    gi['zorder'] = 2001

    if P.REAL_SCALE:
        gi['scale'] = 0.02

    return gi


def h_mid_gi_():  # 20

    gi = {}
    gi['scale'] = 0.27
    gi['phi'] = 0
    gi['speed_gi'] = -2
    gi['tilt'] = 0
    gi['max_alpha'] = 0.55
    gi['min_alpha'] = 0.04
    gi['zorder'] = 2002

    return gi


def h_red_gi_():  # 20

    gi = {}
    gi['scale'] = 0.27
    gi['phi'] = 0
    gi['speed_gi'] = 4
    gi['tilt'] = 0
    gi['max_alpha'] = 0.45
    gi['min_alpha'] = 0.14
    gi['zorder'] = 2003

    return gi


def red_gi_():  # 20

    gi = {}
    gi['scale'] = 0.3
    gi['phi'] = 0
    gi['speed_gi'] = 4.5
    gi['tilt'] = 0
    gi['max_alpha'] = 0.3
    gi['min_alpha'] = 0.1
    gi['zorder'] = 2004

    return gi


def mid_gi_():  # 20

    gi = {}
    gi['scale'] = 0.3
    gi['phi'] = 0
    gi['speed_gi'] = -2
    gi['tilt'] = 0
    gi['max_alpha'] = 0.6
    gi['min_alpha'] = 0.1
    gi['zorder'] = 2005

    return gi


def light_gi_():  # 20

    gi = {}
    gi['scale'] = 0.3
    gi['phi'] = 0
    gi['speed_gi'] = 1
    gi['tilt'] = 0
    gi['max_alpha'] = 0.5
    gi['min_alpha'] = 0.1
    gi['zorder'] = 2006

    return gi


def h_light_gi_():  # 20

    gi = {}
    gi['scale'] = 0.26
    gi['phi'] = 0
    gi['speed_gi'] = 5
    gi['tilt'] = 0
    gi['max_alpha'] = 0.55
    gi['min_alpha'] = 0.2
    gi['zorder'] = 2007

    return gi