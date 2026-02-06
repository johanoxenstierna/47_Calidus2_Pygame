
import numpy as np
from copy import deepcopy
import json

import P

from O0_info import sun_gi, R_gi
from O0_info.O1 import a_mercury_gi, b_venus_gi, c_earth_gi, d_mars_gi, f_jupiter_gi, z_astro0_gi, g_saturn_gi, h_uranus_gi, i_neptune_gi

def _genesis():

    '''
    Creates instance of each info and stores in dict
    '''

    USE_T = 1
    USE_SAVED_R = 1

    # UNTOUCHED REAL VALUES solar_system_info = {
    #     '2_Mercury': {'AU': 0.387, 'period_days': 88},
    #     '3_Venus': {'AU': 0.723, 'period_days': 225},
    #     '4_Earth': {'AU': 1.000, 'period_days': 365},
    #     '4_GSS': {'AU': 0.000282, 'period_days': 1},
    #     '4_Moon': {'AU': 0.00257, 'period_days': 27},  # around Earth
    #     '5_Mars': {'AU': 1.524, 'period_days': 687},
    #     '6_Jupiter': {'AU': 5.203, 'period_days': 4332},
    #     '6_Europa': {'AU': 0.00448, 'period_days': 4},  # around Jupiter
    #     '6_Ganymede': {'AU': 0.00715, 'period_days': 7},  # around Jupiter
    #     '6_Io': {'AU': 0.00282, 'period_days': 2},  # around Jupiter
    #     '9_Neptune': {'AU': 30, 'period_days': 60195}
    # }

    # solar_system_info = {
    #     '2_Mercury': {'AU': 0.387, 'period_days': 88},
    #     '3_Venus': {'AU': 0.723, 'period_days': 225},
    #     '4_Earth': {'AU': 1.000, 'period_days': 365},
    #     '4_GSS': {'AU': 0.20282, 'period_days': 120},
    #     '4_Moon': {'AU': 0.50257, 'period_days': 150},  # around Earth
    #     '5_Mars': {'AU': 1.524, 'period_days': 687},
    #     '6_Jupiter': {'AU': 5.203, 'period_days': 1332},
    #     '6_Europa': {'AU': 0.20448, 'period_days': 400},  # around Jupiter
    #     '6_Ganymede': {'AU': 0.40715, 'period_days': 170},  # around Jupiter
    #     '6_Io': {'AU': 0.30282, 'period_days': 120},  # around Jupiter
    # }

    # Settings override (overrides solar_system_info) =========
    T = {
        '2_Mercury':  {'r':  50, 'phi': 0 * 2 * np.pi, 'period_days':  150, 'y_squeeze': 0.15, 'tilt': 0.1 * np.pi, 'scale': 0.05},
        '3_Venus':    {'r': 100, 'phi': 0 * 2 * np.pi, 'period_days':  200, 'y_squeeze': 0.15, 'tilt': 0.1 * np.pi, 'scale': 0.15},
        '4_Earth':    {'r': 280, 'phi': 0 * 2 * np.pi, 'period_days':  365, 'y_squeeze': 0.15, 'tilt': 0.1 * np.pi, 'scale': 0.1},
        '4_GSS':      {'r':  30, 'phi': 0 * 2 * np.pi, 'period_days':  100, 'y_squeeze': 0.30, 'tilt': 0.2 * np.pi, 'scale': 0.4},
        '4_Moon':     {'r':  50, 'phi': 0 * 2 * np.pi, 'period_days':  200, 'y_squeeze': 0.15, 'tilt': 0.1 * np.pi, 'scale': 0.1},
        '4_NEA':      {'r':  80, 'phi': 0 * 2 * np.pi, 'period_days':  300, 'y_squeeze': 0.15, 'tilt': 0.1 * np.pi, 'scale': 0.2},
        '5_Mars':     {'r': 400, 'phi': 0 * 2 * np.pi, 'period_days':  700, 'y_squeeze': 0.15, 'tilt': 0.1 * np.pi, 'scale': 0.15},
        '6_Jupiter':  {'r': 900, 'phi': 0 * 2 * np.pi, 'period_days': 4332, 'y_squeeze': 0.20, 'tilt': 0.1 * np.pi, 'scale': 0.4},
        '6_Europa':   {'r':  60, 'phi': 0 * 2 * np.pi, 'period_days':  300, 'y_squeeze': 0.15, 'tilt': 0.1 * np.pi, 'scale': 0.07},
        '6_Ganymede': {'r':  50, 'phi': 0 * 2 * np.pi, 'period_days':  250, 'y_squeeze': 0.15, 'tilt': 0.15 * np.pi,'scale': 0.05},
        '6_Io':       {'r':  35, 'phi': 0 * 2 * np.pi, 'period_days':  200, 'y_squeeze': 0.20, 'tilt': 0.2 * np.pi, 'scale': 0.05},
        '9_Neptune':  {'r': 1300, 'phi': 0 * 2 * np.pi, 'period_days': 8000, 'y_squeeze': 0.40, 'tilt': 0.07 * np.pi, 'scale': 0.12}
    }

    # T = convert_to_project(T, solar_system_info)  # ONLY RELEVANT FOR REAL_SCALE

    gis = {}

    # if 'Calidus' in P.OBJ_TO_SHOW:  # EXPL
    gis['0_placeholder'] = sun_gi.sunplaceholder_gi_()
    gis['0_black'] = sun_gi.black_gi_()
    gis['0_sunshining'] = sun_gi.sunshining_gi_()

    if 'Sun' in P.OBJ_TO_SHOW:
        gis['0_red'] = sun_gi.red_gi_()
        gis['0_mid'] = sun_gi.mid_gi_()
        gis['0_light'] = sun_gi.light_gi_()
        gis['0h_red'] = sun_gi.h_red_gi_()
        gis['0h_mid'] = sun_gi.h_mid_gi_()
        gis['0h_light'] = sun_gi.h_light_gi_()

    if 'Astro0' in P.OBJ_TO_SHOW:
        gis['Astro0'] = z_astro0_gi.astro0_gi_()

    if 'Astro0b' in P.OBJ_TO_SHOW:
        gis['Astro0b'] = z_astro0_gi.astro0b_gi_()

    if USE_T:
        for key, val in T.items():
            gis[key] = val
    else:
        if '2_Mercury' in P.OBJ_TO_SHOW:
            gis['2_Mercury'] = a_mercury_gi.mercury_gi_()

        if '3_Venus' in P.OBJ_TO_SHOW:
            gis['3_Venus'] = b_venus_gi.venus_gi_()

        if '4_Earth' in P.OBJ_TO_SHOW:
            gis['4_Earth'] = c_earth_gi.earth_gi_()

        if '4_GSS' in P.OBJ_TO_SHOW:
            gis['4_GSS'] = c_earth_gi.gss_gi_()

        if '4_Moon' in P.OBJ_TO_SHOW:
            gis['4_Moon'] = c_earth_gi.moon_gi_()

        if '4_NEA' in P.OBJ_TO_SHOW:
            gis['4_NEA'] = c_earth_gi.nea_gi_()

        if '5_Mars' in P.OBJ_TO_SHOW:
            gis['5_Mars'] = d_mars_gi.mars_gi_()

        if '6_Jupiter' in P.OBJ_TO_SHOW:
            gis['6_Jupiter'] = f_jupiter_gi.jupiter_gi_()

        if '6_Europa' in P.OBJ_TO_SHOW:
            gis['6_Europa'] = f_jupiter_gi.europa_gi_()

        if '6_Ganymede' in P.OBJ_TO_SHOW:
            gis['6_Ganymede'] = f_jupiter_gi.ganymede_gi_()

        if '6_Io' in P.OBJ_TO_SHOW:
            gis['6_Io'] = f_jupiter_gi.io_gi_()

        if 'Saturn' in P.OBJ_TO_SHOW:
            gis['Saturn'] = g_saturn_gi.saturn_gi_()

        if 'Uranus' in P.OBJ_TO_SHOW:
            gis['Uranus'] = h_uranus_gi.uranus_gi_()

        if '9_Neptune' in P.OBJ_TO_SHOW:
            gis['9_Neptune'] = i_neptune_gi.neptune_gi_()

    gis['Rockets'] = None  # to avoid incl_frames error.
    if 'Rockets' in P.OBJ_TO_SHOW:
        if USE_SAVED_R == 0:
            gis['Rockets'] = R_gi.R_gi_()
            gis['Rockets'] = R_gi.translate(gis['Rockets'])
            with open('./O0_info/R_gi_save.json', 'w') as f:
                json.dump(gis['Rockets'], f, indent=4)
        else:
            with open('./O0_info/R_gi_save_2.json', 'r') as f:
                gis['Rockets'] = json.load(f)

    for gi_id, gi in gis.items():
        if type(gi) is dict:  # non rocket
            gi_keys = gi.keys()

            if USE_T:
                if gi_id in T:
                    for gi_key in gi_keys:
                        gi[gi_key] = T[gi_id][gi_key]

            if 'period_days' in gi_keys:
                period_frames = gi['period_days'] / P.SPEED_MULTIPLIER
                gi['w'] = -w_from_period_frames(period_frames)

    return gis


def w_from_period_frames(period_frames, cw=True):
    """Radians per frame"""
    s = -1.0 if cw else 1.0
    return s * (2.0 * np.pi / period_frames)  # rad/frame


def convert_to_project(T, solar_system_info, r_jupiter=800):
    """
    Return a copy of T with 'r' and 'period_days' updated from solar_system_info.

    - Distances use AU from Sun for planets and AU from parent for moons (as given).
    - All radii are scaled so that Jupiter's AU maps to r_jupiter pixels.
    """
    T2 = deepcopy(T)
    au_jup = float(solar_system_info['6_Jupiter']['AU'])

    for name, params in T2.items():
        info = solar_system_info.get(name)
        if not info:
            continue  # skip entries not in the info dict

        # AU -> px using Jupiter as the baseline
        r_px = (float(info['AU']) / au_jup) * float(r_jupiter)
        params['r'] = int(round(r_px))

        # Period in Earth days (or around parent for moons)
        params['period_days'] = info['period_days']

    return T2


def gen_incl_frames(R_gi: list) -> np.ndarray:

    """Generate animation-speed array from rocket schedule."""
    if P.SKIP_FRAMES == 0:
        return np.arange(0, P.FRAMES_STOP, dtype=np.uint32)

    max_speed = 1000.0

    aspeed = np.full(P.FRAMES_STOP, fill_value=max_speed, dtype=np.float32)

    # constants (can be tuned)
    ramp_len = 10000
    stay_len = 300

    # sort events by init_frame
    init_frames = sorted([d["init_frame"] for d in R_gi])

    ramp_up = np.linspace(1, max_speed, ramp_len)
    ramp_do = np.linspace(max_speed, 1, ramp_len)

    sink = np.ones((ramp_len * 2 + stay_len,), dtype=np.float32)
    sink[0:ramp_len] = ramp_do
    sink[ramp_len + stay_len:] = ramp_up

    # OBS earth will use 730 frames for 1 rotation if SPEED_MULTIPLIER = 0.5
    aspeed[0:365] = np.ones((365,))  # example: 1 full rotation at speed 1
    aspeed[365:365 + ramp_len] = ramp_up

    for i in init_frames:
        i_start = i - ramp_len
        i_end = i + stay_len + ramp_len
        aspeed[i_start:i_end] = sink


    incl = []
    t = 0.0
    while t < P.FRAMES_STOP:
        incl.append(int(t))
        _aspeed = aspeed[int(t)]
        t += _aspeed  # advance by current speed

    incl = np.array(incl, dtype=np.uint32)
    # optional sentinel
    incl = np.append(incl, np.uint32(P.FRAMES_STOP + 1))  # instead of assertion

    return incl
