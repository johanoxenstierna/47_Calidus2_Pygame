

import P
from src.objects.abstract_pygame import AbstractPygameObject
from src.helpers_distributions import *
from src_preproc import image2_bank
from src.load_save_pics import load_pics_DL


class O1C(AbstractPygameObject):

    """
    O1 objects are everything that is orbiting the sun, so planets, moons and asteroid belt (astro0).
    'calidus_astro' represent sun blends and flares that behave in a similar way as astro0 (png's with alpha=0 in the middle).
    """

    def __init__(_s, o1_id, gi, pics, parent, type):
        AbstractPygameObject.__init__(_s)

        _s.id = o1_id
        _s.gi = gi
        _s.drawn = 1  # begins as drawn
        _s.zorder = None
        _s.type = type
        _s.parent = parent  # parent
        _s.children = []  # added in gen_objects currently
        _s.pic = pics[0]  # the png
        _s.pics = pics
        _s.radius = [_s.pic.shape[0] / 2, _s.pic.shape[1] / 2]  # if scale not changed?
        if _s.type in ['body']:
            _s.radiuss = np.full((P.N_ORBIT_SAMPLES,), fill_value=_s.pic.shape[0] / 2).astype(np.float32)
        # else:
        #     _s.radiuss = np.full((P.FRAMES_STOP,), fill_value=_s.pic.shape[0] / 2).astype(np.float32)

        # _s.moons = {}

        _s.xy0 = None
        _s.xy2 = None

        _s.zorders = None
        _s.alphas = None
        _s.scale = None

        # _s.alphas_DL = []
        # _s.zorders_DL = []
        _s.DL = []
        _s.frame_ss = [0, P.FRAMES_STOP]

        # ONLY 4_Earth
        _s.years_days = None
        _s.ax_year_day = None

    def gen_orbit(_s):

        """
        Naming convention REFACTORED
        NOW local-only (orbits centered on 0, 0) here; parent composition happens at draw time:

        xy0: circular coordinates centered on (0, 0) (heliocentric) – STORED in _s (for rockets and debug)
        xy1: xy0 after y_squeeze – not stored in _s
        xy2: xy1 after tilt – STORED in _s
        xy2_abs: computed at draw/update as parent.xy_abs + xy_local[i_orbit] – not stored here
        xy0_abs: Used by rocket1 (Hohmann rockets) by combining xy0's (if it's a moon).
        """

        # tot_dist = _s.gi['speed_gi'] * P.FRAMES_TOT_BODIES
        # num_rot = tot_dist / 6000  # 6_Jupiter: num_rut=1 when speed_mult=1 & 4000 frames

        # if _s.id == '4_Earth':
        #     _s.years_days = _s._years_days(num_rot)
        #
        # if P.REAL_SCALE:  # ???
        #     pdf = -np.log(np.linspace(1, 100, 100))
        #     pdf += abs(min(pdf))
        #     pdf = min_max_normalize_array(pdf, y_range=[10, 40])

        _s.alphas = np.full((P.N_ORBIT_SAMPLES,), fill_value=0)
        _s.zorders = np.full((P.N_ORBIT_SAMPLES,), dtype=int, fill_value=1)

        # # Generate the elliptical motion for the planet
        # MOVE TO infos
        # y_squeeze = 0.08
        # if _s.id in ['Astro0b', '6_Jupiter']:
        #     y_squeeze = 0.15
        # elif _s.id in ['6_Jupiter']:
        #     y_squeeze = 0.3
        #     if P.REAL_SCALE == 1:
        #         y_squeeze = 0.15
        # elif _s.id in ['Saturn', 'Uranus', 'Neptune']:
        #     y_squeeze = 0.35
        #     if P.REAL_SCALE == 1:
        #         y_squeeze = 0.15

        # thetas = np.linspace(0 + _s.gi['phi'], num_rot * 2 * np.pi + _s.gi['phi'], P.FRAMES_TOT_BODIES)
        # thetas = _s.gi['phi'] + np.arange(P.FRAMES_TOT_BODIES) * _s.gi['omega']
        phis = _s.gi['phi'] + np.linspace(0, 2 * np.pi, P.N_ORBIT_SAMPLES, endpoint=False)
        _s.xy0 = np.stack([np.sin(phis) * _s.gi['r'], -np.cos(phis) * _s.gi['r']], axis=1, dtype=np.float32)
        _s.speed_xy0 = _s.compute_speed()

        # xy1 = squeezed
        xy1 = np.copy(_s.xy0)
        xy1[:, 1] *= _s.gi['y_squeeze']

        # xy2 = tilted. OBS this same functionality is found in rotate_xy() in rocket_helpers
        cos_phi = np.cos(_s.gi['tilt'])
        sin_phi = np.sin(_s.gi['tilt'])
        x_rot = cos_phi * xy1[:, 0] - sin_phi * xy1[:, 1]
        y_rot = sin_phi * xy1[:, 0] + cos_phi * xy1[:, 1]

        _s.xy2 = np.empty_like(xy1)
        _s.xy2[:, 0] = x_rot
        _s.xy2[:, 1] = y_rot

        # WTF PEND DEL
        # _s.vxy = np.gradient(_s.xy0, axis=0)  # ONLY USED FOR ZORDER
        # inds_neg = np.where(_s.vxy[:, 0] >= 0)[0]  # moving right  OBS ONLY WORKS FOR ageWISE THEN
        # # inds_neg = np.where(_s.xy0[:, 0] < 0)[0]  # DOESNT WORK BECAUSE IT JUST DOESNT
        # _s.zorders[inds_neg] *= -1
        # _s.zorders *= _s.gi['r']  # this is crude, but it prevents rockets from going in and out of planets
        # # _s.zorders += _s.parent.zorders  # HERE PROBABLY NEED TO ADD I_MOVE SOMEHOW

        front = _s.xy0[:, 1] > 0  # y-down screen: >0 is front
        # asdf = np.where(front, 1, -1)  # put 1 where true and -1 otherwise
        _s.zorders = (np.where(front, 1, -1) * int(_s.gi['r'])).astype(int)

        if _s.parent.id == '0':  # NON GSS/4_Moon
            _s.scale = np.copy(_s.xy0[:, 1])
            _s.scale = min_max_normalize_array(_s.scale, y_range=[0.8 * _s.gi['scale'], _s.gi['scale']])
        else:
            _s.scale = np.full((P.N_ORBIT_SAMPLES,), fill_value=_s.gi['scale'])  # GSS, 4_Moon

        _s.radiuss *= _s.scale

        if P.GEN_DL_PIC_BANK == '9_Neptune':  #_s.id:
            image2_bank.build_and_save_bank(_s)
            if P.USE_DL:  # MAYBE TEMP...
                _s.pics = load_pics_DL(f'./pictures/bodies/{_s.id}/')  #
            # exit()

    def gen_calidus_astro(_s, pi_offset_distr):

        # _s.xy = np.zeros((P.N_ORBIT_SAMPLES, 2))
        # _s.xy[:, 0] += _s.parent.xy[:, 0] #- _s.radius[1] // 2
        # _s.xy[:, 1] += _s.parent.xy[:, 1] #- _s.radius[0] // 2
        # _s.xy = _s.parent.xy0_abs

        # _s.zorders = np.full((P.FRAMES_STOP,), dtype=np.uint16, fill_value=_s.gi['zorder'])
        _s.zorder = _s.gi['zorder']
        # _s.scale = np.full((P.FRAMES_STOP,), fill_value=_s.gi['scale'])
        _s.scale = _s.gi['scale']
        # _s.radiuss *= _s.scale

        tot_dist = _s.gi['speed_gi'] * P.FRAMES_STOP
        num_alpha = tot_dist / 2000 * P.SPEED_MULTIPLIER
        num_rot = tot_dist / 4000 * P.SPEED_MULTIPLIER

        _s.alphas = 0.5 * (np.sin(np.linspace(pi_offset_distr, pi_offset_distr + num_alpha * 2 * np.pi, P.FRAMES_STOP)) + 1).astype(np.float32)
        _s.alphas = min_max_normalize_array(_s.alphas, y_range=[_s.gi['min_alpha'], _s.gi['max_alpha']])

        # _s.rotation = -np.linspace(pi_offset_distr, pi_offset_distr + num_rot * 2 * np.pi, P.FRAMES_TOT_BODIES)
        _s.rotation = -np.linspace(0, num_rot * np.pi, P.FRAMES_STOP, dtype=np.float32)

        if _s.id == '0_black':
            _s.rotation = np.full((P.FRAMES_STOP,), fill_value=0)
            _s.alphas = np.full((P.FRAMES_STOP,), fill_value=1)
        # _s.rotation = np.full((P.FRAMES_TOT_BODIES,), fill_value=0)

    def gen_astro0(_s):

        """
        OBS A LOT OF THIS IS SET DIRECTLY IN ani_helpers
        """

        # _s.xy = np.zeros((P.FRAMES_TOT_BODIES, 2))
        # _s.xy[:, 0] += _s.parent.xy[:, 0]  # - _s.radius[1] // 2
        # _s.xy[:, 1] += _s.parent.xy[:, 1]  # - _s.radius[0] // 2

        # _s.zorders = np.full((P.FRAMES_TOT_BODIES,), dtype=int, fill_value=_s.gi['zorder'])
        # _s.scale = np.full((P.FRAMES_TOT_BODIES,), fill_value=_s.gi['scale'])

        _s.zorder = _s.gi['zorder']
        _s.scale = _s.gi['scale']

        tot_dist = _s.gi['speed_gi'] * P.FRAMES_STOP
        num_rot = tot_dist / 4000 * P.SPEED_MULTIPLIER

        # _s.alphas = np.full((P.FRAMES_STOP,), fill_value=0.99)
        # _s.alpha = 255
        _s.rotation = np.linspace(0, num_rot * np.pi, P.FRAMES_STOP, dtype=np.float32)

    def _years_days(_s, num_rot):
        
        # ads
        days = np.linspace(1, int(num_rot * 365), P.FRAMES_TOT_BODIES, dtype=np.int32)

        years_days = []

        for d in days:
            years = d // 365
            rem_days = d % 365
            if years < 1:
                years_days.append(f"Day {rem_days}")
            else:
                years_days.append(f"Year {years} Day {rem_days}")

        return years_days

    def get_i_orbit(_s, i):
        """
        Return index/indices into the one-orbit sample table (length = P.N_ORBIT_SAMPLES).
        - If i is an int: return a single int index.
        - If i is a range: return a list of indices, one per frame in the range.
        OBS: independent of phi/tilt; this is purely the circular baseline (xy0).

        A = (i % period_frames)  # just i until 2048, then restart
        B = A / period_frames  # ratio complete, then multiply with 2048
        """
        period_frames = _s.gi['period_days'] / P.SPEED_MULTIPLIER
        f = lambda t: int(((t % period_frames) / period_frames) * P.N_ORBIT_SAMPLES)

        if isinstance(i, int):
            return f(i)
        elif isinstance(i, range):
            return [f(t) for t in i]
        else:
            raise TypeError("get_i_orbit expects int or range")

    def compute_speed(_s):
        # frames per full orbit, consistent with get_i_orbit()
        period_frames = _s.gi['period_days'] / P.SPEED_MULTIPLIER
        # constant pixels/frame along the circular xy0 path
        return (2 * np.pi * _s.gi['r']) / period_frames




