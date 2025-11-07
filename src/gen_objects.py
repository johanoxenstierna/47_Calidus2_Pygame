from scipy.stats import beta
from src.helpers_distributions import *

import P as P
from src.objects.o0 import O0C
from src.objects.o1 import O1C
from src.objects.rocket1 import Rocket1
from src.objects.rocket2 import Rocket2
from src.objects.rocket3 import Rocket3
from src.load_save_pics import *
from src.objects.rocket_helpers import jittered_range


class GenObjects:

    """
    """

    def __init__(self):

        self.pics = {}
        load_pics(self.pics, './pictures/backgr/')
        load_pics(self.pics, './pictures/bodies1/')  # bank generation AND Astro0
        load_pics(self.pics, './pictures/0_sun/')

        load_pics_bodies(self.pics)

        self.gis = None
        self._years_days = None

    def gen_base_object(self):
        """
        Base objects.
        """

        # for o0_id in P.O0_TO_SHOW:  # number_id
        o0_gi = self.gis['0_placeholder']
        pic = self.pics['0_red']  # HAVE TO to get radius
        # O0[o0_id] = O0C(pic=None, gi=o0_gi)  # No pic CURRENTLY
        o0 = O0C(pic=pic, gi=o0_gi)  # NEEDS pic for radius

        # pi_offset_distr = [0, 0.3333 * 2 * np.pi, 0.6666 * 2 * np.pi]  # v0
        pi_offset_distr = [0.3333 * 2 * np.pi, 0.6666 * 2 * np.pi, 0]  # v1

        if P.REAL_SCALE == 0:
            pic_black = self.pics['0_black']
            gi = self.gis['0_black']
            o1 = O1C(o1_id='0_black', gi=gi, pics=[pic_black], parent=o0, type='0_static')  # THE PIC IS ALWAYS TIED TO 1 INSTANCE?
            o1.zorder = gi['zorder']
            o1.alpha = gi['alpha']
            o1.scale = gi['scale']
            o0.O1['0_black'] = o1

        pic_sun = self.pics['0_sunshiningTilted']
        gi = self.gis['0_sunshining']
        o1 = O1C(o1_id='0_sunshining', gi=gi, pics=[pic_sun], parent=o0, type='0_static')  # THE PIC IS ALWAYS TIED TO 1 INSTANCE?
        o1.zorder = gi['zorder']
        o1.alpha = gi['alpha']
        if 'Sun' not in P.OBJ_TO_SHOW and P.REAL_SCALE == False:
            o1.alpha = 0.1
        o1.scale = gi['scale']
        o0.O1['0_sunshining'] = o1

        if 'Sun' in P.OBJ_TO_SHOW:

            pic_red = self.pics['0_red']
            gi = self.gis['0_red']
            o1 = O1C(o1_id='0_red', gi=gi, pics=[pic_red], parent=o0, type='0_')  # THE PIC IS ALWAYS TIED TO 1 INSTANCE?
            o1.gen_calidus_astro(pi_offset_distr[0])
            o0.O1['0_red'] = o1

            pic_mid = self.pics['0_mid']
            gi = self.gis['0_mid']
            o1 = O1C(o1_id='0_mid', gi=gi, pics=[pic_mid], parent=o0, type='0_')
            o1.gen_calidus_astro(pi_offset_distr[1])
            o0.O1['0_mid'] = o1

            pic_light = self.pics['0_light']
            gi = self.gis['0_light']
            o1 = O1C(o1_id='0_light', gi=gi, pics=[pic_light], parent=o0, type='0_')
            o1.gen_calidus_astro(pi_offset_distr[2])
            o0.O1['0_light'] = o1

            pic_0h_red = self.pics['0h_red']
            gi = self.gis['0h_red']
            o1 = O1C(o1_id='0h_red', gi=gi, pics=[pic_0h_red], parent=o0, type='0_')
            o1.gen_calidus_astro(pi_offset_distr[0])
            o0.O1['0h_red'] = o1

            pic_0h_mid = self.pics['0h_mid']
            gi = self.gis['0h_mid']
            o1 = O1C(o1_id='0h_mid', gi=gi, pics=[pic_0h_mid], parent=o0, type='0_')
            o1.gen_calidus_astro(pi_offset_distr[1])
            o0.O1['0h_mid'] = o1

            pic_0h_light = self.pics['0h_light']
            gi = self.gis['0h_light']
            o1 = O1C(o1_id='0h_light', gi=gi, pics=[pic_0h_light], parent=o0, type='0_')
            o1.gen_calidus_astro(pi_offset_distr[2])
            o0.O1['0h_light'] = o1

        return o0

    def gen_planets_moons(self, o0):
        """
        """

        # time0 = time.time()

        if '2_Mercury' in P.OBJ_TO_SHOW:
            gi = self.gis['2_Mercury']
            pics = self.pics['2_Mercury']
            o1mercury = O1C(o1_id='2_Mercury', gi=gi, pics=pics, parent=o0, type='body')  # THE PIC IS ALWAYS TIED TO 1 INSTANCE?
            o1mercury.gen_orbit()
            o0.O1['2_Mercury'] = o1mercury

        if '3_Venus' in P.OBJ_TO_SHOW:
            gi = self.gis['3_Venus']
            pics = self.pics['3_Venus']
            o1venus = O1C(o1_id='3_Venus', gi=gi, pics=pics, parent=o0, type='body')  # THE PIC IS ALWAYS TIED TO 1 INSTANCE?
            o1venus.gen_orbit()
            o0.O1['3_Venus'] = o1venus

        if '4_Earth' in P.OBJ_TO_SHOW:
            gi = self.gis['4_Earth']
            pics = self.pics['4_Earth']
            o1earth = O1C(o1_id='4_Earth', gi=gi, pics=pics, parent=o0, type='body')  # THE PIC IS ALWAYS TIED TO 1 INSTANCE?
            o1earth.gen_orbit()
            # o1nauvis.gen_DL()
            o1earth.children = ['4_Moon', '4_GSS', '4_NEA']
            o0.O1['4_Earth'] = o1earth

            for child_id in o1earth.children:
                if child_id in P.OBJ_TO_SHOW:
                    gi = self.gis[child_id]
                    pics = self.pics[child_id]
                    _o1 = O1C(o1_id=child_id, gi=gi, pics=pics, parent=o1earth, type='body')  # THE PIC IS ALWAYS TIED TO 1 INSTANCE?
                    _o1.gen_orbit()
                    # _o1.gen_DL()
                    o0.O1[child_id] = _o1

        if '5_Mars' in P.OBJ_TO_SHOW:
            gi = self.gis['5_Mars']
            pics = self.pics['5_Mars']
            o1mars = O1C(o1_id='5_Mars', gi=gi, pics=pics, parent=o0, type='body')  # THE PIC IS ALWAYS TIED TO 1 INSTANCE?
            o1mars.gen_orbit()
            o0.O1['5_Mars'] = o1mars

        if '6_Jupiter' in P.OBJ_TO_SHOW:
            gi = self.gis['6_Jupiter']
            pics = self.pics['6_Jupiter']  # TEMP

            o1jupiter = O1C(o1_id='6_Jupiter', gi=gi, pics=pics, parent=o0, type='body')  # THE PIC IS ALWAYS TIED TO 1 INSTANCE?
            o1jupiter.gen_orbit()
            # o1jupiter.gen_DL()
            o1jupiter.children = ['6_Europa', '6_Ganymede', '6_Io']
            o0.O1['6_Jupiter'] = o1jupiter

            for child_id in o1jupiter.children:
                if child_id in P.OBJ_TO_SHOW:
                    # _o1 = None
                    gi = self.gis[child_id]
                    pics = self.pics[child_id]
                    _o1 = O1C(o1_id=child_id, gi=gi, pics=pics, parent=o1jupiter, type='body')  # THE PIC IS ALWAYS TIED TO 1 INSTANCE?
                    _o1.gen_orbit()
                    o0.O1[child_id] = _o1

        if 'Astro0' in P.OBJ_TO_SHOW:
            gi = self.gis['Astro0']
            pic_planet = self.pics['z_Astro0_masked']
            o1astro0 = O1C(o1_id='Astro0', gi=gi, pics=[pic_planet], parent=o0, type='astro')  # THE PIC IS ALWAYS TIED TO 1 INSTANCE?
            o1astro0.gen_astro0()
            o0.O1['Astro0'] = o1astro0

        if 'Astro0b' in P.OBJ_TO_SHOW:
            gi = self.gis['Astro0b']
            pics = self.pics['Astro0b']
            _o1 = O1C(o1_id='Astro0b', gi=gi, pics=pics, parent=o0, type='body')  # THE PIC IS ALWAYS TIED TO 1 INSTANCE?
            _o1.gen_orbit()
            # _o1.gen_DL()
            o0.O1['Astro0b'] = _o1

        if 'Saturn' in P.OBJ_TO_SHOW:
            gi = self.gis['Saturn']
            pics = self.pics['Saturn']
            o1saturn = O1C(o1_id='Saturn', gi=gi, pics=pics, parent=o0, type='body')  # THE PIC IS ALWAYS TIED TO 1 INSTANCE?
            o1saturn.gen_orbit()
            o1saturn.gen_DL()
            o0.O1['Saturn'] = o1saturn

        if 'Uranus' in P.OBJ_TO_SHOW:
            gi = self.gis['Uranus']
            pics = self.pics['Uranus']
            o1uranus = O1C(o1_id='Uranus', gi=gi, pics=pics, parent=o0, type='body')  # THE PIC IS ALWAYS TIED TO 1 INSTANCE?
            o1uranus.gen_orbit()
            o1uranus.gen_DL()
            o0.O1['Uranus'] = o1uranus

        if 'Neptune' in P.OBJ_TO_SHOW:
            gi = self.gis['Neptune']
            pics = self.pics['Neptune']
            o1neptune = O1C(o1_id='Neptune', gi=gi, pics=pics, parent=o0, type='body')  # THE PIC IS ALWAYS TIED TO 1 INSTANCE?
            o1neptune.gen_orbit()
            o1neptune.gen_DL()
            o0.O1['Neptune'] = o1neptune

        return o0

    def gen_rockets(self, o0):
        """
        This includes some writing to gi. We cannot do this writing in genesis_infos.py because
        body objects are needed.
        """

        R = []

        print("Generating rocket solutions")
        for roc_gi in self.gis['Rockets']:

            p0 = o0.O1[roc_gi['od'][0]]
            p1 = o0.O1[roc_gi['od'][1]]

            if (p1.parent is p0) or (p0.parent is p1):
                roc_gi['destination_type'] = 'orbit'
            else:
                roc_gi['destination_type'] = 'inter'

            if 'xy0_timezerotopi' not in roc_gi:  # OBS ONLY PRODUCED ONCE PER OD
                rocket1 = Rocket1(gi=roc_gi, p0=p0, p1=p1).gen_rocket_motion()
                xy0_notime = rocket1.xy0
                rocket2 = Rocket2(gi=roc_gi, p0=p0, p1=p1, xy0_notime=xy0_notime).gen_rocket_motion()
                roc_gi['xy0_timezerotopi'] = rocket2.xy0

            # GOOD BUT CANT BE USED WHEN DEBUGGING CUZ 1/4 CHANCE ANY INIT_FRAME WILL BE USED==========
            if 'init_frames' not in roc_gi:
                init_frames = self.gen_init_frames(p0, p1, roc_gi)
                roc_gi['init_frames'] = init_frames
            else:
                if roc_gi['init_frames'][-1] != P.FRAMES_STOP:
                    raise Exception("pre-generating init_frames only works if P.FRAMES_STOP kept constant")
            roc_gi['init_frames'].pop()
            # roc_gi['init_frames'] = [60]
            # ===============

            for init_frame in roc_gi['init_frames']:
                rocket3 = Rocket3(gi=roc_gi, p0=p0, p1=p1, init_frame=init_frame).gen_rocket_motion()
                # R.append(rocket1)  # P0 SHOULD NOT MOVE, BECAUSE IT IS SHIFTED BY IT
                # R.append(rocket2)  # JUST ADDS TIME
                R.append(rocket3)
            print("generated a rocket od")

        R = sorted(R, key=lambda x:x.init_frame)

        print("\nPRODUCED init frames ===================")
        print("total: " + str(len(R)) + '\n')
        for k, roc in enumerate(R):
            print(f"roc.id: {roc.id}, init_frame: {roc.init_frame}, od={roc.p0.id, roc.p1.id}")

        return R
    
    def gen_init_frames(self, p0, p1, roc_gi):
        """
        Launch frames (init_frames) for a rocket going from body p0 to body p1.
        hPerR: 'average time distance in hours between each rocket launch'
        fdist_avg: average time distance in frames between each rocket launch.
        Using standard settings for the Factorio simulation (600 game hours = 6000 years),
        we get 1 hour in game = 10 Earth years IRL
        This leads to too few rockets per year, so we need to reduce
        We generate candidate init_frames evenly spaced.
        OBS This function is VERY inefficient as it produces rocket3's most of which won't be used.
        But it doesn't matter because it's doing a pre-generation that can then be re-used.
        """

        init_frames = []  # output

        # roc_gi['t'] = 20  # time of first launch (in game hours)
        # roc_gi['hPerR'] = 10

        ''' 
        
        Discussion on frame translations: see R_gi.py
        
        '''

        # init_frame0 = roc_gi['t'] * 10 * standard_earth_period_frames  # * 10 bcs 1h=10years
        # init_frame0 = int(roc_gi['init_frame'] / 1)  # without this there would be a bunch of rotations before init_frame
        # fdist_avg = roc_gi['hPerR'] * 10 * standard_earth_period_frames  # framesDist. 1 hour in game = 10 earth years
        # fdist_avg = int(roc_gi['fdist'] / 1)
        CANDS_FACTOR_DIVISOR = 4  #so 4x more candidates than used
        if roc_gi['destination_type'] == 'orbit':
            CANDS_FACTOR_DIVISOR = 1

        base_step = max(20, roc_gi['fdist'] // CANDS_FACTOR_DIVISOR)  # MAX 1 launch every 0.3 s

        # init_frame_cands0 = list(range(init_frame0, P.FRAMES_STOP, fdist_avg // CANDS_FACTOR))  # so 4x more candidates than used
        init_frame_cands0 = jittered_range(roc_gi['init_frame'], P.FRAMES_STOP, base_step=base_step, rand_step=base_step // 3)

        if roc_gi['destination_type'] == 'orbit':
            init_frames.append(P.FRAMES_STOP)
            return init_frame_cands0

        init_frame_cands1 = np.zeros((len(init_frame_cands0), 2), dtype=int)  # 0: init_frame 1: len of rocket
        init_frame_cands1[:, 0] = np.asarray(init_frame_cands0)

        for ii, init_frame in enumerate(init_frame_cands0):
            rocket3 = Rocket3(gi=roc_gi, p0=p0, p1=p1, init_frame=init_frame)  # INCL COASTING
            # if ii == 1:
            #     asdf = 5
            # try:
            rocket3.gen_rocket_motion()
            # except:
            #     print(f"DEBUG ii: {ii}")
            init_frame_cands1[ii, 1] = len(rocket3.xy2)

        if len(init_frame_cands1) > CANDS_FACTOR_DIVISOR:
            init_frames = self.prune_init_frames(init_frame_cands1, CANDS_FACTOR_DIVISOR)
        else:
            init_frames = init_frame_cands1[:, 0].tolist()

        init_frames.append(P.FRAMES_STOP)  # used for a guard to make sure P.FRAMES_STOP is kept when this is queried
        return init_frames

    def prune_init_frames(self, init_frame_cands, cands_factor):

        """
        :param init_frame_cands: array where col=0 gives an init_frame and col=1 gives an int
        cands_factor: the number of init_frames to be kept is 1/cand_factor
        :return: init_frames: list

        """

        k = init_frame_cands.shape[0] // cands_factor
        idx = np.argsort(init_frame_cands[:, 1])  # ascending
        init_frames0 = init_frame_cands[idx]
        init_frames1 = init_frames0[0:k, :]
        init_frames2 = init_frames1[:, 0].tolist()
        return sorted(init_frames2)

    # def gen_init_framesOLD(self, p0, p1, roc_gi, destination_type):
    #     """
    #     For rockets based on distances (dp0p1) and distance gradients between two planets.
    #     The sampling is beta-discrete from an argsort.
    #
    #     init_frames_cands ALLOWS all frames in planet motions. Cleaned up AFTER selection
    #     The ddist are argsorted, then samples are taken from this argsort.
    #     """
    #
    #     num_to_select = P.FRAMES_TOT // roc_gi['init_frame_step']
    #
    #     dist = -np.asarray([np.linalg.norm(p0.xy[i, :] - p1.xy[i, :]) for i in range(len(p0.xy))])
    #     dist_grad = np.gradient(dist, axis=0)  # MUST BE BASED ON unsorted
    #
    #     dist = min_max_normalize_array(dist, y_range=[0, 1])
    #     dist_grad = min_max_normalize_array(dist_grad, y_range=[0, 1])  # seems to work for neg vals
    #
    #     ddd = 0.3 * dist + 0.7 * dist_grad
    #     ddd = min_max_normalize_array(ddd, y_range=[0, 1])
    #
    #     ddd_inds_sorted = np.argsort(ddd)[::-1]  #
    #
    #     '''OBS this pdf gives the probability that a frame will be sampled based on ddd
    #     loc=30 means that the 30 frames when p0 and p1 are too close won't be sampled '''
    #     pdf_dist = beta.pdf(x=np.arange(0, len(p0.xy)), loc=30, a=2, b=6, scale= len(p0.xy))  # 2 6
    #     if destination_type == 'orbit':
    #         pdf_dist = beta.pdf(x=np.arange(0, len(p0.xy)), loc=30, a=2, b=2, scale=1 * len(p0.xy))  # 2 6
    #     pdf_dist /= np.sum(pdf_dist)  # only works for pos?
    #     # dist_inds_sorted_subset = np.random.choice(dist_inds_sorted, size=len(p0.xy) // 2, replace=False, p=pdf_dist)
    #     # dist_inds_sorted_subset = np.random.choice(dist_inds_sorted, size=num_to_select * 2, replace=False, p=pdf_dist)
    #     init_frames = np.sort(np.random.choice(ddd_inds_sorted, size=num_to_select, replace=False, p=pdf_dist))
    #     init_frames = init_frames[np.where(init_frames > 5)]
    #     # init_frames = init_frames[np.where(init_frames + 2000 + roc_gi['frames_max'] < P.FRAMES_TOT_BODIES)]
    #     init_frames = init_frames[np.where(init_frames + 2000 < P.FRAMES_TOT_BODIES)]
    #
    #     # min_distance_integers =
    #     # filtered = [arr[0]]  # Always keep the first element
    #     # for num in arr[1:]:
    #     #     if num - filtered[-1] >= 100:
    #     #         filtered.append(num)
    #
    #     init_frames = list(init_frames)
    #     return init_frames

