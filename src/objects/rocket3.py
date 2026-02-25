

import numpy as np
from scipy.stats import beta

from src.helpers_distributions import *
from src.helpers import *
import P
from src.objects.rocket_helpers import *
from src.objects.abstract_pygame_rocket import AbstractPygameRocket

class Rocket3(AbstractPygameRocket):

    """Lightweight rocket"""

    def __init__(self, gi, p0, p1, init_frame):
        AbstractPygameRocket.__init__(self)

        self.init_frame = init_frame
        self.id = p0.id + '_' + p1.id
        # self.gi = gi  # general info  # maybe not needed: Just extract what's needed here in constructor
        self.p0 = p0  # body that rocket takes off from
        self.p1 = p1  # body that rocket lands on
        self.xy0_timezerotopi = gi['xy0_timezerotopi']  # Hohmann transfer p0 -> p1 altitudes with 0 -> pi on unit circle.
        self.destination_type = gi['destination_type']

        self.C = self.p0
        if self.p0.parent is self.p1:  # Currently this doesnt handle the inter + moon case (because it's custom-built)
            self.C = self.p1

        # self.xy0 = np.zeros((0, 2), dtype=np.float32)  # output
        # self.xy2 = np.zeros((0, 2), dtype=np.float32)  # output
        self.alphas = None
        self.zorders = np.zeros((0, 2), dtype=np.uint32)

    def gen_rocket_motion(self):
        """
        xy0_timezerotopi means that we have array xy '0' (0 means all body orbits are circular) with coordinates of a rocket
        going from orbit of body p0 to orbit body p1. If destination_type is 'orbit', one of the bodies is heliocentric and the other a moon
        and 'inter' means travel between two heliocentric bodies (note p0 and p1 may be moons,
        but for inter transfers xy0_timezerotopi uses their parents).
        'time' means that the array has been smoothed using a speed schedule 0->1 or 1->0
        and that len(xy0_timezerotopi) has been computed using these same speeds.
        'zerotopi' means two things. 1. That we follow a convention where we work with an ellipse focus around [0, 0]
        (needs verification). 2. That, in a 0->2pi rotation sense, the transfer always starts
        at pi, and then goes clockwise to the opposite side of the circle 0/2pi.
        On the screen, this translates to 'starts down and goes up'. Hence, xy0_timezerotopi only works correct in terms
        of hitting orbit altitudes, and in this function we rotate it etc. to get exact p0->p1 matching.

        """
        takeoff_frames = int(120 / P.SPEED_MULTIPLIER)  # WE NEED HARDCODED FOR COASTING/TRANSFER ROTATION, BUT RAND COULD BE ADDED HERE
        landing_frames = int(120 / P.SPEED_MULTIPLIER)  #

        # XY_TARGET -> PHI FOR ROTATION ===============================
        if self.destination_type == 'inter':
            """OBS currently this is 'from-start-to-start' and the coasting function makes sure p1 reached by matching
            the angle velocities of p0 and p1. Also strictly xy0. xy2 is handled in takeoff/landing"""
            p1tp = top_parent(self.p1)
            xy_target = p1tp.xy0[p1tp.get_i_orbit(self.init_frame + takeoff_frames)]  # HELIOCENTRIC
        else:
            """
            Orbit case.
            if self.C is not self.p1, then we have Earth -> Moon, And that WORKS without big changes because 
            EARTH is HELIOCENTRIC, so it's same as inter but simpler because we don't need coasting. 
            Hence, we cannot work with 'from-start-to-start' in Earth -> Moon BECAUSE WE DON'T DO ANY COASTING, 
            we don't need to do any speed matching etc, hence use of word 'arrival'.
            if self.C is self.p1 (Moon -> Earth), we need p0's position RELATIVE to p1 aka C. 
            So... in essence, it's just p0's position
            Crucially, coordinates in self.xy0_timezerotopi for this case ARE CENTERED ON THE PARENT PLANET, NOT THE SUN, 
            """
            # if self.C is not self.p1:  # Earth -> Moon
            if self.p0.parent.id == '0':  # Earth -> Moon
                frame_arrival = self.init_frame + takeoff_frames + len(self.xy0_timezerotopi)
                xy_target = self.p1.xy0[self.p1.get_i_orbit(frame_arrival)]  # HELIOCENTRIC
            elif self.p0.parent.id != '0':   # Moon -> Earth
                frame_start = self.init_frame + takeoff_frames
                xy_target = -self.p0.xy0[self.p0.get_i_orbit(frame_start)]  # CENTERED ON PARENT PLANET. FLIPPED. OBS invert WORKS CUZ ITS A SINGLE COORDINATE
            else:
                raise Exception("Something VERY weird")

        phi1 = phi_from_coord(xy_target)

        delta0, delta1 = phi1, phi1  # deltas are used to rotate xy0_timezerotopi

        # xy0_spiral0 = None
        xy0_coast = None
        # xy0_spiral1 = None

        # COASTING AND ROTATION ================================
        if self.destination_type == 'inter':
            """Note that phi0 is only used in this case and not in orbit, because in the orbit case one of the 
            bodies is [0, 0] so only one angle is enough."""
            p0tp = top_parent(self.p0)
            p0tp_xy_start = p0tp.xy0[p0tp.get_i_orbit(self.init_frame + takeoff_frames)]
            phi0 = phi_from_coord(p0tp_xy_start)
            # p0_xy_start = self.p0.xy0[self.p0.get_i_orbit(self.init_frame + takeoff_frames)]
            # phi0 = phi_from_coord(p0_xy_start)
            # eps = np.random.uniform(0.05, 0.2)
            # eps = 0.5
            xy0_coast, phi_burn = self.gen_coast(phi0, phi1)
            delta1 = phi_burn - np.pi  # OBS THIS IS BECAUSE xy0_timezerotopi STARTS AT pi AND NOT ZERO

        xy0_rotated = rotate_xy(self.xy0_timezerotopi, delta1)  # MIN_FRAMES = 60

        if self.destination_type == 'inter' and len(xy0_coast) > 10:  # THERE ARE CASED WHEN NO COAST FOUND (because a floor is used)
            B0 = min(80, len(xy0_coast) // 2 * 2, len(xy0_rotated) // 2 * 2)
            blend0 = crossfade_B_frames(xy0_coast, xy0_rotated, B0)
            # xy0_rotated = np.vstack((xy0_coast, xy0_rotated))  # DEBUG
            xy0_rotated = np.vstack((xy0_coast[:(len(xy0_coast) - B0 // 2)], blend0, xy0_rotated[(B0 // 2):]))

        xy2_rotated = self.y_squeeze_tilt(xy0_rotated)

        # FROM THIS POINT, EVERYTHING IS xy2 (unless debugging). It WILL run with P.XY01=0, but won't be correct======================
        # TAKEOFF ========================
        xy2_takeoff = self.takeoff(xy2_rotated, takeoff_frames)
        B1 = min(80, len(xy2_takeoff) // 2 * 2, len(xy0_rotated) // 2 * 2)
        blend1 = crossfade_B_frames(xy2_takeoff, xy2_rotated, B1)
        # xy2_rotated = np.vstack((xy2_takeoff, xy2_rotated))  # DEBUG
        xy2_rotated = np.vstack((xy2_takeoff[:len(xy2_takeoff) - B1 // 2], blend1, xy2_rotated[B1 // 2:]))

        # ================================

        # # LANDING =======================
        if self.destination_type == 'inter' and self.p1.parent.id != '0':  # moon
            xy2_rotated_cutoff = int(len(xy2_rotated) * 0.8)
            xy2_rotated = xy2_rotated[:xy2_rotated_cutoff, :]
            # v0 = slope_at_idx(xy2_rotated, len(xy2_rotated) - 1, side='before')
            i_range_landing = range(self.init_frame + len(xy2_rotated),
                                    self.init_frame + len(xy2_rotated) + landing_frames)
            p1xy2 = np.copy(self.p1.xy2[self.p1.get_i_orbit(i_range_landing)])  # [0, 0] is Jupiter
            p1tpxy2 = np.copy(self.p1.parent.xy2[self.p1.parent.get_i_orbit(i_range_landing)])  # [0, 0] is Sun
            p1xy2 += p1tpxy2  # [0, 0] is Sun
            B2 = 120
            blend2 = crossfade_B_frames(xy2_rotated, p1xy2, B2)
            xy2_rotated = np.vstack((xy2_rotated[:len(xy2_rotated) - B2 // 2], blend2, p1xy2[B2 // 2:]))

        ## ================================

        self.xy2 = xy2_rotated
        self.frame_ss = [self.init_frame, self.init_frame + len(self.xy2)]

        self.zorders = self.gen_zorders()
        self.alphas = self.gen_alpha(xy2_takeoff, xy0_coast, xy2_rotated)

        if self.destination_type == 'orbit':  # everything assumes [0, 0] center, and here [0, 0] is a planet (also for moon -> parent)
            i_range = range(self.frame_ss[0], self.frame_ss[0] + len(self.xy2))
            C_helio = helio_xy2_over(self.C, i_range)  # THIS INCLUDES SHIFTING WITH PARENT
            self.xy2 += C_helio
        self.xy = self.xy2 + np.array([960, 540], dtype=np.float32)

        return self

    def gen_coast(self, phi0, phi1, eps=0.05, alpha=1.5, k_max=1):
        """
        Pre-phasing coast at phasing radius r_p to align arrival phase for the Hohmann half-ellipse.
        Inputs:
          phi0, phi1 : phase (rad) at frame 0 in your screen convention (phi=0 at [0,-r])
          r0, r1     : source/target radii (pixels)
          w0, w1     : signed angular rates (rad/frame). Include CW/CCW in the sign.
          T_tr       : time-of-flight of the half-ellipse in frames (len(xy0_timezerotopi_rotated))
        Params:
          eps        : phasing offset (r_p = r0*(1±eps))
          alpha      : ω(r) scaling exponent; 1.5 ≈ Kepler, 1.0 looks fine visually too
            Exponent controlling how angular rate scales with radius during pre-phasing:
            ω(r) = sign(w0) * |w0| * (r0 / r) ** alpha
            - alpha = 1.5  → Kepler-like (ω ∝ r^(-3/2)); physically plausible.
            - alpha = 1.0  → linear “gamey” scaling; visually OK and simpler.
            - Larger alpha → bigger ω difference at r_p → shorter t_pre (but implies larger Δv).
            - Smaller alpha → smaller ω difference → longer t_pre (gentler).
          k_max      : wrap search range for minimal nonnegative t_pre
        Returns:
          xy0_coast  : (N,2) float32 points at r_p, starting at phi0 (screen basis).
                       Empty array => skip pre-coast (burn now).
        """

        # phi0 -= 0.5 * np.pi  # NO. phi is ALWAYS non-transformed. Below, theta is used instead for this.
        # phi1 -= 0.5 * np.pi

        p0tp = top_parent(self.p0)
        p1tp = top_parent(self.p1)

        # r0 = self.p0.gi['r']
        # r1 = self.p1.gi['r']
        # w0 = self.p0.gi['w']
        # w1 = self.p1.gi['w']

        r0 = p0tp.gi['r']
        r1 = p1tp.gi['r']
        w0 = p0tp.gi['w']
        w1 = p1tp.gi['w']

        T_tr = len(self.xy0_timezerotopi)

        def wrap2pi(x):  # to do with 0-2pi convention
            y = np.fmod(x, 2 * np.pi)
            return y + 2 * np.pi if y < 0.0 else y

        # Phase error if we burned now
        Delta = wrap2pi(phi1 + w1 * T_tr - (phi0 + np.pi))  # in [0, 2π)

        # Try inside (faster) and outside (slower)
        r_candidates = []
        if r0 * (1.0 - eps) > 0:
            r_candidates.append(r0 * (1.0 - eps))
        r_candidates.append(r0 * (1.0 + eps))

        best = None  # (t_pre, r_p, w_p)
        for r_p in r_candidates:
            # angular rate at phasing radius (signed)
            w_p = w0 * (r0 / r_p) ** alpha
            denom = (w_p - w1)
            if abs(denom) < 1e-12:
                continue
            # allow negative/positive wraps to find smallest nonnegative t_pre
            for k in range(-k_max, k_max + 1):
                t_pre = (Delta + 2 * np.pi * k) / denom
                if t_pre >= 0.0:
                    if (best is None) or (t_pre < best[0]):
                        best = (t_pre, r_p, w_p)

        if best is None:
            # Graceful fallback: no pre-coast; burn now
            # return np.empty((0, 2), dtype=np.float32)
            raise Exception("No coast found: Try increasing k_max")

        t_pre, r_p, w_p = best

        num_frames = max(int(np.floor(t_pre)), 0)  # snap to an integer number of frames
        if num_frames == 0:
            phi_burn = (phi0) % (2 * np.pi)  # burn now; no pre-coast
            return np.empty((0, 2), dtype=np.float32), phi_burn

        t = np.arange(num_frames)
        angles_phi = (phi0 + w_p * t) % (2 * np.pi)  # screen-basis φ(t)
        theta = phi_to_theta(angles_phi) # math-basis θ
        xy0_coast = np.column_stack((r_p * np.cos(theta), r_p * np.sin(theta))).astype(np.float32, copy=False)

        phi_burn = (phi0 + w_p * num_frames) % (2 * np.pi)  # phi angle at NEXT frame (start of ellipse)
        return xy0_coast, phi_burn

    def gen_zorders(self):

        zorders = np.full((len(self.xy2),), dtype=int, fill_value=10)
        vxy_t = np.gradient(self.xy2, axis=0)
        inds_neg = np.where(vxy_t[:, 0] >= 0)[0]  # SO, right motion is behind and left in front (clockwise)
        zorders[inds_neg] *= -10
        i_orbit_move_p0 = self.p0.get_i_orbit(range(self.frame_ss[0], self.frame_ss[1]))
        i_orbit_move_p0par = self.p0.parent.get_i_orbit(range(self.frame_ss[0], self.frame_ss[1]))

        if self.p0.parent.id == '0':
            zp0andpar = self.p0.zorders[i_orbit_move_p0] + 2000
        else:
            zp0andpar = self.p0.parent.zorders[i_orbit_move_p0par] + 2000

        zorders += zp0andpar
        return zorders

    def gen_alpha(self, xy2_takeoff, xy0_coast, xy_full):
        """
        OBS: Only works with the lengths of these arrays.
        """

        TEMP = 0

        # len_xy0_spiral0 = 0 if xy2_takeoff is None else len(xy2_takeoff)
        len_xy2_takeoff = len(xy2_takeoff)
        len_xy0_coast = 0 if xy0_coast is None else len(xy0_coast)
        len_xy_full = len(xy_full)
        # len_xy0_spiral1 = 0 if xy0_spiral1 is None else len(xy0_spiral1)

        alphas = np.full((len(xy_full),), fill_value=125, dtype=np.uint8)

        num_twinkle = np.random.randint(low=1, high=8)
        if self.destination_type == 'orbit':  # No coast
            # alphas[len(xy0_rotated) - len(xy0_spiral1):] = 255
            # return alphas
            num_twinkle = np.random.randint(low=1, high=3)

        inds = np.random.randint(low=10, high=len(xy_full) - 100, size=num_twinkle)
        for ind in inds:
            num_frames = np.random.randint(low=20, high=50)
            pdf = -beta.pdf(x=np.arange(0, num_frames), a=2, b=2, loc=0, scale=num_frames)
            pdf = min_max_normalize_array(pdf, y_range=[0, 125])
            alphas[ind:ind + num_frames] = pdf

        # alphas1 = np.full((len1,), fill_value=125, dtype=np.uint8)
        # if len(alphas2) > 30:
        #     """If significant burn then add alpha"""
        #     NUM = len2
        #     pdf = beta.pdf(x=np.arange(0, NUM), a=2, b=8, loc=0, scale=NUM)
        #     pdf = min_max_normalize_array(pdf, y_range=[125, 255]).astype(np.uint8)
        #     alphas2[:] = pdf

        # if TEMP:
        #     alphas[0:len_xy0_spiral0] = 255  # takeoff
            # alphas[len(xy0_spiral0):len(xy0_spiral0) + len(xy0_coast)] = 255  # coast
            # alphas[len_xy0_spiral0 + len_xy0_coast:len_xy0_rotated - len_xy0_spiral1] = 255  # transfer
            # alphas[len(xy0_rotated) - len(xy0_spiral1):] = 255  # landing

        # alphas = np.concatenate((alphas1, alphas2))

        return alphas

    def y_squeeze_tilt(self, xy0_rotated):

        """Only used for coasting and transfer and ONLY ON TOP PARENTS.
        takeoff/landing fix the rest
        """
        p0tp = top_parent(self.p0)
        p1tp = top_parent(self.p1)

        xy1 = np.copy(xy0_rotated)

        y_squeeze_avg = np.mean([p0tp.gi['y_squeeze'], p1tp.gi['y_squeeze']])
        xy1[:, 1] *= y_squeeze_avg

        tilt_avg = np.mean([p0tp.gi['tilt'], p1tp.gi['tilt']])
        xy2 = rotate_xy(xy1, delta=tilt_avg)

        return xy2

    def post_smoothing(self):
        #FINALIZE LANDING AND ADD POST_SMOOTHING
        pass

    def takeoff(self, xy2_rotated, takeoff_frames):

        """

        """

        i_range_takeoff = range(self.init_frame, self.init_frame + takeoff_frames)
        p0_takeoff = self.p0.xy2[self.p0.get_i_orbit(i_range_takeoff), :]  # ALWAYS AT p0

        # a -> b the target coords that takeoff go into. Assume [0, 0] start
        if self.destination_type == 'orbit' and self.p0.parent.id == '0':
            a, b = xy2_rotated[0, :], xy2_rotated[1, :]  # works because xy2_rotated starts at [0, 0]
        elif self.destination_type == 'orbit' and self.p0.parent.id != '0':  # xy2_rotated starts at moon where earth is [0, 0]
            a, b = xy2_rotated[0, :] - p0_takeoff[-1, :], xy2_rotated[1, :] - p0_takeoff[-1, :]
        elif self.destination_type == 'inter' and self.p0.parent.id == '0':  # xy2_rotated starts at planet where sun is [0, 0]
            # a, b = xy2_rotated[0, :], xy2_rotated[1, :]  # doesnt work because xy2_rotated doesnt start at [0, 0]
            a, b = xy2_rotated[0, :] - p0_takeoff[-1, :], xy2_rotated[1, :] - p0_takeoff[-1, :]  # ZERO CENTER ON P0 REQUIRED. WHEN? At the end of takeof
        elif self.destination_type == 'inter' and self.p0.parent.id != '0':  # xy2_rotated starts at moon where sun is [0, 0]

            p0tp_takeoff = self.p0.parent.xy2[self.p0.parent.get_i_orbit(i_range_takeoff), :]  # tp=top parent

            # a, b = xy2_rotated[0, :], xy2_rotated[1, :]  # relative to sun

            # a = xy2_rotated[0, :] - p0tp_takeoff[-1, :]  # relative to earth
            # b = xy2_rotated[1, :] - p0tp_takeoff[-1, :]

            a = xy2_rotated[0, :] - p0tp_takeoff[-1, :] - p0_takeoff[-1, :]  # relative to moon
            b = xy2_rotated[1, :] - p0tp_takeoff[-1, :] - p0_takeoff[-1, :]

        c, undersh_oversh = closest_point_to_origin_on_line(a, b)
        # print(f"roc.id: {self.id} undersh_oversh: {undersh_oversh} init_frame: {self.init_frame}")
        a_semi, b_semi, theta, u_major, u_minor = ellipse_from_abc(a, b, c)  # if it starts perpendicular we get circle
        xy2_takeoff = gen_xy_takeoff(a_semi, b_semi, u_major, u_minor, takeoff_frames, undersh_oversh, a, b, c)

        # i_range_takeoff = range(self.init_frame, self.init_frame + len(xy2_takeoff))
        # p0_takeoff = self.p0.xy2[self.p0.get_i_orbit(i_range_takeoff), :]  # ALWAYS AT p0

        if self.destination_type == 'orbit' and self.p0.parent.id == '0':
            pass
        elif self.destination_type == 'orbit' and self.p0.parent.id != '0':  # moon
            xy2_takeoff += p0_takeoff
        elif self.destination_type == 'inter' and self.p0.parent.id == '0':
            xy2_takeoff += p0_takeoff
        elif self.destination_type == 'inter' and self.p0.parent.id != '0':  # moon
            xy2_takeoff += self.p0.parent.xy2[self.p0.parent.get_i_orbit(i_range_takeoff), :]  # centers on earth instead of sun at [0, 0]
            xy2_takeoff += p0_takeoff  # without this it starts at earth
            # pass
        return xy2_takeoff

    def landing(self):
        pass
