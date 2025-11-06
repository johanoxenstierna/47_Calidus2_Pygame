

import numpy as np
import P

from src.helpers import *
from src.objects.rocket_helpers import *
from src.objects.abstract_pygame_rocket import AbstractPygameRocket

class Rocket2(AbstractPygameRocket):
    """
    Deterministic time/placement layer for a rocket1 path.
    - Input: a rocket1.Rocket instance with xy0_C (central-frame path), p0, p1, destination_type, init_frame.
    - Output: .xy (display coords), plus .xy0 (heliocentric circular coords).
    - Case implemented: 'orbit' (parent–child). 'inter' is stubbed for a later pass.
    - No PID, no y_squeeze/tilt (handled later), no exact phase-matching/coast yet (k=0 for now).
    """

    def __init__(self, gi, p0, p1, xy0_notime):
        AbstractPygameRocket.__init__(self)
        self.DISPLAY = 1  # Just frame_ss xy BOTH ARE OVERWRITTEN

        self.id = 'Not done yet (maybe not needed)'
        self.gi = gi
        self.zorder = 2000
        self.type = 'rocket'
        self.p0 = p0
        self.p1 = p1
        self.destination_type = gi['destination_type']

        self.C = self.p0  # this stuff is used here cuz we only do tranfers between top parents in Rocket2
        if self.p0.parent is self.p1:
            self.C = self.p1

        # storage
        self.xy0_notime = xy0_notime
        self.xy0 = np.zeros((0, 2), dtype=np.float32)  # heliocentric xy0 (Sun-centered)


    def gen_rocket_motion(self):

        self.xy0 = self.xy0_time_sampled(self.xy0_notime)

        # ONLY FOR DISPLAY (NOT USED FOR OTHER ROCKETS) ================
        if self.DISPLAY == True:
            # if len(self.xy0) > P.FRAMES_STOP:
            #     raise Exception(f"len(xy0) = {len(self.xy0)} Decrease period_frames")

            i_range = range(0, len(self.xy0))
            xy0 = np.copy(self.xy0)
            if self.destination_type == 'orbit':
                C_helio = helio_xy0_over(self.C, i_range)
                xy0 = (xy0 + C_helio).astype(np.float32)

            self.xy = xy0 + np.array([960, 540], dtype=np.float32)
            self.frame_ss = [0, len(xy0)]

        return self

    def xy0_time_sampled(self, xy0_notime):
        """
        Populate xy0 (heliocentric)
        Implements:
          - 'orbit' (parent–child) using central-body per-frame placement
          - 'inter'  (heliocentric leg; e.g., Earth -> Jupiter)
        """

        if self.destination_type == 'orbit':

            # This is RELATIVE speed so no worry about the zero.
            speed0 = self.p0.speed_xy0 if (self.p0.parent is self.C) else 0.0
            speed1 = self.p1.speed_xy0 if (self.p1.parent is self.C) else 0.0

        else: # xy0_C from rocket1 is already HELIOCENTRIC in the inter case (built from top-parent radii),
            P0_top = top_parent(self.p0)  # e.g., Earth if p0 is Moon or Earth
            P1_top = top_parent(self.p1)  # e.g., Jupiter if p1 is Io or Jupiter

            speed0 = P0_top.speed_xy0
            speed1 = P1_top.speed_xy0

        if speed0 < 1e-6:
            speed0 = max(speed1 * 0.25, 0.1)

        # Arc-length + v-profile guardrails
        speed_profile = build_speed_profile(xy0_notime, speed0, speed1)
        xy0_C_time = resample_by_time(xy0_notime, speed_profile)  # HENCE,
        xy0 = xy0_C_time.astype(np.float32)

        return xy0

