import numpy as np
from src.objects.rocket_helpers import *
from src.objects.abstract_pygame_rocket import AbstractPygameRocket


class Rocket1(AbstractPygameRocket):
    """
    rocket1: geometry-only Hohmann-like transfer in circular (xy0) space.
    - Builds the path in a central frame (Sun- or planet-centric) with the central body at a focus.
    - Outputs:
        _s.xy0_C  : path in the chosen central frame (debugging clarity)
        _s.xy0    : heliocentric (Sun-centered) circular coords for each frame
        _s.xy     : display coords (xy0 + [960, 540]) for P.XY01==0 viewing
    - No tilt/y_squeeze, no velocity shaping, no phase-matching.
    """

    def __init__(_s, gi, p0, p1):
        AbstractPygameRocket.__init__(_s)

        _s.DISPLAY = 1

        _s.init_frame = 0
        _s.id = 'Not done yet (maybe not needed)'
        _s.gi = gi
        _s.zorder = 2000
        _s.type = 'rocket'
        _s.p0 = p0
        _s.p1 = p1
        _s.destination_type = gi['destination_type']

        _s.C = _s.p0
        if _s.p0.parent is _s.p1:
            _s.C = _s.p1

        _s.xy0 = np.zeros((0, 2), dtype=np.float32)   # heliocentric xy0 (Sun-centered)

    # ---------- main ----------
    def gen_rocket_motion(_s):
        _s.xy0 = _s.xy0_hohmann()

        # ONLY FOR DISPLAY (NOT USED FOR OTHER ROCKETS) ================
        if _s.DISPLAY == True:  # serves as a temporary guard
            i_range = range(0, 300)
            xy0 = np.copy(_s.xy0)
            if _s.destination_type == 'orbit':
                C_helio = helio_xy0_over(_s.C, i_range)
                xy0 = (xy0 + C_helio).astype(np.float32)

            _s.xy = xy0 + np.array([960, 540], dtype=np.float32)
            _s.frame_ss = [0, 300]

        return _s

    def xy0_hohmann(_s):

        """Produces an """

        if _s.destination_type == 'orbit':

            if _s.p1.parent is _s.p0:  # Earth -> Moon: Set Earth r to zero
                r0 = 30  # Earth  # exit altitude
                r1 = _s.p1.gi['r'] #- 30  # Moon approach altitude
            elif _s.p0.parent is _s.p1:  # Moon -> Earth:
                r0 = _s.p0.gi['r'] - 15  # Earth # exit altitude  WITHOUT THIS THERE WONT BE TAKEOFF
                r1 = 1  # Moon  # The more, the further from Earth it will stop.
            else:
                raise Exception("moon -> moon ?")

            # Hohmann ellipse parameters (focus at origin of C-frame)
            a = 0.5 * (r0 + r1)  # semi-major: i.e. full linear distance/2
            c = (r1 - r0) / 2.0  # eccentricity (distance to focus)
            b = np.sqrt(max(a * a - c * c, 0.0))  # semi-minor.

            # θ = 0 → π always; θ=0 is periapsis at r0, θ=π is apoapsis at r1
            theta = np.linspace(0.0, np.pi, 300, endpoint=True)

            # rotate to convention
            x_f = -b * np.sin(theta)
            y_f = a * np.cos(theta) - c

            xy0 = np.stack((x_f, y_f), axis=1) #@ rot.T  # still C-centric
            xy0 = xy0.astype(np.float32)

            return xy0

        else:
            # ----- inter (heliocentric) -----

            r0 = _s.p0.gi['r']
            r1 = _s.p1.gi['r']

            if _s.p0.parent.id != '0':
                r0 = _s.p0.parent.gi['r']

            if _s.p1.parent.id != '0':
                r1 = _s.p1.parent.gi['r']

            a = 0.5 * (r0 + r1)
            c = (r1 - r0) / 2.0
            b = np.sqrt(max(a * a - c * c, 0.0))

            theta = np.linspace(0.0, np.pi, 300, endpoint=True)

            x_f = -b * np.sin(theta)
            y_f = a * np.cos(theta) - c

            # In inter, the central frame is already heliocentric, so xy0_C == xy0
            xy0 = np.stack((x_f, y_f), axis=1) #@ rot.T
            # _s.xy0 = xy0_C.astype(np.float32)
            xy0 = xy0.astype(np.float32)

            return xy0


