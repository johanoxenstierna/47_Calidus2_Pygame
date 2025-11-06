# from src.gen_extent_triangles import *
# from src.objects.abstract_all import AbstractObject
from src.objects.abstract_pygame import AbstractPygameObject
import P as P
import numpy as np
import random


class O0C(AbstractPygameObject):

    def __init__(_s, pic, gi):
        super().__init__()
        _s.id = '0'
        _s.gi = gi
        _s.pic = pic
        _s.O1 = {}

        _s.radius = [int(pic.shape[0] / 2), int(pic.shape[1] / 2)]

        _s.xy = np.zeros((P.N_ORBIT_SAMPLES, 2), dtype=np.float32)
        _s.vxy = np.zeros((P.N_ORBIT_SAMPLES, 2), dtype=np.float32)

        # _s.xy[:, 0] = P.MAP_DIMS[0] / 2
        # _s.xy[:, 1] = P.MAP_DIMS[1] / 2

        _s.xy0_abs = [int(P.MAP_DIMS[0] / 2), int(P.MAP_DIMS[1] / 2)]
        _s.xy2_abs = [int(P.MAP_DIMS[0] / 2), int(P.MAP_DIMS[1] / 2)]

        # _s.xy1 = np.full((P.N_ORBIT_SAMPLES, 2), fill_value=0).astype(np.float32)
        # _s.xy1 = np.copy(_s.xy0)

        _s.zorders = np.full((P.N_ORBIT_SAMPLES,), fill_value=2000).astype(int)
        # _s.gi['r'] = 0

    def get_i_orbit(_s, i):
        return 0