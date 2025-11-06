
import P


class AbstractObject:
    """
    """

    def __init__(_s):
        # _s.drawn = 0  # 0: not drawn, 1: start drawing, 2. continue drawing, 3. end drawing, 4: dynamic flag usage
        _s.age = 0

        _s.pic = None
        _s.pics = []
        _s.type = None
        _s.zorder = None
        _s.frame_ss = [None, None]

    # def set_frame_ss(_s, ii, num_frames):
    #     """start stop frames to show the object"""
    #     _s.frame_ss = [ii, ii + num_frames]

    # def set_age(_s, i):
    #     """
    #     The layer classes don't have access to the ax, so
    #     this essentially tells the ax what to do.
    #     """
    #
    #     if i == _s.frame_ss[0]:
    #         _s.drawn = 1
    #     elif i > _s.frame_ss[0] and i < _s.frame_ss[1] - 1:
    #         _s.drawn = 2  # continue. needed bcs ani_update_step will create a new D otherwise
    #         _s.age += 1
    #     elif i >= _s.frame_ss[1] - 1:
    #         _s.drawn = 3  # end drawing
    #         _s.age = 99999  # to make sure error will happen if tried to use
    #     else:  # NEEDED BCS OTHERWISE _s.drawn just stays on 3
    #         _s.drawn = 0


