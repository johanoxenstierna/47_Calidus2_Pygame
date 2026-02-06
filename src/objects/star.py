
import numpy as np
import P

from src.objects.abstract_pygame import AbstractPygameObject


class Star(AbstractPygameObject):

    def __init__(self, star_info):

        AbstractPygameObject.__init__(self)
        self.info = star_info
        self.type = 'star'

        self.rgb = np.zeros(shape=(P.FRAMES_STOP, 3), dtype=np.uint8)
        self.frame_ss = [0, P.FRAMES_STOP]

    def twinkle_brightness(self):

        t = np.arange(P.FRAMES_STOP)
        val = self.info['base_brightness'] * (
            1.0
            + 0.25 * np.sin(self.info['twinkle_rate'] * t + self.info['phase'])
            + 0.10 * np.sin(2 * self.info['twinkle_rate'] * t + 1.7 * self.info['phase'])
        )
        b = np.uint8(np.clip(val, 0, 255))

        self.rgb[:, 0] = b
        self.rgb[:, 1] = b
        self.rgb[:, 2] = b

        PALETTE = np.array([
            # [255, 255, 255],  # white
            [255, 240, 200],  # warm
            [200, 220, 255],  # cool
            [255, 220, 220],  # reddish
            [220, 255, 240],  # greenish
        ], dtype=np.uint8)

        self.tint = PALETTE[np.random.randint(len(PALETTE))]  # (3,)
        self.rgb[:] = (self.tint * (b[:, None] / 255)).astype(np.uint8)





