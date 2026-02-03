
import numpy as np

from src.objects.abstract_pygame_rocket import AbstractPygameRocket

class Rocket3000(AbstractPygameRocket):

    def __init__(self, gi):

        AbstractPygameRocket.__init__(self)

        self.gi = gi

        self.xy = np.zeros(shape=(gi['num_frames'], 2), dtype=np.float32)
        self.xy[:, 0] = np.linspace(gi['od'][0][0], gi['od'][1][0], gi['num_frames'])
        self.xy[:, 1] = np.linspace(gi['od'][0][1], gi['od'][1][1], gi['num_frames'])

        self.frame_ss = [gi['start_frame'], gi['start_frame'] + gi['num_frames']]
        self.zorders = np.full((gi['num_frames'],), fill_value=2010)
        self.alphas = np.full((gi['num_frames'],), fill_value=255, dtype=np.uint8)




