import numpy as np
from pathlib import Path
import pygame

from src.helpers import *
from src.helpers_distributions import *
from src.objects.rocket_helpers import *

import matplotlib.pyplot as plt

fig = plt.figure(figsize=[8, 6])

x = np.linspace(0, 1, num=100)
y = smoothstep(x, edge0=0.5, edge1=1)

plt.plot(x, y)

plt.show()




