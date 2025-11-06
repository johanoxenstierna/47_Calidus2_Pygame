
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
MAP_DIMS = (1920, 1080)

# FRAMES_START = 0
FRAMES_STOP = 20000  #30000  OBS: Will be bugs if lower than, say,
# FRAMES_TOT_BODIES = FRAMES_STOP - 25
# FRAMES_TOT = FRAMES_STOP - FRAMES_START

FPS = 60  # 17:31
SPEED_MULTIPLIER = 1  #latest: 0.4  # Makes objects move faster
WRITE = 0  # OBS USE CORRECT VIDEO NUM
ONE_ROCKET = 0

SS = 1  # SSAA (super sampling anti aliasing) SET TO 2 FOR HIGHER QUALITY
LU_2X_ZOOM = 0#(480, 280)
SS_RENDER = SS * (2 if LU_2X_ZOOM else 1)  #
REAL_SCALE = 0

OBJ_TO_SHOW = []
OBJ_TO_SHOW.append('Rockets')
# OBJ_TO_SHOW.append('Sun')
# OBJ_TO_SHOW.append('3_Venus')
# OBJ_TO_SHOW.append('2_Mercury')
OBJ_TO_SHOW.append('4_Earth')
OBJ_TO_SHOW.append('4_GSS')
# OBJ_TO_SHOW.append('4_Moon')
# OBJ_TO_SHOW.append('4_NEA')
# OBJ_TO_SHOW.append('5_Mars')
# OBJ_TO_SHOW.append('Astro0')
# OBJ_TO_SHOW.append('Astro0b')
OBJ_TO_SHOW.append('6_Jupiter')
OBJ_TO_SHOW.append('6_Io')
# OBJ_TO_SHOW.append('6_Ganymede')
# OBJ_TO_SHOW.append('6_Europa')
# OBJ_TO_SHOW.append('Saturn')
# OBJ_TO_SHOW.append('Uranus')
# OBJ_TO_SHOW.append('Neptune')
# OBJ_TO_SHOW.append('YearsDays')

VID_SINGLE_WORD = '_'
for word in OBJ_TO_SHOW:
    VID_SINGLE_WORD += word[0] + '_'

XY01 = 1  # OBS ROCKET TAKEOFF LANDING USE bodies xy2 i.e. AFTER y_squeeze and tilt. So generally only use this for bodies or when testing xy0
GEN_DL_PIC_BANK = 0  #'6_Jupiter' #'4_Moon' #'4_Earth'  #  DEFAULT = 0 OBS PHI==0 WHEN GENERATING. ALSO, WHENEVER CHANGING ANY GI PARAMETERS ACTUALLY
USE_DL = 0
N_ORBIT_SAMPLES = 2048
N_ORBIT_IMAGES = 32



