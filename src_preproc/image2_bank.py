
import numpy as np
from typing import Iterable, Set
from PIL import Image

from src.helpers import *
from src.load_save_pics import save_png
from src_preproc.image1_mask import *


def build_and_save_bank(o1):
    """
    Walk frames, map to N angle slots, render each slot once, save as bank_###.png.
    """

    # name: str,
    # base_rgba: np.ndarray,
    # xy_abs: np.ndarray,  # shape (F, 2) absolute canvas coords of the body (xy3)
    # sun_xy: np.ndarray,  # shape (F, 2) or a single (cx, cy)
    # softness: float = 0.08,
    # shadow: float = 0.65,
    # highlight: float = 1.05,

    N = 32  # N/4 IS AN INT
    body_rotations_deg = gen_body_rotations_deg(N, max_deg=30)
    o1tp = top_parent(o1)  # Moons use same as parent
    y_squeezes = gen_y_squeezes(N, y_squeeze=o1tp.gi['y_squeeze'])

    body_rgba = o1.pic  # img = np.array(Image.open(f).convert("RGBA") )   ndarray: (d, d, 4)
    base = body_rgba

    # PATH_OUT = (P.BASE_DIR / './pictures/bodies_x/').resolve()
    PATH_OUT = (P.BASE_DIR / f'./pictures/bodies/{o1.id}/').resolve()  # CAREFUL
    [f.unlink() for f in Path(PATH_OUT).iterdir() if f.is_file()]  # REMOVES PREV

    if o1.gi['phi'] > 0:  # phi is ALWAYS 0-2pi
        raise Exception("phi not zero")
        # print("phi not zero for image2_bank -> Setting to zero!")  # SCREWS UP MOONS
        # o1.gi['phi'] = 0

    H, W, _ = o1.pic.shape
    d = min(H, W)

    xy_sun = [0, 0] #ALWAYS

    chunks_xy0 = np.array_split(o1.xy0, N)
    chunks_xy2 = np.array_split(o1.xy2, N)

    for k in range(len(chunks_xy0)):

        im = Image.fromarray(base, mode="RGBA")
        im_rot = im.rotate(body_rotations_deg[k], resample=Image.BICUBIC, expand=False)
        body_rot = np.array(im_rot, dtype=np.uint8)
        body_rgba = body_rot

        chunk_xy0 = chunks_xy0[k]
        chunk_xy2 = chunks_xy2[k]

        _xy0 = np.array([np.mean(chunk_xy0[:, 0]), np.mean(chunk_xy0[:, 1])])
        _xy2 = np.array([np.mean(chunk_xy2[:, 0]), np.mean(chunk_xy2[:, 1])])
        # mask = gen_mask_xy0(d, _xy0)
        mask = gen_mask(d, y_squeeze=y_squeezes[k], softness=5)
        mask = flip_rotate_mask(mask, _xy0, _xy2)

        # mask_to_png(mask, './tempMASK.png')

        body_masked = apply_mask(body_rgba, mask,
            shadow=0.05,  # less=more shadow
            highlight=1.3,  # more=more light
        )

        name = f"{k:02d}"
        # save_png(body_masked, name, './pictures/bodies_x/')
        save_png(body_masked, name, PATH_OUT)

# HELPERS =================================================
def angle_from_xy(xy) -> float:
    th = np.arctan2(xy[1], xy[0])
    return float(th if th >= 0 else th + 2*np.pi)


def gen_body_rotations_deg(N: int, max_deg: float = 30.0) -> np.ndarray:
    """
    OBS theta used here!
    N rotation angles in DEGREES for one CLOCKWISE orbit starting at top [0,-y]:
      top -> 0°, right -> +max, bottom -> 0°, left -> -max, back to top -> 0°.
    Positive here means tilt “down to the right”, which aligns with a CLOCKWISE visual tilt.
    PIL.Image.rotate expects CCW-positive degrees, so pass these directly if you want
    clockwise visual tilt (PIL uses CCW; supplying +deg rotates CCW).
    """
    k = np.arange(N, dtype=np.float32)
    # Start at top, move clockwise: theta = π/2 - 2π*k/N
    theta = (np.pi / 2.0) - (2.0 * np.pi * k / N)
    # Map: right(+x)-> +max, top->0, left-> -max, bottom->0
    rot_deg = max_deg * np.cos(theta)
    return rot_deg


def gen_y_squeezes(N: int, y_squeeze):
    """
    y_squeeze is only usable for location [0, -r] (phi=0)
    y_squeezes is the array input for building the masks
    """
    a = np.linspace(y_squeeze, 1, num=N // 4, endpoint=False)
    b = np.linspace(1, y_squeeze, num=N // 4, endpoint=False)
    c = np.hstack((a, b, a, b))
    return c

