
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from src.load_save_pics import *


def _smoothstep01(t: np.ndarray) -> np.ndarray:
    """Cubic smoothstep on [0,1]."""
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def gen_circular_alpha_mask(
    diameter: int,
    low: float = 0.90,         # start fading at % of radius
    high: float = 0.95,        # fully faded by % of radius
    min_alpha: float = 0.0,    # alpha at the very edge
    max_alpha: float = 1.0,    # alpha at the center
    gamma: float = 1.0         # >1 steeper near edge, <1 softer
) -> np.ndarray:
    """
    Returns a float32 alpha mask in [0,1], with center ~max_alpha and edge ~min_alpha.
    'low' and 'high' are fractions of the radius that control the fade band.

    gamma > 1: pushes values closer to 0 for most of the fade → fade feels softer, more gradual.
    gamma < 1: pushes values closer to 1 for most of the fade → fade feels sharper near the cutoff.
    """
    assert 0.0 <= low < high <= 1.0
    r = diameter / 2.0
    Y, X = np.mgrid[0:diameter, 0:diameter]
    Xc, Yc = X - r, Y - r
    R = np.sqrt(Xc*Xc + Yc*Yc)
    nr = R / r  # normalized radius: 0 at center, 1 at edge

    # Map normalized radius into a 0..1 fade coordinate across [low, high]
    t = (nr - low) / max(1e-8, (high - low))
    t = _smoothstep01(t)

    if gamma != 1.0:
        # Shape the fade curve (optional)
        t = np.clip(t, 0.0, 1.0) ** gamma

    # Interpolate between max_alpha (center) and min_alpha (edge)
    alpha = max_alpha * (1.0 - t) + min_alpha * t

    # Zero outside the disk (optional; keeps a clean circular sprite)
    alpha = np.where(nr <= 1.0, alpha, 0.0).astype(np.float32)
    return alpha


def apply_rim_alpha(rgba_u8: np.ndarray, rim_alpha: np.ndarray, mode: str = "multiply") -> np.ndarray:
    """
    Apply a rim alpha mask to an RGBA image.
    - mode="multiply": newA = oldA * rim_alpha
    - mode="replace":  newA = rim_alpha
    Returns a new uint8 RGBA.
    """
    assert rgba_u8.ndim == 3 and rgba_u8.shape[2] == 4
    H, W, _ = rgba_u8.shape
    if rim_alpha.shape != (H, W):
        raise ValueError("rim_alpha size must match image")

    out = rgba_u8.copy()
    a = out[..., 3].astype(np.float32) / 255.0
    if mode == "multiply":
        a_new = np.clip(a * rim_alpha.astype(np.float32), 0.0, 1.0)
    elif mode == "replace":
        a_new = np.clip(rim_alpha.astype(np.float32), 0.0, 1.0)
    else:
        raise ValueError("mode must be 'multiply' or 'replace'")
    out[..., 3] = (a_new * 255.0 + 0.5).astype(np.uint8)
    return out


def crop_to_content(rgba_u8: np.ndarray, margin: int = 0) -> np.ndarray:
    """
    Crop an RGBA image to the bounding box of nonzero alpha.

    Parameters
    ----------
    rgba_u8 : np.ndarray
        HxWx4 uint8 RGBA.
    margin : Extra pixels of padding around the detected content.
    OBS this ONLY makes sense when the raw image borders have been cut a lot.
    Because that adds 0 alpha which previously wasn't there

    Returns
    -------
    np.ndarray
        Cropped RGBA uint8 image.
    """
    assert rgba_u8.ndim == 3 and rgba_u8.shape[2] == 4, "Need RGBA image"
    alpha = rgba_u8[..., 3]
    ys, xs = np.nonzero(alpha > 0)
    if len(xs) == 0 or len(ys) == 0:
        return rgba_u8  # fully transparent

    x0, x1 = xs.min(), xs.max()
    y0, y1 = ys.min(), ys.max()
    x0 = max(0, x0 - margin)
    y0 = max(0, y0 - margin)
    x1 = min(rgba_u8.shape[1], x1 + margin + 1)
    y1 = min(rgba_u8.shape[0], y1 + margin + 1)
    return rgba_u8[y0:y1, x0:x1, :]


def downscale(
    rgba_u8: np.ndarray,
    diameter: int,
    *,
    pad_to_square: bool = True
) -> np.ndarray:
    """
    Downscale an RGBA image to fit within `diameter` (px) while preserving aspect.
    Optionally pad to an exact square (diameter x diameter) with transparent pixels.

    Parameters
    ----------
    rgba_u8 : np.ndarray
        Input RGBA uint8 image (H, W, 4).
    diameter : int
        Target max dimension.
    pad_to_square : bool
        If True, center the result on a (diameter x diameter) transparent canvas.

    Returns
    -------
    np.ndarray
        RGBA uint8 result.
    """
    if rgba_u8.dtype != np.uint8 or rgba_u8.ndim != 3 or rgba_u8.shape[2] != 4:
        raise ValueError("rgba_u8 must be uint8 with shape (H, W, 4).")

    h, w, _ = rgba_u8.shape
    if h == 0 or w == 0:
        return rgba_u8

    # scale factor to fit max dimension
    scale = diameter / max(h, w)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))

    im = Image.fromarray(rgba_u8, mode="RGBA")
    im_resized = im.resize((new_w, new_h), resample=Image.LANCZOS)

    if not pad_to_square:
        return np.array(im_resized, dtype=np.uint8)

    # center on transparent square canvas
    canvas = Image.new("RGBA", (diameter, diameter), (0, 0, 0, 0))
    off_x = (diameter - new_w) // 2
    off_y = (diameter - new_h) // 2
    canvas.paste(im_resized, (off_x, off_y), im_resized)
    return np.array(canvas, dtype=np.uint8)


def show_png(rgba_u8: np.ndarray, title: str = "") -> None:
    """
    Quick visualization using matplotlib.
    """

    assert rgba_u8.ndim == 3 and rgba_u8.shape[2] == 4
    plt.figure(figsize=(5,5))
    # matplotlib expects RGBA float or uint8; it will ignore alpha on white bg
    plt.imshow(rgba_u8)
    if title:
        plt.title(title)
    plt.axis('off')
    plt.show()


if __name__ == '__main__':
    # mask = gen_binary_mask(50, xyz=[1, 0, 0])

    PATH_IN = './pictures/bodies0_raw/'
    PATH_OUT = './pictures/bodies_x/'

    SETTINGS = {
        '2_Mercury': {'low': 0.95, 'high': 0.98, 'margin': 0, 'diameter': 100},
        '3_Venus': {'low': 0.95, 'high': 0.98, 'margin': 0, 'diameter': 100},
        '4_Earth': {'low': 0.95, 'high': 0.98, 'margin': 0, 'diameter': 200},
        '4_GSS': {'low': 0.95, 'high': 0.98, 'margin': 0, 'diameter': 20},
        '4_Moon': {'low': 0.80, 'high': 0.90, 'margin': 0, 'diameter': 50},
        '5_Mars': {'low': 0.95, 'high': 0.98, 'margin': 0, 'diameter': 100},
        '6_Jupiter': {'low': 0.95, 'high': 0.98, 'margin': 0, 'diameter': 100},
        '6_Io': {'low': 0.95, 'high': 0.98, 'margin': 0, 'diameter': 100},
        '6_Ganymede': {'low': 0.95, 'high': 0.98, 'margin': 0, 'diameter': 100},
        '6_Europa': {'low': 0.95, 'high': 0.98, 'margin': 0, 'diameter': 100},
    }

    # OBS margin no effect if pic already reasonably bounded

    NAME = '2_Mercury'
    S = SETTINGS[NAME]

    pics = {}
    load_pics(pics, PATH_IN)

    # 1) Get the base RGBA (e.g., 'Jupiter') and confirm shape
    body_rgba = pics[NAME]  # HxWx4, uint8

    # 2) Build the light mask
    H, W, _ = body_rgba.shape
    d = min(H, W)  # if image is square, d=H=W

    mask_rim = gen_circular_alpha_mask(d, low=S['low'], high=S['high'], gamma=1)  # 0..1  more=more spread out
    body_rimmed = apply_rim_alpha(body_rgba, mask_rim)
    body_rimmed_cropped = crop_to_content(body_rimmed, margin=S['margin'])
    body_rimmed_cropped_downscaled = downscale(body_rimmed_cropped, diameter=S['diameter'], pad_to_square=True)

    # show_png(body_rimmed_cropped_downscaled)
    save_png(body_rimmed_cropped_downscaled, name=NAME, filepath=PATH_OUT)
