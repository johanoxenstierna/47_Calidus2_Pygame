
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import time
from pathlib import Path  # used to save png's to file

from src.helpers import phi_from_coord
from src.load_save_pics import load_pics
import P


# ==== helpers ================================================================

def make_relative_grid(diameter: int):
    """
    Returns coordinate grids centered at the image center and radial distances.
    """
    r = diameter / 2
    Y, X = np.mgrid[0:diameter, 0:diameter]
    x_rel = X - r
    y_rel = Y - r
    radius = np.sqrt(x_rel**2 + y_rel**2)
    return x_rel, y_rel, radius

def _normalize(v: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    n = np.linalg.norm(v) + eps
    return v / n

def _smoothstep01(t: np.ndarray) -> np.ndarray:
    """Cubic smoothstep on [0,1]."""
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)

# ==== masks ==================================================================

def gen_mask_xy0(diameter: int, xy0, softness: float = 0.08,
) -> np.ndarray:
    """
    DOES NOT USE A STANDARDIZED MASK FROM [0, -r], instead light direction done inside here.
    OBS ONLY FOR LOOKING STRAIGHT ABOVE SOLAR SYSTEM
    Continuous 0..1 lighting mask from 2D-projected light direction.
    1.0 = fully lit, 0.0 = fully shadowed.
    Uses only x,y (z ignored for now, as agreed).

    Parameters
    ----------
    diameter : int
        Square image size in px.
    xy: Relative sun position. Only (x,y) are used here.
    softness : float
        Width of the terminator transition as a fraction of radius (≈0.02..0.15).

    Returns
    -------
    mask : float32 array in [0, 1], outside circle is 0.
    """
    x = xy0[0]
    y = xy0[1]
    r = diameter / 2.0

    # Project light into the image plane and normalize
    L2 = np.array([x, y], dtype=float)
    n = np.hypot(L2[0], L2[1])
    if n < 1e-12:
        raise ValueError("gen_mask_xyz: light direction has zero x and y; undefined terminator.")
    ux, uy = L2 / n

    # Grid
    Y, X = np.mgrid[0:diameter, 0:diameter]
    Xc = X - r
    Yc = Y - r
    R = np.sqrt(Xc*Xc + Yc*Yc)
    inside = (R <= r)

    # Signed distance along light direction (scaled to ~[-1,1])
    s = (Xc * ux + Yc * uy) / (r + 1e-8)

    # Soft terminator around s=0. Positive s -> shadow side; negative s -> lit side
    k = max(1e-6, float(softness))
    mask = np.clip(0.5 * (1.0 - s / k), 0.0, 1.0)

    # Zero outside the disk
    mask *= inside.astype(np.float32)
    return mask.astype(np.float32)


def gen_mask(d: int, y_squeeze: float, softness: float = 2.0) -> np.ndarray:
    """
    Standardized for relative sun position [0, -1], i.e., it's behind the sun.
    y_squeeze in [0, 1]. It's the multiplication factor of the y component in a perfectly circular orbit.
    y_squeeze = 1.0 → straight line y=r (-1 out top half y<r)  (we're looking straight from above solar system)
    y_squeeze = 0.5 → parabola through (0,r) → (r,0.5r) → (d,r), (we're looking at solar system from 45 degrees)
    y_squeeze = 0.0 → keep full circle (no -1's)  (we're looking straight from the side of solar system)

    Output:
      float32 mask with values

    Notes:
      - 'y_squeeze' is flipped for legacy reasons (see docstring).
      - Uses a circular-cap (not a parabola) so small updated y_squeeze (e.g. 0.1)
        gives a slim crescent of -1 near the top.
    """
    r = d / 2.0
    x_rel, y_rel, radius = make_relative_grid(d)

    # Base: outside=0, inside=1
    out = np.zeros((d, d), dtype=np.float32)
    inside = radius <= r
    out[inside] = 1.0

    # circular cap geometry (same shape, now used for a SOFT boundary)
    s = float(np.clip((1.0 - float(np.clip(y_squeeze, 0.0, 1.0))) * r, 0.0, r - 1e-6))
    if s <= 1e-6:
        # boundary degenerates to the top chord at y=r
        X = np.arange(d, dtype=np.float32)
        y_cut = np.full_like(X, r, dtype=np.float32)
    else:
        t = (s * s - r * r) / (2.0 * s)  # helper for center offset
        c = r - t  # circle center y
        R2 = (r - s - c) ** 2  # circle radius^2 along y
        X = np.arange(d, dtype=np.float32)
        dx2 = (X - r) ** 2
        y_cut = c - np.sqrt(np.maximum(R2 - dx2, 0.0), dtype=np.float32)  # (d,)

    # signed distance proxy to boundary: negative = "above" (shadow), positive = "below" (light)
    Y = np.arange(d, dtype=np.float32)[:, None]  # (d,1)
    dy = Y - y_cut[None, :]  # (d,d)

    # soft logistic ramp centered at 0 → outputs 0..1 with 0.5 on the boundary
    # softness = pixel distance from boundary to reach ~10%/90%.
    k = np.log(9.0) / max(1e-6, float(softness))  # 10–90% over ±softness px
    light = 1.0 / (1.0 + np.exp(-k * dy))  # 0..1

    # zero outside the disk
    light = (light * inside.astype(np.float32))

    return light.astype(np.float32)

    # # Convert to a circular-cap "sagitta" depth s in [0, r]
    # # We want small *updated* y_squeeze  → slim crescent (deep cap),
    # # so set s = (1 - ys_eff) * r.
    # s = (1.0 - y_squeeze) * r
    # s = float(np.clip(s, 0.0, r - 1e-6))  # avoid degenerate geometry
    #
    # if s <= 1e-6:
    #     # Straight chord at y = r (top half carved)
    #     Y = np.arange(d, dtype=np.float32)[:, None]
    #     cut = (Y < r)  # "above" the chord
    #     # mark carved region only where inside circle
    #     out[cut & inside] = -1.0
    #     return out
    #
    # # Geometry of a circle that passes through (0, r) and (d, r) and dips to (r, r - s)
    # # Let center be (r, c) with c = r - t, where:
    # #    t = (s^2 - r^2) / (2s)
    # t = (s * s - r * r) / (2.0 * s)
    # c = r - t
    # # Radius of that circle
    # R2 = (r - s - c) ** 2  # == (t - s)^2
    #
    # # Compute the y-threshold curve y_cut(x) = c - sqrt(R^2 - (x - r)^2),
    # # and carve everything "above" it (smaller y).
    # X = np.arange(d, dtype=np.float32)
    # dx2 = (X - r) ** 2
    # # Guard against tiny numeric negatives inside sqrt
    # under_sqrt = np.maximum(R2 - dx2, 0.0)
    # y_cut = c - np.sqrt(under_sqrt, dtype=np.float32)  # shape (d,)
    #
    # Y = np.arange(d, dtype=np.float32)[:, None]        # (d,1)
    # above = (Y < y_cut[None, :])                       # (d,d)
    #
    # # Apply carve only inside the planetary disk
    # out[above & inside] = -1.0
    #
    #
    # return out


def flip_rotate_mask(mask: np.ndarray, _xy0, _xy2) -> np.ndarray:
    """
    Using both 0 and 2 probably works because tilt is not supposed to change orbital position.
    (although the operation it uses is a rotation).
    Rotate mask to align with phi(_xy2), optionally flip (invert) for _xy0[1] > 0.
    Inputs/Outputs:
      - mask: float32 in [0,1]
      - returns float32 in [0,1]
    """
    # angle
    phi = phi_from_coord(_xy2)
    phi_deg = float(np.degrees(phi))
    if _xy0[1] > 0:
        phi_deg -= 180.0

    # PIL expects mode 'F' for float; fill with 0 (dark) outside
    im = Image.fromarray(mask, mode="F")
    im_rot = im.rotate(-phi_deg, resample=Image.BICUBIC, expand=False, fillcolor=0.0)

    out = np.array(im_rot, dtype=np.float32)

    # flip = invert brightness when _xy0 is "below" (legacy semantics)
    if _xy0[1] > 0:
        out = 1.0 - out

    # cubic/lanczos can overshoot -> clamp to [0,1]
    np.clip(out, 0.0, 1.0, out=out)
    return out


# --- sRGB <-> Linear helpers (piecewise accurate) ---

def srgb_to_linear(arr: np.ndarray) -> np.ndarray:
    """arr: float in [0,1], returns float in [0,1] (linear)."""
    a = 0.055
    # piecewise sRGB EOTF
    low  = arr <= 0.04045
    out = np.empty_like(arr, dtype=np.float32)
    out[low]  = arr[low] / 12.92
    out[~low] = ((arr[~low] + a) / (1 + a)) ** 2.4
    return out

def linear_to_srgb(arr: np.ndarray) -> np.ndarray:
    """arr: float in [0,1] (linear), returns float in [0,1] (sRGB)."""
    a = 0.055
    # piecewise sRGB OETF
    low  = arr <= 0.0031308
    out = np.empty_like(arr, dtype=np.float32)
    out[low]  = arr[low] * 12.92
    out[~low] = (1 + a) * (arr[~low] ** (1/2.4)) - a
    return out


# --- core lighting application ---

def apply_mask(
    rgba_u8: np.ndarray,
    light_mask: np.ndarray,
    shadow: float = 0.65,
    highlight: float = 1.05,
    use_linear: bool = False,
) -> np.ndarray:
    """
    Apply per-pixel lighting to an RGBA image using a 0..1 'light_mask'
    (1=fully lit, 0=fully shadowed). Returns uint8 RGBA.

    - rgba_u8: HxWx4 uint8 RGBA
    - light_mask: HxW float32/float64 in [0,1]
    - shadow/highlight are multiplicative gains
    - use_linear: if True, convert sRGB->linear, apply, convert back
    """
    assert rgba_u8.ndim == 3 and rgba_u8.shape[2] == 4, "rgba_u8 must be HxWx4"
    H, W, _ = rgba_u8.shape
    if light_mask.shape != (H, W):
        raise ValueError("light_mask must match image size (H, W)")

    # split channels
    rgb = rgba_u8[..., :3].astype(np.float32) / 255.0
    a   = rgba_u8[..., 3:4]  # keep alpha as uint8 slice

    # compute per-pixel gain
    # gain = light_mask*highlight + (1-light_mask)*shadow
    gain = (light_mask.astype(np.float32)[..., None] * (highlight - shadow)) + shadow

    if use_linear:
        # more accurate shading
        rgb_lin = srgb_to_linear(rgb)
        rgb_shaded = np.clip(rgb_lin * gain, 0.0, 1.0)
        rgb_out = linear_to_srgb(rgb_shaded)
    else:
        # quick shading directly in sRGB space
        rgb_out = np.clip(rgb * gain, 0.0, 1.0)

    out = (rgb_out * 255.0 + 0.5).astype(np.uint8)
    out = np.concatenate([out, a], axis=-1)
    return out


def mask_to_png(mask: np.ndarray, filename: str = "temp_mask.png") -> None:
    """
    Save a mask as a grayscale PNG image. Supports both boolean and float inputs.
    For float inputs, values are expected in [0, 1].
    """
    if mask.dtype == bool:
        img_array = np.where(mask, 255, 0).astype(np.uint8)
        mode = 'L'
    elif np.issubdtype(mask.dtype, np.floating):
        img_array = np.clip(mask * 255.0, 0, 255).astype(np.uint8)
        mode = 'L'
    else:
        raise ValueError("Unsupported mask dtype")

    Image.fromarray(img_array, mode=mode).save(filename)


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

    mask = gen_mask(100, y_squeeze=0.3, softness=10)
    # mask = gen_mask_xy0(diameter=100, xy0=[0, -1])
    mask_to_png(mask, './tempMASK.png')



    aa = 5