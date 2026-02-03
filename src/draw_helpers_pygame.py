import pygame
import numpy as np
import textwrap

import P


def to_surface(arr):
    """Convert a numpy RGBA uint8 array (H, W, 4) into a Pygame surface."""

    if arr.dtype != np.uint8:
        raise TypeError(f"Expected uint8 array, got {arr.dtype}")

    if arr.ndim != 3 or arr.shape[2] != 4:
        raise ValueError(f"Expected shape (H, W, 4), got {arr.shape}")

    # Pygame uses (W, H, C), so transpose
    arr = np.transpose(arr, (1, 0, 2))  # (H, W, C) → (W, H, C)

    rgb = arr[:, :, :3]
    alpha = arr[:, :, 3]

    surf = pygame.Surface(arr.shape[:2], pygame.SRCALPHA).convert_alpha()
    pygame.surfarray.blit_array(surf, rgb)
    pygame.surfarray.pixels_alpha(surf)[:, :] = alpha
    return surf


def gen_backgr(pics):
    """Returns a background surface created from two overlaid images."""
    base_arr = pics['backgr_b']
    overlay_arr = pics['backgr_55']

    base_surf = to_surface(base_arr)
    overlay_surf = to_surface(overlay_arr)
    overlay_surf.set_alpha(int(0.2 * 255))

    base_surf.blit(overlay_surf, (0, 0))
    return base_surf


def get_final_frame(world_surf: pygame.Surface) -> pygame.Surface:
    W, H = P.MAP_DIMS

    if not P.LU_2X_ZOOM:
        # Normal path: world is (W*render_ss, H*render_ss); downscale to screen
        return pygame.transform.smoothscale(world_surf, (W, H))

    lu_x, lu_y = P.LU_2X_ZOOM  # base MAP coords (top-left origin)
    # For 2x zoom we want a crop that represents W/2 x H/2 in base coords.
    # In "render_ss" pixels, that crop is (W/2 * render_ss, H/2 * render_ss).
    sw = int((W // 2) * P.SS_RENDER)
    sh = int((H // 2) * P.SS_RENDER)

    # Convert LU base coords to render_ss pixels
    sx = int(lu_x * P.SS_RENDER)
    sy = int(lu_y * P.SS_RENDER)

    ww, wh = world_surf.get_size()
    sx = max(0, min(sx, ww - sw))
    sy = max(0, min(sy, wh - sh))

    view = world_surf.subsurface(pygame.Rect(sx, sy, sw, sh))
    # Final step is a **downscale** to (W, H) -> preserves quality
    return pygame.transform.smoothscale(view, (W, H))


def _get_viewport_info():
    """Return (sx, sy, sw, sh) in SS_RENDER pixels of the source rect sampled into the screen."""
    W, H = P.MAP_DIMS
    SSR = P.SS_RENDER
    if not P.LU_2X_ZOOM:
        sx = sy = 0
        sw, sh = int(W * SSR), int(H * SSR)
    else:
        lu_x, lu_y = P.LU_2X_ZOOM  # base MAP coords, y-down
        sw, sh = int((W // 2) * SSR), int((H // 2) * SSR)
        sx, sy = int(lu_x * SSR), int(lu_y * SSR)
        ww, wh = int(W * SSR), int(H * SSR)
        sx = max(0, min(sx, ww - sw))
        sy = max(0, min(sy, wh - sh))
    return sx, sy, sw, sh


def screen_to_world(mx: int, my: int):
    """Map screen (pixels in 1920x1080 window) → base MAP coords (no SS)."""
    W, H = P.MAP_DIMS
    SSR = P.SS_RENDER
    sx, sy, sw, sh = _get_viewport_info()
    scale_x = sw / W
    scale_y = sh / H
    wx_ss = sx + mx * scale_x
    wy_ss = sy + my * scale_y
    return (wx_ss / SSR, wy_ss / SSR)  # base-map coords (y-down)


def draw_HUD_debug(i, screen: pygame.Surface, font=None, color=(255, 255, 255)):
    """Draw a HUD overlay after get_final_frame(), so size is stable and always visible."""

    font = pygame.font.SysFont(None, 24)
    if font is None:
        font = pygame.font.SysFont(None, 18)
    mx, my = pygame.mouse.get_pos()
    wx, wy = screen_to_world(mx, my)
    txt = f"screen=({mx},{my})  world=({wx:.1f},{wy:.1f} i={i})"
    surf = font.render(txt, True, color)
    screen.blit(surf, (10, 10))


# ~100 words of placeholder copy.


def _pick_font(size: int) -> pygame.font.Font:

    name = "ubuntu"
    path = pygame.font.match_font(name, bold=False, italic=False)
    if path:
        return pygame.font.Font(path, size)
    # return pygame.font.SysFont(None, size)
    # FONT_PREFS = [
    #     "ubuntucondensed", "ubuntu", "dejavusansmono", "liberationsansnarrow",
    #     "nimbussansnarrow", "dejavusans", "notomono", "freesans"
    # ]


def draw_HUD(i: int, screen: pygame.Surface, pos=(800, 900), color=(255, 255, 255)):
    """
    Draws:
      1) 'Year ####  Day ###' at pos with ~30pt font.
      2) ~100-word paragraph beneath with ~12pt font.
    Time rules: start at Year 1750 Day 0. 1 frame = P.SPEED_MULTIPLIER days.
    Clamped to Year 7750.
    """
    # --- time math ---
    days_total = int(i * P.SPEED_MULTIPLIER)
    max_days = 6000 * 365  # 1750 -> 7750 inclusive of 6000 years at 365 days/year
    days_total = max(0, min(days_total, max_days))

    years_passed, day_of_year = divmod(days_total, 365)
    year = 1750 + years_passed
    # year = 2250 + years_passed * 100
    # Safety in case of clamp at the very end:
    if year >= 7750:
        year = 7750
        day_of_year = min(day_of_year, 364)

    # --- fonts ---
    title_font = _pick_font(40)
    # text_font  = _pick_font(12)

    # --- render title line ---
    title = f"Year {year}   Day {day_of_year}"
    title_surf = title_font.render(title, True, color)
    x, y = pos
    screen.blit(title_surf, (x, y))

    # # --- render paragraph beneath (wrap to a sensible width) ---
    # max_text_width = 450
    # lines = _wrap_text(_LOREM, text_font, max_text_width)
    # line_h = text_font.get_linesize()
    # ty = y + title_surf.get_height() + 6  # small spacing under title
    # for line in lines:
    #     surf = text_font.render(line, True, color)
    #     screen.blit(surf, (x, ty))
    #     ty += line_h

    # _LOREM = (
    #     "Lorem ipsum dolor sit amet, consectetur adipiscing elit. "
    #     "Integer nec odio. Praesent libero. Sed cursus ante dapibus diam. "
    #     "Sed nisi. Nulla quis sem at nibh elementum imperdiet. Duis sagittis ipsum. "
    #     "Praesent mauris. Fusce nec tellus sed augue semper porta. Mauris massa. "
    # )

#
# def _wrap_text(text: str, font: pygame.font.Font, max_width: int):
#     # Greedy wrap by measuring pixel width of candidate lines.
#     words = text.split()
#     lines, cur = [], []
#     for w in words:
#         test = (" ".join(cur + [w])) if cur else w
#         if font.size(test)[0] <= max_width:
#             cur.append(w)
#         else:
#             if cur:
#                 lines.append(" ".join(cur))
#             cur = [w]
#     if cur:
#         lines.append(" ".join(cur))
#     return lines