import numpy as np


def helio_xy0_over(body, i_range):
    """heliocentric xy0 for body over frames i_range (no display offset)."""
    idx = body.get_i_orbit(i_range)
    xy = body.xy0[idx]  # relative to parent
    if body.parent.id != '0':  # i.e. it's a moon
        xy = xy + body.parent.xy0[idx]
    return xy


def helio_xy2_over(body, i_range):
    """heliocentric xy0 for body over frames i_range (no display offset)."""
    idx = body.get_i_orbit(i_range)
    xy = body.xy2[idx]  # relative to parent
    if body.parent.id != '0':  # i.e. it's a moon
        xy = xy + body.parent.xy2[idx]
    return xy


def rel_to_C_over(body, C, i_range):
    """
    Was intended for inter cases where we want to get top_parent coordinates,
    but it was then probably replaced by other code achieving the same thing more explicitly with cases.
    C-centric xy0 for body over frames i_range.
    """
    xy_helio = helio_xy0_over(body, i_range)
    C_helio  = helio_xy0_over(C,    i_range)

    return xy_helio - C_helio


def arc_length(xy):
    diffs_len = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    diffs_len_cum = np.concatenate([[0.0], np.cumsum(diffs_len)])
    return diffs_len_cum, float(diffs_len_cum[-1])


def smoothstep01(x):
    # clamp then cubic smoothstep
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def build_speed_profile(xy, speed0, speed1):
    """
    :param xy: array where constant inter-frame speed is assumed. Needed to get arc length
    :param speed0: desired starting speed per frame
    :param speed1: desired ending speed per frame
    """
    # Arc-length + v-profile guardrails
    diffs_len_cum, L = arc_length(xy)
    if L < 1e-6:
        raise Exception("L can never be zero")

    diffs_len_cum01 = diffs_len_cum / L
    ease = smoothstep01(diffs_len_cum01)  # creates sigmoid looking thing (instead of input linear)

    # --- speed guardrails (A) ---
    MIN_FRAMES, MAX_FRAMES = 0, 3000  # min frames checked elsewhere currently (TODO)
    min_v_floor = max(L / MAX_FRAMES, 0.05)
    speed_profile = (1.0 - ease) * speed0 + ease * speed1

    # makes sure speed per frame is always at least min_v
    speed_profile = np.maximum(speed_profile, min_v_floor).astype(np.float32)
    return speed_profile


def resample_by_time(xy_notime, speed_profile):
    """
    Given a path (N,2) parameterized by cumulative arclength and a per-point speed (pixels/frame),
    compute per-frame samples. speed_profile length must equal path length N.
    OBS THIS CHANGES NUMBER OF FRAMES (obviously)
    """
    diffs_len_cum, L = arc_length(xy_notime)
    # avoid zero-length issues
    if L < 1e-6:
        return xy_notime.copy()

    # per-segment len and speed (use midpoints)
    diffs_len = np.diff(diffs_len_cum)
    v_mid = 0.5 * (speed_profile[:-1] + speed_profile[1:])
    v_mid = np.maximum(v_mid, 1e-6)

    dt = diffs_len / v_mid                           # frames per segment (since v is in pixels/frame)
    t = np.concatenate([[0.0], np.cumsum(dt)])
    total_frames = max(int(np.round(t[-1])), 1)

    # Build uniform frame times 0..total_frames
    t_uniform = np.arange(0, total_frames + 1, dtype=np.float32)

    # For each axis, interpolate by t
    x = xy_notime[:, 0]
    y = xy_notime[:, 1]
    x_out = np.interp(t_uniform, t, x)
    y_out = np.interp(t_uniform, t, y)
    out = np.stack([x_out, y_out], axis=1).astype(np.float32)

    if len(out) <= 60:
        raise Exception("Time resampling resulted in less than 60 frames (shouldnt happen but if so fix needed)")

    return out


def rotate_xy(xy, delta):
    "xy array of coordinates rotated."
    c, s = np.cos(delta), np.sin(delta)
    R = np.array([[c, -s], [s, c]], dtype=xy.dtype)
    return (xy @ R.T).astype(np.float32, copy=False)


def slope_at_idx(xy, idx, side, k=3):
    """
    Least-squares slope (per frame) at xy[idx] using k samples on one side.
    side chooses which one-sided window to fit for the tangent:

    Before/after is not super important, but it gives capability to avoid getting samples from areas that wont be used
    when blending.
    before → use samples before idx → approach slope.
    after → use samples after idx → depart slope.

    Used to determine velocity of array xy. Say B=80,
    then we want V from xy_first 40 from the end and V from xy_second 40 from beginning
    e.g. xy_first is the first array to be used for crossfade,
    xy_first: side='before/left',  n=97,  i0=54, i1=57
    xy_secon: side='after/right', n=461, i0=40, i1=43

    """
    n = len(xy); k = max(1, min(k, n-1))
    if side == "before":
        i0, i1 = max(0, idx-k), idx
    elif side == "after":
        i0, i1 = idx, min(n-1, idx+k)
    else:
        raise Exception("side needs to be before or after")

    frames = np.arange(i0, i1+1, dtype=np.float32)[:, None]  # (m,1)
    xy_used = xy[i0:i1+1]                                     # (m,2)
    # Fit xy_used ≈ a*frames + b → slope a
    x0 = frames - frames.mean()
    denom = (x0**2).sum()
    if denom == 0:
        # fallback to finite difference
        return (xy[min(n-1, idx+1)] - xy[max(0, idx-1)]) * 0.5
    a = (x0 * (xy_used - xy_used.mean(axis=0))).sum(axis=0) / denom
    return a  # per frame


def crossfade_B_frames(xy0, xy1, B, k=3):
    """
    C1 seam with fixed length: output len == len(xy0) + len(xy1).
    Replaces last M=B/2 of xy0 and first M of xy1 with a B-frame cubic-Hermite blend
    that matches positions and per-frame velocities at both ends.
    """

    if xy0 is None:
        return xy1

    assert B >= 2 and B % 2 == 0, "B must be even and >= 2"
    M = B // 2
    n0, n1 = len(xy0), len(xy1)
    assert n0 > M and n1 > M, "arrays too short for requested B"

    # Endpoints to match
    P0 = xy0[n0 - M]      # first removed point from xy0
    P1 = xy1[M]           # first kept point from xy1 after the removed head

    # Per-frame velocities at the endpoints (least-squares over k samples)
    V0 = slope_at_idx(xy0, n0 - M, side="before", k=k)  # get V M indicies from end
    V1 = slope_at_idx(xy1, M,      side="after", k=k)  # get V M indicies from start

    # Cubic Hermite over B frames, t in [0,1], scale tangents by (B-1)
    t = np.linspace(0.0, 1.0, B, dtype=np.float32)[:, None]
    h00 =  2*t**3 - 3*t**2 + 1
    h10 =      t**3 - 2*t**2 + t
    h01 = -2*t**3 + 3*t**2
    h11 =      t**3 -   t**2
    m0 = V0 * (B - 1)
    m1 = V1 * (B - 1)
    blend = (h00*P0 + h10*m0 + h01*P1 + h11*m1).astype(np.float32, copy=False)
    return blend


def spiral(num_frames, r, direction, tf_or_ld):
    """
    Make a one-turn clockwise spiral from Opi -> 2pi (in phi space) (starting and ending at [0, -r]
    """
    if tf_or_ld == 'takeoff':
        rs = np.linspace(0, r, num=num_frames)
    elif tf_or_ld == 'landing':
        rs = np.linspace(r, 0, num=num_frames)

    phis = np.linspace(0, 2 * np.pi, num=num_frames)

    if direction == 'cw':
        x = rs * np.sin(phis)
        y = -rs * np.cos(phis)
    elif direction == 'ccw':
        x = -rs * np.sin(phis)
        y = -rs * np.cos(phis)
    else:
        raise Exception('cw or ccw not specified')

    spiral = np.stack([x, y], axis=1) #+ xy_start

    return spiral


def jittered_range(start: int, stop: int, base_step: int, rand_step: int):
    """
    Generate [start, ..., < stop] with variable step = base_step + randint(-b, b).
    Ensures each step >= min_step (default 1) so it always advances.
    """

    out = [start]
    cur = start
    while True:
        step = base_step + np.random.randint(-rand_step, rand_step)
        nxt = cur + step
        if nxt >= stop:     # match range(): stop is exclusive
            break
        out.append(nxt)
        cur = nxt

    return sorted(out)


def closest_point_to_origin_on_line(a, b):
    """Return point c on the infinite line through floats a,b with minimal distance to [0,0]."""
    d = b - a
    denom = float(d @ d)
    if denom == 0.0:
        return a.copy()  # degenerate: a==b
    t = - float(a @ d) / denom
    c = a + t * d

    #undersh_oversh === A @ B gives dot product: How aligned in same direction ====
    u_ab = (b - a).astype(float)
    nu = np.linalg.norm(u_ab)
    u_ab = u_ab / (nu if nu > 0 else 1.0)

    undersh_oversh = "overshoot" if float(u_ab @ (c - a)) > 0.0 else "undershoot"
    return c, undersh_oversh


def ellipse_from_abc(a, b, c, k=2.0, eps=1e-3, sym_tol=0.2):
    """
    From two points a,b (defining a line), build an ellipse centered at 0.
    Returns (a_semi, b_semi, angle), where angle is the major-axis angle (radians, CCW from +x).
    - Minor axis is along c (closest point on line ab to origin); length = 2*b_semi.
    - Major axis is perpendicular to c; length = 2*a_semi.
    - If a,b straddle symmetrically → circle (a=b).
    - Else a grows with projected span along the major direction; k controls “squish”.
    """

    rc = np.linalg.norm(c)

    # unit axes: minor along c, major perpendicular
    if rc < eps:
        u_minor = np.array([0.0, 1.0])             # fallback when line crosses origin
    else:
        u_minor = c / rc
    u_major = np.array([u_minor[1], -u_minor[0]])   # rotate CW (perp)

    # ensure major axis points in correct direction
    if (b - a) @ u_major < 0:
        u_major = -u_major

    # semi-minor
    b_semi = max(rc, eps)

    # projections of a,b onto major axis
    pa = float(a @ u_major)
    pb = float(b @ u_major)

    # symmetric straddle? -> circle through c
    sym = (np.sign(pa) != np.sign(pb)) and (
        abs(abs(pa) - abs(pb)) / max(abs(pa), abs(pb), 1e-9) < sym_tol
    )
    if sym:
        a_semi = b_semi
    else:
        span = max(abs(pa), abs(pb))
        ratio = span / max(b_semi, eps)
        # ensure we at least cover span; grow beyond by k*(ratio-1)
        a_semi = max(span, b_semi * (1.0 + k * max(0.0, ratio - 1.0)))

    # cap major when the line passes ~through the origin (flat case a)
    CAP_RC = 1e-2  # or use 10*eps
    if rc < CAP_RC:
        a_semi = min(a_semi, min(np.linalg.norm(a), np.linalg.norm(b)))

    # Aspect cap to tame extreme flats
    R_MAX = 5.0  # tune 3–5
    if b_semi > 0 and a_semi / b_semi > R_MAX:
        a_semi = R_MAX * b_semi

    # major-axis theta (for drawing/rotation)
    # theta = float(np.arctan2(u_major[1], u_major[0]))
    theta = float(np.mod(np.arctan2(u_major[1], u_major[0]), np.pi))
    return a_semi, b_semi, theta, u_major, u_minor  # u_major and u_minor are given as [x, y] directions on unit circle


def gen_xy_takeoff(a_semi, b_semi, u_major, u_minor, takeoff_frames, undersh_oversh, a, b, c):
    """
    Start at [0,0], point toward c (= b_semi*u_minor), do exactly 1 rotation, end at c.
    Constant speed via arc-length reparam. Returns (N,2) float32.
    """

    # oversample curve
    M = 1200
    s = np.linspace(0.0, 1.0, M, dtype=np.float64)

    # start at t = π/2 + 2π (E points along +u_minor ⇒ toward c), decrease to π/2
    q = 1.6  # end-ease (>=1). Increase slightly if still failing.
    s_e = 1.0 - (1.0 - s) ** q
    t = (np.pi / 2.0 + 2.0 * np.pi) - 2.0 * np.pi * s_e

    E = (a_semi * np.cos(t)[:, None] * u_major[None, :] +
         b_semi * np.sin(t)[:, None] * u_minor[None, :])    # ellipse frame
    ovs_takeoff = (s[:, None] * E)   # spiral out (oversampled)

    if undersh_oversh == "undershoot":
        # need to decide how many frames to use
        num_frames_repair = 600
        line_x = np.linspace(ovs_takeoff[-1, 0], a[0], num_frames_repair + 1, endpoint=False)  # + 1 cuz first removed
        line_y = np.linspace(ovs_takeoff[-1, 1], a[1], num_frames_repair + 1, endpoint=False)
        line_xy = np.column_stack((line_x[1:], line_y[1:]))
        ovs_takeoff = np.vstack((ovs_takeoff, line_xy))

    # # # constant speed resample ===================
    # d = np.linalg.norm(np.diff(ovs_takeoff, axis=0), axis=1)  # diff([1, 2, 4, 8]) = [1, 2, 4]  distances between diffs
    # L = np.concatenate(([0.0], np.cumsum(d)))
    # L /= L[-1] if L[-1] > 0 else 1.0  # 0-1 normalization (since cumsum)
    # tau = np.linspace(0.0, 1.0, takeoff_frames, dtype=np.float64)
    # x = np.interp(tau, L, ovs_takeoff[:, 0])
    # y = np.interp(tau, L, ovs_takeoff[:, 1])  # linear interpolation
    # xy2_takeoff = np.column_stack((x, y)).astype(np.float32)
    speed1 = np.linalg.norm(b - a)
    xy2_takeoff = resample_ramped_speed(ovs_takeoff, takeoff_frames, speed1=speed1)

    # Overshoot: Replace resampled ellipse tail with straight line
    dist_c_to_a = np.linalg.norm(a - c)
    if undersh_oversh == 'overshoot' and dist_c_to_a > 3:
        u_ab = (b - a)
        u_ab /= (np.linalg.norm(u_ab) or 1.0)  # ensures 0-1
        xy2_takeoff = trim_overshoot(xy2_takeoff, u_ab, a, b, c)
        num_frames_repair = takeoff_frames - len(xy2_takeoff)

        line_x = np.linspace(xy2_takeoff[-1, 0], a[0], num_frames_repair + 1, endpoint=False)  # + 1 cuz first removed
        line_y = np.linspace(xy2_takeoff[-1, 1], a[1], num_frames_repair + 1, endpoint=False)
        line_xy = np.column_stack((line_x[1:], line_y[1:]))
        xy2_takeoff = np.vstack((xy2_takeoff, line_xy))

    # sanity: start at 0, end at c, last step smooth-ish
    xy2_takeoff[0] = 0.0
    xy2_takeoff[-1] = a
    # assert abs(float(xy2_takeoff[-2, 1] - xy2_takeoff[-1, 1])) < 6, "end not smooth in y"

    return xy2_takeoff


def resample_ramped_speed(ovs_takeoff, takeoff_frames, speed1=5.0, v_min=1e-3):
    L_cum, L = arc_length(ovs_takeoff)
    T = int(takeoff_frames) - 1
    if T <= 0:
        raise ValueError("takeoff_frames must be >= 2")

    t = np.linspace(0.0, 1.0, T, dtype=np.float64)
    S = smoothstep01(t).sum()
    denom = (T - S)
    v0 = (L - speed1 * S) / denom if denom > 1e-9 else speed1

    v = v0 + (speed1 - v0) * smoothstep01(t)
    v = np.maximum(v, v_min)
    v *= (L / v.sum())

    s = np.concatenate(([0.0], np.cumsum(v)))  # length == takeoff_frames
    tau = s / L

    tau_path = L_cum / L
    x = np.interp(tau, tau_path, ovs_takeoff[:, 0])
    y = np.interp(tau, tau_path, ovs_takeoff[:, 1])
    return np.column_stack((x, y)).astype(np.float32)


def trim_overshoot(xy2_takeoff, u_ab, a, b, c, min_keep=30, tol=1e-6):
    """
    Force undershoot wrt line a->b. Keep at least min_keep points.
    Chooses a single best cut index near the end instead of popping many times.
    """

    # # search only in the tail to avoid huge trims
    # i0 = max(min_keep, len(xy2_takeoff) - 120)            # scan window start
    # proj_pos = (c - xy2_takeoff[i0:]) @ u_ab           # position projection along a->b
    # dv = np.diff(xy2_takeoff[i0:], axis=0)
    # proj_vel = dv @ u_ab                      # velocity projection along a->b
    #
    # # candidate indices where both projections no longer point "toward c"
    # ok = np.where((proj_pos[:-1] <= tol) & (proj_vel <= tol))[0]
    # if len(ok) > 0:
    #     cut = i0 + int(ok[0] + 1)          # keep up to this point
    # else:
    #     # fallback: pick the index with the smallest (closest to 0) position projection
    #     j = int(np.argmin(np.abs(proj_pos[:-1])))
    #     cut = i0 + j + 1
    #
    # # enforce keep floor
    # cut = max(cut, min_keep)
    # return xy2_takeoff[:cut].astype(np.float32)

    # AFTER (trim by signed projection; ensure we search far enough back)
    # scan_max = 600  # widen if your spirals are long
    # i0 = max(min_keep, len(xy2_takeoff) - scan_max)

    # REWRITE YOURSELF. WHILE LOOP UNTIL xy2_takeoff[-2] -> xy2_takeoff[-1]
    # i0 = min_keep
    # ADF =(c - xy2_takeoff)
    # s = (c - xy2_takeoff) @ u_ab  # signed distance along a->b
    # tail = s[i0:]
    #
    # # first index in the tail where we stop overshooting (s <= 0)
    # hits = np.nonzero(tail <= 0.0)[0]
    # if len(hits):
    #     cut = i0 + int(hits[0])  # keep up to this index (inclusive)
    # else:
    #     # never crosses: pick closest approach to the perpendicular (s ~ 0)
    #     j = int(np.argmin(np.abs(tail)))
    #     cut = i0 + j
    #
    # cut = max(min_keep, min(cut, len(xy2_takeoff) - 1))
    # xy2_takeoff = xy2_takeoff[:cut + 1]
    # return xy2_takeoff

    dv = np.diff(xy2_takeoff, axis=0)

    i_last_alignment_v = -999
    i_last_alignment_p = -999
    ovs_desperation = False

    for i in range(len(xy2_takeoff) - 1, 0, -1):  # only goes down to i=1 but that's desirable
        c0 = xy2_takeoff[i - 1]
        c1 = xy2_takeoff[i]

        dv0 = dv[i - 1]  #

        alignment_v = dv0 @ u_ab  # pos=same moving in same direction
        alignment_p = (a - c0) @ u_ab  # pos = behind

        # if alignment_v > 0 and i_last_alignment_v < 0:
        #     i_last_alignment_v = i
        # if alignment_p > 0 and i_last_alignment_p < 0:
        #     i_last_alignment_p = i
        if alignment_v > 0 and alignment_p > 0:
            # OBS high i here means lots is KEPT
            xy2_takeoff = xy2_takeoff[:i, :]
            break

        if i < min_keep:
            ovs_desperation = True
            break

    # temp desperation: remove last quarter
    if ovs_desperation:
        # print("overshoot desperation")
        xy2_takeoff = xy2_takeoff[0:int(0.75 * len(xy2_takeoff)), :]

    return xy2_takeoff


def post_smoothing():  # TODO
    pass