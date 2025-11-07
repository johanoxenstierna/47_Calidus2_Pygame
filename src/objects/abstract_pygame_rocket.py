import P


class AbstractPygameRocket:
    def __init__(_s):
        _s.xy = None
        _s.age = 0
        _s.frame_ss = [None, None]  # start and stop frames
        _s.type = 'rocket'

    def update_draw(_s, D_scene):

        x, y = _s.xy[_s.age] * P.SS_RENDER
        z = _s.zorders[_s.age]
        alpha = _s.alphas[_s.age]
        # alpha = 255

        # Convert color from float to 0–255
        # color = tuple(int(c * 255) for c in _s.color[_s.age])
        color = 255#int(_s.color[_s.age] * 255)

        # Radius: 1 or 2 depending on resolution
        radius = 1

        # Store drawing command into D
        D_scene.append((z, _s.type, (color, (int(x), int(y)), radius, alpha)))