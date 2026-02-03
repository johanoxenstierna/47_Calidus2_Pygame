

import P
from src.objects.abstract_all import AbstractObject
from src.draw_helpers_pygame import *

from src.helpers_distributions import *

class AbstractPygameObject(AbstractObject):
    
    def __init__(_s):
        AbstractObject.__init__(_s)

        _s.surf = None
        _s.rect = None

        # ONLY USED BY o1 'body' type:
        # _s.surfs_DL = []
        # _s.rects_DL = []
        _s.DL = []
        _s.lifetime_dl = 200
        if P.USE_DL == 0:
            _s.lifetime_dl = P.N_ORBIT_SAMPLES

        _s.alpha_dl = np.sin(np.linspace(0, np.pi, _s.lifetime_dl))

        # _s.alpha_dl = np.clip(_s.alpha_dl, 0, 0.5)
        _s.alpha_dl = min_max_normalize_array(_s.alpha_dl, y_range=[0, 0.99])
        _s.offset_samples = None

    def start_draw(_s, D_scene):
        """
        Called when object becomes visible (drawn == 1).
        """

        if _s.type == 'body':

            S = P.N_ORBIT_SAMPLES / P.N_ORBIT_IMAGES
            _s.offset_samples = round((_s.gi['phi'] % (2 * np.pi)) * P.N_ORBIT_SAMPLES / (2 * np.pi))

            for k in range(len(_s.pics)):
                pic = _s.pics[k]
                surf = to_surface(pic)  # Assume self.pic is a NumPy array

                center = round(k * S)
                # start_i_orbit = (center - _s.lifetime_dl // 2) % P.N_ORBIT_SAMPLES

                start_i_orbit = (center - _s.offset_samples - _s.lifetime_dl // 2) % P.N_ORBIT_SAMPLES
                if _s.parent.id != '0':
                    start_i_orbit = (center - _s.parent.offset_samples - _s.lifetime_dl // 2) % P.N_ORBIT_SAMPLES

                _s.DL.append([surf, start_i_orbit])

        elif _s.type == '0_static':

            _s.surf = to_surface(_s.pic)

            r = int(_s.surf.get_height() / 2 * _s.scale) * P.SS_RENDER
            size = 2 * r

            _s.surf = pygame.transform.smoothscale(_s.surf, (size, size))
            _s.surf.set_alpha(int(_s.alpha * 255))

            _s.rect = _s.surf.get_rect()

            _s.rect.topleft = (P.MAP_DIMS[0] * P.SS_RENDER / 2 - r, P.MAP_DIMS[1] * P.SS_RENDER / 2 - r)
            D_scene.append((_s.zorder, _s.type, (_s.surf, _s.rect.topleft)))

        elif _s.type in ['0_', 'astro']:
            _s.surf = to_surface(_s.pic)
            _s.rect = _s.surf.get_rect()

            D_scene.append((_s.zorder, _s.type, (_s.surf, _s.rect.topleft)))

        elif _s.type == 'star':
            pass

    def update_draw(_s, D_scene, i):
        """
        This is doing work for ALL the animated objects (except rockets),
        so there's some redundancy for all if else.
        """

        if _s.type == 'body':  # planets/moons/space stations
            # offset_samples = round((_s.gi['phi'] % 2 * np.pi) * 32 / (2 * np.pi))
            DL_active = []

            i_orbit_move = _s.get_i_orbit(i)  # for position (6_Io around 6_Jupiter)
            i_orbit_light = i_orbit_move
            z = _s.zorders[i_orbit_move] + _s.parent.zorders[_s.parent.get_i_orbit(i)]
            if _s.parent.id != '0':
                i_orbit_light = _s.parent.get_i_orbit(i)
                z += 2000

            # xy..._abs means that parent coords have been added
            if P.XY01 == 0:  # No y_squeeze + tilt (ie circular)
                _s.xy0_abs = _s.parent.xy0_abs + _s.xy0[i_orbit_move]
                _s.xy_abs = _s.xy0_abs
            else:  # y_squeeze + tilt
                _s.xy2_abs = _s.parent.xy2_abs + _s.xy2[i_orbit_move]
                _s.xy_abs = _s.xy2_abs

            scale = _s.scale[i_orbit_move]

            cx = int(round(_s.xy_abs[0] * P.SS_RENDER))
            cy = int(round(_s.xy_abs[1] * P.SS_RENDER))

            for k, dl in enumerate(_s.DL):

                start_i_orbit = dl[1]

                d = (i_orbit_light - start_i_orbit) % P.N_ORBIT_SAMPLES  #

                if d < _s.lifetime_dl: # and dl[2] < 0:  # IS IT WITHIN TIME WINDOW WHEN ACTIVE?

                    surf = dl[0]

                    # SCALING ====
                    w, h = surf.get_size()
                    surf = pygame.transform.smoothscale(surf, (int(w * scale * P.SS_RENDER),
                                                               int(h * scale * P.SS_RENDER)))

                    # ALPHA ======
                    alpha = _s.alpha_dl[d]
                    surf.set_alpha(int(alpha * 255))

                    if P.USE_DL == 0:
                        surf.set_alpha(255)

                    # z = _s.zorders_DL[k][_s.age]  # NOT DONE YET

                    DL_active.append((surf, d, alpha, z))

            for surf, d, alpha, z in sorted(DL_active, key=lambda x: x[1]):  # d is now z_order. Youngest objects first
                rect = surf.get_rect(center=(cx, cy))
                D_scene.append((z, _s.type, (surf, rect.topleft)))

        elif _s.type == '0_static':
            pass
            # D_scene.append((_s.zorder, _s.type, (_s.surf, _s.rect.topleft)))
        elif _s.type in ['0_']:
            xy = _s.parent.xy0_abs
            rot = _s.rotation[_s.age]
            # scale = _s.scale[_s.age]
            scale = _s.scale
            surf = _s.surf

            # cx = int(round(xy[0] * P.SS_RENDER))
            # cy = int(round(xy[1] * P.SS_RENDER))

            # SCALING ====
            w, h = surf.get_size()
            surf = pygame.transform.smoothscale(surf, (int(w * scale * P.SS_RENDER),
                                                       int(h * scale * P.SS_RENDER)))

            # ROTATION ====
            surf = pygame.transform.rotate(surf, -np.degrees(rot))
            # rect = surf.get_rect(center=(int(xy[0]), int(xy[1])))
            rect = surf.get_rect(center=(xy[0] * P.SS_RENDER, xy[1] * P.SS_RENDER))

            # ALPHA ====
            # alpha_f = _s.alphas[_s.age]
            surf.set_alpha(int(_s.alphas[_s.age] * 255))
            # surf.set_alpha(250)

            # z = _s.zorders[_s.age]
            z = _s.zorder
            D_scene.append((z, _s.type, (surf, rect.topleft)))

        elif _s.type == 'astro':

            surf = _s.surf
            # alpha = int(_s.alphas[_s.age] * 255)
            # z = int(_s.zorders[_s.age])
            z = _s.zorder
            rot_dynamic = -np.degrees(_s.rotation[_s.age])  # negate for Pygame rotation
            rot_fixed = -np.degrees(0.3)  # static tilt ≈ -17.2°

            # Step 1: First rotation (dynamic)
            surf = pygame.transform.rotate(surf, rot_dynamic)

            # Step 2: Scale
            w, h = surf.get_size()
            surf = pygame.transform.smoothscale(surf, (int(w * 1.3 * P.SS_RENDER), int(h * 0.25 * P.SS_RENDER)))

            # Step 3: Second rotation (static tilt)
            surf = pygame.transform.rotate(surf, rot_fixed)

            # Step 4: Set alpha
            surf.set_alpha(255)

            # Step 5: Positioning
            xy = _s.parent.xy0_abs
            # rect = surf.get_rect(center=(P.MAP_DIMS[0] / 2, P.MAP_DIMS[1] / 2))  # rotate_around anchor
            rect = surf.get_rect(center=(xy[0] * P.SS_RENDER, xy[1] * P.SS_RENDER))  # rotate_around anchor

            D_scene.append((z, _s.type, (surf, rect.topleft)))

        elif _s.type == 'star':
            """(())"""
            # D_scene.append((z, _s.type, (rgba, (int(x), int(y)), radius)))
            D_scene.append((1000, _s.type, \
                            (
                                tuple(_s.rgb[i, :].tolist()) + (255,),
                                (int(_s.info['xy'][0]), int(_s.info['xy'][1])),
                                _s.info['radius'])
                            )
                           )





