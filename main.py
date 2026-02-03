

# import P
# import pygame

import numpy as np
import random

import P

random.seed(3)  # ONLY HERE.000000000000000000000
np.random.seed(3)  # ONLY HERE
import time

from src.gen_objects import GenObjects
from src.genesis_infos import _genesis, gen_incl_frames
from src.draw_helpers_pygame import *
from src.write_to_file import VideoWriter

g = GenObjects()
g.gis = _genesis()

incl_frames = gen_incl_frames(g.gis['Rockets'])

o0 = g.gen_base_object()
g.gen_planets_moons(o0)
g.gen_stars(o0)

if 'Rockets' in P.OBJ_TO_SHOW:
    # R = g.gen_rockets(o0)
    R = g.gen_rockets3000()  # ONLY ONE OR THE OTHER
else:
    R = None

print("Done preprocessing =============================\n")
'''PYGAME INIT =============================================='''

pygame.init()

W, H = P.MAP_DIMS
ZOOM_ON = bool(P.LU_2X_ZOOM)

screen = pygame.display.set_mode((W, H))
world_surf = pygame.Surface((int(W * P.SS_RENDER), int(H * P.SS_RENDER)), flags=pygame.SRCALPHA)
# world_surf = pygame.Surface((int(W), int(H)), flags=pygame.SRCALPHA)  # PEND DEL

backgr_surf = gen_backgr(g.pics)
# backgr_scene = pygame.transform.smoothscale(backgr_surf, (W*P.SS_RENDER, H*P.SS_RENDER))  # PEND DEL
backgr_screen = pygame.transform.smoothscale(backgr_surf, (W, H))
# backgr_surf = gen_backgr(g.pics)  # PEND DEL

if P.WRITE:
    vw = VideoWriter('./vids/vid_' + str(P.WRITE) + '_' + P.VID_SINGLE_WORD + '_' + '.mp4', P.MAP_DIMS, P.FPS)

D = []

print("Done Pygame init =============================\n")  # OBS D is always empty here
'''ANIMATION LOOP ==========================================='''

clock = pygame.time.Clock()
time0 = time.perf_counter()
i = 0
i_step = 1
incl_idx = 0

running = True
while running:  # good so time can be
# for i in range(0, P.FRAMES_STOP):

    if i % 100 == 0:
        print(i)

    # i_step = i_steps[i]

    # If ESC pressed then quit =============
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                running = False
    # =======================================

    # =======================================
    D_scene = []  # Drawables (objects shown in the animation): Rebuilt each iteration!
    for o1_id, o1 in o0.O1.items():
        if i == o1.frame_ss[0]:
            o1.start_draw(D_scene)  # creates surfaces and rects
        elif i > o1.frame_ss[0]:# and i < o1.frame_ss[1]:
            o1.age += i_step  # bodies no longer use this! But _0 still does!
            o1.update_draw(D_scene, i)
        # elif i >= o1.frame_ss[1]:
        #     pass

    if 'Rockets' in P.OBJ_TO_SHOW:
        for rocket in R:
            if i == rocket.frame_ss[0]:
                rocket.update_draw(D_scene)
            elif i > rocket.frame_ss[0] and i < rocket.frame_ss[1] - i_step:
                rocket.age += i_step
                rocket.update_draw(D_scene)
            elif i == rocket.frame_ss[1]:
                pass

    if i < incl_frames[incl_idx]:
        i += 1
        continue  # not yet reached next included frame
    elif i == incl_frames[incl_idx]:
        incl_idx += 1  # move to next target
        # draw this frame

    # BLITTING CODE ==================================
    # world_surf.blit(backgr_scene, (0, 0))  #
    world_surf.fill((0,0,0,0))

    for _, type, tuple in sorted(D_scene, key=lambda x:x[0]):
        if type == 'rocket':
            # color = (tuple[0], tuple[0], tuple[0])
            color = (255, 255, 255)
            pos = tuple[1]
            r = tuple[2]
            alpha = tuple[3]

            # if alpha < 255:
            surface = pygame.Surface((r * 2, r * 2), pygame.SRCALPHA)
            pygame.draw.circle(surface, color + (alpha,), (r, r), r)
            world_surf.blit(surface, (pos[0] - r, pos[1] - r))
            # else:
            # pygame.draw.circle(screen, color, pos, r)
        elif type == 'star':
            rgba = tuple[0]
            pos = tuple[1]
            r = tuple[2]
            # alpha = tuple[3]

            # if alpha < 255:
            surface = pygame.Surface((r * 2, r * 2), pygame.SRCALPHA)
            pygame.draw.circle(surface, rgba, (r, r), r)
            world_surf.blit(surface, (pos[0] - r, pos[1] - r))
        else:
            # screen.blit(tuple[0], tuple[1])
            world_surf.blit(tuple[0], tuple[1])

    final = get_final_frame(world_surf)
    screen.blit(backgr_screen, (0, 0))
    screen.blit(final, (0, 0))

    if P.WRITE == 0:
        draw_HUD_debug(i, screen)  # overlay stays crisp & always visible

    # draw_HUD(i, screen)

    if P.WRITE:
        vw.write_frame(screen)  # or vw.write_frame(final) if your writer accepts a Surface  # <--
    # else:
    pygame.display.flip()  # single flip per frame  # <--

    clock.tick(P.FPS)
    i += i_step

    if i >= P.FRAMES_STOP - 1:
        print("DONE LOOP ==========================================================")
        running = False

    if running == False:
        break

time1 = time.perf_counter() - time0
print(f"time1: {time1}")
# min_vid = (P.FRAMES_STOP / P.FPS) / 60
# sec_per_1_min = int(time1 / min_vid)
# print("sec_per_1_min: " + str(sec_per_1_min))

if P.WRITE:
    vw.close()

pygame.quit()

