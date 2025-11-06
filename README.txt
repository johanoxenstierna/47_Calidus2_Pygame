
Goal: Pygame solar‑system + rocket travel animation. The final output is an mp4 video, so it's ok if the code isn't super fast, but we're still trying to keep things relatively fast.
Render path: draw to world surface (supersampled), then crop/scale via 2× zoom (optional) → blit to 1080p screen → (optional) write video.
Coords: origin at top‑left; y increases down (NumPy rows convention).
Time base: frame index i; orbital motion derives from i (with optional global speed multiplier).

===
Project modules and chronology of running
P.py - most important/global parameters/options. They are read only (in this project).
O0_info/ - O1 contains one info dict per planet (and dicts inside the planet dicts for moons). There is also one dict for "astro0" i.e. the asteroid belt outpost but it behaves same way as a planet. sun_gi.py contains infos for the sun animation (multiple blended sun pictures). R_gi.py contains infos for rockets.
src/ ...
genesis_infos.py - loads and initializes infos from O0_info.
gen_objects.py -  class that holds almost everything that is then used in pygame loop.
load_save_pics.py and m_functions.py are helping.
image_mask.py - work in progress to generate shadow/light for bodies oribiting the sun.
objects/ - o0 is the placeholder object for all objects in the solar system. o1 contains all (or almost) logic of how solar system bodies are moving around in the solar system 0-2pi. rocket is similar but for the rockets. The abstract modules are used to avoid redundancy.
main.py - runs the pygame loop after calling preprocessing functions for solar system bodies/rockets etc. Should be simple and short.

===
Types of objects (as denoted in e.g. main.py and abstract_pygame.py)
backgr - background png (blends of 2 images)
body - planets/moons. Note that body animations use blending of multiple png's. Note 'pics_planet' and 'DL' denotes 'DarkLight' and basically several pictures are blended based on where the body is in its orbit.
0_static - a non-moving png used as part of the sun visualization
0_ - moving png's for the sun visualization
astro - moving (rotating) png for asteroid belt where the alpha layer is very much used.
rocket - visualized using dots by pygame.draw.circle

===

Coordinates like 'xy_t' and 'xy' for bodies denote the 'center' of the body.



