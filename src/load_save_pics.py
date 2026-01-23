
from pathlib import Path
from PIL import Image
import numpy as np
import P


def load_pics(pics, filepath):
    """
    """
    PATH_IN = (P.BASE_DIR / filepath).resolve()
    png_paths = [f for f in PATH_IN.iterdir() if f.is_file() and f.suffix == '.png']

    # load into dict {stem: numpy array}
    for f in png_paths:
        img = Image.open(f).convert("RGBA")  # force RGBA for consistency
        pics[f.stem] = np.array(img)  # shape (H, W, 4), dtype=uint8


    # pics['backgr_b'] = imread('./pictures/backgr_b.png')
    # pics['backgr_55'] = imread('./pictures/backgr_55.png')
    #
    # # if 'Calidus' in P.OBJ_TO_SHOW:
    # pics['0_black'] = imread('./pictures/Calidus1/0_cal/0_black.png')
    # pics['0_sun'] = imread('./pictures/Calidus1/0_cal/0_sunR.png')
    # pics['0_red'] = imread('./pictures/Calidus1/0_cal/0_light.png')  # OBS
    # pics['0_mid'] = imread('./pictures/Calidus1/0_cal/0_mid.png')
    # pics['0_light'] = imread('./pictures/Calidus1/0_cal/0_light.png')
    # pics['0h_red'] = imread('./pictures/Calidus1/0_cal/0h_red.png')
    # pics['0h_mid'] = imread('./pictures/Calidus1/0_cal/0h_mid.png')
    # pics['0h_light'] = imread('./pictures/Calidus1/0_cal/0h_light.png')
    #
    # if '2_Mercury' in P.OBJ_TO_SHOW:
    #     pics['2_Mercury'] = [imread('./pictures/Calidus1/planets/1_OgunD.png'),
    #                       imread('./pictures/Calidus1/planets/1_Ogun.png'),
    #                       imread('./pictures/Calidus1/planets/1_OgunL.png')]
    #
    # if '3_Venus' in P.OBJ_TO_SHOW:
    #     pics['3_Venus'] = [imread('./pictures/Calidus1/planets/2_VenusD.png'),
    #                       imread('./pictures/Calidus1/planets/2_Venus.png'),
    #                       imread('./pictures/Calidus1/planets/2_VenusL.png')]
    #
    # # ------------------------
    #
    # if '4_Earth' in P.OBJ_TO_SHOW:
    #     pics['4_Earth'] = [imread('./pictures/Calidus1/planets/3_NauvisD.png'),
    #                       imread('./pictures/Calidus1/planets/3_Nauvis.png'),
    #                       imread('./pictures/Calidus1/planets/3_NauvisL.png')]
    #
    # if 'GSS' in P.OBJ_TO_SHOW:
    #     pics['GSS'] = [imread('./pictures/Calidus1/planets/3_GSSD.png'),
    #                    imread('./pictures/Calidus1/planets/3_GSS.png')]
    #     '''IN o1.py: OBS _s.scale = np.ones((P.FRAMES_TOT_BODIES,))'''
    #
    # if '4_Moon' in P.OBJ_TO_SHOW:
    #     pics['4_Moon'] = [imread('./pictures/Calidus1/planets/3_MolliD.png'),
    #                      imread('./pictures/Calidus1/planets/3_Molli.png'),
    #                      imread('./pictures/Calidus1/planets/3_MolliL.png')]
    #
    # # ----------------------
    #
    # if 'Mars' in P.OBJ_TO_SHOW:
    #     pics['Mars'] = [imread('./pictures/Calidus1/planets/4_MarsD.png'),
    #                       imread('./pictures/Calidus1/planets/4_Mars.png'),
    #                       imread('./pictures/Calidus1/planets/4_MarsL.png')]
    #
    # if '6_Jupiter' in P.OBJ_TO_SHOW:
    #     pics['6_Jupiter'] = [imread('./pictures/Calidus1/planets/6_JupiterD.png'),
    #                       imread('./pictures/Calidus1/planets/6_Jupiter.png'),
    #                       imread('./pictures/Calidus1/planets/6_JupiterL.png')]
    #
    # if 'Everglade' in P.OBJ_TO_SHOW:
    #     pics['Everglade'] = [imread('./pictures/Calidus1/planets/6_EvergladeD.png'),
    #                       imread('./pictures/Calidus1/planets/6_Everglade.png'),
    #                       imread('./pictures/Calidus1/planets/6_EvergladeL.png')]
    #
    # if 'Petussia' in P.OBJ_TO_SHOW:
    #
    #     pics['Petussia'] = [imread('./pictures/Calidus1/planets/6_PetussiaD.png'),
    #                          imread('./pictures/Calidus1/planets/6_Petussia.png'),
    #                          imread('./pictures/Calidus1/planets/6_PetussiaL.png')]
    #
    # if 'Astro0' in P.OBJ_TO_SHOW:
    #     pics['Astro0'] = imread('./pictures/Calidus1/z_Astro0_masked.png')
    #
    # if 'Astro0b' in P.OBJ_TO_SHOW:
    #     pics['Astro0b'] = [imread('./pictures/Calidus1/planets/3_GSSD.png'),
    #                        imread('./pictures/Calidus1/planets/3_GSS.png')]
    #
    # if 'Saturn' in P.OBJ_TO_SHOW:
    #     pics['Saturn'] = [imread('./pictures/Calidus1/planets/7_SaturnD.png'),
    #                       imread('./pictures/Calidus1/planets/7_Saturn.png'),
    #                       imread('./pictures/Calidus1/planets/7_SaturnL.png')]
    #
    # if 'Uranus' in P.OBJ_TO_SHOW:
    #     pics['Uranus'] = [imread('./pictures/Calidus1/planets/8_UranusD.png'),
    #                       imread('./pictures/Calidus1/planets/8_Uranus.png'),
    #                       imread('./pictures/Calidus1/planets/8_UranusL.png')]
    #
    # if 'Neptune' in P.OBJ_TO_SHOW:
    #     pics['Neptune'] = [imread('./pictures/Calidus1/planets/9_NeptuneD.png'),
    #                       imread('./pictures/Calidus1/planets/9_Neptune.png'),
    #                       imread('./pictures/Calidus1/planets/9_NeptuneL.png')]
    #


def load_pics_bodies(pics):

    for id in P.OBJ_TO_SHOW:

        if id in ['Rockets', 'Sun', 'Astro0']:
            continue

        # if P.GEN_DL_PIC_BANK == id or P.USE_DL == 0:
        if P.USE_DL == 0:  # we don't know whether there is anything in bodies/6_Jupiter
            pics[id] = [pics[id]]  # OBS OBS FOR GENERATION SEE BELOW
        # elif P.USE_DL == 1 and P.GEN_DL_PIC_BANK == '9_Neptune':  # generate and show
        #     pics[id] = [pics[id]]
        elif P.USE_DL == 1 and P.GEN_DL_PIC_BANK == '9_Neptune':  # generate ALL and show
            pics[id] = [pics[id]]  # IF FAIL, CHECK P.OBJ_TO_SHOW OR CHANGE 1 TO ID
        else:
            pics[id] = load_pics_DL(f'./pictures/bodies/{id}/')  # only show. HENCE, DONT USE THIS FOR BANK GEN

def load_pics_DL(filepath):

    PATH_IN = (P.BASE_DIR / filepath).resolve()
    png_paths = [f for f in PATH_IN.iterdir() if f.is_file() and f.suffix == '.png']
    png_paths_sorted = sorted(png_paths, key=lambda f: int(f.stem))
    pics = []
    for i in range(len(png_paths_sorted)):
        img = Image.open(png_paths_sorted[i]).convert("RGBA")  # force RGBA for consistency
        pics.append(np.array(img))
    return pics


def save_png(
    rgba_u8: np.ndarray,
    name: str,
    filepath: str = "./pictures/bodies1/"
) -> Path:
    """
    Save a uint8 RGBA image (H x W x 4) to <filepath>/<name>.png

    - If `filepath` is relative, it's interpreted relative to P.BASE_DIR (project root).
    - The directory is created if it doesn't exist.
    - The function returns the resolved Path to the saved file.

    Parameters
    ----------
    rgba_u8 : np.ndarray
        3D uint8 array of shape (H, W, 4).
    name : str
        File base name, with or without '.png' extension.
    filepath : str
        Directory to save into (default './pictures/bodies1/').

    Returns
    -------
    Path
        Absolute path to the saved PNG.
    """
    # Validate input
    if rgba_u8.dtype != np.uint8 or rgba_u8.ndim != 3 or rgba_u8.shape[2] != 4:
        raise ValueError("rgba_u8 must be a uint8 array with shape (H, W, 4).")

    # Resolve directory (relative to project root if not absolute)
    out_dir = Path(filepath)
    if not out_dir.is_absolute():
        out_dir = (Path(P.BASE_DIR) / filepath).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # Ensure .png extension
    filename = name if name.lower().endswith(".png") else f"{name}.png"
    out_path = out_dir / filename

    # Save
    Image.fromarray(rgba_u8, mode="RGBA").save(out_path)
    return out_path
