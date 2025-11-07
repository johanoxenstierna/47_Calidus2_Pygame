
import numpy as np
import P

def R_gi_():

    R = []

    if P.ONE_ROCKET == 1:
        R = [
            {'t': 20,
             # 'od': ['6_Io', '4_GSS'],
             'od': ['4_Earth', '6_Io'],
             # 'od': ['4_Earth', '4_GSS'],
             'type': 'first',
             'hPerR': 1,
             'c': 'Sputnik - ISS (67)'
             }
        ]

        # R = [{
        #         'init_frame': 60,
        #         # 'od': ['4_Earth', '6_Io'],
        #         'od': ['5_Io', '4_Moon'],
        #         'type': 'first',
        #         'fdist': 10000,
        #         'c': 'asdf'
        # }
        # ]

    else:
        R = [
            {'t': 20,
             'od': ['Nauvis', 'GSS'],
             'type': 'first',
             'hPerR': 10,
             'c': 'Sputnik - ISS (67)'
             },
            {'t': 68,
             'od': ['Nauvis', 'GSS'],
             'type': 'mtrl',
             'hPerR': 1,
             'c': "GSS factory: Most of these rockets carry construction materials and water (108)."
             },
            {'t': 69,
             'od': ['Nauvis', 'Corus'],
             'type': 'first',
             'hPerR': 20,
             'c': "Moon factory. Same expansion programme (109)"
             },
            {'t': 140,
             'od': ['Corus', 'GSS'],
             'type': 'mtrl',
             'hPerR': 5,
             'c': "Material from Moon to GSS"
             },
            {'t': 170,
             'od': ['Nauvis', 'Petussia'],
             'type': 'first',
             'hPerR': 30,
             'c': 'Mars factory'
             },
            {'t': 175,
             'od': ['Petussia', 'GSS'],
             'type': 'mtrl',
             'hPerR': 4,
             'c': 'Material from Mars to GSS. Doubles through time'
             },
            {'t': 190,
             'od': ['GSS', 'Nauvis'],
             'type': 'mtrl',
             'hPerR': 2,
             'c': 'Material from GSS to Earth (mostly scrap)'
             },
            {'t': 250,
             'od': ['Nauvis', 'Perodome'],
             'type': 'first',
             'hPerR': 40,
             'c': 'Venus factory'
             },
            {'t': 255,
             'od': ['Perodome', 'GSS'],
             'type': 'mtrl',
             'hPerR': 5,
             'c': 'Material from Venus to GSS'
             },
            {'t': 270,
             'od': ['Nauvis', 'Everglade'],
             'type': 'first',
             'hPerR': 15,
             'c': 'Io factory'
             },
            {'t': 275,
             'od': ['Everglade', 'GSS'],
             'type': 'mtrl',
             'hPerR': 20,
             'c': 'Material from Io to GSS'
             },
            {'t': 316,
             'od': ['Nauvis', 'GSS'],
             'type': 'mtrl',
             'hPerR': 0.5,
             'c': 'Expanded deliveries to GSS'
             },
            {'t': 317,
             'od': ['GSS', 'Corus'],
             'type': 'water',
             'hPerR': 5,
             'c': 'Mostly water for expanded production'
             },
            {'t': 318,
             'od': ['GSS', 'Petussia'],
             'type': 'water',
             'hPerR': 20,
             'c': 'Mostly water for expanded production'
             },
            {'t': 330,
             'od': ['Nauvis', 'Cyclops'],
             'type': 'first',
             'hPerR': 40,
             'c': 'Ganymede factory'
             },
            {'t': 335,
             'od': ['Cyclops', 'GSS'],
             'type': 'mtrl',
             'hPerR': 20,
             'c': 'Material from Ganymede to GSS.'
             },
            {'t': 350,
             'od': ['Nauvis', 'Astro0'],
             'type': 'first',
             'hPerR': 30,
             'c': 'Asteroid Belt factory'
             },
            {'t': 355,
             'od': ['Astro0', 'GSS'],
             'type': 'mtrl',
             'hPerR': 30,
             'c': 'Material from Asteroid Belt to GSS'
             },
            {'t': 384,
             'od': ['Nauvis', 'Molli'],
             'type': 'first',
             'hPerR': 20,
             'c': 'Near Earth Asteroid factory'
             },
            {'t': 390,
             'od': ['Molli', 'GSS'],
             'type': 'mtrl',
             'hPerR': 2,
             'c': 'Mostly rare petrochemical products'
             },
            {'t': 391,
             'od': ['Molli', 'GSS'],
             'type': 'water',
             'hPerR': 2,
             'c': 'Earth-based water at GSS is finally replaced by water from Moon and NEAs.'
             },
            {'t': 392,
             'od': ['Molli', 'Nauvis'],
             'type': 'mtrl',
             'hPerR': 2,
             'c': 'Rare but highly demanded materials from NEAs'
             },
            {'t': 450,
             'od': ['GSS', 'Astro0'],
             'type': 'mtrl',
             'hPerR': 20,
             'c': 'Advanced building materials to Asteroid Belt'
             },
            {'t': 480,
             'od': ['Molli', 'Astro0'],
             'type': 'mtrl',
             'hPerR': 20,
             'c': 'Mostly rare petrochemical products'
             },
            {'t': 510,
             'od': ['Petussia', 'Astro0'],
             'type': 'mtrl',
             'hPerR': 20,
             'c': 'Water compression chemicals.'
             },
            {'t': 530,
             'od': ['Astro0', 'Petussia'],
             'type': 'water',
             'hPerR': 15,
             'c': 'Water from Asteroid Belt to Mars.'
             }
        ]

    return R


def translate(_R_gi):

    """
    Nauvis = Earth
    GSS = Geostationary Space Station
    Corus = Moon
    Petussia = Mars
    Perodome = Venus
    Everglade = Io
    Cyclops = Ganymede
    Astro0 = Asteroid belt outpost between Mars Jupiter.
    Molli = Near Earth Asteroid with ample water.
    Ogun = Mercury
    Darkflare = interstellar object
    """

    key = {
        'Nauvis': '4_Earth',
        'GSS': '4_GSS',
        'Corus': '4_Moon',
        'Petussia': '5_Mars',
        'Perodome': '3_Venus',
        'Everglade': '6_Io',
        'Cyclops': '6_Ganymede',
        'Astro0': 'Astro0b',
        'Molli': '4_NEA'
    }

    '''
    NEED TO DECIDE ANIMATION FRAMES TO GET t -> init_frame
    1h   = 10 years   = 3650 frames
    10h  = 100 years  = 36500 frames
    100h = 1000 years = 365000 frames
    600h = 6000 years = 2190000 frames
     
    # 20h = 200 years and
    # if earth 1 year=365 frames -> 36500 frames = 100 years -> 73000 = 200 years
    # 600h = 6000 years  -> 
    # 365000days = 1000 years -> 6000 years = 2 190 000 days and then /=1000 gives 2190 total frames for full animation.
    '''
    # BB = key.keys()
    # aasdf = sorted(_R_gi, key=lambda x:x['t'])
    standard_earth_period_frames = int(365 / P.SPEED_MULTIPLIER)

    for gi in _R_gi:
        for k, _od in enumerate(gi['od']):
            if _od in key.keys():
                gi['od'][k] = key[_od]
            # _od = key[_od]  # does not work

        '''1 hour in game = 10 years IRL. 1 year = 365 frames, so 3650 frames'''
        gi['init_frame'] = int(gi['t'] * 10 * standard_earth_period_frames / 500) # * 10 bcs 1h=10years  (t is in hours)
        gi['fdist'] = int(gi['hPerR'] * 10 * standard_earth_period_frames / 1000)  # OBS MORE DIV MAKES MORE OF THEM

    # for gi in _R_gi:  # after debug
    #     del gi['t']
    #     del gi['hPerR']

    _R_gi1 = []
    for roc_gi in _R_gi:
        if roc_gi['od'][0] in P.OBJ_TO_SHOW and roc_gi['od'][1] in P.OBJ_TO_SHOW:
            _R_gi1.append(roc_gi)

    return _R_gi1
