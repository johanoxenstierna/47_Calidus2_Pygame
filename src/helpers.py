
import numpy as np


def top_parent(body):
    """
    Return the top-level parent.
    """
    if body.parent.id == '0':  # top
        return body
    else:
        return body.parent


def phi_from_coord(xy):
    """
    0pi starts at [0, -y] and then goes clockwise one rotation to 2pi.
    There are no negative phi's.
    """
    phi = (np.pi / 2 - np.arctan2(-xy[1], xy[0])) % (2 * np.pi)
    return phi


def phi_to_theta(phi):
    """Screen φ (0 at [0,-r], clockwise) → math θ (0 at [+r,0], CCW)."""
    return phi - np.pi/2


def theta_to_phi(theta):
    """Math θ → screen φ."""
    return (theta + np.pi/2) % (2 * np.pi)