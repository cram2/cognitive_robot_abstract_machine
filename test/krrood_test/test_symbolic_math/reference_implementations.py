from typing_extensions import Tuple, Union, List

import numpy as np

from semantic_digital_twin.datastructures.types import AnyMatrix4x4, NpMatrix4x4
from semantic_digital_twin.spatial_types.derivatives import Derivatives


def shortest_angular_distance(from_angle, to_angle):
    """Given 2 angles, this returns the shortest angular
    difference.  The inputs and ouputs are of course radians.

    The result would always be -pi <= result <= pi. Adding the result
    to "from" will always get you an equivelent angle to "to".
    """
    return normalize_angle(to_angle - from_angle)


def normalize_angle(angle):
    """Normalizes the angle to be -pi to +pi
    It takes and returns radians."""
    a = normalize_angle_positive(angle)
    if a > np.pi:
        a -= 2.0 * np.pi
    return a


def normalize_angle_positive(angle):
    """Normalizes the angle to be 0 to 2*pi
    It takes and returns radians."""
    return angle % (2.0 * np.pi)
