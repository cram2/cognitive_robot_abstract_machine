from __future__ import annotations

from dataclasses import dataclass


@dataclass
class FieldOfView:
    """
    Represents the field of view of a camera sensor, defined by the vertical and horizontal angles of the camera's view.
    """

    vertical_angle: float
    """
    The vertical angle of the camera's field of view, in radians.
    """

    horizontal_angle: float
    """
    The horizontal angle of the camera's field of view, in radians.
    """
