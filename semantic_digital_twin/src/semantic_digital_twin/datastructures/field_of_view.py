from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class FieldOfView:
    """
    Represents the field of view of a camera sensor, defined by the vertical and
    horizontal angles of the camera's view.
    """

    vertical_angle: float = field(default=np.pi / 2)
    """
    The vertical angle of the camera's field of view, in radians.
    """

    horizontal_angle: float = field(default=np.pi / 2)
    """
    The horizontal angle of the camera's field of view, in radians.
    """
