"""
Names shared across RoboKudo: the package, its test ROS node and its loggers, and
:class:`Region3DWithName`.
"""

from enum import Enum
from dataclasses import dataclass

PACKAGE_NAME: str = "robokudo"
"""The name of the RoboKudo package"""

TEST_ROS_NODE_NAME: str = "robokudo_test"
"""The name used for RoboKudo test nodes"""

NAME: str = "robokudo"
"""The base name for RoboKudo"""

#:
LOGGING_IDENTIFIER_MAIN: str = PACKAGE_NAME
"""Logging name constant for core RoboKudo functionality. Other RK packages may choose to use others!"""

LOGGING_IDENTIFIER_MAIN_EXECUTABLE: str = "robokudo.main"
"""Logging name constant for core RoboKudo functionality. Other RK packages may choose to use others!"""

LOGGING_IDENTIFIER_QUERY: str = "robokudo.query"
"""Logging name constant for core RoboKudo functionality. Other RK packages may choose to use others!"""


@dataclass(kw_only=True)
class Region3DWithName:
    """
    A named 3D region with position, orientation and size.

    This class represents a 3D region in space with a name identifier.
    It includes position coordinates, quaternion orientation, and size
    dimensions.
    """

    class PoseType(Enum):
        """Enum representing the supported pose types for regions."""

        EULER = "euler"
        QUATERNION = "quaternion"

    name: str = ""
    """Identifier for this region"""

    position_x: float = 0.0
    """X coordinate of the region's position"""

    position_y: float = 0.0
    """Y coordinate of the region's position"""

    position_z: float = 0.0
    """Z coordinate of the region's position"""

    orientation_x: float = 0.0
    """X component of the euler or quaternion orientation"""

    orientation_y: float = 0.0
    """Y component of the euler or quaternion orientation"""

    orientation_z: float = 0.0
    """Z component of the euler or quaternion orientation"""

    orientation_w: float = 1.0
    """W component of the quaternion orientation, unused when `pose_type` is `PoseType.EULER`"""

    x_size: float = 0.0
    """Size of the region in X dimension"""

    y_size: float = 0.0
    """Size of the region in Y dimension"""

    z_size: float = 0.0
    """Size of the region in Z dimension"""

    frame: str = ""
    """The reference frame this region is defined in."""

    pose_type: PoseType = PoseType.QUATERNION
    """Type of the pose defining the regions orientation."""
