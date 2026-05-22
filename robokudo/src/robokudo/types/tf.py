"""Transform and pose types for Robokudo.

This module provides types for representing poses, positions, and transforms
in 3D space. It includes both timestamped and non-timestamped variants.

The module supports:

* 6-DOF poses (translation and rotation)
* 3-DOF positions (translation only)
* Reference frame specifications
* Timestamped variants
* Parent-child frame relationships
"""

from dataclasses import dataclass, field
from typing_extensions import List
from builtin_interfaces.msg import Time
from robokudo.types.core import Type


@dataclass
class Pose(Type):
    """6-DOF pose representation.

    Represents a full 6-degree-of-freedom pose with:
    * 3D translation (x,y,z)
    * 3D rotation as quaternion (x,y,z,w)
    """

    rotation: List[float] = field(default_factory=list)
    """
    Quaternion rotation of the pose in x,y,z,w order
    """

    translation: List[float] = field(default_factory=list)
    """
    Translation of the pose in x,y,z order
    """


@dataclass
class Position(Type):
    """3-DOF position representation.

    Represents a 3-degree-of-freedom position with:
    * 3D translation (x,y,z)
    """

    translation: List[float] = field(default_factory=list)


@dataclass
class StampedPose(Pose):
    """Timestamped 6-DOF pose with reference frame."""

    frame: str = str("")
    """
    Reference frame identifier
    """

    timestamp: Time = field(default_factory=Time)


@dataclass
class StampedPosition(Position):
    """Timestamped 3-DOF position with reference frame."""

    frame: str = str("")
    """
    Reference frame identifier
    """

    timestamp: Time = field(default_factory=Time)


@dataclass
class StampedTransform(StampedPose):
    """Timestamped transform between coordinate frames.

    Extends StampedPose with:
    * Child frame ID for specifying transform target

    Used to represent coordinate transformations between frames.
    """

    child_frame: str = str("")
    """
    Target frame identifier
    """
