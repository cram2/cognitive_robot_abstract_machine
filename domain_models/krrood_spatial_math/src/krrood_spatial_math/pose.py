"""
Poses and orientations: frames given in another frame, which a rigid change of frame
moves by composing with them.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from typing_extensions import Self, TypeVar

from krrood_spatial_math.coordinate_system import (
    CoordinateRepresentation,
    RigidCoordinateChange,
)

# %% where a frame lies and which way it faces


class Pose(CoordinateRepresentation, ABC):
    """
    Where a frame's origin lies and which way its axes point, given in another frame:

    the value of the rigid change from the posed frame into the frame it is given in.
    """

    @abstractmethod
    def to_coordinate_change(self) -> RigidCoordinateChange:
        """
        :return: The change from the posed frame into the frame this pose is given in.
        """

    @classmethod
    @abstractmethod
    def from_coordinate_change(cls, *, change: RigidCoordinateChange) -> Self:
        """
        :param change: A change from the posed frame into another frame.
        :return: The pose of the posed frame, given in the target frame of ``change``.
        """

    def apply_transformation(self, *, transformation: RigidCoordinateChange) -> Self:
        return transformation.express_pose(value=self)


GenericPose = TypeVar("GenericPose", bound=Pose)
"""
The kind of pose a rigid change re-expresses.
"""


# %% which way a frame faces


class Orientation(CoordinateRepresentation, ABC):
    """
    Which way a frame's axes point, given in another frame: the value of the rotation
    between them.

    It has no position, so a change of frame turns it but never shifts it.
    """

    @abstractmethod
    def to_coordinate_change(self) -> RigidCoordinateChange:
        """
        :return: The rotation from the oriented frame into the frame this orientation
            is given in.
        """

    @classmethod
    @abstractmethod
    def from_coordinate_change(cls, *, change: RigidCoordinateChange) -> Self:
        """
        :param change: A rotation from the oriented frame into another frame.
        :return: The orientation of the oriented frame, given in the target frame of
            ``change``.
        """

    def apply_transformation(self, *, transformation: RigidCoordinateChange) -> Self:
        return self.from_coordinate_change(
            change=transformation.rotation @ self.to_coordinate_change()
        )
