"""
Cartesian frames fixed on another frame, which express the values of both.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from krrood_spatial_math.coordinate_system import GenericCoordinateRepresentation
from krrood_spatial_math.embedded_plane import EmbeddedPlane
from krrood_spatial_math.exceptions import ReferenceFrameMismatchError
from krrood_spatial_math.spatial_types import (
    HomogeneousTransformationMatrix3Dimensional,
)

from .frames import NamedFrame

# %% frames other frames are fixed on


@dataclass(eq=False, kw_only=True)
class MountFrame(NamedFrame):
    """
    A Cartesian frame that other frames are fixed on, so it expresses values given in
    itself or in any :class:`MountedFrame` fixed on it.
    """

    def transform_into_own_frame(
        self, *, value: GenericCoordinateRepresentation
    ) -> GenericCoordinateRepresentation:
        value_frame = value.reference_frame
        if value_frame is self:
            return value
        if not (
            isinstance(value_frame, MountedFrame)
            and value_frame.mount_T_self.reference_frame is self
        ):
            raise ReferenceFrameMismatchError(
                expected_frame=self, actual_frame=value_frame
            )
        return value_frame.mount_T_self @ value


# %% frames fixed on another frame


@dataclass(eq=False)
class MountedFrame(NamedFrame):
    """
    A Cartesian frame fixed on another frame, so it expresses values given in itself, in
    that frame, or in another frame fixed on that frame.
    """

    mount_T_self: HomogeneousTransformationMatrix3Dimensional = field(kw_only=True)
    """
    The change from this frame to the frame it is fixed on, which is its reference
    frame; its child frame becomes this frame.
    """

    def __post_init__(self):
        self.mount_T_self = self.mount_T_self.copy_with_new_reference_frames(
            self.mount_T_self.reference_frame, self
        )

    @property
    def embedded_plane(self) -> EmbeddedPlane:
        """
        The plane of the frame it is fixed on if the change it is fixed by keeps that
        plane, otherwise its own x-y plane, unlike :attr:`NamedFrame.embedded_plane`.
        """
        if self.mount_T_self.keeps_x_y_plane:
            return self.mount_T_self.reference_frame.embedded_plane
        return EmbeddedPlane(frame_3_dimensional=self)

    def transform_into_own_frame(
        self, *, value: GenericCoordinateRepresentation
    ) -> GenericCoordinateRepresentation:
        value_frame = value.reference_frame
        if value_frame is self:
            return value
        mount = self.mount_T_self.reference_frame
        if value_frame is mount:
            return self.mount_T_self.inverse() @ value
        if not (
            isinstance(value_frame, MountedFrame)
            and value_frame.mount_T_self.reference_frame is mount
        ):
            raise ReferenceFrameMismatchError(
                expected_frame=mount, actual_frame=value_frame
            )
        return self.mount_T_self.inverse() @ (value_frame.mount_T_self @ value)
