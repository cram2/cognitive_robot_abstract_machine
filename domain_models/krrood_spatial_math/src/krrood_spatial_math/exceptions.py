"""
Errors of spatial types, frames and the spaces they chart.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from typing_extensions import TYPE_CHECKING

from krrood.exceptions import DataclassException

if TYPE_CHECKING:
    from krrood_spatial_math.cartesian_frame_3_dimensional import (
        CartesianFrame3Dimensional,
    )
    from krrood_spatial_math.coordinate_system import CoordinateSystem
    from krrood_spatial_math.spatial_types import (
        Pose3Dimensional,
        SymbolicSpatialType,
        Vector3Dimensional,
    )

# %% spatial types


@dataclass
class SpatialTypesError(DataclassException):
    """
    A spatial type was used in a way it does not support.
    """


@dataclass
class InsufficientVectorsError(SpatialTypesError):
    """
    Raised when a rotation matrix is constructed from fewer than two vectors.
    """

    x: Vector3Dimensional | None
    y: Vector3Dimensional | None
    z: Vector3Dimensional | None

    def error_message(self) -> str:
        return (
            f"from_vectors requires at least two of the vectors x, y, z to be provided; "
            f"the third is computed via the cross product."
            f"X Vector was: {self.x}"
            f"Y Vector was: {self.y}"
            f"Z Vector was: {self.z}"
        )

    def suggest_correction(self) -> str:
        return ""


@dataclass
class ReferenceFrameMismatchError(SpatialTypesError):
    """
    Raised when a value is given in another frame than an operation on it requires.
    """

    expected_frame: CoordinateSystem | None
    """
    The frame the operation requires, or ``None`` if it requires no frame.
    """

    actual_frame: CoordinateSystem | None
    """
    The frame that was found instead, or ``None`` if the value has no frame.
    """

    context: str | None = None
    """
    Description of the value whose frame was checked, e.g. the argument name.
    """

    def error_message(self) -> str:
        checked_value = self.context or "the given value"
        return (
            f"Expected {checked_value} to be expressed in "
            f"{self._describe(frame=self.expected_frame)}, "
            f"but it is expressed in {self._describe(frame=self.actual_frame)}."
        )

    @staticmethod
    def _describe(*, frame: CoordinateSystem | None) -> str:
        """
        :param frame: The frame to describe.
        :return: The frame as it reads in the error message.
        """
        if frame is None:
            return "no frame"
        return f"frame '{frame.name}'"

    def suggest_correction(self) -> str:
        return (
            "express it in the expected frame first, with "
            "expected_frame.express_in_own_frame(value=value), or construct it with the "
            "expected frame."
        )


@dataclass(kw_only=True)
class FrameMeasuresNoLengthsError(SpatialTypesError):
    """
    Raised when a length, an angle or a distance is asked of values given in a kind of
    frame whose Gram matrix is not known, such as a pixel frame of unknown pixel size.
    """

    frame_kind: type
    """
    The kind of frame the values are given in.
    """

    def error_message(self) -> str:
        return (
            f"{self.frame_kind.__name__} does not know the lengths of its axes or the "
            f"angles between them, so its coordinates alone give no lengths or angles."
        )

    def suggest_correction(self) -> str:
        return (
            "change the values into a Cartesian frame first, or give their kind of "
            "frame a gram_matrix."
        )


@dataclass(kw_only=True)
class SpaceMeasuresNoLengthsError(SpatialTypesError):
    """
    Raised when a length, an angle or a distance is asked of values of a space that is
    not Euclidean, which has no inner product.
    """

    space_kind: type
    """
    The kind of space the values belong to.
    """

    def error_message(self) -> str:
        return (
            f"{self.space_kind.__name__} has no inner product, so no lengths or angles."
        )

    def suggest_correction(self) -> str:
        return "derive the space from EuclideanSpace if it has lengths and angles."


@dataclass(kw_only=True)
class MissingAxisCoordinatesError(SpatialTypesError):
    """
    Raised when a point or vector type is defined without an attribute for one of the
    axes of its kind of frame, so it cannot hold a coordinate along that axis.
    """

    value_type: type
    """
    The point or vector type that was defined.
    """

    axes: tuple[StrEnum, ...]
    """
    The axes of its kind of frame that it holds no coordinate for.
    """

    def error_message(self) -> str:
        missing = ", ".join(self.axes)
        return f"{self.value_type.__name__} has no attribute for the axes {missing}."

    def suggest_correction(self) -> str:
        return (
            "declare a field or property named after each axis of the frame kind, as "
            "its axes classproperty names them."
        )


@dataclass
class MissingReferenceFrameError(SpatialTypesError):
    """
    Represents an error that occurs when a spatial type lacks a reference frame, even
    though it is required for the current operation.
    """

    spatial_type: SymbolicSpatialType
    """
    Spatial type that lacks a reference frame.
    """

    def error_message(self) -> str:
        return f"Spatial type {self.spatial_type} has no reference frame, but this operation requires one."

    def suggest_correction(self) -> str:
        return "construct it with an explicit reference frame."


# %% frames of one space


@dataclass(kw_only=True)
class FrameOutsideSpaceError(SpatialTypesError):
    """
    Raised when a frame charts no known space, so no change relates it to another frame.
    """

    frame: CoordinateSystem
    """
    The frame whose space is not known.
    """

    def error_message(self) -> str:
        return f"Frame '{self.frame.name}' charts no known space."

    def suggest_correction(self) -> str:
        return (
            "give the frame the space it charts, such as by adding a body to its world "
            "first."
        )


@dataclass(kw_only=True)
class FramesOfDifferentSpacesError(SpatialTypesError):
    """
    Raised when a value is to be expressed in a frame of another space, whose points
    relate to its own only through a map between the spaces.
    """

    frame: CoordinateSystem
    """
    The frame the value was to be expressed in.
    """

    value_frame: CoordinateSystem
    """
    The frame the value is given in.
    """

    def error_message(self) -> str:
        return (
            f"Frame '{self.value_frame.name}' charts another space than frame "
            f"'{self.frame.name}'."
        )

    def suggest_correction(self) -> str:
        return (
            "express the value in a frame of its own space, or map it between the "
            "spaces first."
        )


# %% values and changes on the x-y plane of a frame


@dataclass(kw_only=True)
class NoPlanarTransformationError(SpatialTypesError):
    """
    A change between two three-dimensional frames has no rigid counterpart between the
    frames on their x-y planes.
    """

    reference_frame: CartesianFrame3Dimensional | None
    """
    The frame the change turns coordinates into.
    """

    child_frame: CartesianFrame3Dimensional | None
    """
    The frame the change turns coordinates from.
    """


@dataclass(kw_only=True)
class FramesOnDifferentPlanesError(NoPlanarTransformationError):
    """
    Raised when the x-y planes of two frames are different planes, so no change between
    frames on one plane relates them.
    """

    def error_message(self) -> str:
        return (
            f"The x-y plane of {self.child_frame} is not the x-y plane of "
            f"{self.reference_frame}: their z-axes are not parallel, or the child's "
            f"origin is off the reference frame's plane."
        )

    def suggest_correction(self) -> str:
        return (
            "project the values onto one plane with an EmbeddedPlaneProjection instead of "
            "changing their coordinates."
        )


@dataclass(kw_only=True)
class MirroredPlaneFramesError(NoPlanarTransformationError):
    """
    Raised when two frames span one plane but their z-axes point to opposite sides of
    it, so the frames on that plane have opposite handedness and the change between them
    is a reflection, not a rigid change.
    """

    def error_message(self) -> str:
        return (
            f"{self.child_frame} and {self.reference_frame} span one x-y plane, but "
            f"their z-axes point in opposite directions."
        )

    def suggest_correction(self) -> str:
        return (
            "relate frames whose z-axes point the same way, for instance by turning one "
            "of them half a turn about its x-axis."
        )


@dataclass(kw_only=True)
class UnknownEmbeddedPlaneRelationError(NoPlanarTransformationError):
    """
    Raised when nothing tells whether two frames span one x-y plane: a frame charts no
    known space, or the change between them is not constant and names no child frame.
    """

    def error_message(self) -> str:
        return (
            f"It is not known whether {self.child_frame} and {self.reference_frame} "
            f"span one x-y plane."
        )

    def suggest_correction(self) -> str:
        return (
            "give both frames the space they chart, or name the child frame of a "
            "change that is not constant."
        )


@dataclass(kw_only=True)
class XAxisNormalToPlaneError(SpatialTypesError):
    """
    Raised when a pose is flattened onto a plane its x-axis is normal to, so the x-axis
    gives the flattened pose no heading.
    """

    pose: Pose3Dimensional
    """
    The pose that was flattened.
    """

    def error_message(self) -> str:
        return (
            f"The x-axis of {self.pose} is normal to the x-y plane of "
            f"{self.pose.reference_frame}, so it has no heading on that plane."
        )

    def suggest_correction(self) -> str:
        return "flatten the pose onto a plane its x-axis is not normal to."


# %% serialized Cartesian frames


@dataclass
class MissingCartesianFrame3DimensionalResolver(SpatialTypesError):
    """
    Raised when a serialized coordinate representation refers to a three-dimensional
    Cartesian frame, but nothing was given to deserialize it with that finds the frame.
    """

    frame_id_key: str
    """
    Where the identifier of the frame sits in the serialized coordinate representation.
    """

    def error_message(self) -> str:
        return (
            f"The serialized coordinate representation refers to a frame by "
            f"'{self.frame_id_key}', but the keyword arguments of from_json carry no "
            f"CartesianFrame3DimensionalResolver."
        )

    def suggest_correction(self) -> str:
        return (
            "pass a CartesianFrame3DimensionalResolver that knows the frame among the keyword "
            "arguments of from_json."
        )
