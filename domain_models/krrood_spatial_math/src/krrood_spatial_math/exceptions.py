from __future__ import annotations

from dataclasses import dataclass

from typing_extensions import TYPE_CHECKING

from krrood.exceptions import DataclassException

if TYPE_CHECKING:
    from krrood_spatial_math.cartesian_frame_3d import CartesianFrame3D
    from krrood_spatial_math.spatial_types import Vector3

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

    x: Vector3 | None
    y: Vector3 | None
    z: Vector3 | None

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
    expected_frame: CartesianFrame3D | None
    """
    The frame the operation requires, or ``None`` if it requires no frame.
    """

    actual_frame: CartesianFrame3D | None
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
            f"{self._describe(self.expected_frame)}, "
            f"but it is expressed in {self._describe(self.actual_frame)}."
        )

    @staticmethod
    def _describe(frame: CartesianFrame3D | None) -> str:
        """
        :param frame: The frame to describe.
        :return: The frame as it reads in the error message.
        """
        if frame is None:
            return "no frame"
        return f"frame '{frame.name}'"

    def suggest_correction(self) -> str:
        return (
            "transform it into the expected frame first, e.g. via "
            "world.transform(value, target_frame), or construct it with the expected frame."
        )


# %% serialized Cartesian frames


@dataclass
class MissingCartesianFrame3DResolver(SpatialTypesError):
    """
    Raised when a serialized spatial type refers to a three-dimensional Cartesian frame,
    but nothing was given to deserialize it with that finds the frame.
    """

    frame_id_key: str
    """
    Where the identifier of the frame sits in the serialized spatial type.
    """

    def error_message(self) -> str:
        return (
            f"The serialized spatial type refers to a frame by "
            f"'{self.frame_id_key}', but the keyword arguments of from_json carry no "
            f"CartesianFrame3DResolver."
        )

    def suggest_correction(self) -> str:
        return (
            "pass the keyword arguments of a CartesianFrame3DResolver that knows the "
            "frame, e.g. **resolver.create_kwargs(), to from_json."
        )
