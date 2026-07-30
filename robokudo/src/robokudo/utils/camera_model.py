"""Camera model calculations shared by camera interfaces."""

from __future__ import annotations

from dataclasses import dataclass
from math import radians, tan


@dataclass(frozen=True)
class PinholeCameraParameters:
    """Intrinsic pinhole camera parameters."""

    width: int
    """Image width in pixels."""

    height: int
    """Image height in pixels."""

    focal_length_x: float
    """Horizontal focal length in pixels."""

    focal_length_y: float
    """Vertical focal length in pixels."""

    center_x: float
    """Horizontal principal point in pixels."""

    center_y: float
    """Vertical principal point in pixels."""


def pinhole_camera_parameters_from_horizontal_field_of_view(
    width: int, height: int, horizontal_field_of_view_degrees: float
) -> PinholeCameraParameters:
    """Calculate pinhole intrinsics from image size and horizontal field of view.

    :param width: Image width in pixels.
    :param height: Image height in pixels.
    :param horizontal_field_of_view_degrees: Horizontal camera field of view.
    :return: Pinhole camera intrinsic parameters.
    :raises ValueError: If the image size or field of view is invalid.
    """
    if width <= 0:
        raise ValueError("Image width must be greater than zero.")
    if height <= 0:
        raise ValueError("Image height must be greater than zero.")
    if (
        horizontal_field_of_view_degrees <= 0.0
        or horizontal_field_of_view_degrees >= 180.0
    ):
        raise ValueError("Horizontal field of view must be between 0 and 180 degrees.")

    field_of_view_radians = radians(horizontal_field_of_view_degrees)
    focal_length = (0.5 * width) / tan(0.5 * field_of_view_radians)

    return PinholeCameraParameters(
        width=width,
        height=height,
        focal_length_x=focal_length,
        focal_length_y=focal_length,
        center_x=(width - 1.0) / 2.0,
        center_y=(height - 1.0) / 2.0,
    )
