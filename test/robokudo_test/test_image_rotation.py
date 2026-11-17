"""
Image rotations preserve matching pixels and camera calibration.
"""

from __future__ import annotations

# %% Imports
import numpy
import pytest
from robokudo.io.ros_camera_without_depth_interface import (
    ROSCameraWithoutDepthInterface,
)
from robokudo.io.exceptions import UnsupportedImageRotation
from robokudo.types.image import ImageRotation
from semantic_digital_twin.datastructures.camera_model import CameraDistortionModel


# %% Image and calibration rotation
@pytest.mark.parametrize("rotation", list(ImageRotation))
def test_image_rotation_matches_intrinsics(rotation: ImageRotation) -> None:
    """
    Each rotation produces the expected pixels and intrinsic matrix.
    """
    image = numpy.arange(6, dtype=numpy.uint8).reshape(2, 3)
    intrinsic_matrix = numpy.array(
        [[10.0, 0.0, 0.5], [0.0, 20.0, 0.25], [0.0, 0.0, 1.0]]
    )
    interface = object.__new__(ROSCameraWithoutDepthInterface)
    rotated_image, rotated_intrinsics, dimensions = (
        interface.rotate_image_and_intrinsics(image, intrinsic_matrix, rotation)
    )
    if rotation == ImageRotation.COUNTERCLOCKWISE_90:
        expected_image = numpy.array([[2, 5], [1, 4], [0, 3]])
        expected_intrinsics = numpy.array(
            [[20.0, 0.0, 0.25], [0.0, 10.0, 1.5], [0.0, 0.0, 1.0]]
        )
    elif rotation == ImageRotation.CLOCKWISE_90:
        expected_image = numpy.array([[3, 0], [4, 1], [5, 2]])
        expected_intrinsics = numpy.array(
            [[20.0, 0.0, 0.75], [0.0, 10.0, 0.5], [0.0, 0.0, 1.0]]
        )
    else:
        expected_image = numpy.array([[5, 4, 3], [2, 1, 0]])
        expected_intrinsics = numpy.array(
            [[10.0, 0.0, 1.5], [0.0, 20.0, 0.75], [0.0, 0.0, 1.0]]
        )
    numpy.testing.assert_array_equal(rotated_image, expected_image)
    numpy.testing.assert_array_equal(rotated_intrinsics, expected_intrinsics)
    assert dimensions == (expected_image.shape[1], expected_image.shape[0])


@pytest.mark.parametrize("rotation", list(ImageRotation))
def test_image_rotation_updates_tangential_distortion(rotation: ImageRotation) -> None:
    """
    Tangential distortion follows the rotated image axes.
    """
    coefficients = (0.1, 0.2, 0.3, 0.4, 0.5)
    expected = {
        ImageRotation.COUNTERCLOCKWISE_90: (0.1, 0.2, -0.4, 0.3, 0.5),
        ImageRotation.CLOCKWISE_90: (0.1, 0.2, 0.4, -0.3, 0.5),
        ImageRotation.HALF_TURN: (0.1, 0.2, -0.3, -0.4, 0.5),
    }
    assert (
        ROSCameraWithoutDepthInterface.rotate_camera_distortion(
            coefficients, CameraDistortionModel.PLUMB_BOB.value, rotation
        )
        == expected[rotation]
    )


# %% Unsupported rotations
@pytest.mark.parametrize("rotation", ["unsupported", None])
def test_intrinsics_reject_unsupported_rotation(rotation: str | None) -> None:
    """
    Intrinsics rotation identifies an unsupported orientation.
    """
    interface = object.__new__(ROSCameraWithoutDepthInterface)
    with pytest.raises(UnsupportedImageRotation) as error:
        interface.rotate_camera_intrinsics(numpy.eye(3), (3, 2), rotation)
    assert error.value.rotation == rotation


@pytest.mark.parametrize("rotation", ["unsupported", None])
def test_image_rejects_unsupported_rotation(rotation: str | None) -> None:
    """
    Image rotation identifies an unsupported orientation.
    """
    interface = object.__new__(ROSCameraWithoutDepthInterface)
    with pytest.raises(UnsupportedImageRotation) as error:
        interface.rotate_image_and_intrinsics(
            numpy.zeros((2, 3)), numpy.eye(3), rotation
        )
    assert error.value.rotation == rotation


@pytest.mark.parametrize("rotation", ["unsupported", None])
@pytest.mark.parametrize("distortion_model", list(CameraDistortionModel))
def test_distortion_rejects_unsupported_rotation(
    rotation: str | None,
    distortion_model: CameraDistortionModel,
) -> None:
    """
    Distortion rejects unsupported orientations for every camera model.
    """
    with pytest.raises(UnsupportedImageRotation) as error:
        ROSCameraWithoutDepthInterface.rotate_camera_distortion(
            (0.1, 0.2, 0.3, 0.4, 0.5), distortion_model.value, rotation
        )
    assert error.value.rotation == rotation
