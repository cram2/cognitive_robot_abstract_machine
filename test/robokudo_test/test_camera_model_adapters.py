# %% Imports

import pytest

import numpy as np
from sensor_msgs.msg import CameraInfo

from robokudo.io.exceptions import (
    CameraDistortionModelMissing,
    UnsupportedCameraDistortionModel,
)
from robokudo.exceptions import InvalidCameraObservation

from robokudo.io.camera_model_adapters import (
    Open3DCameraModelAdapter,
    RosCameraModelAdapter,
)
from semantic_digital_twin.spatial_types import Point2
from semantic_digital_twin.datastructures.camera_model import (
    CameraDistortionModel,
    NoCameraDistortion,
    PinholeCameraModel,
    PlumbBobCameraDistortion,
)
from semantic_digital_twin.datastructures.camera_resolution import CameraResolution


# %% Camera model conversion
def test_ros_camera_model_adapter_reads_camera_info():
    camera_info = CameraInfo()
    camera_info.width = 640
    camera_info.height = 480
    camera_info.distortion_model = CameraDistortionModel.PLUMB_BOB.value
    camera_info.d = [0.1, -0.2, 0.01, 0.02, 0.03]
    camera_info.k = [500.0, 0.0, 321.0, 0.0, 510.0, 239.0, 0.0, 0.0, 1.0]

    model = RosCameraModelAdapter.from_camera_info(camera_info)

    assert model.resolution == CameraResolution(width=640, height=480)
    assert np.array_equal(
        model.intrinsic_matrix, np.asarray(camera_info.k).reshape(3, 3)
    )
    assert model.distortion.to_ordered_coefficients() == tuple(camera_info.d)


def test_ros_camera_model_adapter_accepts_empty_model_with_zero_coefficients():
    camera_info = CameraInfo()
    camera_info.width = 640
    camera_info.height = 480
    camera_info.distortion_model = ""
    camera_info.d = [0.0] * CameraDistortionModel.PLUMB_BOB.coefficient_count
    camera_info.k = [500.0, 0.0, 321.0, 0.0, 510.0, 239.0, 0.0, 0.0, 1.0]

    model = RosCameraModelAdapter.from_camera_info(camera_info)

    assert isinstance(model.distortion, NoCameraDistortion)


def test_ros_camera_model_adapter_writes_camera_info():
    model = PinholeCameraModel(
        image_resolution=CameraResolution(width=640, height=480),
        focal_length_x=500.0,
        focal_length_y=510.0,
        principal_point=Point2(x=321.0, y=239.0),
        distortion=PlumbBobCameraDistortion(radial_coefficient_1=0.1),
    )

    camera_info = RosCameraModelAdapter.to_camera_info(
        camera_model=model,
        frame_id="camera_optical_frame",
    )

    assert camera_info.header.frame_id == "camera_optical_frame"
    assert camera_info.width == model.resolution.width
    assert camera_info.height == model.resolution.height
    assert np.array_equal(
        np.asarray(camera_info.k).reshape(3, 3), model.intrinsic_matrix
    )
    assert camera_info.distortion_model == model.distortion.model.value
    assert tuple(camera_info.d) == model.distortion.to_ordered_coefficients()


def test_open3d_camera_model_adapter_creates_intrinsic():
    model = PinholeCameraModel(
        image_resolution=CameraResolution(width=640, height=480),
        focal_length_x=500.0,
        focal_length_y=510.0,
        principal_point=Point2(x=321.0, y=239.0),
    )

    camera_intrinsic = Open3DCameraModelAdapter.to_intrinsic(model)

    assert camera_intrinsic.width == model.resolution.width
    assert camera_intrinsic.height == model.resolution.height
    assert np.array_equal(camera_intrinsic.intrinsic_matrix, model.intrinsic_matrix)


# %% Invalid distortion metadata


def test_camera_info_requires_model_for_nonzero_distortion() -> None:
    """
    Nonzero coefficients without a model raise the specific missing-model error.
    """
    camera_info = CameraInfo()
    camera_info.d = [0.1, 0.0, 0.0, 0.0, 0.0]
    with pytest.raises(CameraDistortionModelMissing) as error:
        RosCameraModelAdapter.from_camera_info(camera_info)
    assert error.value.coefficients == tuple(camera_info.d)
    assert isinstance(error.value, InvalidCameraObservation)


def test_camera_info_rejects_unsupported_distortion_model() -> None:
    """
    An unknown distortion model raises the specific unsupported-model error.
    """
    camera_info = CameraInfo()
    camera_info.distortion_model = "unsupported"
    with pytest.raises(UnsupportedCameraDistortionModel) as error:
        RosCameraModelAdapter.from_camera_info(camera_info)
    assert error.value.distortion_model == camera_info.distortion_model
    assert isinstance(error.value, InvalidCameraObservation)
