"""
Conversions between semantic camera models and library boundary types.
"""

from __future__ import annotations

# %% Imports

from dataclasses import dataclass

import numpy as np
import open3d
from sensor_msgs.msg import CameraInfo

from robokudo.io.exceptions import (
    CameraDistortionModelMissing,
    UnsupportedCameraDistortionModel,
)
from semantic_digital_twin.spatial_types import Point2
from semantic_digital_twin.datastructures.camera_model import (
    CameraDistortion,
    CameraDistortionModel,
    NoCameraDistortion,
    PinholeCameraModel,
)
from semantic_digital_twin.datastructures.camera_resolution import CameraResolution

# %% ROS camera information


@dataclass(frozen=True)
class RosCameraModelAdapter:
    """
    Convert pinhole calibration at the ROS message boundary.
    """

    @staticmethod
    def from_camera_info(camera_info: CameraInfo) -> PinholeCameraModel:
        """
        Return a semantic model parsed from a ROS camera-info message.

        :param camera_info: Calibration supplied by a ROS image stream.
        :return: Semantic pinhole calibration.
        :raises CameraDistortionModelMissing: If nonzero coefficients have no model.
        :raises UnsupportedCameraDistortionModel: If the model is unsupported.
        """
        distortion_models = {
            model.value: model
            for model in CameraDistortionModel
            if model is not CameraDistortionModel.NONE
        }
        if camera_info.distortion_model == "":
            if any(coefficient != 0.0 for coefficient in camera_info.d):
                raise CameraDistortionModelMissing(coefficients=tuple(camera_info.d))
            distortion = NoCameraDistortion()
        elif camera_info.distortion_model in distortion_models:
            distortion_model = distortion_models[camera_info.distortion_model]
            distortion = CameraDistortion.from_ordered_coefficients(
                model=distortion_model,
                coefficients=camera_info.d,
            )
        else:
            raise UnsupportedCameraDistortionModel(
                distortion_model=camera_info.distortion_model
            )

        return PinholeCameraModel(
            image_resolution=CameraResolution(
                width=camera_info.width,
                height=camera_info.height,
            ),
            focal_length_x=camera_info.k[0],
            focal_length_y=camera_info.k[4],
            principal_point=Point2(x=camera_info.k[2], y=camera_info.k[5]),
            distortion=distortion,
        )

    @staticmethod
    def to_camera_info(
        camera_model: PinholeCameraModel,
        frame_id: str,
    ) -> CameraInfo:
        """
        Return a ROS camera-info message for a semantic model.

        :param camera_model: Effective image calibration.
        :param frame_id: Optical frame stored in the message header.
        :return: ROS camera-info message matching the model.
        """
        camera_info = CameraInfo()
        camera_info.header.frame_id = frame_id
        camera_info.width = camera_model.resolution.width
        camera_info.height = camera_model.resolution.height
        if camera_model.distortion.model == CameraDistortionModel.NONE:
            camera_info.distortion_model = CameraDistortionModel.PLUMB_BOB.value
            camera_info.d = [0.0] * CameraDistortionModel.PLUMB_BOB.coefficient_count
        else:
            camera_info.distortion_model = camera_model.distortion.model.value
            camera_info.d = list(camera_model.distortion.to_ordered_coefficients())
        camera_info.k = camera_model.intrinsic_matrix.reshape(9).tolist()
        camera_info.r = np.eye(3).reshape(9).tolist()
        camera_info.p = [
            camera_model.focal_length_x,
            0.0,
            float(camera_model.principal_point.x),
            0.0,
            0.0,
            camera_model.focal_length_y,
            float(camera_model.principal_point.y),
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
        ]
        return camera_info


# %% Open3D camera information


@dataclass(frozen=True)
class Open3DCameraModelAdapter:
    """
    Convert pinhole calibration at the Open3D boundary.
    """

    @staticmethod
    def to_intrinsic(
        camera_model: PinholeCameraModel,
    ) -> open3d.camera.PinholeCameraIntrinsic:
        """
        Return Open3D calibration matching a semantic model.

        :param camera_model: Effective image calibration.
        :return: Open3D intrinsic object.
        """
        return open3d.camera.PinholeCameraIntrinsic(
            camera_model.resolution.width,
            camera_model.resolution.height,
            camera_model.focal_length_x,
            camera_model.focal_length_y,
            float(camera_model.principal_point.x),
            float(camera_model.principal_point.y),
        )
