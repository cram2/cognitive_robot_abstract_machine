"""
ROS camera interface for RGB-only cameras in RoboKudo.

This module provides an interface for ROS cameras that only provide RGB images
without depth information. It supports:

* Synchronized RGB image and camera info subscription
* Image rotation with intrinsics adjustment
* Transform lookup between camera and world frames
* Thread-safe data handling

The module is used for cameras like:

* Hand cameras on robotic arms
* Standard USB cameras with ROS drivers
* Network cameras with ROS interfaces
"""

from __future__ import annotations

# %% Imports
import logging
from collections.abc import Sequence

import cv2
import message_filters
import numpy as np
from message_filters import Subscriber
from sensor_msgs.msg import CameraInfo, Image
from typing_extensions import Any, TYPE_CHECKING

from krrood.patterns.called_positionally import called_positionally

from robokudo.cas import CASViews
from robokudo.exceptions import InvalidCameraObservation
from robokudo.types.cv import ColorToDepthRatio
from robokudo.types.image import ImageRotation
from robokudo.io.camera_interface import CameraFrame, ROSCameraInterface
from robokudo.io.exceptions import UnsupportedImageRotation
from semantic_digital_twin.datastructures.camera_model import CameraDistortionModel

if TYPE_CHECKING:
    import numpy.typing as npt
    from semantic_digital_twin.world import World

    from robokudo.cas import CAS

logger = logging.getLogger(__name__)


# %% RGB-only camera interface
class ROSCameraWithoutDepthInterface(ROSCameraInterface):
    """
    A ROS camera interface for RGB-only cameras.

    This class handles cameras that provide only RGB images without depth data. It
    synchronizes RGB images with camera calibration information and supports image
    rotation with proper intrinsics adjustment.
    """

    def __init__(self, camera_config: Any) -> None:
        """
        Initialize the RGB-only camera interface.

        Sets up ROS subscribers and synchronization for RGB image and camera info
        topics. Also initializes internal data structures and thread safety.

        :param camera_config: Configuration for the camera interface
        """
        super().__init__(camera_config)

        self.color_subscriber: Subscriber = message_filters.Subscriber(
            camera_config.topic_color, Image
        )
        """
        Subscriber for RGB image topic.
        """
        self.camera_info_subscriber: Subscriber = message_filters.Subscriber(
            camera_config.topic_camera_info, CameraInfo
        )
        """
        Subscriber for camera info topic.
        """
        synchronizer = message_filters.ApproximateTimeSynchronizer(
            [self.color_subscriber, self.camera_info_subscriber],
            queue_size=10,
            slop=0.4,
        )
        synchronizer.registerCallback(self.callback)

        logger.info("Subscribed to: ")
        logger.info(f"  {camera_config.topic_color}")
        logger.info(f"  {camera_config.topic_camera_info}")

        self.color_to_depth_ratio: ColorToDepthRatio = ColorToDepthRatio()
        """
        Always (1.0, 1.0) since no depth data.
        """
        logger.info("ROSCameraWithoutDepthInterface initialized")

    def rotate_camera_intrinsics(
        self,
        *,
        intrinsic_matrix: npt.NDArray,
        image_size: tuple[int, int],
        rotation: ImageRotation = ImageRotation.COUNTERCLOCKWISE_90,
    ) -> tuple[npt.NDArray, tuple[int, int]]:
        """
        Adjust camera intrinsics matrix for image rotation.

        Computes the new camera intrinsics matrix and image dimensions after applying a
        rotation to the image. Supports 90° counter-clockwise, 90° clockwise, and 180°
        rotations.

        :param intrinsic_matrix: Original camera intrinsic matrix of shape (3, 3)
        :param image_size: Original image dimensions as (width, height)
        :param rotation: Rotation applied to the image.
        :return: Tuple of (new intrinsics matrix, new image dimensions)
        :raises UnsupportedImageRotation: If the image rotation is unsupported.
        """
        width, height = image_size
        focal_length_x = intrinsic_matrix[0, 0]
        focal_length_y = intrinsic_matrix[1, 1]
        principal_point_x = intrinsic_matrix[0, 2]
        principal_point_y = intrinsic_matrix[1, 2]

        if rotation == ImageRotation.COUNTERCLOCKWISE_90:
            rotated_intrinsics = np.array(
                [
                    [focal_length_y, 0, principal_point_y],
                    [0, focal_length_x, width - 1.0 - principal_point_x],
                    [0, 0, 1],
                ]
            )
            new_size = (height, width)

        elif rotation == ImageRotation.CLOCKWISE_90:
            rotated_intrinsics = np.array(
                [
                    [focal_length_y, 0, height - 1.0 - principal_point_y],
                    [0, focal_length_x, principal_point_x],
                    [0, 0, 1],
                ]
            )
            new_size = (height, width)

        elif rotation == ImageRotation.HALF_TURN:
            rotated_intrinsics = np.array(
                [
                    [focal_length_x, 0, width - 1.0 - principal_point_x],
                    [0, focal_length_y, height - 1.0 - principal_point_y],
                    [0, 0, 1],
                ]
            )
            new_size = (width, height)

        else:
            raise UnsupportedImageRotation(rotation=rotation)

        return rotated_intrinsics, new_size

    @staticmethod
    def rotate_camera_distortion(
        *, coefficients: Sequence[float], distortion_model: str, rotation: ImageRotation
    ) -> tuple[float, ...]:
        """
        Express lens distortion in the rotated image-axis convention.

        :param coefficients: ROS distortion coefficients for the original image.
        :param distortion_model: ROS distortion model name.
        :param rotation: Rotation applied to the delivered image.
        :return: Distortion coefficients matching the rotated image axes.
        :raises InvalidCameraObservation: If tangential terms are missing.
        :raises UnsupportedImageRotation: If the image rotation is unsupported.
        """
        if rotation not in tuple(ImageRotation):
            raise UnsupportedImageRotation(rotation=rotation)

        tangential_models = (
            CameraDistortionModel.PLUMB_BOB.value,
            CameraDistortionModel.RATIONAL_POLYNOMIAL.value,
        )
        if distortion_model not in tangential_models:
            return tuple(coefficients)
        if len(coefficients) < 4:
            raise InvalidCameraObservation(
                reason=(
                    f"the ROS distortion model '{distortion_model}' requires "
                    "two tangential coefficients"
                )
            )

        rotated_coefficients = list(coefficients)
        tangential_x = coefficients[2]
        tangential_y = coefficients[3]
        if rotation == ImageRotation.COUNTERCLOCKWISE_90:
            rotated_coefficients[2] = -tangential_y
            rotated_coefficients[3] = tangential_x
        elif rotation == ImageRotation.CLOCKWISE_90:
            rotated_coefficients[2] = tangential_y
            rotated_coefficients[3] = -tangential_x
        elif rotation == ImageRotation.HALF_TURN:
            rotated_coefficients[2] = -tangential_x
            rotated_coefficients[3] = -tangential_y
        return tuple(rotated_coefficients)

    def rotate_image_and_intrinsics(
        self,
        *,
        image: npt.NDArray,
        intrinsic_matrix: npt.NDArray,
        rotation: ImageRotation = ImageRotation.COUNTERCLOCKWISE_90,
    ) -> tuple[npt.NDArray, npt.NDArray, tuple[int, int]]:
        """
        Rotate an image and adjust its camera intrinsics accordingly.

        :param image: Input image to rotate
        :param intrinsic_matrix: Camera intrinsics matrix
        :param rotation: Rotation applied to the image.
        :return: Tuple of (rotated image, new intrinsics matrix, new dimensions)
        :raises UnsupportedImageRotation: If the image rotation is unsupported.
        """
        height, width = image.shape[:2]
        if rotation == ImageRotation.COUNTERCLOCKWISE_90:
            rotated_image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
        elif rotation == ImageRotation.CLOCKWISE_90:
            rotated_image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        elif rotation == ImageRotation.HALF_TURN:
            rotated_image = cv2.rotate(image, cv2.ROTATE_180)
        else:
            raise UnsupportedImageRotation(rotation=rotation)

        rotated_intrinsics, new_size = self.rotate_camera_intrinsics(
            intrinsic_matrix=intrinsic_matrix,
            image_size=(width, height),
            rotation=rotation,
        )
        return rotated_image, rotated_intrinsics, new_size

    @called_positionally
    def callback(
        self,
        color_data: Image,
        camera_info: CameraInfo | None = None,
    ) -> None:
        """
        Make the synchronized RGB image and camera info messages the latest frame.

        .. note:: A frame arriving while the previous frame is read is dropped.

        :param color_data: RGB image message
        :param camera_info: Camera calibration info message, defaults to None
        """
        self.store_frame(
            frame=CameraFrame.from_messages(
                color_data=color_data, camera_info=camera_info
            )
        )

    def rotate_color_image_and_calibration(
        self,
        *,
        color_image: npt.NDArray,
        camera_info: CameraInfo,
        rotation: ImageRotation,
    ) -> npt.NDArray:
        """
        Rotate ``color_image`` and adapt the image size, intrinsics and distortion of
        ``camera_info`` to the rotated image.

        :param color_image: The color image ``camera_info`` calibrates.
        :param camera_info: The calibration, changed in place.
        :param rotation: Rotation applied to the image.
        :return: The rotated color image.
        :raises UnsupportedImageRotation: If the image rotation is unsupported.
        """
        rotated_image, rotated_intrinsics, (width, height) = (
            self.rotate_image_and_intrinsics(
                image=color_image,
                intrinsic_matrix=np.array(camera_info.k).reshape(3, 3),
                rotation=rotation,
            )
        )
        camera_info.width = width
        camera_info.height = height
        camera_info.k = rotated_intrinsics.flatten()
        camera_info.d = self.rotate_camera_distortion(
            coefficients=camera_info.d,
            distortion_model=camera_info.distortion_model,
            rotation=rotation,
        )
        return rotated_image

    def set_data(self, *, cas: CAS, world: World) -> None:
        """
        Put the latest color image and camera observation into the CAS, rotating the
        image and adapting its calibration if a rotation is configured.

        :param cas: Common Analysis Structure to update
        :param world: The world perception perceives into
        """
        with self.lock:
            if not self._has_new_data:
                return
            frame = self.latest_frame
            color_image = frame.color_image
            if self.camera_config.rotate_image is not None:
                color_image = self.rotate_color_image_and_calibration(
                    color_image=color_image,
                    camera_info=frame.camera_info,
                    rotation=self.camera_config.rotate_image,
                )

            cas.set(view_name=CASViews.COLOR_IMAGE, value=color_image)
            cas.set(view_name=CASViews.DEPTH_IMAGE, value=None)
            cas.set(view_name=CASViews.COLOR_TO_DEPTH_RATIO, value=ColorToDepthRatio())
            self.store_latest_frame_observation(cas=cas, world=world)

            self._has_new_data = False
