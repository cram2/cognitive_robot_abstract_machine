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

from threading import Lock

import cv2
import message_filters
import numpy as np
import open3d as o3d
from cv_bridge import CvBridge
from message_filters import Subscriber
from sensor_msgs.msg import CameraInfo, Image
from typing_extensions import Any, Tuple, TYPE_CHECKING, Optional, List

from robokudo.cas import CASViews
from robokudo.io.camera_interface import ROSCameraInterface

if TYPE_CHECKING:
    import numpy.typing as npt


# TODO Needs to be migrated to ROS2 for new RGB-only use cases
class ROSCameraWithoutDepthInterface(ROSCameraInterface):
    """A ROS camera interface for RGB-only cameras.

    This class handles cameras that provide only RGB images without depth data.
    It synchronizes RGB images with camera calibration information and supports
    image rotation with proper intrinsics adjustment.
    """

    def __init__(self, camera_config: Any) -> None:
        """Initialize the RGB-only camera interface.

        Sets up ROS subscribers and synchronization for RGB image and camera info
        topics. Also initializes internal data structures and thread safety.

        :param camera_config: Configuration for the camera interface
        """
        super().__init__(camera_config)

        # TODO Handle type hints like 'compressed', 'compressedDepth' and 'raw'
        self.color_subscriber: Subscriber = message_filters.Subscriber(
            camera_config.topic_color, Image
        )
        """Subscriber for RGB image topic"""

        self.cam_info_subscriber: Subscriber = message_filters.Subscriber(
            camera_config.topic_cam_info, CameraInfo
        )
        """Subscriber for camera info topic"""

        ts = message_filters.ApproximateTimeSynchronizer(
            [self.color_subscriber, self.cam_info_subscriber], queue_size=10, slop=0.4
        )
        ts.registerCallback(self.callback)

        self.rk_logger.info("Subscribed to: ")
        self.rk_logger.info(f"  {camera_config.topic_color}")
        self.rk_logger.info(f"  {camera_config.topic_cam_info}")

        self.color: Optional[Tuple[npt.NDArray]] = None
        """Latest RGB image data"""

        self.cam_info: Optional[CameraInfo] = None
        """Latest camera calibration info"""

        self.cam_intrinsic: Optional[o3d.camera.PinholeCameraIntrinsic] = None
        """Camera intrinsic parameters in Open3D format"""

        self.color2depth_ratio: Optional[Tuple[float, float]] = (1.0, 1.0)
        """Always (1.0, 1.0) since no depth data"""

        self.cam_translation: Optional[List[float]] = None
        """Camera translation from TF"""

        self.cam_quaternion: Optional[List[float]] = None
        """Camera rotation as quaternion from TF"""

        self.timestamp: Optional[float] = None
        """Timestamp of latest data"""

        self.lock: Lock = Lock()
        """Thread synchronization lock"""

        # hack because my rosbag has image topic
        self.bridge: CvBridge = CvBridge()
        """Bridge for converting between ROS and OpenCV images"""

        print("ROSCameraWithoutDepthInterface initialized")

    def rotate_camera_intrinsics(
        self, K: npt.NDArray, image_size: Tuple[int, int], rotation: str = "90_ccw"
    ) -> Tuple[npt.NDArray, Tuple[int, int]]:
        """Adjust camera intrinsics matrix for image rotation.

        Computes the new camera intrinsics matrix and image dimensions after
        applying a rotation to the image. Supports 90° counter-clockwise,
        90° clockwise, and 180° rotations.

        :param K: Original camera intrinsic matrix of shape (3, 3)
        :param image_size: Original image dimensions as (width, height)
        :param rotation: '90_ccw', '90_cw', '180' (default is 90° counter-clockwise)
        :return: Tuple of (new intrinsics matrix, new image dimensions)
        :raises ValueError: If rotation type is not supported
        """
        width, height = image_size
        f_x = K[0, 0]
        f_y = K[1, 1]
        c_x = K[0, 2]
        c_y = K[1, 2]

        if rotation == "90_ccw":
            new_K = np.array([[f_y, 0, c_y], [0, f_x, width - c_x], [0, 0, 1]])
            new_size = (height, width)

        elif rotation == "90_cw":
            new_K = np.array([[f_y, 0, height - c_y], [0, f_x, c_x], [0, 0, 1]])
            new_size = (height, width)

        elif rotation == "180":
            new_K = np.array([[f_x, 0, width - c_x], [0, f_y, height - c_y], [0, 0, 1]])
            new_size = (width, height)

        else:
            raise ValueError(
                "Unsupported rotation type. Use '90_ccw', '90_cw', or '180'."
            )

        return new_K, new_size

    def rotate_image_and_intrinsics(
        self, img: npt.NDArray, K: npt.NDArray, rotation: str = "90_ccw"
    ) -> Tuple[npt.NDArray, npt.NDArray, Tuple[int, int]]:
        """
        Rotate an image and adjust its camera intrinsics accordingly.

        :param img: Input image to rotate
        :param K: Camera intrinsics matrix
        :param rotation: Type of rotation to apply, defaults to '90_ccw'
        :return: Tuple of (rotated image, new intrinsics matrix, new dimensions)
        :raises ValueError: If rotation type is not supported
        """
        h, w = img.shape[:2]
        if rotation == "90_ccw":
            img_rotated = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        elif rotation == "90_cw":
            img_rotated = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        elif rotation == "180":
            img_rotated = cv2.rotate(img, cv2.ROTATE_180)
        else:
            raise ValueError("Unsupported rotation type.")

        K_new, new_size = self.rotate_camera_intrinsics(K, (w, h), rotation)
        return img_rotated, K_new, new_size

    def callback(
        self,
        color_data: Image,
        cam_info: Optional[CameraInfo] = None,
    ) -> None:
        """Process synchronized RGB image and camera info messages.

        TODO make this generic. handle the encoding and order properly.
        For standard and compressed images.
        this might also depend on the fix of image_transport_plugins being published as a package.
        Startpoint can be found at the bottom of this method.

        :param color_data: RGB image message
        :param cam_info: Camera calibration info message, defaults to None
        """

        # hack becasue my rosbag has image topics
        # color_arr = np.fromstring(color_data.data, np.uint8)
        self.lock.acquire()
        # self.color = cv2.imdecode(color_arr, cv2.IMREAD_COLOR)
        self.color = self.bridge.imgmsg_to_cv2(color_data, desired_encoding="bgr8")
        self.timestamp = color_data.header.stamp

        self.cam_info = cam_info

        if not self.lookup_transform():
            self._has_new_data = False
            self.lock.release()
            return

        self._has_new_data = True
        self.lock.release()

    def set_data(self, cas: CAS) -> None:
        """
        Update the Common Analysis Structure with latest camera data.

        This method:
        * Applies any configured image rotation
        * Updates camera intrinsics accordingly
        * Sets RGB image, camera info, and transform data in the CAS
        * Handles thread synchronization

        :param cas: Common Analysis Structure to update
        """
        if not self.has_new_data():
            return

        self.lock.acquire()

        # Construct o3d camera intrinsics from cam info in CAS
        self.cam_intrinsic = o3d.camera.PinholeCameraIntrinsic()
        width = self.cam_info.width
        height = self.cam_info.height

        fx = self.cam_info.K[0]
        cx = self.cam_info.K[2]
        fy = self.cam_info.K[4]
        cy = self.cam_info.K[5]
        if self.camera_config.rotate_image is None:
            self.cam_intrinsic.set_intrinsics(width, height, fx, fy, cx, cy)
        else:
            # Rotate the image as desired AND fix camera parameters.
            K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
            # 1) calculate cam intrinsics
            # 2) modify ROS cam info setting
            rotated_img, K_rotated, new_size = self.rotate_image_and_intrinsics(
                self.color, K=K, rotation=self.camera_config.rotate_image
            )
            self.color = rotated_img
            k_flat = K_rotated.flatten()
            fx = k_flat[0]
            cx = k_flat[2]
            fy = k_flat[4]
            cy = k_flat[5]
            width, height = new_size
            # K is an immutable Tuple, so we have to override it completely
            self.cam_info.K = (fx, 0, cx, 0, fy, cy, 0, 0, 1)
            self.cam_intrinsic.set_intrinsics(width, height, fx, fy, cx, cy)

        cas.set(CASViews.COLOR_IMAGE, self.color)
        cas.set(CASViews.DEPTH_IMAGE, None)
        cas.set(CASViews.CAM_INFO, self.cam_info)
        cas.set(CASViews.CAM_INTRINSIC, self.cam_intrinsic)
        cas.set(CASViews.COLOR2DEPTH_RATIO, (1, 1))
        self.store_cam_to_world_transform(cas, self.timestamp)

        self._has_new_data = False
        self.lock.release()
