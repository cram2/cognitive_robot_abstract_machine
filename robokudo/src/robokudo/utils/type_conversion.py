"""
Type conversion utilities for Robokudo.

This module provides functions for converting between different type representations:

* ROS message types to Python dictionaries
* Robokudo annotations to ROS geometry messages
* Pose and position annotations to transform matrices

The module supports bidirectional conversion between:
* ROS CameraInfo messages
* Geometry messages (Pose, PoseStamped)
* Robokudo annotation types
* Transform matrices
"""

from __future__ import annotations

import numpy as np
import open3d as o3d
from geometry_msgs.msg import Pose, PoseStamped
from rosidl_runtime_py import message_to_ordereddict
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import Header
from typing_extensions import TYPE_CHECKING, Dict

from robokudo.utils.cv_bridge_workaround import CVBridgeWorkaround
from robokudo.utils.o3d_helper import get_obb_from_size_and_transform
from robokudo.utils.transform import get_transform_matrix, get_rotation_matrix_from_q

if TYPE_CHECKING:
    import numpy.typing as npt
    from robokudo.types.annotation import (
        PositionAnnotation,
        BoundingBox3DAnnotation,
        PoseAnnotation,
    )


_CV_BRIDGE_WORKAROUND = CVBridgeWorkaround()


def ros_camera_info_to_dict(camera_info: CameraInfo) -> Dict:
    """
    Convert ROS CameraInfo message to dictionary.

    :param camera_info: ROS CameraInfo message
    :return: Dictionary representation of camera info
    """
    return message_to_ordereddict(camera_info)


def ros_camera_info_from_dict(dict_camera_info: Dict) -> CameraInfo:
    """
    Convert dictionary to ROS CameraInfo message.

    :param dict_camera_info: Dictionary containing camera info
    :return: ROS CameraInfo message
    """
    camera_info = CameraInfo()

    # set_message_fields(camera_info, dict_camera_info)  # Fails due to passing stamp as a positional argument to Header.__init__

    if dict_camera_info.get("header", None):
        if dict_camera_info["header"].get("frame_id", None):
            camera_info.header.frame_id = dict_camera_info["header"]["frame_id"]
        if dict_camera_info["header"].get("stamp", None):
            camera_info.header.stamp.sec = dict_camera_info["header"]["stamp"]["secs"]
            camera_info.header.stamp.nanosec = dict_camera_info["header"]["stamp"][
                "nsecs"
            ]

    if dict_camera_info.get("height", None):
        camera_info.height = dict_camera_info["height"]
    if dict_camera_info.get("width", None):
        camera_info.width = dict_camera_info["width"]

    if dict_camera_info.get("D", None) is not None:
        camera_info.d = dict_camera_info["D"]
    if dict_camera_info.get("K", None) is not None:
        camera_info.k = dict_camera_info["K"]
    if dict_camera_info.get("R", None) is not None:
        camera_info.r = dict_camera_info["R"]
    if dict_camera_info.get("P", None) is not None:
        camera_info.p = dict_camera_info["P"]

    if dict_camera_info.get("binning_x", None):
        camera_info.binning_x = dict_camera_info["binning_x"]
    if dict_camera_info.get("binning_y", None):
        camera_info.binning_y = dict_camera_info["binning_y"]

    if dict_camera_info.get("roi", None):
        for key in dict_camera_info["roi"]:
            if hasattr(camera_info.roi, key):
                setattr(camera_info.roi, key, dict_camera_info["roi"][key])

    return camera_info


# RK Type Annotations


def get_geometry_msgs_pose_from_position_annotation(
    position_annotation: PositionAnnotation,
) -> Pose:
    """
    Convert position annotation to ROS Pose message.

    Creates a Pose with position from annotation and identity quaternion.

    :param position_annotation: Position annotation to convert
    :return: ROS Pose message
    """
    pose = Pose()
    pose.position.x = position_annotation.translation[0]
    pose.position.y = position_annotation.translation[1]
    pose.position.z = position_annotation.translation[2]
    pose.orientation.x = float(0)
    pose.orientation.y = float(0)
    pose.orientation.z = float(0)
    pose.orientation.w = float(1)
    return pose


def get_geometry_msgs_pose_from_pose_annotation(
    pose_annotation: PoseAnnotation,
) -> Pose:
    """
    Convert pose annotation to ROS Pose message.

    :param pose_annotation: Pose annotation to convert
    :return: ROS Pose message
    """
    pose = Pose()
    pose.position.x = pose_annotation.translation[0]
    pose.position.y = pose_annotation.translation[1]
    pose.position.z = pose_annotation.translation[2]
    pose.orientation.x = pose_annotation.rotation[0]
    pose.orientation.y = pose_annotation.rotation[1]
    pose.orientation.z = pose_annotation.rotation[2]
    pose.orientation.w = pose_annotation.rotation[3]
    return pose


def get_geometry_msgs_pose_stamped_from_pose_annotation(
    pose_annotation: PoseAnnotation,
    header: Header,
) -> PoseStamped:
    """
    Convert pose annotation to ROS PoseStamped message.

    :param pose_annotation: Pose annotation to convert
    :param header: Header to use for stamped pose
    :return: ROS PoseStamped message
    """
    pose_stamped = PoseStamped()

    pose = get_geometry_msgs_pose_from_pose_annotation(pose_annotation)

    pose_stamped.pose = pose
    pose_stamped.header = header

    return pose_stamped


def get_geometry_msgs_pose_stamped_from_position_annotation(
    position_annotation: PositionAnnotation,
    header: Header,
) -> PoseStamped:
    """
    Convert position annotation to ROS PoseStamped message.

    Creates a PoseStamped with position from annotation and identity quaternion.

    :param position_annotation: Position annotation to convert
    :param header: Header to use for stamped pose
    :return: ROS PoseStamped message
    """
    pose_stamped = PoseStamped()

    pose = get_geometry_msgs_pose_from_position_annotation(position_annotation)

    pose_stamped.pose = pose
    pose_stamped.header = header

    return pose_stamped


def get_transform_matrix_from_pose_annotation(
    pose_annotation: PoseAnnotation,
) -> npt.NDArray:
    """
    Convert pose annotation to 4x4 transform matrix.

    :param pose_annotation: Pose annotation to convert
    :return: 4x4 homogeneous transform matrix
    """
    transform_matrix = get_transform_matrix(
        get_rotation_matrix_from_q(
            np.array(
                [
                    pose_annotation.rotation[0],
                    pose_annotation.rotation[1],
                    pose_annotation.rotation[2],
                    pose_annotation.rotation[3],
                ]
            )
        ),
        np.array(
            [
                pose_annotation.translation[0],
                pose_annotation.translation[1],
                pose_annotation.translation[2],
            ]
        ),
    )
    return transform_matrix


def get_o3d_obb_from_bounding_box_annotation(
    bounding_box_annotation: BoundingBox3DAnnotation,
) -> o3d.geometry.OrientedBoundingBox:
    """
    Convert a bounding box annotation to an open3d oriented bounding box.

    :param bounding_box_annotation: Bounding box annotation to convert
    :return: Open3D OrientedBoundingBox representation of bounding box annotation
    """
    bb_size = np.array(
        [
            bounding_box_annotation.x_length,
            bounding_box_annotation.y_length,
            bounding_box_annotation.z_length,
        ]
    )
    transform_matrix = get_transform_matrix_from_pose_annotation(
        bounding_box_annotation.pose
    )
    return get_obb_from_size_and_transform(bb_size, transform_matrix)


def o3d_camera_intrinsics_from_ros_camera_info(
    camera_info: CameraInfo,
) -> o3d.camera.PinholeCameraIntrinsic:
    """
    Convert ROS CameraInfo to Open3D camera intrinsics.

    :param camera_info: ROS camera info message
    :return: Open3D camera intrinsics
    """
    camera_intrinsic = o3d.camera.PinholeCameraIntrinsic()
    width = camera_info.width
    height = camera_info.height
    fx = camera_info.k[0]
    cx = camera_info.k[2]
    fy = camera_info.k[4]
    cy = camera_info.k[5]
    camera_intrinsic.set_intrinsics(width, height, fx, fy, cx, cy)

    return camera_intrinsic


def convert_ros_to_cv_image(ros_image: Image) -> npt.NDArray:
    """
    Convert ROS image message to OpenCV image.

    :param ros_image: ROS image message
    :return: OpenCV image array
    :raises ValueError: If conversion fails
    """
    cv_rgb_image = _CV_BRIDGE_WORKAROUND.imgmsg_to_cv2(ros_image)
    return cv_rgb_image


def convert_cv_to_ros_image(cv_image: npt.NDArray) -> Image:
    """
    Convert OpenCV image to ROS image message.

    :param cv_image: OpenCV image array
    :return: ROS image message
    :raises ValueError: If conversion fails
    """
    ros_rgb_image = _CV_BRIDGE_WORKAROUND.cv2_to_imgmsg(
        cv_image, encoding="passthrough"
    )
    return ros_rgb_image
