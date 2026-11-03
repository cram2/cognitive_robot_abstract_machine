"""
Type conversion utilities for Robokudo.

This module provides functions for converting between different type representations:

* Robokudo annotations to ROS geometry messages
* Pose and position annotations to transform matrices

The module supports bidirectional conversion between:
* Geometry messages (Pose, PoseStamped)
* Robokudo annotation types
* Transform matrices
"""

from __future__ import annotations

# %% Imports
import numpy as np
import open3d
from geometry_msgs.msg import Pose, PoseStamped
from sensor_msgs.msg import Image
from std_msgs.msg import Header
from typing_extensions import TYPE_CHECKING

from robokudo.utils.cv_bridge_workaround import CVBridgeWorkaround
from robokudo.utils.o3d_helper import create_oriented_bounding_box
from robokudo.utils.transform import get_transform_matrix, get_rotation_matrix_from_q

if TYPE_CHECKING:
    import numpy.typing as npt
    from robokudo.types.annotation import (
        PositionAnnotation,
        BoundingBox3DAnnotation,
        PoseAnnotation,
    )


_CV_BRIDGE_WORKAROUND = CVBridgeWorkaround()


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
) -> open3d.geometry.OrientedBoundingBox:
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
    return create_oriented_bounding_box(bb_size, transform_matrix)


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
