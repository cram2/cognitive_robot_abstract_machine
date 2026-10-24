from __future__ import annotations

# %% Imports
import sys

import numpy as np
import std_msgs.msg

from robokudo.types.annotation import PoseAnnotation, PositionAnnotation
from robokudo.utils.type_conversion import (
    get_geometry_msgs_pose_from_pose_annotation,
    get_geometry_msgs_pose_from_position_annotation,
    get_geometry_msgs_pose_stamped_from_pose_annotation,
    get_geometry_msgs_pose_stamped_from_position_annotation,
    get_transform_matrix_from_pose_annotation,
)


class TestUtilsTypeConversion(object):
    def test_get_geometry_msgs_pose_from_position_annotation(self) -> None:
        position_ann = PositionAnnotation()
        position_ann.translation = np.random.rand(3)

        pose_msg = get_geometry_msgs_pose_from_position_annotation(position_ann)

        assert pose_msg.position.x == position_ann.translation[0]
        assert pose_msg.position.y == position_ann.translation[1]
        assert pose_msg.position.z == position_ann.translation[2]

        assert pose_msg.orientation.x == 0.0
        assert pose_msg.orientation.y == 0.0
        assert pose_msg.orientation.z == 0.0
        assert pose_msg.orientation.w == 1.0

    def test_get_geometry_msgs_pose_from_pose_annotation(self) -> None:
        pose_ann = PoseAnnotation()
        pose_ann.translation = np.random.rand(3)
        pose_ann.rotation = np.random.rand(4)

        pose_msg = get_geometry_msgs_pose_from_pose_annotation(pose_ann)

        assert pose_msg.position.x == pose_ann.translation[0]
        assert pose_msg.position.y == pose_ann.translation[1]
        assert pose_msg.position.z == pose_ann.translation[2]

        assert pose_msg.orientation.x == pose_ann.rotation[0]
        assert pose_msg.orientation.y == pose_ann.rotation[1]
        assert pose_msg.orientation.z == pose_ann.rotation[2]
        assert pose_msg.orientation.w == pose_ann.rotation[3]

    def test_get_geometry_msgs_pose_stamped_from_pose_annotation(self) -> None:
        pose_ann = PoseAnnotation()
        pose_ann.translation = np.random.rand(3)
        pose_ann.rotation = np.random.rand(4)

        header = std_msgs.msg.Header()
        header.frame_id = "some_weird_non_default_frame_id"
        header.stamp.sec = np.random.randint(sys.maxsize)
        header.stamp.nanosec = np.random.randint(sys.maxsize)

        pose_msg = get_geometry_msgs_pose_stamped_from_pose_annotation(pose_ann, header)

        assert pose_msg.header.frame_id == header.frame_id
        assert pose_msg.header.stamp.sec == header.stamp.sec
        assert pose_msg.header.stamp.nanosec == header.stamp.nanosec

        assert pose_msg.pose.position.x == pose_ann.translation[0]
        assert pose_msg.pose.position.y == pose_ann.translation[1]
        assert pose_msg.pose.position.z == pose_ann.translation[2]

        assert pose_msg.pose.orientation.x == pose_ann.rotation[0]
        assert pose_msg.pose.orientation.y == pose_ann.rotation[1]
        assert pose_msg.pose.orientation.z == pose_ann.rotation[2]
        assert pose_msg.pose.orientation.w == pose_ann.rotation[3]

    def test_get_geometry_msgs_pose_stamped_from_position_annotation(self) -> None:
        position_ann = PositionAnnotation()
        position_ann.translation = np.random.rand(3)

        header = std_msgs.msg.Header()
        header.frame_id = "some_weird_non_default_frame_id"
        header.stamp.sec = np.random.randint(sys.maxsize)
        header.stamp.nanosec = np.random.randint(sys.maxsize)

        pose_msg = get_geometry_msgs_pose_stamped_from_position_annotation(
            position_ann, header
        )

        assert pose_msg.header.frame_id == header.frame_id
        assert pose_msg.header.stamp.sec == header.stamp.sec
        assert pose_msg.header.stamp.nanosec == header.stamp.nanosec

        assert pose_msg.pose.position.x == position_ann.translation[0]
        assert pose_msg.pose.position.y == position_ann.translation[1]
        assert pose_msg.pose.position.z == position_ann.translation[2]

        assert pose_msg.pose.orientation.x == 0.0
        assert pose_msg.pose.orientation.y == 0.0
        assert pose_msg.pose.orientation.z == 0.0
        assert pose_msg.pose.orientation.w == 1.0

    def test_get_transform_matrix_from_pose_annotation(self) -> None:
        pose_ann = PoseAnnotation()
        pose_ann.translation = np.random.rand(3)
        pose_ann.rotation = [0.707, 0.0, 0.707, 0.0]

        transform_matrix = get_transform_matrix_from_pose_annotation(pose_ann)

        assert np.allclose(
            transform_matrix[:3, :3],
            np.array([[0.0, 0.0, 1.0], [0.0, -1.0, 0.0], [1.0, 0.0, 0.0]]),
        )
        assert np.allclose(transform_matrix[:3, 3], pose_ann.translation)
