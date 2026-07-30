import sys

import numpy as np
import sensor_msgs.msg
import std_msgs.msg
import open3d as o3d

from robokudo.types.annotation import PoseAnnotation, PositionAnnotation
from robokudo.utils.type_conversion import (
    ros_camera_info_to_dict,
    ros_camera_info_from_dict,
    get_geometry_msgs_pose_from_pose_annotation,
    get_geometry_msgs_pose_from_position_annotation,
    get_geometry_msgs_pose_stamped_from_pose_annotation,
    get_geometry_msgs_pose_stamped_from_position_annotation,
    get_transform_matrix_from_pose_annotation,
    o3d_camera_intrinsics_from_ros_camera_info,
)


class TestUtilsTypeConversion(object):
    def test_ros_camera_info_to_dict(self):
        kinect_camera_info = sensor_msgs.msg.CameraInfo()
        kinect_camera_info.width = 1024
        kinect_camera_info.height = 1280
        kinect_camera_info.k[0] = 1050.0
        kinect_camera_info.k[2] = 1050.0
        kinect_camera_info.k[4] = 639.5
        kinect_camera_info.k[5] = 479.5

        camera_info_dict = ros_camera_info_to_dict(kinect_camera_info)

        assert camera_info_dict["width"] == 1024
        assert camera_info_dict["height"] == 1280
        assert camera_info_dict["k"] == [
            1050.0,
            0.0,
            1050.0,
            0.0,
            639.5,
            479.5,
            0.0,
            0.0,
            0.0,
        ]

    def test_ros_camera_info_from_dict(self):
        camera_info_dict = {
            "header": {"frame_id": "camera_link", "stamp": {"secs": 100, "nsecs": 100}},
            "width": 1024,
            "height": 1280,
            "K": [1050.0, 0.0, 1050.0, 0.0, 639.5, 479.5, 0.0, 0.0, 0.0],
            "D": np.random.rand(5),
            "P": np.random.rand(12),
            "R": np.random.rand(9),
            "binning_x": 2.0,
            "binning_y": 2.0,
            "roi": {
                "x_offset": 5,
                "y_offset": 10,
                "height": 15,
                "width": 20,
                "do_rectify": True,
            },
        }

        kinect_camera_info = ros_camera_info_from_dict(camera_info_dict)

        assert kinect_camera_info.header.frame_id == "camera_link"
        assert kinect_camera_info.header.stamp.sec == 100
        assert kinect_camera_info.header.stamp.nanosec == 100

        assert kinect_camera_info.width == 1024
        assert kinect_camera_info.height == 1280
        assert np.all(camera_info_dict["K"] == kinect_camera_info.k)
        assert np.all(camera_info_dict["D"] == kinect_camera_info.d)
        assert np.all(camera_info_dict["P"] == kinect_camera_info.p)

        assert kinect_camera_info.binning_x == 2.0
        assert kinect_camera_info.binning_y == 2.0

        assert kinect_camera_info.roi.x_offset == 5
        assert kinect_camera_info.roi.y_offset == 10
        assert kinect_camera_info.roi.height == 15
        assert kinect_camera_info.roi.width == 20
        assert kinect_camera_info.roi.do_rectify == True

    def test_ros_camera_info_from_dict_header_frame_id_only(self):
        camera_info_dict = {"header": {"frame_id": "camera_link"}}

        kinect_camera_info = ros_camera_info_from_dict(camera_info_dict)

        assert isinstance(kinect_camera_info, sensor_msgs.msg.CameraInfo)
        assert kinect_camera_info.header.frame_id == "camera_link"

    def test_ros_camera_info_from_dict_header_stamp_only(self):
        camera_info_dict = {"header": {"stamp": {"secs": 100, "nsecs": 100}}}

        kinect_camera_info = ros_camera_info_from_dict(camera_info_dict)

        assert isinstance(kinect_camera_info, sensor_msgs.msg.CameraInfo)
        assert kinect_camera_info.header.stamp.sec == 100
        assert kinect_camera_info.header.stamp.nanosec == 100

    def test_ros_camera_info_from_dict_empty_dict(self):
        kinect_camera_info = ros_camera_info_from_dict({})
        assert isinstance(kinect_camera_info, sensor_msgs.msg.CameraInfo)

    def test_ros_camera_info_from_dict_invalid_dict(self):
        camera_info_dict = {
            "header": {"invalid_key": "invalid_value"},
            "invalid_key": "invalid_value",
            "roi": {"invalid_key": "invalid_value"},
        }
        kinect_camera_info = ros_camera_info_from_dict(camera_info_dict)
        assert isinstance(kinect_camera_info, sensor_msgs.msg.CameraInfo)

    def test_get_geometry_msgs_pose_from_position_annotation(self):
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

    def test_get_geometry_msgs_pose_from_pose_annotation(self):
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

    def test_get_geometry_msgs_pose_stamped_from_pose_annotation(self):
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

    def test_get_geometry_msgs_pose_stamped_from_position_annotation(self):
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

    def test_get_transform_matrix_from_pose_annotation(self):
        pose_ann = PoseAnnotation()
        pose_ann.translation = np.random.rand(3)
        pose_ann.rotation = [0.707, 0.0, 0.707, 0.0]

        transform_matrix = get_transform_matrix_from_pose_annotation(pose_ann)

        assert np.allclose(
            transform_matrix[:3, :3],
            np.array([[0.0, 0.0, 1.0], [0.0, -1.0, 0.0], [1.0, 0.0, 0.0]]),
        )
        assert np.allclose(transform_matrix[:3, 3], pose_ann.translation)

    def test_o3d_camera_intrinsics_from_ros_camera_info(self):
        kinect_camera_info = sensor_msgs.msg.CameraInfo()
        kinect_camera_info.width = 1024
        kinect_camera_info.height = 1280
        kinect_camera_info.k[0] = 1050.0
        kinect_camera_info.k[2] = 1050.0
        kinect_camera_info.k[4] = 639.5
        kinect_camera_info.k[5] = 479.5

        o3d_intrinsics = o3d_camera_intrinsics_from_ros_camera_info(kinect_camera_info)

        assert isinstance(o3d_intrinsics, o3d.camera.PinholeCameraIntrinsic)
        assert o3d_intrinsics.width == kinect_camera_info.width
        assert o3d_intrinsics.height == kinect_camera_info.height
        assert o3d_intrinsics.intrinsic_matrix[0][0] == kinect_camera_info.k[0]
        assert o3d_intrinsics.intrinsic_matrix[0][2] == kinect_camera_info.k[2]
        assert o3d_intrinsics.intrinsic_matrix[1][1] == kinect_camera_info.k[4]
        assert o3d_intrinsics.intrinsic_matrix[1][2] == kinect_camera_info.k[5]
