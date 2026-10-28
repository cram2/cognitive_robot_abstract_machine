from __future__ import annotations

from collections.abc import Iterator
from datetime import timedelta
from threading import Thread

import numpy as np
import pytest
from geometry_msgs.msg import TransformStamped
from rclpy.node import Node
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.world_entity import Body
from sensor_msgs.msg import CameraInfo

from robokudo.cas import CAS
from robokudo.descriptors.camera_configs.config_kinect_robot import KinectCameraConfig
from robokudo.exceptions import UnknownWorldFrameError
from robokudo.io.camera_interface import KinectCameraInterface
from robokudo.io.tf_listener_proxy import TFListenerProxy
from robokudo.utils.cv_bridge_workaround import CVBridgeWorkaround

from ._perception_world import FrameName

# %% camera receiving frames


@pytest.fixture
def camera_interface(node: Node) -> Iterator[KinectCameraInterface]:
    """
    A camera interface receiving raw images, which looks its camera pose up in TF
    against the test frames, where TF holds the camera at the world frame.
    """
    camera_interface = KinectCameraInterface(
        KinectCameraConfig(
            color_hints="raw",
            depth_hints="raw",
            hi_res_mode=False,
            tf_from=FrameName.CAMERA,
            tf_to=FrameName.WORLD,
        )
    )
    world_T_camera = TransformStamped()
    world_T_camera.header.frame_id = FrameName.WORLD
    world_T_camera.child_frame_id = FrameName.CAMERA
    world_T_camera.transform.rotation.w = 1.0
    camera_interface.tf_buffer.set_transform_static(world_T_camera, "test")
    yield camera_interface
    TFListenerProxy().clear_node(node)


def receive_a_frame(camera_interface: KinectCameraInterface) -> None:
    """
    Hand ``camera_interface`` a color image, a depth image and their camera info, as its
    subscriptions do.

    :param camera_interface: The camera interface receiving the frame.
    """
    bridge = CVBridgeWorkaround()
    camera_interface.callback(
        bridge.cv2_to_imgmsg(np.zeros((4, 4, 3), dtype=np.uint8), "bgr8"),
        bridge.cv2_to_imgmsg(np.zeros((4, 4), dtype=np.float32), "32FC1"),
        CameraInfo(width=4, height=4, k=[1.0, 0.0, 2.0, 0.0, 1.0, 2.0, 0.0, 0.0, 1.0]),
    )


def world_without_the_world_frame() -> World:
    """
    :return: A world whose only body is not the world frame the camera is measured
        against.
    """
    world = World()
    with world.modify_world():
        world.add_body(Body(name=PrefixedName("root_of_another_world")))
    return world


# %% failed camera placement


class TestFailedCameraPlacement:
    def test_camera_keeps_receiving_after_its_pose_could_not_be_placed(
        self, camera_interface: KinectCameraInterface
    ) -> None:
        receive_a_frame(camera_interface)
        with pytest.raises(UnknownWorldFrameError):
            camera_interface.set_data(CAS(), world_without_the_world_frame())

        receiving = Thread(
            target=receive_a_frame, args=(camera_interface,), daemon=True
        )
        receiving.start()
        receiving.join(timeout=timedelta(seconds=5).total_seconds())

        assert not receiving.is_alive()
