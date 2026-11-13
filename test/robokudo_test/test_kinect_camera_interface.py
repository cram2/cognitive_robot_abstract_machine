from __future__ import annotations

from collections.abc import Iterator
from datetime import timedelta
from threading import Thread, Timer

import numpy as np
import pytest
from builtin_interfaces.msg import Time
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

from ._perception_world import FrameName, camera_pose

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


def receive_a_frame(
    camera_interface: KinectCameraInterface, acquisition_time: Time | None = None
) -> None:
    """
    Hand ``camera_interface`` a color image, a depth image and their camera info, as its
    subscriptions do.

    :param camera_interface: The camera interface receiving the frame.
    :param acquisition_time: When the frame was taken; the start of time if None.
    """
    bridge = CVBridgeWorkaround()
    color_image = bridge.cv2_to_imgmsg(np.zeros((4, 4, 3), dtype=np.uint8), "bgr8")
    if acquisition_time is not None:
        color_image.header.stamp = acquisition_time
    camera_interface.callback(
        color_image,
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
            camera_interface.set_data(cas=CAS(), world=world_without_the_world_frame())

        receiving = Thread(
            target=receive_a_frame, args=(camera_interface,), daemon=True
        )
        receiving.start()
        receiving.join(timeout=timedelta(seconds=5).total_seconds())

        assert not receiving.is_alive()


# %% camera pose at acquisition time


@pytest.fixture
def turning_camera_interface(node: Node) -> Iterator[KinectCameraInterface]:
    """
    A camera interface receiving raw images, which looks its camera pose up in TF
    against the test frames, where TF has not yet located the camera.
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
    yield camera_interface
    TFListenerProxy().clear_node(node)


def locate_camera(
    camera_interface: KinectCameraInterface, acquisition_time: Time, x: float
) -> None:
    """
    Tell the TF buffer of ``camera_interface`` where its camera was at
    ``acquisition_time``, as its TF subscription does.

    :param camera_interface: The camera interface whose camera is located.
    :param acquisition_time: When the camera was there.
    :param x: Position of the camera along the world x axis then.
    """
    world_T_camera = TransformStamped()
    world_T_camera.header.frame_id = FrameName.WORLD
    world_T_camera.header.stamp = acquisition_time
    world_T_camera.child_frame_id = FrameName.CAMERA
    world_T_camera.transform.translation.x = x
    world_T_camera.transform.rotation.w = 1.0
    camera_interface.tf_buffer.set_transform(world_T_camera, "test")


def read_frame(camera_interface: KinectCameraInterface) -> CAS:
    """
    Read the frame ``camera_interface`` holds into a new CAS, as the collection reader
    does.

    :param camera_interface: The camera interface holding a frame.
    :return: The CAS.
    """
    assert camera_interface.has_new_data()
    cas = CAS()
    camera_interface.set_data(cas=cas, world=World())
    return cas


class TestCameraPoseAtAcquisitionTime:
    def test_frame_is_placed_where_the_camera_was_when_it_was_taken(
        self, turning_camera_interface: KinectCameraInterface
    ) -> None:
        locate_camera(turning_camera_interface, Time(sec=1), x=1.0)
        locate_camera(turning_camera_interface, Time(sec=2), x=2.0)
        receive_a_frame(turning_camera_interface, Time(sec=1))

        observation = read_frame(turning_camera_interface).require_camera_observation()

        np.testing.assert_allclose(
            observation.world_T_camera.to_np(), camera_pose(x=1.0).to_np()
        )

    def test_frame_waits_for_the_camera_pose_of_its_acquisition_time(
        self, turning_camera_interface: KinectCameraInterface
    ) -> None:
        locate_camera(turning_camera_interface, Time(sec=1), x=1.0)
        receive_a_frame(turning_camera_interface, Time(sec=2))
        late_pose = Timer(
            interval=timedelta(milliseconds=100).total_seconds(),
            function=locate_camera,
            args=(turning_camera_interface, Time(sec=2)),
            kwargs={"x": 2.0},
        )
        late_pose.start()

        observation = read_frame(turning_camera_interface).require_camera_observation()

        late_pose.join()
        np.testing.assert_allclose(
            observation.world_T_camera.to_np(), camera_pose(x=2.0).to_np()
        )

    def test_frame_without_a_camera_pose_is_dropped(
        self, turning_camera_interface: KinectCameraInterface
    ) -> None:
        turning_camera_interface.camera_config.camera_pose_timeout = timedelta()
        locate_camera(turning_camera_interface, Time(sec=1), x=1.0)
        receive_a_frame(turning_camera_interface, Time(sec=2))

        assert not turning_camera_interface.has_new_data()

    def test_frame_replacing_a_checked_frame_is_placed_at_its_own_pose(
        self, turning_camera_interface: KinectCameraInterface
    ) -> None:
        locate_camera(turning_camera_interface, Time(sec=1), x=1.0)
        locate_camera(turning_camera_interface, Time(sec=2), x=2.0)
        receive_a_frame(turning_camera_interface, Time(sec=1))
        assert turning_camera_interface.has_new_data()
        receive_a_frame(turning_camera_interface, Time(sec=2))

        cas = CAS()
        turning_camera_interface.set_data(cas=cas, world=World())

        np.testing.assert_allclose(
            cas.require_camera_observation().world_T_camera.to_np(),
            camera_pose(x=2.0).to_np(),
        )

    def test_frame_arriving_while_a_frame_is_read_does_not_wait(
        self, turning_camera_interface: KinectCameraInterface
    ) -> None:
        with turning_camera_interface.lock:
            receiving = Thread(
                target=receive_a_frame,
                args=(turning_camera_interface, Time(sec=1)),
                daemon=True,
            )
            receiving.start()
            receiving.join(timeout=timedelta(seconds=5).total_seconds())

            assert not receiving.is_alive()
