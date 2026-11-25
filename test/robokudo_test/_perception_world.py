"""
The world perception perceives into, as tests place cameras in and share it.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import timedelta
from enum import StrEnum

from rclpy.node import Node
from semantic_digital_twin.adapters.ros.world_fetcher import FetchWorldServer
from semantic_digital_twin.adapters.ros.world_synchronizer import WorldSynchronizer
from krrood_spatial_math.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.datastructures.camera_model import (
    CameraModality,
    PinholeCameraModel,
)
from semantic_digital_twin.datastructures.camera_resolution import CameraResolution
from semantic_digital_twin.datastructures.field_of_view import FieldOfView
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import (
    Connection6DoF,
    FixedConnection,
)
from semantic_digital_twin.world_description.world_entity import Body

from robokudo.cas import CAS
from robokudo.descriptors.camera_configs.base_camera_config import BaseCameraConfig
from robokudo.descriptors.camera_configs.components import TfComponent
from robokudo.io.camera_interface import CameraInterface, ROSCameraInterface

# %% frames


class FrameName(StrEnum):
    """
    The names of the frames tests place cameras against.
    """

    WORLD = "map"
    """
    The frame the camera pose is given in.
    """

    CAMERA = "camera"
    """
    The frame of the camera.
    """

    HEAD = "head"
    """
    The frame of the head a robot's camera is mounted on.
    """


class RobotName(StrEnum):
    """
    The names of the robots in test worlds.
    """

    ROBOT = "robot"
    """
    A robot whose parts carry its name as prefix, as the parts of a robot loaded from its
    description do.
    """


# %% waiting


def wait_until(
    condition: Callable[[], bool],
    timeout: timedelta = timedelta(seconds=5),
    poll_interval: timedelta = timedelta(milliseconds=50),
) -> bool:
    """
    Wait for ``condition`` to hold, at most ``timeout``.

    :param condition: The condition to wait for.
    :param timeout: How long to wait, long enough for a change to travel between
        synchronized worlds or a thread to stop.
    :param poll_interval: How long to sleep between two checks of ``condition``.
    :return: Whether the condition held in time.
    """
    deadline = time.monotonic() + timeout.total_seconds()
    while time.monotonic() < deadline:
        if condition():
            return True
        time.sleep(poll_interval.total_seconds())
    return condition()


# %% camera placement


def camera_pose(x: float = 0.0) -> HomogeneousTransformationMatrix:
    """
    :param x: Position of the camera along the world x axis.
    :return: A camera pose looking along the world frame axes.
    """
    return HomogeneousTransformationMatrix.from_xyz_rpy(x=x)


def cas_seen_from(
    world: World,
    world_T_camera: HomogeneousTransformationMatrix,
    reference_frame: str = FrameName.WORLD,
) -> CAS:
    """
    A CAS whose camera is placed at ``world_T_camera`` in ``world``.

    :param world: The world the camera is placed in.
    :param world_T_camera: Pose of the camera in ``reference_frame``.
    :param reference_frame: The frame the camera pose is measured against.
    :return: The CAS.
    """
    cas = CAS()
    place_camera_of(cas, world, world_T_camera, reference_frame=reference_frame)
    return cas


def camera_interface(*, located_through_tf: bool = False) -> CameraInterface:
    """
    :param located_through_tf: Whether the camera pose is looked up through TF.
    :return: A camera interface that places its camera against the test frames.
    """
    if located_through_tf:
        return ros_camera_interface(lookup_viewpoint=True)
    return CameraInterface(BaseCameraConfig(interface_type="test"))


def ros_camera_interface(*, lookup_viewpoint: bool) -> ROSCameraInterface:
    """
    :param lookup_viewpoint: Whether the camera pose is looked up through TF.
    :return: A ROS camera interface without subscriptions or TF listener, for the
        test frames.
    """
    interface = object.__new__(ROSCameraInterface)
    interface.camera_config = TfComponent(
        tf_from=FrameName.CAMERA,
        tf_to=FrameName.WORLD,
        lookup_viewpoint=lookup_viewpoint,
    )
    interface.lookup_viewpoint = lookup_viewpoint
    return interface


def camera_model() -> PinholeCameraModel:
    """
    :return: The calibration of the cameras of the test worlds.
    """
    return PinholeCameraModel.from_field_of_view(
        resolution=CameraResolution(), field_of_view=FieldOfView()
    )


def place_camera_of(
    cas: CAS,
    world: World,
    world_T_camera: HomogeneousTransformationMatrix,
    interface: CameraInterface | None = None,
    reference_frame: str = FrameName.WORLD,
    camera_frame: str = FrameName.CAMERA,
) -> None:
    """
    Place the camera of ``cas`` at ``world_T_camera`` in ``world`` and store its
    observation in ``cas``.

    :param cas: The CAS whose camera is placed.
    :param world: The world the camera is placed in.
    :param world_T_camera: Pose of the camera in ``reference_frame``.
    :param interface: The camera interface placing the camera; a live camera located
        through TF if omitted.
    :param reference_frame: The frame the camera pose is measured against.
    :param camera_frame: The frame the camera's images are given in.
    """
    interface = (
        interface
        if interface is not None
        else camera_interface(located_through_tf=True)
    )
    interface.store_camera_observation(
        cas=cas,
        world=world,
        camera_model=camera_model(),
        camera_frame=camera_frame,
        timestamp_nanoseconds=0,
        modalities=(CameraModality.COLOR, CameraModality.DEPTH),
        world_T_camera=interface.bind_world_T_camera(
            world=world,
            reference_frame=reference_frame,
            camera_frame=camera_frame,
            world_T_camera=world_T_camera,
        ),
    )


# %% robot camera


def world_with_a_head_camera(robot_name: str | None = None) -> World:
    """
    :param robot_name: The prefix of the names of the head and the camera, as a robot
        loaded from its description prefixes its parts; no prefix if None.
    :return: A world whose camera hangs below a head, as on a robot, 1.5 m above the
        world frame.
    """
    world = World()
    world_body = Body(name=PrefixedName(name=FrameName.WORLD))
    head_body = Body(name=PrefixedName(name=FrameName.HEAD, prefix=robot_name))
    camera_body = Body(name=PrefixedName(name=FrameName.CAMERA, prefix=robot_name))
    with world.modify_world():
        world.add_body(body=world_body)
        world.add_connection(
            connection=Connection6DoF.create_with_dofs(
                parent=world_body, child=head_body, world=world
            )
        )
        world.add_connection(
            connection=FixedConnection(parent=head_body, child=camera_body)
        )
    head_body.parent_connection.origin = HomogeneousTransformationMatrix.from_xyz_rpy(
        z=1.5, reference_frame=world_body
    )
    return world


# %% shared world


@dataclass(kw_only=True)
class SharedWorld:
    """
    The world another process serves and keeps synchronized, such as the world of the
    process executing robot actions.
    """

    world: World
    """
    The served world.
    """

    server: FetchWorldServer
    """
    Serves the world to processes that join.
    """

    synchronizer: WorldSynchronizer
    """
    Keeps the served world in sync with the processes that joined.
    """

    @classmethod
    def served_through(cls, *, node: Node, world: World) -> SharedWorld:
        """
        :param node: The node of the process serving ``world``.
        :param world: The world to serve.
        :return: ``world``, served and synchronized through ``node``.
        """
        return cls(
            world=world,
            server=FetchWorldServer(node=node, world=world),
            synchronizer=WorldSynchronizer(node=node, _world=world),
        )

    def close(self) -> None:
        """
        Stop serving and synchronizing the world.
        """
        self.synchronizer.close()
        self.server.close()
