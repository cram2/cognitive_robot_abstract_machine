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
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import (
    Connection6DoF,
    FixedConnection,
)
from semantic_digital_twin.world_description.world_entity import Body

from robokudo.cas import CAS
from robokudo.io.camera_placement import CameraPlacement, MeasuredCameraPose

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
) -> CAS:
    """
    A CAS whose camera is placed at ``world_T_camera`` in ``world``.

    :param world: The world the camera is placed in.
    :param world_T_camera: Pose of the camera in the world frame.
    :return: The CAS.
    """
    cas = CAS()
    place_camera_of(cas, world, world_T_camera)
    return cas


def place_camera_of(
    cas: CAS,
    world: World,
    world_T_camera: HomogeneousTransformationMatrix,
) -> None:
    """
    Place the camera of ``cas`` at ``world_T_camera`` in ``world``.

    :param cas: The CAS whose camera is placed.
    :param world: The world the camera is placed in.
    :param world_T_camera: Pose of the camera in the world frame.
    """
    CameraPlacement(world).place(
        cas,
        MeasuredCameraPose(
            world_frame=FrameName.WORLD,
            camera_frame=FrameName.CAMERA,
            world_T_camera=world_T_camera,
        ),
    )


# %% robot camera


def world_with_a_head_camera() -> World:
    """
    :return: A world whose camera hangs below a head, as on a robot, 1.5 m above the
        world frame.
    """
    world = World()
    world_body = Body(name=PrefixedName(FrameName.WORLD))
    head_body = Body(name=PrefixedName(FrameName.HEAD))
    camera_body = Body(name=PrefixedName(FrameName.CAMERA))
    with world.modify_world():
        world.add_body(world_body)
        world.add_connection(
            Connection6DoF.create_with_dofs(
                parent=world_body, child=head_body, world=world
            )
        )
        world.add_connection(FixedConnection(parent=head_body, child=camera_body))
    head_body.parent_connection.origin = HomogeneousTransformationMatrix.from_xyz_rpy(
        z=1.5, reference_frame=world_body
    )
    return world


# %% shared world


@dataclass
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
    def served_through(cls, node: Node, world: World) -> SharedWorld:
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
