"""
The world perception perceives into, as tests place cameras in and share it.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import timedelta

from semantic_digital_twin.adapters.ros.world_fetcher import FetchWorldServer
from semantic_digital_twin.adapters.ros.world_synchronizer import WorldSynchronizer
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.world import World

from robokudo.cas import CAS
from robokudo.camera_placement import CameraPlacement

WORLD_FRAME = "map"
"""
Frame the camera pose is given in.
"""

CAMERA_FRAME = "camera"
"""
Frame of the camera.
"""

SYNCHRONIZATION_TIMEOUT = timedelta(seconds=5)
"""
How long a change may take to travel between synchronized worlds.
"""

# %% waiting for synchronization


def wait_until(condition: Callable[[], bool]) -> bool:
    """
    Wait for ``condition`` to hold, at most :data:`SYNCHRONIZATION_TIMEOUT`.

    :param condition: The condition to wait for.
    :return: Whether the condition held in time.
    """
    deadline = time.monotonic() + SYNCHRONIZATION_TIMEOUT.total_seconds()
    while time.monotonic() < deadline:
        if condition():
            return True
        time.sleep(0.05)
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
    cas.world_frame = WORLD_FRAME
    cas.camera_frame = CAMERA_FRAME
    cas.camera_to_world_transform = CameraPlacement(world).place(
        WORLD_FRAME, CAMERA_FRAME, world_T_camera
    )


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

    def close(self) -> None:
        """
        Stop serving and synchronizing the world.
        """
        self.synchronizer.close()
        self.server.close()
