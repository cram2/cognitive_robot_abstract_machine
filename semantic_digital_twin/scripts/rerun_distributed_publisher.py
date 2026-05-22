from __future__ import annotations

import math
import os
import threading
import time

import rclpy
from ament_index_python.packages import get_package_share_directory
from typing_extensions import List

from semantic_digital_twin.adapters.ros.world_fetcher import FetchWorldServer
from semantic_digital_twin.adapters.ros.world_synchronizer import (
    ModelSynchronizer,
    StateSynchronizer,
)
from semantic_digital_twin.adapters.urdf import URDFParser
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import Connection

UR_TYPE = "ur5"
"""Universal Robots model to load (e.g. ur3/ur5/ur10)."""

UR_XACRO = os.path.join(
    get_package_share_directory("ur_description"),
    "urdf",
    "ur.urdf.xacro",
)
"""Path to the UR description xacro (requires the ``ur_description`` package)."""


def load_example_world() -> World:
    """Parse the UR robot xacro into a world.

    :return: The populated world.
    """
    return URDFParser.from_xacro(
        UR_XACRO, mappings={"ur_type": UR_TYPE, "name": UR_TYPE}
    ).parse()


def movable_connections(world: World) -> List[Connection]:
    """Return the connections whose position can be driven (i.e. have a DOF).

    :param world: The world to inspect.
    :return: The list of connections with a settable position.
    """
    return [
        connection for connection in world.connections if hasattr(connection, "raw_dof")
    ]


def wiggle(world: World, duration_seconds: float = 10.0) -> None:
    """Sweep the world's movable joints with a sine wave to show live updates.

    :param world: The world whose joints are driven.
    :param duration_seconds: How long to keep moving the joints.
    """
    connections = movable_connections(world)
    if not connections:
        return
    start = time.time()
    while time.time() - start < duration_seconds:
        phase = time.time() - start
        for index, connection in enumerate(connections):
            connection.position = 0.6 * math.sin(phase + index)
        time.sleep(0.02)


def main() -> None:
    """Own the authoritative world and serve it over ROS until interrupted."""
    rclpy.init()
    node = rclpy.create_node("semdt_publisher")
    threading.Thread(target=rclpy.spin, args=(node,), daemon=True).start()

    world = load_example_world()
    FetchWorldServer(node=node, world=world)  # serves the initial model on request
    StateSynchronizer(_world=world, node=node)  # publishes state changes
    ModelSynchronizer(_world=world, node=node)  # publishes model changes

    print(
        "Publisher running; start rerun_distributed_viewer.py elsewhere. Ctrl-C to stop."
    )
    try:
        while True:
            wiggle(world)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
