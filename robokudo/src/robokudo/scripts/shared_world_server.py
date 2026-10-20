"""
Serve a shared world for RoboKudo to join, standing in for the process that executes
robot motions.

The world holds only the ``map`` frame. RoboKudo started with ``_synchronize_world``
fetches it, and every object it perceives is added to this world as well; the
server prints each body it learns about and shows the world in RViz::

    python -m robokudo.scripts.shared_world_server
    python -m robokudo.scripts.main _ae semdt_shared_world_from_files _synchronize_world
"""

from __future__ import annotations

from dataclasses import dataclass, field
from uuid import UUID

import rclpy
from rclpy.executors import ExternalShutdownException, SingleThreadedExecutor
from rclpy.node import Node
from semantic_digital_twin.adapters.ros.visualization.viz_marker import (
    VizMarkerPublisher,
)
from semantic_digital_twin.adapters.ros.world_fetcher import FetchWorldServer
from semantic_digital_twin.adapters.ros.world_synchronizer import WorldSynchronizer
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.world_entity import Body
from typing_extensions import Set

WORLD_FRAME = "map"
"""
Name of the root frame of the served world.
"""

REPORT_PERIOD = 1.0
"""
How often, in seconds, the server reports bodies it has not reported yet.
"""

# %% served world


@dataclass
class SharedWorldServer:
    """
    A world served to the processes that join it and kept in sync with them.
    """

    node: Node
    """
    The node the world is served and synchronized through.
    """

    world: World = field(init=False)
    """
    The served world, holding only :data:`WORLD_FRAME` until others change it.
    """

    reported_body_ids: Set[UUID] = field(init=False, default_factory=set)
    """
    The bodies already reported.
    """

    server: FetchWorldServer = field(init=False)
    """
    Hands the world to processes that join.
    """

    synchronizer: WorldSynchronizer = field(init=False)
    """
    Keeps the world in sync with the processes that joined.
    """

    marker_publisher: VizMarkerPublisher = field(init=False)
    """
    Shows the world in RViz.
    """

    def __post_init__(self) -> None:
        self.world = World()
        with self.world.modify_world():
            self.world.add_body(Body(name=PrefixedName(WORLD_FRAME)))
        self.reported_body_ids = {body.id for body in self.world.bodies}
        self.server = FetchWorldServer(node=self.node, world=self.world)
        self.synchronizer = WorldSynchronizer(node=self.node, _world=self.world)
        self.marker_publisher = VizMarkerPublisher(_world=self.world, node=self.node)

    def report_new_bodies(self) -> None:
        """
        Print the bodies that joined the world since the last report, at their current
        position.
        """
        for body in self.world.bodies:
            if body.id in self.reported_body_ids:
                continue
            self.reported_body_ids.add(body.id)
            x, y, z = body.global_transform.to_np()[:3, 3]
            print(
                f"New body {body.name.name} ({body.id}) at x={x:.2f} y={y:.2f} z={z:.2f}"
            )

    def close(self) -> None:
        """
        Stop serving and synchronizing the world.
        """
        self.synchronizer.close()
        self.server.close()


# %% entry point


def main() -> None:
    """
    Serve the world until interrupted.
    """
    rclpy.init()
    node = rclpy.create_node("shared_world_server")
    served_world = SharedWorldServer(node=node)
    node.create_timer(REPORT_PERIOD, served_world.report_new_bodies)
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    print(f"Serving a world with root '{WORLD_FRAME}'. Start RoboKudo now.")
    try:
        executor.spin()
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        served_world.close()
        executor.shutdown()
        node.destroy_node()
        rclpy.try_shutdown()

if __name__ == "__main__":
    main()
