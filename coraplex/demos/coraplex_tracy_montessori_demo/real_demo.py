"""
The montessori demo carried out by the real Tracy, through giskard.

Run it from this folder::

    python real_demo.py

The world is not built here but fetched from the running stack, so the robot the plan
addresses is the one the controllers already hold, and a synchronizer keeps it in step
with what the robot actually does. The board and the pieces are then stood in that world.
"""

from __future__ import annotations

import os
import signal
import subprocess
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import timedelta
from typing_extensions import Iterator

import rclpy
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node

from coraplex.plans.context_extensions import RobotAccess
from coraplex.plans.executors import RealRobotPlanExecutor
from cramph.context import StatechartContext
from demo import DemoRunner, ShapeSortingPlan
from semantic_digital_twin.adapters.ros.world_fetcher import fetch_world_from_service
from semantic_digital_twin.adapters.ros.world_synchronizer import WorldSynchronizer
from semantic_digital_twin.robots.tracy import Tracy
from semantic_digital_twin.world import World


@dataclass
class RunningRobot:
    """
    The real Tracy's stack, up and reachable: its world and the node the demo talks
    through.
    """

    world: World
    """
    The world fetched from the stack, kept in step with what the robot does.
    """

    node: Node
    """
    The demo's own node, spun on a thread of its own.
    """


@dataclass
class RealRunner(DemoRunner):
    """
    Carries the sorting plan out on the real robot, bringing its controllers up first.
    """

    controller_launch_command: list[str] = field(
        default_factory=lambda: [
            "ros2",
            "launch",
            "giskardpy_ros",
            "giskardpy_tracy_standalone.launch.py",
        ]
    )
    """
    The command that brings the robot's controllers up.
    """

    controller_start_up_time: timedelta = timedelta(seconds=8)
    """
    How long the controllers are given to come up before anything talks to them.
    """

    world_fetch_timeout: timedelta = timedelta(seconds=300)
    """
    How long fetching the world may take: as long as giskardpy's own client waits, since
    the server may still be parsing the robot's description.
    """

    node_name: str = "tracy_montessori_demo_real"
    """
    The name of the demo's own ROS node.
    """

    def run(self) -> None:
        with self.running_robot() as robot:
            self.scene.populate(robot.world)
            [tracy] = robot.world.get_semantic_annotations_by_type(Tracy)
            executor = RealRobotPlanExecutor(
                StatechartContext.with_extensions(robot.world, [RobotAccess(tracy)]),
                ros_node=robot.node,
                collision_avoidance=False,
            )
            executor.compile(ShapeSortingPlan(scene=self.scene).build(executor.context))
            executor.execute()
            executor.close()

    @contextmanager
    def running_robot(self) -> Iterator[RunningRobot]:
        """
        Bring the robot's controllers up and fetch the world they hold; shut them down
        when the demo is done.

        :return: The running robot, for as long as the context lasts.
        """
        controllers = subprocess.Popen(
            self.controller_launch_command, start_new_session=True
        )
        time.sleep(self.controller_start_up_time.total_seconds())
        try:
            rclpy.init()
            node = rclpy.create_node(self.node_name)
            executor = MultiThreadedExecutor()
            executor.add_node(node)
            threading.Thread(
                target=executor.spin, daemon=True, name="rclpy-executor"
            ).start()

            world = fetch_world_from_service(
                node=node, timeout_seconds=self.world_fetch_timeout.total_seconds()
            )
            WorldSynchronizer(_world=world, node=node)
            yield RunningRobot(world=world, node=node)
        finally:
            os.killpg(os.getpgid(controllers.pid), signal.SIGTERM)
            controllers.wait()


if __name__ == "__main__":
    RealRunner().run()
