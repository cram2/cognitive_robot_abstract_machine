"""
The montessori demo carried out by the real Tracy, through giskard.

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
from dataclasses import dataclass
from typing import TYPE_CHECKING, Iterator

import rclpy
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node

from coraplex.datastructures.dataclasses import Context
from coraplex.datastructures.enums import ExecutionType
from coraplex.execution_environment import ExecutionEnvironment
from semantic_digital_twin.adapters.ros.world_fetcher import fetch_world_from_service
from semantic_digital_twin.adapters.ros.world_synchronizer import WorldSynchronizer
from semantic_digital_twin.robots.tracy import Tracy
from semantic_digital_twin.world import World

if TYPE_CHECKING:
    from demo import MontessoriEnvironment


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


@contextmanager
def running_robot() -> Iterator[RunningRobot]:
    """
    Bring the robot's controllers up and fetch the world they hold; shut them down when
    the demo is done.

    :return: The running robot, for as long as the context lasts.
    """
    giskard_process = subprocess.Popen(
        ["ros2", "launch", "giskardpy_ros", "giskardpy_tracy_standalone.launch.py"],
        start_new_session=True,
    )
    # Give the controllers time to come up before anything talks to them.
    time.sleep(8)
    try:
        rclpy.init()
        node = rclpy.create_node("tracy_montessori_demo_real")
        executor = MultiThreadedExecutor()
        executor.add_node(node)
        threading.Thread(
            target=executor.spin, daemon=True, name="rclpy-executor"
        ).start()

        # As long as giskardpy's own client waits: the server may still be parsing the
        # robot's description, and a shorter budget would expire.
        world = fetch_world_from_service(node=node, timeout_seconds=300)
        WorldSynchronizer(_world=world, node=node)
        yield RunningRobot(world=world, node=node)
    finally:
        os.killpg(os.getpgid(giskard_process.pid), signal.SIGTERM)
        giskard_process.wait()


def run(environment: MontessoriEnvironment, robot: RunningRobot) -> None:
    """
    Carry the sorting plan out on the real robot.

    :param environment: The scene, its board and pieces already stood in the fetched
        world.
    :param robot: The running robot the scene was built in.
    """
    [tracy] = robot.world.get_semantic_annotations_by_type(Tracy)
    context = Context(
        world=robot.world,
        robot=tracy,
        ros_node=robot.node,
        evaluate_conditions=False,
    )
    with ExecutionEnvironment(
        execution_type=ExecutionType.REAL,
        collision_avoidance=False,
    ):
        environment.build_plan(context).perform()
