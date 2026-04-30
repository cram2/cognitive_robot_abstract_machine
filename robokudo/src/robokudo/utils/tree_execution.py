"""Tree execution utilities for RoboKudo.

This module provides utilities for executing behavior trees in ROS environments.
It supports:

* One-shot tree execution with GUI option
* Rate-limited tree ticking
* Status monitoring and completion detection
* ROS time handling

Dependencies:

* py_trees for behavior tree functionality
* rospy for ROS integration
"""

from __future__ import annotations

import rclpy
from py_trees.blackboard import Blackboard
from py_trees.common import Status
from py_trees.decorators import OneShot
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from typing_extensions import TYPE_CHECKING, Optional

from robokudo.garden import grow_tree
from robokudo.utils.tree import setup_with_descendants_rk
from robokudo.identifier import BBIdentifier

if TYPE_CHECKING:
    from py_trees.behaviour import Behaviour


def run_tree_once(
    tree: Behaviour,
    node: Node,
    include_gui: bool = False,
    max_iterations: int = 500,
    tick_rate: int = 5,
) -> Optional[Status]:
    """Execute a behavior tree once with monitoring.

    This function:
    * Grows the tree with optional GUI
    * Sets up all descendants
    * Ticks the tree at specified rate
    * Monitors for completion via OneShot decorator
    * Handles ROS time exceptions

    :param tree: Root behavior of tree to execute
    :param node: a ros node
    :param include_gui: Whether to show GUI visualization, defaults to False
    :param max_iterations: Maximum number of ticks before timeout, defaults to 500
    :param tick_rate: Rate to tick tree in Hz, defaults to 5
    :return: Final tree status (SUCCESS/FAILURE) or None if timed out
    """
    ae_root = grow_tree(tree, node, include_gui=include_gui, run_once=True)
    setup_with_descendants_rk(ae_root)

    tick_count = 0
    final_status = None
    timer = None

    def tick_tree() -> None:
        nonlocal tick_count, final_status, timer
        ae_root.tick()
        tick_count += 1

        assert isinstance(ae_root.root, OneShot)
        one_shot = ae_root.root

        done = (
            one_shot.status in (Status.SUCCESS, Status.FAILURE)
            or tick_count >= max_iterations
        )
        if done:
            final_status = one_shot.status
            # stop scheduling future ticks
            if timer is not None:
                timer.cancel()

    timer = node.create_timer(1.0 / tick_rate, tick_tree)

    executor = MultiThreadedExecutor(num_threads=2)
    executor.add_node(node)

    # Fetch the Action Server, if present. It's a separate node.
    blackboard = Blackboard()
    if blackboard.exists(BBIdentifier.QUERY_SERVER):
        executor.add_node(blackboard.get(BBIdentifier.QUERY_SERVER))

    # spin only this node until we’re done (don’t touch global rclpy lifecycle)
    while final_status is None and rclpy.ok(context=node.context):
        # print("spinning node", node.get_name(), id(node)) # in case of missed goal callbacks, debug spinned node
        executor.spin_once(timeout_sec=0.1)

    return final_status
