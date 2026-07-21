"""
Tree execution utilities for RoboKudo.

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
from typing import Callable, TypeVar
from typing_extensions import TYPE_CHECKING, Optional

from robokudo.garden import grow_tree
from robokudo.utils.tree import setup_with_descendants_rk
from robokudo.identifier import BBIdentifier

if TYPE_CHECKING:
    from py_trees.behaviour import Behaviour
    from py_trees_ros.trees import BehaviourTree
    from robokudo.cas import CAS

_T = TypeVar("_T")


def _tick_tree_until(
    ae_root: BehaviourTree,
    node: Node,
    stop_condition: Callable[[int], Optional[_T]],
    tick_rate: int,
) -> Optional[_T]:
    """
    Tick an already setup tree until ``stop_condition`` returns a result.
    """
    tick_count = 0
    result = None
    timer = None
    executor = MultiThreadedExecutor(num_threads=2)
    executor_nodes = []

    def tick_tree() -> None:
        nonlocal tick_count, result, timer
        ae_root.tick()
        tick_count += 1
        result = stop_condition(tick_count)

        if result is not None and timer is not None:
            timer.cancel()

    try:
        timer = node.create_timer(1.0 / tick_rate, tick_tree)

        executor.add_node(node)
        executor_nodes.append(node)

        # Fetch the Action Server, if present. It's a separate node.
        blackboard = Blackboard()
        if blackboard.exists(BBIdentifier.QUERY_SERVER):
            query_server_node = blackboard.get(BBIdentifier.QUERY_SERVER)
            executor.add_node(query_server_node)
            executor_nodes.append(query_server_node)

        # Spin only these nodes until we're done; don't touch global rclpy lifecycle.
        while result is None and rclpy.ok(context=node.context):
            # print("spinning node", node.get_name(), id(node)) # in case of missed goal callbacks, debug spinned node
            executor.spin_once(timeout_sec=0.1)

        return result
    finally:
        if timer is not None:
            timer.cancel()
            node.destroy_timer(timer)
        for executor_node in executor_nodes:
            executor.remove_node(executor_node)
        executor.shutdown()


def run_tree_once(
    tree: Behaviour,
    node: Node,
    include_gui: bool = False,
    max_iterations: int = 500,
    tick_rate: int = 5,
) -> Optional[Status]:
    """
    Execute a behavior tree once with monitoring.

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

    assert isinstance(ae_root.root, OneShot)
    one_shot = ae_root.root

    def stop_condition(tick_count: int) -> Optional[Status]:
        if (
            one_shot.status in (Status.SUCCESS, Status.FAILURE)
            or tick_count >= max_iterations
        ):
            return one_shot.status
        return None

    return _tick_tree_until(ae_root, node, stop_condition, tick_rate)


def run_tree_until_successful_cas_count(
    tree: Behaviour,
    node: Node,
    expected_cas_count: int,
    include_gui: bool = False,
    max_iterations: int = 500,
    tick_rate: int = 20,
    on_successful_cas: Optional[Callable[[CAS], None]] = None,
) -> list:
    """
    Execute a tree until a requested number of distinct successful CASes exist.

    The tree is setup once without a ``OneShot`` wrapper so continuously running
    pipelines can process multiple percepts using the same ROS node.
    ``on_successful_cas`` is called immediately after each new successful CAS is
    observed.
    """
    if expected_cas_count < 0:
        raise ValueError("expected_cas_count must be non-negative")
    if expected_cas_count == 0:
        return []

    ae_root = grow_tree(tree, node, include_gui=include_gui, run_once=False)
    setup_with_descendants_rk(ae_root)

    seen_cas_ids = set()
    successful_cas_instances = []

    def stop_condition(tick_count: int) -> Optional[list]:
        if tree.status is Status.SUCCESS:
            cas = tree.cas
            if cas.cas_id not in seen_cas_ids:
                seen_cas_ids.add(cas.cas_id)
                successful_cas_instances.append(cas)
                if on_successful_cas is not None:
                    on_successful_cas(cas)

        if len(successful_cas_instances) >= expected_cas_count:
            return successful_cas_instances

        if tick_count >= max_iterations:
            raise TimeoutError(
                f"Expected {expected_cas_count} successful CASes, "
                f"got {len(successful_cas_instances)} after {tick_count} ticks"
            )

        return None

    result = _tick_tree_until(ae_root, node, stop_condition, tick_rate)
    return result if result is not None else successful_cas_instances
