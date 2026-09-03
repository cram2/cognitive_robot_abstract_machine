"""
Giskard-owned ROS lifecycle with process-local shared node access.
"""

from __future__ import annotations

import functools
import traceback
from threading import RLock, Thread
from time import sleep

import rclpy
from rclpy import Future
from rclpy.action import ActionClient
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from semantic_digital_twin.adapters.ros.node_registry import ROSNodeRegistry

executor: MultiThreadedExecutor | None = None
spinner_thread: Thread | None = None


# ROS2 Jazzy race condition fix for ActionClient
# See https://github.com/ros2/rclpy/issues/1589
_original_action_client_init = ActionClient.__init__


@functools.wraps(_original_action_client_init)
def _patched_action_client_init(self, *args, **kwargs):
    if not hasattr(self, "_lock"):
        self._lock = RLock()
    _original_action_client_init(self, *args, **kwargs)


ActionClient.__init__ = _patched_action_client_init


def get_node() -> Node:
    """
    Return the process-local ROS node owned by Giskard's lifecycle.

    The node is stored in the lifecycle-neutral semantic digital twin registry so that
    all libraries in this process resolve the same object. Consumers may use the node,
    but must not destroy it, change its executor ownership, or shut down its context.

    :return: Giskard's registered ROS node.
    :raises ROSNodeNotRegisteredError: If :func:`init_node` has not registered a node.
    """
    return ROSNodeRegistry().get()


def spinner_thread_target(node: Node) -> None:
    """
    Thread that runs a multithreaded executor in the background.

    :param node: The Giskard-owned node registered for shared access.
    """
    global executor
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        while rclpy.ok():
            executor.spin_once(timeout_sec=0.1)
            sleep(0.001)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass
    except Exception:
        traceback.print_exc()
    # Avoid touching a destroyed node during shutdown
    try:
        node.get_logger().info(f"{node.get_name()} died.")
    except Exception:
        pass


def wait_for_future_to_complete(future: Future) -> None:
    while rclpy.ok() and not future.done():
        sleep(0.01)


def init_node(node_name: str) -> None:
    """
    Initialize Giskard's ROS node, executor, and spin thread once.

    Giskard owns the complete lifecycle. The node is registered before consumers can
    access it and remains registered until :func:`shutdown` stops the executor. Calling
    this function again while the node is registered is a no-op.

    :param node_name: Name of the node to create.
    """
    global spinner_thread
    registry = ROSNodeRegistry()
    if registry.has_node():
        return
    if not rclpy.ok():
        rclpy.init()
    node = Node(node_name)
    registry.register(node)
    spinner_thread = Thread(
        target=spinner_thread_target,
        args=(node,),
        daemon=True,
        name=f"{node.get_name()} spin",
    )
    spinner_thread.start()


def shutdown() -> None:
    """
    Cleanly shut down the Giskard-owned ROS runtime.

    Node users must already have stopped. The registration is cleared before the node is
    destroyed, as required by the shared-node contract. This also avoids invalid handles
    on subsequent initializations.
    """
    global executor, spinner_thread

    registry = ROSNodeRegistry()
    node = registry.get() if registry.has_node() else None

    if executor is not None:
        executor.shutdown()
    if spinner_thread is not None:
        spinner_thread.join(2.0)
    if node is not None:
        registry.clear(node)
        node.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()

    executor = None
    spinner_thread = None
