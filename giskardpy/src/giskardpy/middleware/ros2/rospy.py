import traceback
from threading import Thread
from time import sleep

import rclpy
from rclpy import Future
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node

node: Node = None
executor: MultiThreadedExecutor = None
spinner_thread: Thread = None


def spinner_thread_target():
    """
    Thread that runs a multithreaded executor in the background
    """
    global node, executor
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
        if node is not None:
            node.get_logger().info(f"{node.get_name()} died.")
    except Exception:
        pass


def wait_for_future_to_complete(future: Future) -> None:
    while rclpy.ok() and not future.done():
        sleep(0.01)


def init_node(node_name: str) -> None:
    """
    Initialise a global ROS2 node and spin thread.
    """
    global node, spinner_thread, executor
    if node is not None:
        return
    rclpy.init()
    node = Node(node_name)
    spinner_thread = Thread(
        target=spinner_thread_target, daemon=True, name=f"{node.get_name()} spin"
    )
    spinner_thread.start()


def shutdown() -> None:
    """
    Cleanly shutdown the ROS2 node, executor and spin thread between tests.
    This avoids InvalidHandle errors on subsequent initialisations.
    """
    global node, executor, spinner_thread

    executor.shutdown()
    spinner_thread.join(2.0)
    node.destroy_node()
    rclpy.shutdown()

    # Reset globals
    executor = None
    spinner_thread = None
    node = None
