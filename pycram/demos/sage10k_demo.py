import threading
import time

import rclpy
import tqdm
from rclpy.executors import SingleThreadedExecutor

from krrood.utils import recursive_subclasses
from pycram.motion_executor import simulated_robot
from pycram.robot_plans.actions.sage10k_actions import *
from semantic_digital_twin.adapters.ros.visualization.viz_marker import (
    VizMarkerPublisher,
)
from semantic_digital_twin.orm.ormatic_interface import *  # type: ignore

demo = Sage10kCraftsmanLobbyDemo()


def run_demo(demo: Sage10kAbstractDemo):
    demo.create_world()
    if not rclpy.ok():
        rclpy.init()
    node = rclpy.create_node("test_node")

    executor = SingleThreadedExecutor()
    executor.add_node(node)

    thread = threading.Thread(target=executor.spin, daemon=True, name="rclpy-executor")
    thread.start()
    time.sleep(0.1)

    viz_marker_publisher = VizMarkerPublisher(_world=demo.world, node=node)
    viz_marker_publisher.with_tf_publisher()

    with simulated_robot:
        demo.plan.perform()

    viz_marker_publisher.stop()
    del demo


# pbar = tqdm.tqdm(recursive_subclasses(Sage10kAbstractDemo))
pbar = tqdm.tqdm([Sage10kCraftsmanLobbyDemo])
for demo in pbar:
    pbar.set_postfix({"Current Scene": demo.scene_url.name})
    run_demo(demo())
