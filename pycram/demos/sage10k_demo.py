import threading
import time

import rclpy
from rclpy.executors import SingleThreadedExecutor

from pycram.motion_executor import simulated_robot
from pycram.robot_plans.actions.sage10k_actions import (
    Sage10kGymDemo,
    Sage10kTVStudioDemo,
    Sage10kCraftsmanLobbyDemo,
    Sage10kTropicalWarehouse,
)
from semantic_digital_twin.adapters.ros.visualization.viz_marker import (
    VizMarkerPublisher,
)
from semantic_digital_twin.orm.ormatic_interface import *  # type: ignore

demo = Sage10kTropicalWarehouse()
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
