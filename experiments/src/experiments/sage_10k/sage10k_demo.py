"""
Sage10k Demos Runner
====================

This script executes all available demos in the Sage10k dataset. It iterates through
all subclasses of Sage10kAbstractDemo, setting up the simulation environment for each,
and executing the predefined robot plans.

Each demo run involves:
1. Creating the simulation world.
2. Initializing a ROS 2 node and executor.
3. Starting a visualization marker publisher for real-time feedback.
4. Performing the robot's plan in a simulated environment.

.. warning::
    Running this script executes all demos in sequence, which takes approximately 20 minutes to complete.
"""

import threading
import time

import rclpy
import tqdm
from rclpy.executors import SingleThreadedExecutor

from experiments.sage_10k.demos import Sage10kAbstractDemoHSRB
from krrood.utils import recursive_subclasses
from cramph.context import StatechartContext
from coraplex.plans.executors import KinematicallySimulatedPlanExecutor
from cramph.statechart import Statechart
from semantic_digital_twin.adapters.ros.visualization.viz_marker import (
    VizMarkerPublisher,
)
from semantic_digital_twin.orm.ormatic_interface import *  # type: ignore


def run_demo(demo: Sage10kAbstractDemoHSRB):
    """
    Runs a single Sage10k demo.

    This function initializes a ROS 2 node, sets up the simulation world, starts a
    visualization marker publisher, and performs the robot's plan.

    :param demo: The demo instance to run.
    """
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

    plan_executor = KinematicallySimulatedPlanExecutor(
        StatechartContext.with_extensions(demo.world, demo.context_extensions)
    )
    statechart = Statechart(context=plan_executor.context)
    statechart.add_node(demo.plan)
    plan_executor.compile(statechart)
    plan_executor.execute()

    viz_marker_publisher.stop()
    del demo


if __name__ == "__main__":
    pbar = tqdm.tqdm(recursive_subclasses(Sage10kAbstractDemoHSRB))
    for demo in pbar:
        pbar.set_postfix({"Current Scene": demo.scene_url.name})
        run_demo(demo())
