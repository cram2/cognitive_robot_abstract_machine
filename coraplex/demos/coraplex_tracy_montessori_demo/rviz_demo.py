"""
The montessori demo drawn in RViz, with Tracy moved kinematically rather than simulated.
Run it from this folder::

    python rviz_demo.py

Watch it with a ``MarkerArray`` display on ``/semworld/viz_marker``, whose durability has
to be set to transient local for the markers to arrive.

Neither publisher is driven by a ROS timer: each one publishes from a callback the world
itself calls when its model or its state changes, so the node needs nothing spinning it.
"""

from __future__ import annotations

from dataclasses import dataclass

import rclpy

from coraplex.plans.context_extensions import RobotAccess
from coraplex.plans.executors import KinematicallySimulatedPlanExecutor
from cramph.context import StatechartContext
from demo import DemoRunner, ShapeSortingPlan
from semantic_digital_twin.adapters.ros.tf_publisher import TFPublisher
from semantic_digital_twin.adapters.ros.visualization.viz_marker import (
    VizMarkerPublisher,
)
from semantic_digital_twin.robots.tracy import Tracy


@dataclass
class RvizRunner(DemoRunner):
    """
    Draws the scene in RViz and carries the sorting plan out kinematically.
    """

    node_name: str = "tracy_montessori_demo_rviz"
    """
    The name of the ROS node the scene is published through.
    """

    def run(self) -> None:
        world = self.scene.build_world()
        [tracy] = world.get_semantic_annotations_by_type(Tracy)

        rclpy.init()
        node = rclpy.create_node(self.node_name)
        TFPublisher(node=node, _world=world)
        VizMarkerPublisher(_world=world, node=node)
        world.notify_state_change()

        executor = KinematicallySimulatedPlanExecutor(
            StatechartContext.with_extensions(world, [RobotAccess(tracy)]),
            ros_node=node,
        )
        executor.compile(ShapeSortingPlan(scene=self.scene).build(executor.context))
        executor.execute()
        executor.close()


if __name__ == "__main__":
    RvizRunner().run()
