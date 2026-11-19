"""
RViz runner for the Tracy screwing demo, publishing transforms and markers via ROS 2.

Run from this folder::

    python rviz_demo.py --real-time
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import rclpy
from coraplex.plans.context_extensions import RobotAccess
from coraplex.plans.executors import KinematicallySimulatedPlanExecutor
from cramph.context import StatechartContext
from cramph.executor import NoPacing, RealTimePacer
from demo import ScrewingDemoRunner, ScrewingSceneKind, build_scene
from semantic_digital_twin.adapters.ros.tf_publisher import TFPublisher
from semantic_digital_twin.adapters.ros.visualization.viz_marker import (
    VizMarkerPublisher,
)


@dataclass(kw_only=True)
class RvizRunner(ScrewingDemoRunner):
    """
    Publishes the screwing scene to RViz and executes the plan kinematically.
    """

    real_time: bool = False
    """
    Whether to pace execution in wall-clock real time for viewing in RViz.
    """

    node_name: str = "tracy_screwing_demo_rviz"
    """
    Name of the ROS 2 node publishing TF and visualization markers.
    """

    def run(self) -> None:
        world, tracy = self.scene.build_world()
        context = StatechartContext.with_extensions(
            world=world, extensions=[RobotAccess(robot=tracy)]
        )
        statechart = self.scene.build_statechart(context=context, tracy=tracy)

        rclpy.init()
        node = rclpy.create_node(self.node_name)
        try:
            TFPublisher(node=node, _world=world)
            VizMarkerPublisher(_world=world, node=node)
            world.notify_state_change()

            executor = KinematicallySimulatedPlanExecutor(
                context=context,
                ros_node=node,
                simulation_time_per_action=self.simulation_time_limit,
                pacer=RealTimePacer() if self.real_time else NoPacing(),
            )
            try:
                executor.compile(statechart=statechart)
                executor.execute()
            finally:
                executor.close()
        finally:
            node.destroy_node()
            rclpy.try_shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scene",
        choices=tuple(ScrewingSceneKind),
        default=ScrewingSceneKind.AIRPLANE.value,
        help="scene to run: nut or airplane (default: airplane)",
    )
    parser.add_argument(
        "--turn-by-hand",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="whether to turn the bolt by hand during insertion (default: True)",
    )
    parser.add_argument(
        "--real-time",
        action="store_true",
        help="pace execution in wall-clock real time for RViz viewing",
    )
    arguments = parser.parse_args()
    selected_scene = build_scene(
        scene_kind=ScrewingSceneKind(arguments.scene),
        turn_by_hand=arguments.turn_by_hand,
    )
    RvizRunner(scene=selected_scene, real_time=arguments.real_time).run()
