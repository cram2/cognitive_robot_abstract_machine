"""
The montessori demo drawn in RViz, with Tracy moved kinematically rather than simulated.

Watch it with a ``MarkerArray`` display on ``/semworld/viz_marker``, whose durability has
to be set to transient local for the markers to arrive.

Neither publisher is driven by a ROS timer: each one publishes from a callback the world
itself calls when its model or its state changes, so the node needs nothing spinning it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import rclpy

from coraplex.datastructures.dataclasses import Context
from coraplex.execution_environment import kinematically_simulated_robot
from semantic_digital_twin.adapters.ros.tf_publisher import TFPublisher
from semantic_digital_twin.adapters.ros.visualization.viz_marker import (
    VizMarkerPublisher,
)
from semantic_digital_twin.robots.tracy import Tracy

if TYPE_CHECKING:
    from demo import MontessoriEnvironment
    from semantic_digital_twin.world import World


def run(environment: MontessoriEnvironment, world: World) -> None:
    """
    Draw the scene in RViz and carry the sorting plan out kinematically.

    :param environment: The scene, with the plan that sorts its pieces.
    :param world: The world the environment stood Tracy, the board and the pieces in.
    """
    [robot] = world.get_semantic_annotations_by_type(Tracy)

    rclpy.init()
    node = rclpy.create_node("tracy_montessori_demo_rviz")
    TFPublisher(node=node, _world=world)
    VizMarkerPublisher(_world=world, node=node)
    world.notify_state_change()

    context = Context(
        world=world, robot=robot, ros_node=node, evaluate_conditions=False
    )
    with kinematically_simulated_robot:
        environment.build_plan(context).perform()
