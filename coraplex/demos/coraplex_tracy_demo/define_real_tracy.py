import threading
import time
from dataclasses import dataclass

import logging
# from suturo_resources.suturo_map import load_environment
from typing_extensions import Tuple, Any

import semantic_digital_twin.exceptions
from coraplex.datastructures import dataclasses
from coraplex.datastructures.dataclasses import Context
from coraplex.datastructures.enums import Arms
# from coraplex.language import SequentialPlan
from coraplex.motion_executor import real_robot
from giskardpy.middleware.ros2 import rospy
# from coraplex.robot_plans import ParkArmsActionDescription
from semantic_digital_twin.adapters.ros.messages import LoadModel
from semantic_digital_twin.adapters.ros.tf_publisher import TFPublisher
from semantic_digital_twin.adapters.ros.visualization.viz_marker import (
    VizMarkerPublisher,
)
from semantic_digital_twin.adapters.ros.world_fetcher import fetch_world_from_service
from semantic_digital_twin.adapters.ros.world_synchronizer import (
    WorldSynchronizer,
    ModelReloadSynchronizer,
)
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.robots.robot_parts import AbstractRobot
from semantic_digital_twin.robots.tracy import Tracy
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import FixedConnection
import numpy as np

from semantic_digital_twin.world_description.geometry import Box, Scale
from semantic_digital_twin.world_description.shape_collection import ShapeCollection
from semantic_digital_twin.world_description.world_entity import Body
from test.krrood_test.dataset.example_classes import Node

logger = logging.getLogger(__name__)

def setup_tracy_context(
    node_name: str = "coraplex_node",
) -> Tuple[Any, World, Tracy, Context]:
    """
    Initializes rclpy, starts a SingleThreadedExecutor in a background thread,
    synchronizes the world model, and returns all relevant objects.

    Returns:
        dict containing:
            - node
            - world
            - robot_view
            - context
    """

    rospy.init_node("demo_node")

    # Fetch world
    world: World = fetch_world_from_service(rospy.node)

    # Synchronizer
    world_sync = WorldSynchronizer(_world=world, node=rospy.node)

    # Optional TF publisher
    # TFPublisher(world=world, node=rospy.node)

    # env_world = load_environment()
    # with world.modify_world():
    #     world.merge_world(env_world)

    # Visualization
    # VizMarkerPublisher(world=world, node=rospy.node)

    # Robot semantic view
    robot_view = world.get_semantic_annotations_by_type(Tracy)[0]

    # Context
    context = Context(
        world,
        robot_view,
        ros_node=rospy.node
    )

    return rospy.node, world, robot_view, context