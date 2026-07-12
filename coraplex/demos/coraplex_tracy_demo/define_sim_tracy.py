from typing import Any

from coraplex.datastructures.dataclasses import Context
from giskardpy.middleware.ros2 import rospy
from semantic_digital_twin.adapters.ros.visualization.viz_marker import VizMarkerPublisher
from semantic_digital_twin.adapters.urdf import URDFParser
from semantic_digital_twin.robots.tracy import Tracy

import rclpy

from semantic_digital_twin.world import World


def setup_sim_tracy(
    node_name: str = "coraplex_node",
) -> tuple[Any, World, Tracy, Context]:

    rospy.init_node("demo_node")


    # %% Robot Setup
    tracy = "package://iai_tracy_description/urdf/tracy.urdf.xacro"
    tracy_parser = URDFParser.from_file(file_path=tracy)
    tracy_world = tracy_parser.parse()
    Tracy.from_world(tracy_world)

    world = tracy_world

    viz = VizMarkerPublisher(_world=world, node=rospy.node)
    viz.with_tf_publisher()

    # Robot semantic view
    robot_view = world.get_semantic_annotations_by_type(Tracy)[0]

    # Context
    context = Context(
        world,
        robot_view,
    )

    return rospy.node, world, robot_view, context