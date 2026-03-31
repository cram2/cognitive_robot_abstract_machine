from time import sleep

import rclpy

from semantic_digital_twin.adapters.ros.visualization.viz_marker import (
    VizMarkerPublisher,
)
from semantic_digital_twin.adapters.urdf import URDFParser
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.robots.abstract_robot import AbstractRobot
from semantic_digital_twin.robots.unitree_g1 import UnitreeG1
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.world_entity import Body

# This script is part of the "How to add robots" example

your_ros2_package_name = "iai_offis_g1_description"
your_path_from_package_root_to_urdf = "urdf/offis_unitree_g1.urdf"
robot_world = URDFParser.from_file(
    f"package://{your_ros2_package_name}/{your_path_from_package_root_to_urdf}"
).parse()
robot_world.visualize_world_structure().show()

world = World()
root = Body(name=PrefixedName(name="map"))
with world.modify_world():
    world.add_body(root)
world.merge_world(robot_world)

rclpy.init()
VizMarkerPublisher(
    _world=world, node=rclpy.create_node("urdf_visualization_node")
).with_tf_publisher()
sleep(2)

your_robot_class: AbstractRobot = UnitreeG1

robot = your_robot_class.from_world(world)

sleep(2)
