import pytest

from pycram.robot_plans.probabilistic_actions import MoveToReach
from semantic_digital_twin.adapters.ros.visualization.viz_marker import (
    VizMarkerPublisher,
)
from semantic_digital_twin.adapters.urdf import URDFParser
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.robots.abstract_robot import AbstractRobot
from semantic_digital_twin.robots.pr2 import PR2
from semantic_digital_twin.world_description.connections import OmniDrive
from semantic_digital_twin.world_description.world_entity import Body


@pytest.fixture
def empty_pr2_world():
    pr2_urdf = "package://iai_pr2_description/robots/pr2_with_ft2_cableguide.xacro"

    pr2_parser = URDFParser.from_file(file_path=pr2_urdf)
    world_with_pr2 = pr2_parser.parse()
    with world_with_pr2.modify_world():
        pr2_root = world_with_pr2.root
        localization_body = Body(name=PrefixedName("odom_combined"))
        world_with_pr2.add_kinematic_structure_entity(localization_body)
        c_root_bf = OmniDrive.create_with_dofs(
            parent=localization_body, child=pr2_root, world=world_with_pr2
        )
        world_with_pr2.add_connection(c_root_bf)
    return world_with_pr2


def test_move_to_reach(empty_pr2_world, rclpy_node):
    pr2 = PR2.from_world(empty_pr2_world)
    training_environment = MoveToReach.training_environment(pr2)
    pub = VizMarkerPublisher(
        _world=training_environment.plan.world,
        node=rclpy_node,
    )
    pub.with_tf_publisher()

    training_environment.generate_episode()
