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


def test_move_to_reach(pr2_world_setup, rclpy_node):
    pr2 = PR2.from_world(pr2_world_setup)
    training_environment = MoveToReach.training_environment(pr2)
    pub = VizMarkerPublisher(
        _world=pr2_world_setup,
        node=rclpy_node,
    )
    pub.with_tf_publisher()
    print(training_environment.plan)
    training_environment.generate_episode()
    print(training_environment.plan)
