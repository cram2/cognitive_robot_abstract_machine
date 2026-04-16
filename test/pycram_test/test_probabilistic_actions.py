from pycram.datastructures.enums import TaskStatus
from pycram.robot_plans.probabilistic_actions import MoveToReach
from semantic_digital_twin.adapters.ros.visualization.viz_marker import (
    VizMarkerPublisher,
)
from semantic_digital_twin.robots.pr2 import PR2

# The alternative mapping needs to be imported for the stretch to work properly
import pycram.alternative_motion_mappings.stretch_motion_mapping  # type: ignore
import pycram.alternative_motion_mappings.tiago_motion_mapping  # type: ignore
from pycram.robot_plans.motions import *  # type: ignore


def test_move_to_reach(pr2_world_copy, rclpy_node):
    [pr2] = pr2_world_copy.get_semantic_annotations_by_type(PR2)
    training_environment = MoveToReach.training_environment(pr2, limit=10)
    pub = VizMarkerPublisher(
        _world=pr2_world_copy,
        node=rclpy_node,
    )
    pub.with_tf_publisher()

    training_environment.generate_episode()

    print(training_environment.plan)
    successful = 0
    for index, child in enumerate(training_environment.plan.root.children):
        if child.status == TaskStatus.SUCCEEDED:
            successful += 1
    print(
        f"{successful} / {len(training_environment.plan.root.children)} tries were successful"
    )
