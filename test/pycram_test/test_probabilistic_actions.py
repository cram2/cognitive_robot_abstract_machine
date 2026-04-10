import pytest

from krrood.entity_query_language.backends import ProbabilisticBackend
from pycram.datastructures.dataclasses import Context
from pycram.datastructures.enums import TaskStatus
from pycram.motion_executor import simulated_robot
from pycram.plans.factories import execute_single
from pycram.robot_plans.actions.core.misc import MoveToReachHub
from pycram.robot_plans.probabilistic_actions import MoveToReach
from semantic_digital_twin.adapters.ros.visualization.viz_marker import (
    VizMarkerPublisher,
)
from semantic_digital_twin.adapters.urdf import URDFParser
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.robots.abstract_robot import AbstractRobot
from semantic_digital_twin.robots.pr2 import PR2
from semantic_digital_twin.spatial_types.spatial_types import Pose
from semantic_digital_twin.world_description.connections import OmniDrive
from semantic_digital_twin.world_description.world_entity import Body


def test_move_to_reach(pr2_world_copy, rclpy_node):
    [pr2] = pr2_world_copy.get_semantic_annotations_by_type(PR2)
    training_environment = MoveToReach.training_environment(pr2, limit=50)
    pub = VizMarkerPublisher(
        _world=pr2_world_copy,
        node=rclpy_node,
    )
    # target_pose = Pose.from_xyz_rpy(
    #     x=1,
    #     y=1,
    #     z=1,
    #     reference_frame=pr2_world_copy.root,
    # )
    #
    # context = Context(
    #     robot=pr2, world=pr2_world_copy, query_backend=ProbabilisticBackend()
    # )
    # manipulator = pr2.manipulators[0]
    # pub.with_tf_publisher()
    # with simulated_robot:
    #     execute_single(
    #         MoveToReach(
    #             standing_pose=target_pose,
    #             manipulator=manipulator,
    #             target_pose=target_pose,
    #             grasp_description=None,
    #         ),
    #         context,
    #     ).perform()

    training_environment.generate_episode()

    successful = 0
    for index, child in enumerate(training_environment.plan.root.children):
        if child.status == TaskStatus.SUCCEEDED:
            successful += 1
    print(
        f"{successful} / {len(training_environment.plan.root.children)} tries were successful"
    )
