import numpy as np

from pycram.datastructures.dataclasses import Context
from pycram.datastructures.enums import ApproachDirection, VerticalAlignment
from pycram.datastructures.grasp import GraspDescription
from pycram.motion_executor import simulated_robot
from pycram.plans.factories import execute_single
from pycram.robot_plans.actions.core.misc import MoveToReach
from semantic_digital_twin.adapters.ros.visualization.viz_marker import (
    VizMarkerPublisher,
)
from semantic_digital_twin.robots.abstract_robot import Manipulator
from semantic_digital_twin.spatial_types.spatial_types import Pose


def test_move_to_reach_empty_world(pr2_world_state_reset, rclpy_node):
    world = pr2_world_state_reset
    VizMarkerPublisher(_world=world, node=rclpy_node).with_tf_publisher()
    context = Context.from_world(world=world)

    move_to_reach = MoveToReach(
        target_pose=Pose.from_xyz_rpy(z=1.2, reference_frame=world.root),
        robot_x=-0.8,
        robot_y=0.0,
        hip_rotation=-0.0,
        grasp_description=GraspDescription(
            approach_direction=ApproachDirection.FRONT,
            vertical_alignment=VerticalAlignment.NoAlignment,
            rotate_gripper=False,
            manipulator=world.get_semantic_annotations_by_type(Manipulator)[0],
        ),
    )

    plan = execute_single(move_to_reach, context=context)
    with simulated_robot:
        plan.perform()
