from copy import deepcopy

import numpy as np
import pytest

from giskardpy.motion_statechart.tasks.cartesian_tasks import CartesianPose
from giskardpy.motion_statechart.tasks.joint_tasks import JointPositionList
from pycram.datastructures.dataclasses import Context
from pycram.datastructures.enums import (
    ApproachDirection,
    VerticalAlignment,
    Arms,
)
from pycram.datastructures.grasp import GraspDescription
from pycram.motion_executor import simulated_robot, real_robot
from pycram.plans.factories import sequential, execute_single
from pycram.plans.plan_node import MotionNode, ActionNode
from pycram.robot_plans.actions.core.navigation import NavigateAction
from pycram.robot_plans.actions.core.pick_up import PickUpAction
from pycram.robot_plans.actions.core.robot_body import MoveTorsoAction
from semantic_digital_twin.datastructures.definitions import TorsoState
from semantic_digital_twin.robots.pr2 import PR2
from semantic_digital_twin.spatial_types import Point3, Quaternion
from semantic_digital_twin.spatial_types.spatial_types import Pose

try:
    from pycram.alternative_motion_mappings.hsrb_motion_mapping import *

    skip_tests = False
except (ImportError, ModuleNotFoundError, AttributeError):
    skip_tests = True


@pytest.mark.skipIf(skip_tests, "Alternative motion mappings not available")
def test_pick_up_motion(immutable_model_world):
    world, view, context = immutable_model_world
    test_world = deepcopy(world)
    test_robot = PR2.from_world(test_world)
    grasp_description = GraspDescription(
        ApproachDirection.FRONT,
        VerticalAlignment.NoAlignment,
        test_robot.left_arm.manipulator,
    )
    pick_up = PickUpAction(
        test_world.get_body_by_name("milk.stl"), Arms.LEFT, grasp_description
    )

    root = sequential(
        children=[
            ActionNode(
                designator=NavigateAction(
                    Pose(
                        Point3.from_iterable([1.7, 1.5, 0]),
                        Quaternion.from_iterable([0, 0, 0, 1]),
                        test_world.root,
                    ),
                    True,
                )
            ),
            MoveTorsoAction(TorsoState.HIGH),
            pick_up,
        ],
        context=Context.from_world(test_world),
    )
    assert pick_up.plan is not None
    with simulated_robot:
        root.perform()

    pick_up_node = root.plan.get_nodes_by_designator_type(PickUpAction)[0]

    motion_nodes = list(
        filter(lambda x: isinstance(x, MotionNode), pick_up_node.descendants)
    )

    assert len(motion_nodes) == 5

    motion_charts = [type(m.designator.motion_chart) for m in motion_nodes]
    assert all(mc is not None for mc in motion_charts)
    assert CartesianPose in motion_charts
    assert JointPositionList in motion_charts


def test_move_motion_chart(immutable_model_world):
    world, view, context = immutable_model_world
    motion = MoveMotion(
        Pose(Point3.from_iterable([1, 1, 1]), reference_frame=world.root)
    )
    plan = execute_single(
        motion,
        context=context,
    )

    msc = motion.motion_chart

    assert msc
    np.testing.assert_equal(msc.goal_pose.to_position().to_np(), np.array([1, 1, 1, 1]))


@pytest.mark.skipIf(skip_tests, "Alternative motion mappings not available")
def test_alternative_mapping(hsr_apartment_world):
    world, view, context = hsr_apartment_world
    move_motion = MoveMotion(
        Pose(Point3.from_iterable([1, 1, 1]), reference_frame=world.root)
    )

    plan = execute_single(move_motion, context=context)

    with real_robot:
        assert move_motion.get_alternative_motion()
        msc = move_motion.motion_chart
        assert NavigateActionServerTask == type(msc)
