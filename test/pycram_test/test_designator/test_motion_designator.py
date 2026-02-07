import os
import unittest
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
from pycram.datastructures.pose import PoseStamped
from pycram.language import SequentialPlan
from pycram.plan import MotionNode
from pycram.motion_executor import simulated_robot, no_execution, real_robot
from pycram.robot_plans import (
    MoveMotion,
    BaseMotion,
    PickUpActionDescription,
    NavigateActionDescription,
    MoveTorsoActionDescription,
    PickUpAction,
)
from semantic_digital_twin.adapters.urdf import URDFParser
from semantic_digital_twin.datastructures.definitions import TorsoState
from semantic_digital_twin.robots.hsrb import HSRB
from semantic_digital_twin.robots.pr2 import PR2

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
    description = PickUpActionDescription(
        test_world.get_body_by_name("milk.stl"), [Arms.LEFT], [grasp_description]
    )

    plan = plan = SequentialPlan(
        Context.from_world(test_world),
        NavigateActionDescription(
            PoseStamped.from_list([1.7, 1.5, 0], [0, 0, 0, 1], test_world.root),
            True,
        ),
        MoveTorsoActionDescription([TorsoState.HIGH]),
        description,
    )
    with simulated_robot:
        plan.perform()

    pick_up_node = plan.get_nodes_by_designator_type(PickUpAction)[0]

    motion_nodes = list(
        filter(lambda x: isinstance(x, MotionNode), pick_up_node.recursive_children)
    )

    assert len(motion_nodes) == 5

    motion_charts = [type(m.designator_ref.motion_chart) for m in motion_nodes]
    assert all(mc is not None for mc in motion_charts)
    assert CartesianPose in motion_charts
    assert JointPositionList in motion_charts


def test_move_motion_chart(immutable_model_world):
    world, view, context = immutable_model_world
    motion = MoveMotion(PoseStamped.from_list([1, 1, 1], frame=world.root))
    SequentialPlan(context, motion)

    msc = motion.motion_chart

    assert msc
    np.testing.assert_equal(msc.goal_pose.to_position().to_np(), np.array([1, 1, 1, 1]))


@pytest.mark.skipIf(skip_tests, "Alternative motion mappings not available")
def test_alternative_mapping(hsr_apartment_world):
    world, view, context = hsr_apartment_world
    move_motion = MoveMotion(PoseStamped.from_list([1, 1, 1], frame=world.root))

    plan = SequentialPlan(context, move_motion)

    with real_robot:
        assert move_motion.get_alternative_motion()
        msc = move_motion.motion_chart
        assert NavigateActionServerTask == type(msc)
