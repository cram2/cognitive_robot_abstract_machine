import gc
from copy import deepcopy

import objgraph

from pycram.datastructures.dataclasses import Context
from pycram.datastructures.enums import Arms, ApproachDirection, VerticalAlignment
from pycram.datastructures.grasp import GraspDescription
from pycram.motion_executor import simulated_robot
from pycram.plans.factories import sequential
from pycram.robot_plans.actions.composite.transporting import TransportAction
from pycram.robot_plans.actions.core.navigation import NavigateAction
from pycram.robot_plans.actions.core.robot_body import MoveTorsoAction
from semantic_digital_twin.adapters.ros.visualization.viz_marker import (
    VizMarkerPublisher,
)
from semantic_digital_twin.datastructures.definitions import TorsoState
from semantic_digital_twin.robots.pr2 import PR2
from semantic_digital_twin.spatial_types.spatial_types import Pose


def test_ref_chain_after_copy(immutable_model_world):
    world, view, c = immutable_model_world
    copy_world = deepcopy(world)
    copy_world.name = "copy_world"
    chain = objgraph.find_ref_chain(world, lambda x: x is copy_world)
    assert chain == [world]


def test_ref_chain_after_copy_with_execute(immutable_model_world):
    world, view, c = immutable_model_world
    copy_world = deepcopy(world)
    copy_world.name = "copy_world"

    copy_context = Context(copy_world, PR2.from_world(copy_world))

    plan = sequential(
        [NavigateAction(Pose.from_xyz_rpy(1, -1, 0, reference_frame=copy_world.root))],
        copy_context,
    )

    with simulated_robot:
        plan.perform()

    gc.collect()
    chain = objgraph.find_ref_chain(world, lambda x: x is copy_world)
    assert chain == [world]


def test_ref_chain_after_copy_with_execute_complex_plan(mutable_model_world):
    world, view, context = mutable_model_world
    copy_world = deepcopy(world)
    copy_world.name = "copy_world"

    copy_context = Context(copy_world, copy_robot := PR2.from_world(copy_world))

    description = TransportAction(
        copy_world.get_body_by_name("milk.stl"),
        Pose.from_xyz_quaternion(3.4, 2.2, 0.95, 0.0, 0.0, 1.0, 0.0, world.root),
        Arms.RIGHT,
        GraspDescription(
            ApproachDirection.RIGHT,
            VerticalAlignment.NoAlignment,
            copy_robot.right_arm.manipulator,
        ),
    )
    plan = sequential([MoveTorsoAction(TorsoState.HIGH), description], copy_context)
    with simulated_robot:
        plan.perform()

    gc.collect()
    chain = objgraph.find_ref_chain(world, lambda x: x is copy_world)
    assert chain == [world]
