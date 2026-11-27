"""
Tests that running a plan in a copy of a world keeps no reference from the original
world to the copy.
"""

from __future__ import annotations

import gc
from copy import deepcopy

from coraplex.robot_plans.actions.composite.transporting import TransportAction
from coraplex.robot_plans.actions.core.navigation import NavigateAction
from coraplex.robot_plans.actions.core.robot_body import MoveTorsoAction
from semantic_digital_twin.adapters.ros.visualization.viz_marker import (
    VizMarkerPublisher,
)
from semantic_digital_twin.datastructures.definitions import TorsoState
from semantic_digital_twin.robots.pr2 import PR2
from krrood_spatial_math.spatial_types import Pose, Pose2D
from semantic_digital_twin.semantic_annotations.semantic_annotations import Milk
from cramph.composites import Sequence

from ..conftest import SAMPLING_SEED
from ..plan_running import context_of, robot_extensions, run_plan
from ..reference_chain import ReferenceSearch


def test_ref_chain_after_copy(pr2_apartment_context):
    world, view, c = pr2_apartment_context
    copy_world = deepcopy(world)
    copy_world.name = "copy_world"
    assert ReferenceSearch(start=world).find_reference_chain(target=copy_world) == []


def test_ref_chain_after_copy_with_execute(pr2_apartment_context):
    """
    After a plan ran in a copy of the world, the original world holds no reference to
    the copy.
    """
    world, view, _ = pr2_apartment_context
    copy_world = deepcopy(world)
    copy_world.name = "copy_world"

    copy_extensions = robot_extensions(
        copy_world.get_semantic_annotation_by_id(id=view.id)
    )

    plan = Sequence(
        nodes=[
            NavigateAction(
                planar_goal_pose=Pose2D.from_xy_yaw(
                    x=1, y=-1, reference_frame=copy_world.root
                )
            )
        ]
    )

    run_plan(plan, copy_extensions)

    gc.collect()
    assert ReferenceSearch(start=world).find_reference_chain(target=copy_world) == []


def test_ref_chain_after_copy_with_execute_complex_plan(pr2_apartment_context):
    world, view, _ = pr2_apartment_context
    copy_world = deepcopy(world)
    copy_world.name = "copy_world"

    copy_robot = copy_world.get_semantic_annotation_by_id(id=view.id)
    copy_extensions = robot_extensions(copy_robot)

    milk = copy_world.get_semantic_annotations_by_type(semantic_annotation_type=Milk)[0]
    description = TransportAction.from_graspable_by_closest_grasps(
        graspable=milk,
        goal_pose=Pose.from_xyz_quaternion(
            3.1, 2.2, 0.95, 0.0, 0.0, 1.0, 0.0, world.root
        ),
        arm=copy_robot.right_arm,
        context=context_of(copy_extensions),
    )
    plan = Sequence(nodes=[MoveTorsoAction(torso_state=TorsoState.HIGH), description])
    run_plan(plan, copy_extensions)

    gc.collect()
    assert ReferenceSearch(start=world).find_reference_chain(target=copy_world) == []
