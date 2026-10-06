from copy import deepcopy

import numpy as np
import pytest
import rclpy
from typing_extensions import Generator, Tuple

from coraplex.alternative_motion_mappings.hsrb_motion_mapping import HSRBMoveMotion
from coraplex.alternative_motion_mappings.stretch_motion_mapping import (
    StretchMoveToolCenterPoint,
    StretchMoveSim,
    StretchMoveReal,
    StretchClose,
)
from coraplex.alternative_motion_mappings.tiago_motion_mapping import TiagoMoveSim
from coraplex.datastructures.dataclasses import Context

from coraplex.locations.locations import ReachabilityLocation, VisibilityLocation
from semantic_digital_twin.spatial_types.spatial_types import Pose
from coraplex.execution_environment import simulated_robot
from coraplex.plans.factories import sequential
from coraplex.robot_plans.actions.core.robot_body import ParkArmsAction, MoveTorsoAction
from semantic_digital_twin.datastructures.definitions import TorsoState
from semantic_digital_twin.robots.robot_parts import AbstractRobot
from ..conftest import right_or_only_arm
from ..world_snapshot import WorldSnapshot

try:
    from semantic_digital_twin.robots.garmi import Garmi
except ImportError:
    Garmi = None
from semantic_digital_twin.robots.hsrb import HSRB
from semantic_digital_twin.robots.pr2 import PR2
from semantic_digital_twin.robots.stretch import Stretch
from semantic_digital_twin.robots.tiago import Tiago
from semantic_digital_twin.spatial_types import (
    HomogeneousTransformationMatrix,
)
from semantic_digital_twin.world import World

from ...conftest import SAMPLING_SEED

# The alternative motion mappings that should be available to the plans in this test module.
# Resolution filters by robot type and execution type, so passing the full set is always safe.
ALTERNATIVE_MOTION_MAPPINGS = [
    HSRBMoveMotion,
    StretchMoveToolCenterPoint,
    StretchMoveSim,
    StretchMoveReal,
    StretchClose,
    TiagoMoveSim,
]


@pytest.fixture(
    scope="module",
    params=[
        # TODO Garmi commented out until we get access to the robot description in CI
        # pytest.param(
        #     "garmi",
        #     marks=pytest.mark.skipif(
        #         Garmi is None,
        #         reason="GARMI semantic annotation not installed",
        #     ),
        # ),
        "hsrb",
        "stretch",
        "tiago",
        "pr2",
    ],
)
def setup_multi_robot_simple_apartment(
    request,
    _hsr_world_setup,
    _stretch_world_setup,
    _tiago_world_setup,
    _pr2_world_setup,
    _simple_apartment_setup,
):
    apartment_copy = deepcopy(_simple_apartment_setup)

    if request.param == "hsrb":
        hsr_copy = deepcopy(_hsr_world_setup)
        apartment_copy.merge_world(hsr_copy)
        view = apartment_copy.get_semantic_annotations_by_type(HSRB)
        view = view[0] if view else HSRB.from_world(apartment_copy)
        view.root.parent_connection.origin = (
            HomogeneousTransformationMatrix.from_xyz_rpy(1.5, 2, 0)
        )
        return apartment_copy, view
    elif request.param == "stretch":
        stretch_copy = deepcopy(_stretch_world_setup)
        apartment_copy.merge_world(
            stretch_copy,
        )
        view = apartment_copy.get_semantic_annotations_by_type(Stretch)
        view = view[0] if view else Stretch.from_world(apartment_copy)
        view.root.parent_connection.origin = (
            HomogeneousTransformationMatrix.from_xyz_rpy(1.5, 2, 0)
        )
        return apartment_copy, view

    elif request.param == "tiago":
        tiago_copy = deepcopy(_tiago_world_setup)
        apartment_copy.merge_world(
            tiago_copy,
        )
        view = apartment_copy.get_semantic_annotations_by_type(Tiago)
        view = view[0] if view else Tiago.from_world(apartment_copy)
        view.root.parent_connection.origin = (
            HomogeneousTransformationMatrix.from_xyz_rpy(1.5, 2, 0)
        )
        return apartment_copy, view

    elif request.param == "pr2":
        pr2_copy = deepcopy(_pr2_world_setup)
        apartment_copy.merge_world(
            pr2_copy,
        )
        view = apartment_copy.get_semantic_annotations_by_type(PR2)
        view = view[0] if view else PR2.from_world(apartment_copy)
        view.root.parent_connection.origin = (
            HomogeneousTransformationMatrix.from_xyz_rpy(1.5, 2, 0)
        )
        return apartment_copy, view

    elif request.param == "garmi":
        if Garmi is None:
            pytest.skip("GARMI semantic annotation not installed")
        garmi_world_setup = request.getfixturevalue("garmi_world_setup")
        garmi_copy = deepcopy(garmi_world_setup)
        apartment_copy.merge_world(
            garmi_copy,
        )
        view = Garmi.from_world(apartment_copy)
        view.root.parent_connection.origin = (
            HomogeneousTransformationMatrix.from_xyz_rpy(1.5, 2, 0)
        )
        return apartment_copy, view


@pytest.fixture
def multiple_robot_simple_apartment_context(
    setup_multi_robot_simple_apartment,
) -> Generator[Tuple[World, AbstractRobot, Context]]:
    """
    The shared simple apartment world with one robot, the robot and a context for both,
    returned to its initial model and state after the test.
    """
    world, view = setup_multi_robot_simple_apartment
    snapshot = WorldSnapshot.capture(world)
    yield world, view, Context(
        world,
        view,
        alternative_motion_mappings=ALTERNATIVE_MOTION_MAPPINGS,
        sampling_seed=SAMPLING_SEED,
    )
    snapshot.restore()


def _floor_distance(pose: Pose, body) -> float:
    """
    :return: How far `pose` stands from `body` along the floor.
    """
    offset = body.global_pose.to_position().to_np()[:2] - pose.to_position().to_np()[:2]
    return float(np.linalg.norm(offset))


def _assert_faces(pose: Pose, body) -> None:
    """
    Assert that a robot standing at `pose` has `body` straight ahead.
    """
    offset = body.global_pose.to_position().to_np()[:2] - pose.to_position().to_np()[:2]
    heading = pose.to_rotation_matrix().to_np()[:2, 0]
    np.testing.assert_allclose(heading, offset / np.linalg.norm(offset), atol=0.05)


def test_new_reachability_location_body(
    multiple_robot_simple_apartment_context, rclpy_node
):
    world, robot, context = multiple_robot_simple_apartment_context

    plan = sequential(
        [ParkArmsAction(context.robot.all_arms), MoveTorsoAction(TorsoState.HIGH)],
        context,
    )
    with simulated_robot:
        plan.perform()

        world.notify_state_change()

        arm = right_or_only_arm(context.robot)
        milk = world.get_body_by_name("milk.stl")
        location = ReachabilityLocation(
            Pose(reference_frame=milk), arm, context=context
        )

        pose = next(iter(location))

    assert pose.reference_frame is world.root
    assert _floor_distance(pose, milk) <= float(arm.approximate_length())
    _assert_faces(pose, milk)


def test_visibility_location_pose(multiple_robot_simple_apartment_context):
    world, robot, context = multiple_robot_simple_apartment_context

    plan = sequential(
        [ParkArmsAction(context.robot.all_arms), MoveTorsoAction(TorsoState.HIGH)],
        context,
    )
    with simulated_robot:
        plan.perform()

        world.notify_state_change()

        location = VisibilityLocation(
            world.get_body_by_name("milk.stl").global_pose, context=context
        )

        pose = next(iter(location))

    assert pose.reference_frame is world.root
    _assert_faces(pose, world.get_body_by_name("milk.stl"))


def test_visibility_location_body(multiple_robot_simple_apartment_context):
    world, robot, context = multiple_robot_simple_apartment_context

    plan = sequential(
        [ParkArmsAction(context.robot.all_arms), MoveTorsoAction(TorsoState.HIGH)],
        context,
    )
    with simulated_robot:
        plan.perform()

        world.notify_state_change()

        location = VisibilityLocation(
            Pose(reference_frame=world.get_body_by_name("milk.stl")), context=context
        )

        pose = next(iter(location))

    assert pose.reference_frame is world.root
    _assert_faces(pose, world.get_body_by_name("milk.stl"))
