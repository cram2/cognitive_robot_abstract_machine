from copy import deepcopy

import pytest
from typing_extensions import Generator, Tuple

from krrood.entity_query_language.backends import ProbabilisticBackend
from krrood.entity_query_language.factories import underspecified
from pycram.datastructures.dataclasses import Context
from pycram.datastructures.enums import Arms
from pycram.locations.locations import (
    CostmapLocation,
    AccessingLocation,
    GiskardLocation,
)
from pycram.motion_executor import simulated_robot
from pycram.plans.factories import sequential
from pycram.robot_plans.actions.core.robot_body import ParkArmsAction, MoveTorsoAction
from semantic_digital_twin.datastructures.definitions import TorsoState
from semantic_digital_twin.robots.abstract_robot import AbstractRobot
from semantic_digital_twin.robots.hsrb import HSRB
from semantic_digital_twin.robots.pr2 import PR2
from semantic_digital_twin.robots.stretch import Stretch
from semantic_digital_twin.robots.tiago import Tiago
from semantic_digital_twin.spatial_types import (
    HomogeneousTransformationMatrix,
    Point3,
    Quaternion,
)
from semantic_digital_twin.spatial_types.spatial_types import Pose
from semantic_digital_twin.world import World


@pytest.fixture(scope="module", params=["hsrb", "stretch", "tiago", "pr2"])
def setup_multi_robot_simple_apartment(
    request,
    hsr_world_setup,
    stretch_world,
    tiago_world,
    pr2_world_setup,
    simple_apartment_setup,
):
    apartment_copy = deepcopy(simple_apartment_setup)

    if request.param == "hsrb":
        hsr_copy = deepcopy(hsr_world_setup)
        apartment_copy.merge_world(hsr_copy)
        view = HSRB.from_world(apartment_copy)
        view.root.parent_connection.origin = (
            HomogeneousTransformationMatrix.from_xyz_rpy(1.5, 2, 0)
        )
        return apartment_copy, view
    elif request.param == "stretch":
        stretch_copy = deepcopy(stretch_world)
        apartment_copy.merge_world(
            stretch_copy,
        )
        view = Stretch.from_world(apartment_copy)
        view.root.parent_connection.origin = (
            HomogeneousTransformationMatrix.from_xyz_rpy(1.5, 2, 0)
        )
        return apartment_copy, view

    elif request.param == "tiago":
        tiago_copy = deepcopy(tiago_world)
        apartment_copy.merge_world(
            tiago_copy,
        )
        view = Tiago.from_world(apartment_copy)
        view.root.parent_connection.origin = (
            HomogeneousTransformationMatrix.from_xyz_rpy(1.5, 2, 0)
        )
        return apartment_copy, view

    elif request.param == "pr2":
        pr2_copy = deepcopy(pr2_world_setup)
        apartment_copy.merge_world(
            pr2_copy,
        )
        view = PR2.from_world(apartment_copy)
        view.root.parent_connection.origin = (
            HomogeneousTransformationMatrix.from_xyz_rpy(1.5, 2, 0)
        )
        return apartment_copy, view


@pytest.fixture
def immutable_multiple_robot_simple_apartment(
    setup_multi_robot_simple_apartment,
) -> Generator[Tuple[World, AbstractRobot, Context]]:
    world, view = setup_multi_robot_simple_apartment
    state = deepcopy(world.state._data)
    yield world, view, Context(world, view)
    world.state._data[:] = state
    world.notify_state_change()


@pytest.fixture
def mutable_multiple_robot_simple_apartment(setup_multi_robot_simple_apartment):
    world, view = setup_multi_robot_simple_apartment
    copy_world = deepcopy(world)
    copy_view = view.from_world(copy_world)
    return copy_world, copy_view, Context(copy_world, copy_view)


def test_reachability_costmap_location(immutable_multiple_robot_simple_apartment):
    world, robot, context = immutable_multiple_robot_simple_apartment

    plan = sequential(
        [ParkArmsAction(Arms.BOTH), MoveTorsoAction(TorsoState.HIGH)],
        context,
    )
    with simulated_robot:
        plan.perform()

    world.notify_state_change()

    location_desig = CostmapLocation(
        world.get_body_by_name("milk.stl").global_pose,
        context=context,
    )
    location = next(iter(location_desig))

    assert len(location.to_position().to_list()) == 4
    assert len(location.to_quaternion().to_list()) == 4


def test_reachability_pose_costmap_location(immutable_multiple_robot_simple_apartment):
    world, robot_view, context = immutable_multiple_robot_simple_apartment
    plan = sequential(
        [
            ParkArmsAction(Arms.BOTH),
            MoveTorsoAction(TorsoState.HIGH),
        ],
        context,
    )
    with simulated_robot:
        plan.perform()
    underspecified_costmap_location = underspecified(CostmapLocation)(
        target=Pose.from_xyz_quaternion(
            -2.7, 0, 1, 0, 0, 0, 1, reference_frame=world.root
        ),
        reachable=True,
        context=context,
        reachable_arm=...,
    )
    underspecified_costmap_location.resolve()
    specified_costmap_location = next(
        ProbabilisticBackend().evaluate(underspecified_costmap_location)
    )

    location = next(iter(specified_costmap_location))

    assert len(location.to_position().to_list()) == 4
    assert len(location.to_quaternion().to_list()) == 4


def test_visibility_costmap_location(immutable_multiple_robot_simple_apartment):
    world, robot_view, context = immutable_multiple_robot_simple_apartment
    plan = sequential(
        [ParkArmsAction(Arms.BOTH), MoveTorsoAction(TorsoState.HIGH)],
        context,
    )
    with simulated_robot:
        plan.perform()
    location_desig = CostmapLocation(
        world.get_body_by_name("milk.stl").global_pose,
        context=context,
        visible=True,
    )
    location = next(iter(location_desig))

    assert len(location.to_position().to_list()) == 4
    assert len(location.to_quaternion().to_list()) == 4


def test_visibility_pose_costmap_location(immutable_multiple_robot_simple_apartment):
    world, robot_view, context = immutable_multiple_robot_simple_apartment
    plan = sequential(
        [
            ParkArmsAction(Arms.BOTH),
            MoveTorsoAction(TorsoState.HIGH),
        ],
        context,
    )
    with simulated_robot:
        plan.perform()
    location_desig = CostmapLocation(
        Pose(Point3.from_iterable([-1, 0, 1.2]), reference_frame=world.root),
        visible=True,
        context=context,
    )

    location = next(iter(location_desig))
    assert len(location.to_position().to_list()) == 4
    assert len(location.to_quaternion().to_list()) == 4


def test_reachability_and_visibility_costmap_location(
    immutable_multiple_robot_simple_apartment,
):
    world, robot_view, context = immutable_multiple_robot_simple_apartment
    plan = sequential(
        [
            ParkArmsAction(Arms.BOTH),
            MoveTorsoAction(TorsoState.HIGH),
        ],
        context,
    )
    with simulated_robot:
        plan.perform()
    world.notify_state_change()
    location_desig = CostmapLocation(
        world.get_body_by_name("milk.stl").global_pose,
        reachable=True,
        visible=True,
        context=context,
        reachable_arm=Arms.BOTH,
    )
    location = next(iter(location_desig))
    assert len(location.to_position().to_list()) == 4
    assert len(location.to_quaternion().to_list()) == 4


def test_accessing_location(immutable_model_world):
    world, robot_view, context = immutable_model_world
    location_desig = AccessingLocation(
        handle=world.get_body_by_name("handle_cab10_m"),
        arm=Arms.RIGHT,
        context=context,
    )
    access_pose = next(iter(location_desig))

    assert len(access_pose.to_position().to_list()) == 4
    assert len(access_pose.to_quaternion().to_list()) == 4


def test_giskard_location_pose(immutable_model_world):
    world, pr2, context = immutable_model_world
    location_desig = GiskardLocation(
        Pose(Point3.from_iterable([1.9, 2, 1]), reference_frame=world.root),
        Arms.RIGHT,
        context=context,
    )

    location = next(iter(location_desig))

    assert len(location.to_position().to_list()) == 4
    assert len(location.to_quaternion().to_list()) == 4


def test_costmap_location_last_result(immutable_multiple_robot_simple_apartment):
    world, robot_view, context = immutable_multiple_robot_simple_apartment
    plan = sequential(
        [
            ParkArmsAction(Arms.BOTH),
            MoveTorsoAction(TorsoState.HIGH),
        ],
        context,
    )
    with simulated_robot:
        plan.perform()
    world.notify_state_change()
    location_desig = CostmapLocation(
        Pose(
            Point3.from_iterable([-2.7, 0, 1]),
            Quaternion.from_iterable([0, 0, 0, 1]),
            world.root,
        ),
        reachable=True,
        context=context,
        reachable_arm=Arms.BOTH,
    )

    location = next(iter(location_desig))
    last_result = location_desig._last_result

    assert len(last_result.to_position().to_list()) == 4
    assert len(last_result.to_quaternion().to_list()) == 4
    assert location == last_result
    assert location is last_result
