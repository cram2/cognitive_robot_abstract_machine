from copy import deepcopy

import pytest
import rclpy
from typing_extensions import Generator, Tuple

import pycram.alternative_motion_mappings.hsrb_motion_mapping  # type: ignore
import pycram.alternative_motion_mappings.stretch_motion_mapping  # type: ignore
import pycram.alternative_motion_mappings.tiago_motion_mapping  # type: ignore
from pycram.datastructures.dataclasses import Context
from pycram.datastructures.enums import Arms
from pycram.locations.factories import (
    reachability_location,
    visibility_location,
    accessing_location,
)
from pycram.motion_executor import simulated_robot
from pycram.plans.factories import sequential
from pycram.robot_plans.actions.core.robot_body import ParkArmsAction, MoveTorsoAction
from semantic_digital_twin.adapters.ros.visualization.viz_marker import (
    VizMarkerPublisher,
)
from semantic_digital_twin.datastructures.definitions import TorsoState
from semantic_digital_twin.robots.abstract_robot import AbstractRobot
from semantic_digital_twin.robots.hsrb import HSRB
from semantic_digital_twin.robots.pr2 import PR2
from semantic_digital_twin.robots.stretch import Stretch
from semantic_digital_twin.robots.tiago import Tiago
from semantic_digital_twin.semantic_annotations.semantic_annotations import (
    Drawer,
    Handle,
)
from semantic_digital_twin.spatial_types import (
    HomogeneousTransformationMatrix,
)
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


def test_new_reachability_location_pose(
    immutable_multiple_robot_simple_apartment, rclpy_node
):
    world, robot, context = immutable_multiple_robot_simple_apartment

    # VizMarkerPublisher(_world=world, node=rclpy_node).with_tf_publisher()

    plan = sequential(
        [ParkArmsAction(Arms.BOTH), MoveTorsoAction(TorsoState.HIGH)],
        context,
    )
    with simulated_robot:
        plan.perform()

        world.notify_state_change()

        location = reachability_location(
            world.get_body_by_name("milk.stl").global_pose, context, Arms.RIGHT
        )

        pose = next(iter(location))
    assert len(pose.to_position().to_list()) == 4
    assert len(pose.to_quaternion().to_list()) == 4


def test_new_reachability_location_body(
    immutable_multiple_robot_simple_apartment, rclpy_node
):
    world, robot, context = immutable_multiple_robot_simple_apartment

    # VizMarkerPublisher(_world=world, node=rclpy_node).with_tf_publisher()

    plan = sequential(
        [ParkArmsAction(Arms.BOTH), MoveTorsoAction(TorsoState.HIGH)],
        context,
    )
    with simulated_robot:
        plan.perform()

        world.notify_state_change()

        location = reachability_location(
            world.get_body_by_name("milk.stl"), context, Arms.RIGHT
        )

        pose = next(iter(location))
    assert len(pose.to_position().to_list()) == 4
    assert len(pose.to_quaternion().to_list()) == 4


def test_merge_reachability_location(immutable_multiple_robot_simple_apartment):
    world, robot, context = immutable_multiple_robot_simple_apartment

    # VizMarkerPublisher(_world=world, node=rclpy_node).with_tf_publisher()

    plan = sequential(
        [ParkArmsAction(Arms.BOTH), MoveTorsoAction(TorsoState.HIGH)],
        context,
    )
    with simulated_robot:
        plan.perform()

        world.notify_state_change()

        location_body = reachability_location(
            world.get_body_by_name("milk.stl"), context, Arms.RIGHT
        )

        location_pose = reachability_location(
            world.get_body_by_name("milk.stl").global_pose, context, Arms.RIGHT
        )

        merged_location = location_body & location_pose
        pose = next(iter(merged_location))

    assert len(pose.to_position().to_list()) == 4
    assert len(pose.to_quaternion().to_list()) == 4


def test_visibility_location_pose(immutable_multiple_robot_simple_apartment):
    world, robot, context = immutable_multiple_robot_simple_apartment

    plan = sequential(
        [ParkArmsAction(Arms.BOTH), MoveTorsoAction(TorsoState.HIGH)],
        context,
    )
    with simulated_robot:
        plan.perform()

        world.notify_state_change()

        location = visibility_location(
            world.get_body_by_name("milk.stl").global_pose, context
        )

        pose = next(iter(location))

    assert len(pose.to_position().to_list()) == 4
    assert len(pose.to_quaternion().to_list()) == 4


def test_visibility_location_body(immutable_multiple_robot_simple_apartment):
    world, robot, context = immutable_multiple_robot_simple_apartment

    plan = sequential(
        [ParkArmsAction(Arms.BOTH), MoveTorsoAction(TorsoState.HIGH)],
        context,
    )
    with simulated_robot:
        plan.perform()

        world.notify_state_change()

        location = visibility_location(world.get_body_by_name("milk.stl"), context)

        pose = next(iter(location))

    assert len(pose.to_position().to_list()) == 4
    assert len(pose.to_quaternion().to_list()) == 4


def test_visibility_reachability_merge(immutable_multiple_robot_simple_apartment):
    world, robot, context = immutable_multiple_robot_simple_apartment

    plan = sequential(
        [ParkArmsAction(Arms.BOTH), MoveTorsoAction(TorsoState.HIGH)],
        context,
    )
    with simulated_robot:
        plan.perform()

        world.notify_state_change()

        location_vis = visibility_location(world.get_body_by_name("milk.stl"), context)

        location_reach = reachability_location(
            world.get_body_by_name("milk.stl"), context, Arms.RIGHT
        )

        location = location_vis & location_reach

        pose = next(iter(location))

    assert len(pose.to_position().to_list()) == 4
    assert len(pose.to_quaternion().to_list()) == 4


def test_accessing_location_pose(immutable_model_world):
    world, robot, context = immutable_model_world
    plan = sequential(
        [
            ParkArmsAction(Arms.BOTH),
            MoveTorsoAction(TorsoState.HIGH),
        ],
        context,
    )
    with simulated_robot:
        plan.perform()

    with world.modify_world():
        world.add_semantic_annotation(
            drawer := Drawer(
                root=world.get_body_by_name("cabinet10_drawer_middle"),
                handle=Handle(root=world.get_body_by_name("handle_cab10_m")),
            )
        )
    location_desig = accessing_location(drawer, context=context, arm=Arms.RIGHT)
    with simulated_robot:
        pose = next(iter(location_desig))

    assert len(pose.to_position().to_list()) == 4
    assert len(pose.to_quaternion().to_list()) == 4
