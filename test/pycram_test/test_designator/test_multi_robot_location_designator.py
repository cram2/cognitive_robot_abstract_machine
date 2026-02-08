from copy import deepcopy

import pytest
import rclpy
from typing_extensions import Generator, Tuple

from pycram.datastructures.dataclasses import Context
from pycram.datastructures.enums import Arms
from pycram.datastructures.pose import PoseStamped
from pycram.designators.location_designator import (
    CostmapLocation,
    ProbabilisticCostmapLocation,
    SemanticCostmapLocation,
    ProbabilisticSemanticLocation,
    GiskardLocation,
    AccessingLocation,
)
from pycram.language import SequentialPlan
from pycram.process_module import simulated_robot
from pycram.robot_plans import (
    NavigateActionDescription,
    MoveTorsoActionDescription,
    ParkArmsActionDescription,
)
from semantic_digital_twin.adapters.ros.tf_publisher import TFPublisher
from semantic_digital_twin.adapters.ros.visualization.viz_marker import (
    VizMarkerPublisher,
)
from semantic_digital_twin.datastructures.definitions import TorsoState
from semantic_digital_twin.robots.abstract_robot import AbstractRobot
from semantic_digital_twin.robots.hsrb import HSRB
from semantic_digital_twin.robots.pr2 import PR2
from semantic_digital_twin.robots.stretch import Stretch
from semantic_digital_twin.robots.tiago import Tiago
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.world import World


@pytest.fixture(scope="session", params=["hsrb", "stretch", "tiago", "pr2"])
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
    state = deepcopy(world.state.data)
    yield world, view, Context(world, view)
    world.state.data = state
    world.notify_state_change()


@pytest.fixture
def mutable_multiple_robot_simple_apartment(setup_multi_robot_simple_apartment):
    world, view = setup_multi_robot_simple_apartment
    copy_world = deepcopy(world)
    copy_view = view.from_world(copy_world)
    return copy_world, copy_view, Context(copy_world, copy_view)


def test_reachability_costmap_location(immutable_multiple_robot_simple_apartment):
    world, robot, context = immutable_multiple_robot_simple_apartment

    plan = SequentialPlan(
        context,
        ParkArmsActionDescription(Arms.BOTH),
        MoveTorsoActionDescription(TorsoState.HIGH),
    )
    with simulated_robot:
        plan.perform()

    world.notify_state_change()

    location_desig = CostmapLocation(
        world.get_body_by_name("milk.stl"), reachable_for=robot
    )
    plan = SequentialPlan(context, NavigateActionDescription(location_desig))
    location = location_desig.resolve()
    assert len(location.position.to_list()) == 3
    assert len(location.orientation.to_list()) == 4
    # assertTrue(Arms.LEFT == location.reachable_arm or Arms.RIGHT == location.reachable_arm)


def test_reachability_pose_costmap_location(immutable_multiple_robot_simple_apartment):
    world, robot_view, context = immutable_multiple_robot_simple_apartment
    plan = SequentialPlan(
        context,
        ParkArmsActionDescription(Arms.BOTH),
        MoveTorsoActionDescription(TorsoState.HIGH),
    )
    with simulated_robot:
        plan.perform()
    location_desig = CostmapLocation(
        PoseStamped.from_list([-2.2, 0, 1], [0, 0, 0, 1], world.root),
        reachable_for=robot_view,
    )
    plan = SequentialPlan(context, NavigateActionDescription(location_desig))
    location = location_desig.resolve()
    assert len(location.position.to_list()) == 3
    assert len(location.orientation.to_list()) == 4


def test_visibility_costmap_location(immutable_multiple_robot_simple_apartment):
    world, robot_view, context = immutable_multiple_robot_simple_apartment
    plan = SequentialPlan(
        context,
        ParkArmsActionDescription(Arms.BOTH),
        MoveTorsoActionDescription(TorsoState.HIGH),
    )
    with simulated_robot:
        plan.perform()
    location_desig = CostmapLocation(
        world.get_body_by_name("milk.stl"), visible_for=robot_view
    )
    plan = SequentialPlan(context, NavigateActionDescription(location_desig))
    location = location_desig.resolve()
    assert len(location.position.to_list()) == 3
    assert len(location.orientation.to_list()) == 4


def test_visibility_pose_costmap_location(immutable_multiple_robot_simple_apartment):
    world, robot_view, context = immutable_multiple_robot_simple_apartment
    plan = SequentialPlan(
        context,
        ParkArmsActionDescription(Arms.BOTH),
        MoveTorsoActionDescription(TorsoState.HIGH),
    )
    with simulated_robot:
        plan.perform()
    location_desig = CostmapLocation(
        PoseStamped.from_list([-1, 0, 1.2], frame=world.root),
        visible_for=robot_view,
    )
    plan = SequentialPlan(context, NavigateActionDescription(location_desig))
    location = location_desig.resolve()
    assert len(location.position.to_list()) == 3
    assert len(location.orientation.to_list()) == 4


def test_reachability_and_visibility_costmap_location(
    immutable_multiple_robot_simple_apartment,
):
    world, robot_view, context = immutable_multiple_robot_simple_apartment
    plan = SequentialPlan(
        context,
        ParkArmsActionDescription(Arms.BOTH),
        MoveTorsoActionDescription(TorsoState.HIGH),
    )
    with simulated_robot:
        plan.perform()
    world.notify_state_change()
    location_desig = CostmapLocation(
        world.get_body_by_name("milk.stl"),
        reachable_for=robot_view,
        visible_for=robot_view,
    )
    plan = SequentialPlan(context, NavigateActionDescription(location_desig))
    location = location_desig.resolve()
    assert len(location.position.to_list()) == 3
    assert len(location.orientation.to_list()) == 4


def test_reachability_probabilistic_costmap_location(
    immutable_multiple_robot_simple_apartment,
):
    world, robot_view, context = immutable_multiple_robot_simple_apartment
    plan = SequentialPlan(
        context,
        ParkArmsActionDescription(Arms.BOTH),
        MoveTorsoActionDescription(TorsoState.HIGH),
    )
    with simulated_robot:
        plan.perform()
    world.notify_state_change()
    location_desig = ProbabilisticCostmapLocation(
        world.get_body_by_name("milk.stl"), reachable_for=robot_view
    )
    plan = SequentialPlan(context, NavigateActionDescription(location_desig))
    location = location_desig.resolve()
    assert len(location.position.to_list()) == 3
    assert len(location.orientation.to_list()) == 4
    # assertTrue(Arms.LEFT == location.reachable_arm or Arms.RIGHT == location.reachable_arm)


def test_reachability_pose_probabilistic_costmap_location(
    immutable_multiple_robot_simple_apartment,
):
    world, robot_view, context = immutable_multiple_robot_simple_apartment

    plan = SequentialPlan(
        context,
        ParkArmsActionDescription(Arms.BOTH),
        MoveTorsoActionDescription(TorsoState.HIGH),
    )
    with simulated_robot:
        plan.perform()
    location_desig = ProbabilisticCostmapLocation(
        PoseStamped.from_list([0.4, 0.6, 0.9], [0, 0, 0, 1], frame=world.root),
        reachable_for=robot_view,
    )
    plan = SequentialPlan(context, NavigateActionDescription(location_desig))
    location = location_desig.resolve()
    assert len(location.position.to_list()) == 3
    assert len(location.orientation.to_list()) == 4
    # assertTrue(Arms.LEFT == location.reachable_arm or Arms.RIGHT == location.reachable_arm)


def test_visibility_probabilistic_costmap_location(
    immutable_multiple_robot_simple_apartment,
):
    world, robot_view, context = immutable_multiple_robot_simple_apartment
    plan = SequentialPlan(
        context,
        ParkArmsActionDescription(Arms.BOTH),
        MoveTorsoActionDescription(TorsoState.HIGH),
    )
    with simulated_robot:
        plan.perform()
    location_desig = ProbabilisticCostmapLocation(
        world.get_body_by_name("milk.stl"), visible_for=robot_view
    )
    plan = SequentialPlan(context, NavigateActionDescription(location_desig))
    location = location_desig.resolve()
    assert len(location.position.to_list()) == 3
    assert len(location.orientation.to_list()) == 4


def test_visibility_pose_probabilistic_costmap_location(
    immutable_multiple_robot_simple_apartment,
):
    world, robot_view, context = immutable_multiple_robot_simple_apartment
    plan = SequentialPlan(
        context,
        ParkArmsActionDescription(Arms.BOTH),
        MoveTorsoActionDescription(TorsoState.HIGH),
    )
    with simulated_robot:
        plan.perform()
    location_desig = ProbabilisticCostmapLocation(
        PoseStamped.from_list([-1, 0, 1.2], frame=world.root),
        visible_for=robot_view,
    )
    plan = SequentialPlan(context, NavigateActionDescription(location_desig))
    location = location_desig.resolve()
    assert len(location.position.to_list()) == 3
    assert len(location.orientation.to_list()) == 4


def test_reachability_and_visibility_probabilistic_costmap_location(
    immutable_multiple_robot_simple_apartment,
):
    world, robot_view, context = immutable_multiple_robot_simple_apartment
    plan = SequentialPlan(
        context,
        ParkArmsActionDescription(Arms.BOTH),
        MoveTorsoActionDescription(TorsoState.HIGH),
    )
    with simulated_robot:
        plan.perform()
    world.notify_state_change()
    location_desig = ProbabilisticCostmapLocation(
        world.get_body_by_name("milk.stl"),
        reachable_for=robot_view,
        visible_for=robot_view,
    )
    plan = SequentialPlan(context, NavigateActionDescription(location_desig))
    location = location_desig.resolve()
    assert len(location.position.to_list()) == 3
    assert len(location.orientation.to_list()) == 4


def test_semantic_location(immutable_model_world):
    world, robot_view, context = immutable_model_world
    location_desig = SemanticCostmapLocation(
        world.get_body_by_name("island_countertop")
    )
    location = location_desig.resolve()
    assert len(location.position.to_list()) == 3
    assert len(location.orientation.to_list()) == 4

    location_desig = SemanticCostmapLocation(
        world.get_body_by_name("island_countertop"),
        for_object=world.get_body_by_name("milk.stl"),
    )
    location = location_desig.resolve()
    assert len(location.position.to_list()) == 3
    assert len(location.orientation.to_list()) == 4


def test_probabilistic_semantic_location(immutable_multiple_robot_simple_apartment):
    world, robot_view, context = immutable_multiple_robot_simple_apartment
    location_desig = ProbabilisticSemanticLocation(
        [world.get_body_by_name("box_2")], link_is_center_link=True
    )
    plan = SequentialPlan(context, NavigateActionDescription(location_desig))
    location = location_desig.resolve()
    assert len(location.position.to_list()) == 3
    assert len(location.orientation.to_list()) == 4

    location_desig = ProbabilisticSemanticLocation(
        [world.get_body_by_name("box")],
        for_object=world.get_body_by_name("milk.stl"),
        link_is_center_link=True,
    )
    plan = SequentialPlan(context, NavigateActionDescription(location_desig))
    location = location_desig.resolve()
    assert len(location.position.to_list()) == 3
    assert len(location.orientation.to_list()) == 4


def test_accessing_location(immutable_model_world):
    world, robot_view, context = immutable_model_world
    location_desig = AccessingLocation(
        world.get_body_by_name("handle_cab10_m"),
        robot_desig=robot_view,
        arm=Arms.RIGHT,
    )
    plan = SequentialPlan(context, NavigateActionDescription(location_desig))
    access_pose = location_desig.resolve()

    assert len(access_pose.position.to_list()) == 3
    assert len(access_pose.orientation.to_list()) == 4


def test_giskard_location_pose(immutable_model_world):
    world, robot_view, context = immutable_model_world
    location_desig = GiskardLocation(
        PoseStamped.from_list([2.1, 2, 1], frame=world.root), Arms.RIGHT
    )
    plan = SequentialPlan(context, NavigateActionDescription(location_desig))

    location = location_desig.resolve()
    assert len(location.position.to_list()) == 3
    assert len(location.orientation.to_list()) == 4


def test_costmap_location_last_result(immutable_multiple_robot_simple_apartment):
    world, robot_view, context = immutable_multiple_robot_simple_apartment
    plan = SequentialPlan(
        context,
        ParkArmsActionDescription(Arms.BOTH),
        MoveTorsoActionDescription(TorsoState.HIGH),
    )
    with simulated_robot:
        plan.perform()
    world.notify_state_change()
    location_desig = CostmapLocation(
        PoseStamped.from_list([-2.2, 0, 1], [0, 0, 0, 1], world.root),
        reachable_for=robot_view,
    )
    plan = SequentialPlan(context, NavigateActionDescription(location_desig))
    location = location_desig.resolve()
    last_result = next(location_desig.last_result)

    assert len(last_result.position.to_list()) == 3
    assert len(last_result.orientation.to_list()) == 4
    assert location == last_result
    assert location is last_result
