from copy import deepcopy

import pytest
import rclpy
from typing_extensions import Generator, List, Tuple

from coraplex.alternative_motion_mappings.hsrb_motion_mapping import HSRBMoveMotion
from coraplex.alternative_motion_mappings.stretch_motion_mapping import (
    StretchMoveToolCenterPoint,
    StretchMoveSim,
    StretchMoveReal,
    StretchClose,
)
from coraplex.alternative_motion_mappings.tiago_motion_mapping import TiagoMoveSim
from coraplex.datastructures.dataclasses import Context

from coraplex.datastructures.enums import Arms, ApproachDirection, VerticalAlignment
from coraplex.datastructures.grasp import GraspDescription
from coraplex.locations.base import DeferredLocation
from coraplex.locations.factories import (
    reachability_location,
    visibility_location,
    accessing_location,
    giskard_reachability_location,
)
from krrood.entity_query_language.factories import variable
from semantic_digital_twin.spatial_types.spatial_types import Pose
from coraplex.execution_environment import simulated_robot
from coraplex.plans.factories import sequential
from coraplex.robot_plans.actions.core.robot_body import ParkArmsAction, MoveTorsoAction
from coraplex.view_manager import ViewManager
from semantic_digital_twin.adapters.ros.visualization.viz_marker import (
    VizMarkerPublisher,
)
from semantic_digital_twin.datastructures.definitions import TorsoState
from semantic_digital_twin.robots.robot_parts import AbstractRobot

try:
    from semantic_digital_twin.robots.garmi import Garmi
except ImportError:
    Garmi = None
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
def immutable_multiple_robot_simple_apartment(
    setup_multi_robot_simple_apartment,
) -> Generator[Tuple[World, AbstractRobot, Context]]:
    world, view = setup_multi_robot_simple_apartment
    state = deepcopy(world.state._data)
    yield world, view, Context(
        world, view, alternative_motion_mappings=ALTERNATIVE_MOTION_MAPPINGS
    )
    world.state._data[:] = state
    world.notify_state_change()


@pytest.fixture
def mutable_multiple_robot_simple_apartment(setup_multi_robot_simple_apartment):
    world, view = setup_multi_robot_simple_apartment
    copy_world = deepcopy(world)
    copy_view = view.from_world(copy_world)
    return (
        copy_world,
        copy_view,
        Context(
            copy_world,
            copy_view,
            alternative_motion_mappings=ALTERNATIVE_MOTION_MAPPINGS,
        ),
    )


def test_deferred_location_factory_runs_at_execution_not_construction():
    """A :class:`DeferredLocation` must not build its :class:`Location` until the EQL
    variable domain is actually consumed.

    ``variable`` wraps the domain in :func:`filter`, whose builtin implementation calls
    :func:`iter` on its argument at construction time. An eagerly-iterating
    ``DeferredLocation`` would therefore run the factory while the plan is being parsed,
    reintroducing the stale-pose bug.
    """
    factory_calls = []

    def build_poses() -> List[Pose]:
        factory_calls.append(True)
        return [Pose.from_xyz_rpy(0.0, 0.0, 0.0)]

    domain_variable = variable(Pose, domain=DeferredLocation(build_poses))

    assert factory_calls == []

    next(iter(domain_variable._re_enterable_domain_generator_), None)

    assert factory_calls == [True]


def test_deferred_location_reflects_state_changed_after_construction():
    """The deferred factory observes the world state as it is when the location is
    consumed, not as it was when the underspecified action was constructed.
    """
    observed_positions = []
    moving_pose = {"value": Pose.from_xyz_rpy(0.0, 0.0, 0.0)}

    def build_poses() -> List[Pose]:
        observed_positions.append(moving_pose["value"].to_position().to_list())
        return [moving_pose["value"]]

    domain_variable = variable(Pose, domain=DeferredLocation(build_poses))

    moving_pose["value"] = Pose.from_xyz_rpy(3.1, 2.2, 0.95)

    next(iter(domain_variable._re_enterable_domain_generator_), None)

    assert observed_positions == [[3.1, 2.2, 0.95, 1.0]]


def test_new_reachability_location_pose(
    immutable_multiple_robot_simple_apartment, rclpy_node
):
    world, robot, context = immutable_multiple_robot_simple_apartment

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


def test_visibility_reachability_merge(
    immutable_multiple_robot_simple_apartment, rclpy_node
):
    world, robot, context = immutable_multiple_robot_simple_apartment

    plan = sequential(
        [ParkArmsAction(Arms.BOTH), MoveTorsoAction(TorsoState.HIGH)],
        context,
    )

    context.ros_node = rclpy_node
    context.debug = True

    with simulated_robot:
        plan.perform()

        world.notify_state_change()

        location_vis = visibility_location(world.get_body_by_name("milk.stl"), context)

        next(iter(location_vis))

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
        world.add_semantic_annotation_recursively(
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


def test_giskard_location_pose(immutable_multiple_robot_simple_apartment):
    world, robot, context = immutable_multiple_robot_simple_apartment
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

        location = giskard_reachability_location(
            world.get_body_by_name("milk.stl"),
            context,
            Arms.RIGHT,
            GraspDescription(
                ApproachDirection.FRONT,
                VerticalAlignment.NoAlignment,
                ViewManager.get_end_effector_view(Arms.RIGHT, robot),
            ),
        )

        pose = next(iter(location))

    assert len(pose.to_position().to_list()) == 4
    assert len(pose.to_quaternion().to_list()) == 4
