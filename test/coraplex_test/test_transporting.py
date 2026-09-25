"""
How a transport moves to an object, fetches it and puts it down.
"""

import numpy as np
import pytest
from typing_extensions import Callable, List, Type

from krrood.entity_query_language.factories import a, variable
from krrood.entity_query_language.query.match import Match
from coraplex.datastructures.dataclasses import Context
from coraplex.datastructures.enums import ReachFraction
from coraplex.execution_environment import simulated_robot
from coraplex.locations.locations import ReachabilityLocation
from coraplex.plans.factories import sequential
from coraplex.plans.plan_node import ActionNode
from coraplex.plans.underspecified import UnderspecifiedNode
from coraplex.robot_plans.actions.base import ActionDescription
from coraplex.robot_plans.actions.core.navigation import (
    FaceAtAction,
    LookAtAction,
    NavigateAction,
)
from coraplex.robot_plans.actions.composite.facing import FaceAndLookAtAction
from coraplex.robot_plans.actions.composite.transporting import (
    MoveAndOpenAction,
    MoveAndPickUpAction,
    MoveAndPlaceAction,
    PickAndPlaceAction,
    TransportAction,
)
from coraplex.robot_plans.actions.core.pick_up import PickUpAction
from coraplex.robot_plans.actions.core.placing import PlaceAction
from coraplex.robot_plans.actions.core.robot_body import MoveTorsoAction
from semantic_digital_twin.semantic_annotations.semantic_annotations import (
    Drawer,
    Handle,
    Milk,
)
from semantic_digital_twin.robots.robot_parts import AbstractRobot, Arm
from semantic_digital_twin.semantic_annotations.mixins import GraspPose, HasGraspPoses
from semantic_digital_twin.spatial_types.spatial_types import Point3, Pose
from semantic_digital_twin.world import World

# %% where the robot stands is tried together with what it does there


def _underspecified_steps(transport: TransportAction) -> List[Type[ActionDescription]]:
    """
    :return: The action types of the steps the transport leaves to be grounded, in
        order.
    """
    return [
        child.designator_type
        for child in transport._action_plan.children
        if isinstance(child, UnderspecifiedNode)
    ]


def _pick_up_the_milk(world: World, context: Context) -> MoveAndPickUpAction:
    """
    :return: A pick-up of the milk, standing wherever its trial finds one that works.
    """
    milk = world.get_semantic_annotations_by_type(Milk)[0]
    milk_pose = milk.root.global_pose
    return a(MoveAndPickUpAction)(
        navigate=a(NavigateAction)(
            target_location=variable(
                Pose,
                domain=ReachabilityLocation(
                    Pose(reference_frame=milk.root),
                    context.robot.right_arm,
                    context=context,
                ),
            )
        ),
        face_and_look_at=a(FaceAndLookAtAction)(
            face_at=a(FaceAtAction)(target=milk_pose),
            look_at=a(LookAtAction)(target=milk_pose),
        ),
        pick_up=a(PickUpAction)(
            grasp=milk.grasp_poses()[0], arm=context.robot.right_arm
        ),
    )


def _place_at(
    target: Pose, placed: HasGraspPoses, context: Context
) -> MoveAndPlaceAction:
    """
    :return: A place of `placed` at `target`, standing wherever its trial finds one
        that works.
    """
    return a(MoveAndPlaceAction)(
        navigate=a(NavigateAction)(
            target_location=variable(
                Pose,
                domain=ReachabilityLocation(
                    target,
                    context.robot.right_arm,
                    context=context,
                ),
            )
        ),
        face_and_look_at=a(FaceAndLookAtAction)(
            face_at=a(FaceAtAction)(target=target),
            look_at=a(LookAtAction)(target=target),
        ),
        place=a(PlaceAction)(object_designator=placed, target_location=target),
    )


def _standing_positions(step: Match) -> ReachabilityLocation:
    """
    :return: The location the standing pose of `step` is drawn from.
    """
    return step.kwargs["navigate"].kwargs["target_location"]._domain_.domain


def _transport_of_the_milk(world: World, context: Context) -> TransportAction:
    return TransportAction(
        pick_up=_pick_up_the_milk(world, context),
        place=_place_at(
            Pose(reference_frame=world.root),
            world.get_semantic_annotations_by_type(Milk)[0],
            context,
        ),
    )


def test_a_transport_grounds_the_steps_it_is_given(mutable_model_world):
    """
    The caller decides what is left open in each step, so the transport grounds the
    steps it was given rather than steps of its own.
    """
    world, robot, context = mutable_model_world
    transport = _transport_of_the_milk(world, context)
    sequential([transport], context)

    assert _underspecified_steps(transport) == [
        MoveAndPickUpAction,
        MoveAndPlaceAction,
    ]


def test_a_transport_tries_a_bounded_number_of_candidates(mutable_model_world):
    """
    Each standing pose is tried by running the step from it, so a step that can succeed
    from nowhere has to give up after a fixed number of them.
    """
    world, robot, context = mutable_model_world
    transport = _transport_of_the_milk(world, context)
    sequential([transport], context)

    limits = [
        child.underspecified_action.expression._limit_
        for child in transport._action_plan.children
        if isinstance(child, UnderspecifiedNode)
    ]

    assert limits == [transport.candidates_to_try] * len(limits)
    assert limits


def test_a_transport_leaves_the_torso_where_it_is(mutable_model_world):
    world, robot, context = mutable_model_world
    transport = _transport_of_the_milk(world, context)
    sequential([transport], context)

    assert not [
        child
        for child in transport._action_plan.children
        if isinstance(child, ActionNode)
        and isinstance(child.designator, MoveTorsoAction)
    ]


def test_a_transport_from_a_grasp_stands_around_the_object_then_the_target(
    mutable_model_world,
):
    """
    Built from a grasp alone, a transport leaves only where the robot stands open: close
    to the object for the pick-up, and close to the target for the place.
    """
    world, robot, context = mutable_model_world
    milk = world.get_semantic_annotations_by_type(Milk)[0]
    target = Pose.from_xyz_rpy(4.0, 1.5, 0.9, reference_frame=world.root)

    transport = TransportAction.from_grasp(
        milk.grasp_poses()[0], target, context.robot.right_arm, context
    )

    pick_up_location = _standing_positions(transport.pick_up)
    place_location = _standing_positions(transport.place)
    assert pick_up_location.target_pose.reference_frame is milk.root
    assert place_location.target_pose is target


# %% picking up and placing without moving


def _pick_and_place_of_the_milk(world: World, arm: Arm) -> PickAndPlaceAction:
    """
    :param arm: The arm that picks the milk up and puts it down.
    :return: A pick-and-place of the milk that tries every grasp it offers.
    """
    milk = world.get_semantic_annotations_by_type(Milk)[0]
    return PickAndPlaceAction(
        pick_up=a(PickUpAction)(
            grasp=variable(GraspPose, domain=milk.grasp_poses()), arm=arm
        ),
        place=a(PlaceAction)(
            object_designator=milk,
            target_location=Pose(reference_frame=world.root),
        ),
    )


def test_a_pick_and_place_grounds_the_steps_it_is_given(mutable_model_world):
    world, robot, context = mutable_model_world
    pick_and_place = _pick_and_place_of_the_milk(world, robot.right_arm)
    sequential([pick_and_place], context)

    assert [
        child.designator_type
        for child in pick_and_place._action_plan.children
        if isinstance(child, UnderspecifiedNode)
    ] == [PickUpAction, PlaceAction]


def test_a_pick_and_place_tries_a_bounded_number_of_candidates(mutable_model_world):
    world, robot, context = mutable_model_world
    pick_and_place = _pick_and_place_of_the_milk(world, robot.right_arm)
    sequential([pick_and_place], context)

    limits = [
        child.underspecified_action.expression._limit_
        for child in pick_and_place._action_plan.children
        if isinstance(child, UnderspecifiedNode)
    ]

    assert limits == [pick_and_place.candidates_to_try] * len(limits)
    assert limits


# %% fetching an object out of a drawer

DRAWER = "cabinet10_drawer_top"
"""
The apartment drawer the transport opens on its way to the object inside it.
"""

DRAWER_HANDLE = "handle_cab10_t"
"""
The handle of :data:`DRAWER`.
"""


def _pick_up_near_a_drawer(world: World, context: Context) -> MoveAndPickUpAction:
    """
    :return: A pick-up of the milk in a world where :data:`DRAWER` is annotated.
    """
    with world.modify_world():
        world.add_semantic_annotation_recursively(
            Drawer(
                root=world.get_body_by_name(DRAWER),
                handle=Handle(root=world.get_body_by_name(DRAWER_HANDLE)),
            )
        )
    move_and_pick_up = MoveAndPickUpAction.from_standing_position(
        standing_position=Pose(reference_frame=world.root),
        grasp=world.get_semantic_annotations_by_type(Milk)[0].grasp_poses()[0],
        arm=context.robot.right_arm,
    )
    sequential([move_and_pick_up], context)
    return move_and_pick_up


def test_opening_a_container_on_the_way_is_tried_with_the_move_to_it(
    mutable_model_world,
):
    """
    An object inside a drawer is fetched by opening the drawer first, from a standing
    pose of its own.
    """
    world, robot, context = mutable_model_world
    move_and_pick_up = _pick_up_near_a_drawer(world, context)

    assert [
        action.type
        for action in move_and_pick_up._make_open_container_actions(
            world.get_body_by_name(DRAWER)
        )
    ] == [MoveAndOpenAction]


def test_opening_a_container_on_the_way_stands_where_it_is_opened_from(
    mutable_model_world,
):
    """
    The robot stands back for opening a container the way it does for any container,
    rather than as close as it would to grasp something that stays put.
    """
    world, robot, context = mutable_model_world
    move_and_pick_up = _pick_up_near_a_drawer(world, context)

    [open_on_the_way] = move_and_pick_up._make_open_container_actions(
        world.get_body_by_name(DRAWER)
    )
    location = _standing_positions(open_on_the_way)

    assert location.reach_fraction == ReachFraction.ACCESSING


# %% moving to an object and picking it up


def test_move_and_pick_up_takes_the_grasp_it_was_given(mutable_model_world):
    """
    The caller chooses the grasp, so the pick-up at the end of the walk takes that one
    rather than whichever grasp the object happens to list first.
    """
    world, robot, context = mutable_model_world
    grasp = world.get_semantic_annotations_by_type(Milk)[0].grasp_poses()[-1]
    move_and_pick_up = MoveAndPickUpAction.from_standing_position(
        standing_position=Pose(reference_frame=world.root),
        grasp=grasp,
        arm=context.robot.left_arm,
    )
    sequential([move_and_pick_up], context)

    pick_ups = [
        child
        for child in move_and_pick_up._action_plan.children
        if isinstance(getattr(child, "designator", None), PickUpAction)
    ]

    assert [pick_up.designator.grasp for pick_up in pick_ups] == [grasp]


def test_move_and_pick_up_approaches_with_the_clearances_it_was_given(
    mutable_model_world,
):
    world, robot, context = mutable_model_world
    approach_clearance, retreat_distance = 0.07, 0.13
    move_and_pick_up = MoveAndPickUpAction.from_standing_position(
        standing_position=Pose(reference_frame=world.root),
        grasp=world.get_semantic_annotations_by_type(Milk)[0].grasp_poses()[0],
        arm=context.robot.left_arm,
        approach_clearance=approach_clearance,
        retreat_distance=retreat_distance,
    )
    sequential([move_and_pick_up], context)

    [pick_up] = [
        child.designator
        for child in move_and_pick_up._action_plan.children
        if isinstance(getattr(child, "designator", None), PickUpAction)
    ]

    assert (pick_up.approach_clearance, pick_up.retreat_distance) == (
        approach_clearance,
        retreat_distance,
    )


# %% placing and opening from a standing position


def _hold_the_milk(world: World, arm: Arm) -> Milk:
    """
    Put the milk in the gripper of `arm`, as a pick-up does.

    :return: The milk.
    """
    milk = world.get_semantic_annotations_by_type(Milk)[0]
    tool_frame = arm.end_effector.tool_frame
    with world.modify_world():
        world.move_branch_with_fixed_connection(milk.root, tool_frame)
    return milk


def test_a_move_and_place_from_a_standing_position_places_the_given_object(
    mutable_model_world,
):
    world, robot, context = mutable_model_world
    milk = world.get_semantic_annotations_by_type(Milk)[0]
    standing_position = Pose(reference_frame=world.root)
    target = Pose.from_xyz_rpy(4.0, 1.5, 0.9, reference_frame=world.root)

    move_and_place = MoveAndPlaceAction.from_standing_position(
        standing_position, target, milk
    )

    assert move_and_place.navigate.target_location is standing_position
    assert move_and_place.face_and_look_at.face_at.target is target
    assert move_and_place.face_and_look_at.look_at.target is target
    assert move_and_place.place.object_designator is milk
    assert move_and_place.place.target_location is target


def test_a_move_and_open_from_a_standing_position_opens_the_given_handle(
    mutable_model_world,
):
    world, robot, context = mutable_model_world
    handle = Handle(root=world.get_body_by_name(DRAWER_HANDLE))
    standing_position = Pose(reference_frame=world.root)

    move_and_open = MoveAndOpenAction.from_standing_position(
        standing_position, handle, context.robot.left_arm
    )

    assert move_and_open.navigate.target_location is standing_position
    assert move_and_open.face_and_look_at.face_at.target.reference_frame is (
        handle.root.global_pose.reference_frame
    )
    assert move_and_open.open_container.handle is handle
    assert move_and_open.open_container.arm is context.robot.left_arm


# %% a move-and-act step acts from where it moved to

STANDING_POSITION = (3.5, 1.5)
"""
Where the move-and-act steps are sent, away from where the robot starts.
"""


def _navigation_targets(action: ActionDescription) -> List[Pose]:
    """
    :return: Every standing pose `action` navigates to, including the ones of the
        actions it is built from.
    """
    action.plan_node.notify()
    return [
        node.designator.target_location
        for node in action.plan_node.plan.get_nodes_by_designator_type(NavigateAction)
    ]


def _standing_pose(world: World) -> Pose:
    return Pose.from_xyz_rpy(*STANDING_POSITION, 0.0, reference_frame=world.root)


def _placing_the_held_milk(world: World, context: Context) -> MoveAndPlaceAction:
    milk = _hold_the_milk(world, context.robot.left_arm)
    return MoveAndPlaceAction.from_standing_position(
        standing_position=_standing_pose(world),
        target_location=Pose.from_xyz_rpy(4.0, 1.5, 0.9, reference_frame=world.root),
        object_designator=milk,
    )


MOVE_AND_ACT_STEPS = {
    "pick up": lambda world, context: MoveAndPickUpAction.from_standing_position(
        standing_position=_standing_pose(world),
        grasp=world.get_semantic_annotations_by_type(Milk)[0].grasp_poses()[0],
        arm=context.robot.left_arm,
    ),
    "place": _placing_the_held_milk,
}


@pytest.mark.parametrize("build", MOVE_AND_ACT_STEPS.values(), ids=MOVE_AND_ACT_STEPS)
def test_a_move_and_act_step_only_ever_stands_where_it_was_sent(
    mutable_model_world, build: Callable[[World, Context], ActionDescription]
):
    """
    Its plan is built before the robot moves, so turning to face the target has to be
    worked out from where the robot is sent rather than from where it stands at first,
    or the robot is sent back there before it acts.
    """
    world, robot, context = mutable_model_world
    step = build(world, context)
    sequential([step], context)

    for target in _navigation_targets(step):
        np.testing.assert_allclose(
            target.to_position().to_np()[:2].ravel(), STANDING_POSITION
        )


def _assert_base_faces(robot: AbstractRobot, target: Point3):
    """
    Assert that the robot's base front points horizontally at `target`.
    """
    world = robot._world
    base_P_target = world.transform(target, robot.mobile_base.root).to_np()[:2]
    np.testing.assert_allclose(
        base_P_target / np.linalg.norm(base_P_target),
        robot.mobile_base.forward_axis.to_np()[:2],
        atol=0.02,
    )


def test_facing_after_navigating_turns_where_the_robot_was_sent(mutable_model_world):
    world, robot, context = mutable_model_world
    target = Pose.from_xyz_rpy(4.0, 2.5, 0.9, reference_frame=world.root)
    plan = sequential(
        [NavigateAction(_standing_pose(world)), FaceAtAction(target)], context
    )

    with simulated_robot:
        plan.perform()

    np.testing.assert_allclose(
        robot.root.global_pose.to_position().to_np()[:2],
        STANDING_POSITION,
        atol=0.03,
    )
    _assert_base_faces(robot, target.to_position())


def test_facing_a_target_given_relative_to_a_body_turns_towards_that_body(
    mutable_model_world,
):
    world, robot, context = mutable_model_world
    milk = world.get_semantic_annotations_by_type(Milk)[0].root
    plan = sequential(
        [
            NavigateAction(_standing_pose(world)),
            FaceAtAction(Pose(reference_frame=milk)),
        ],
        context,
    )

    with simulated_robot:
        plan.perform()

    _assert_base_faces(robot, milk.global_pose.to_position())


def test_facing_and_looking_at_a_target_turns_the_base_and_the_camera_towards_it(
    mutable_model_world,
):
    world, robot, context = mutable_model_world
    target = Pose.from_xyz_rpy(4.0, 2.5, 0.9, reference_frame=world.root)
    plan = sequential(
        [
            NavigateAction(_standing_pose(world)),
            FaceAndLookAtAction(FaceAtAction(target), LookAtAction(target)),
        ],
        context,
    )

    with simulated_robot:
        plan.perform()

    _assert_base_faces(robot, target.to_position())
    camera = robot.get_default_camera()
    camera_P_target = world.transform(target.to_position(), camera.root).to_np()[:3]
    camera_V_forward = camera.forward_facing_axis.to_np()[:3]
    np.testing.assert_allclose(
        camera_P_target / np.linalg.norm(camera_P_target),
        camera_V_forward / np.linalg.norm(camera_V_forward),
        atol=0.02,
    )
