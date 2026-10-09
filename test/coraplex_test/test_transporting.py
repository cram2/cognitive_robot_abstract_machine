"""
How a transport moves to an object, fetches it and puts it down.
"""

import itertools

import numpy as np
import pytest
from typing_extensions import Callable, List, Type

from krrood.entity_query_language.factories import a, variable
from krrood.entity_query_language.query.match import Match
from coraplex.datastructures.dataclasses import Context
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
from coraplex.querying.predicates import IsAmongTheClosestGraspsTo
from coraplex.robot_plans.actions.core.placing import PlaceAction
from coraplex.robot_plans.actions.core.robot_body import MoveTorsoAction
from semantic_digital_twin.semantic_annotations.semantic_annotations import (
    Handle,
    Milk,
)
from semantic_digital_twin.robots.robot_parts import AbstractRobot, Arm
from semantic_digital_twin.grasping.grasp_candidates import (
    GraspCandidate,
    HasGraspCandidates,
)
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.spatial_types.spatial_types import (
    Point3,
    Pose,
    RotationMatrix,
)
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.world_entity import Body

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
            grasp=milk.grasp_candidates()[0], arm=context.robot.right_arm
        ),
    )


def _place_at(
    target: Pose, placed: HasGraspCandidates, context: Context
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
    :return: The location the standing pose of `step` is sampled from.
    """
    return step._kwargs_["navigate"]._kwargs_["target_location"]._domain_.domain


def _transport_of_the_milk(world: World, context: Context) -> TransportAction:
    return TransportAction(
        pick_up=_pick_up_the_milk(world, context),
        place=_place_at(
            Pose(reference_frame=world.root),
            world.get_semantic_annotations_by_type(Milk)[0],
            context,
        ),
    )


def test_a_transport_grounds_the_steps_it_is_given(pr2_apartment_context):
    """
    The caller decides what is left open in each step, so the transport grounds the
    steps it was given rather than steps of its own.
    """
    world, robot, context = pr2_apartment_context
    transport = _transport_of_the_milk(world, context)
    sequential([transport], context)

    assert _underspecified_steps(transport) == [
        MoveAndPickUpAction,
        MoveAndPlaceAction,
    ]


def test_a_transport_leaves_the_torso_where_it_is(pr2_apartment_context):
    world, robot, context = pr2_apartment_context
    transport = _transport_of_the_milk(world, context)
    sequential([transport], context)

    assert not [
        child
        for child in transport._action_plan.children
        if isinstance(child, ActionNode)
        and isinstance(child.designator, MoveTorsoAction)
    ]


def test_a_transport_of_a_graspable_stands_around_the_object_then_the_target(
    pr2_apartment_context,
):
    """
    Built from the object alone, a transport stands close to the object for the pick-up,
    and close to the target for the place.
    """
    world, robot, context = pr2_apartment_context
    milk = world.get_semantic_annotations_by_type(Milk)[0]
    target = Pose.from_xyz_rpy(4.0, 1.5, 0.9, reference_frame=world.root)

    transport = TransportAction.from_graspable_by_closest_grasps(
        milk, target, context.robot.right_arm, context
    )

    pick_up_location = _standing_positions(transport.pick_up)
    place_location = _standing_positions(transport.place)
    assert pick_up_location.target_pose.reference_frame is milk.root
    assert place_location.target_pose is target


# %% picking up and placing without moving


def pick_and_place_of_the_milk(world: World, arm: Arm) -> PickAndPlaceAction:
    """
    :param arm: The arm that picks the milk up and puts it down.
    :return: A pick-and-place of the milk that tries every grasp it offers.
    """
    milk = world.get_semantic_annotations_by_type(Milk)[0]
    return PickAndPlaceAction(
        pick_up=a(PickUpAction)(
            grasp=variable(GraspCandidate, domain=milk.grasp_candidates()), arm=arm
        ),
        place=a(PlaceAction)(
            object_designator=milk,
            target_location=Pose(reference_frame=world.root),
        ),
    )


def test_a_pick_and_place_grounds_the_steps_it_is_given(pr2_apartment_context):
    world, robot, context = pr2_apartment_context
    pick_and_place = pick_and_place_of_the_milk(world, robot.right_arm)
    sequential([pick_and_place], context)

    assert [
        child.designator_type
        for child in pick_and_place._action_plan.children
        if isinstance(child, UnderspecifiedNode)
    ] == [PickUpAction, PlaceAction]


# %% moving to an object and picking it up


def test_move_and_pick_up_takes_the_grasp_it_was_given(pr2_apartment_context):
    """
    The caller chooses the grasp, so the pick-up at the end of the walk takes that one
    rather than whichever grasp the object happens to list first.
    """
    world, robot, context = pr2_apartment_context
    grasp = world.get_semantic_annotations_by_type(Milk)[0].grasp_candidates()[-1]
    move_and_pick_up = MoveAndPickUpAction.from_standing_position(
        standing_position=Pose(reference_frame=world.root),
        grasp=grasp,
        arm=context.robot.left_arm,
    )
    sequential([move_and_pick_up], context)

    pick_ups = [
        child
        for child in move_and_pick_up._action_plan.children
        if isinstance(child, ActionNode) and isinstance(child.designator, PickUpAction)
    ]

    assert [pick_up.designator.grasp for pick_up in pick_ups] == [grasp]


def test_move_and_pick_up_approaches_with_the_clearances_it_was_given(
    pr2_apartment_context,
):
    world, robot, context = pr2_apartment_context
    approach_clearance, retreat_distance = 0.07, 0.13
    move_and_pick_up = MoveAndPickUpAction.from_standing_position(
        standing_position=Pose(reference_frame=world.root),
        grasp=world.get_semantic_annotations_by_type(Milk)[0].grasp_candidates()[0],
        arm=context.robot.left_arm,
        approach_clearance=approach_clearance,
        retreat_distance=retreat_distance,
    )
    sequential([move_and_pick_up], context)

    [pick_up] = [
        child.designator
        for child in move_and_pick_up._action_plan.children
        if isinstance(child, ActionNode) and isinstance(child.designator, PickUpAction)
    ]

    assert (pick_up.approach_clearance, pick_up.retreat_distance) == (
        approach_clearance,
        retreat_distance,
    )


# %% picking up by the grasps closest to where the robot stands

STANDING_DISTANCE = 1.0
"""
How far from the milk, against the world's x-axis, the robot stands, so that the milk
lies straight ahead along that axis.
"""

CLOSER_BY = 0.05
"""
How much nearer the robot, in meters, the closer grasp lies than the milk's own grasps.
"""

RAISED_BY = 0.2
"""
How much higher, in meters, the raised grasp lies than the milk's own grasps.
"""

NON_DEFAULT_NUMBER_OF_GRASPS = 2
"""
A number of closest grasps other than the default, so that the number is seen to be
passed on.
"""


def _standing_behind_the_milk(world: World) -> Pose:
    """
    :return: A standing pose from which the milk lies straight ahead along the world's
        x-axis.
    """
    milk_pose = world.get_semantic_annotations_by_type(Milk)[0].root.global_pose
    return Pose.from_xyz_rpy(
        milk_pose.x - STANDING_DISTANCE, milk_pose.y, 0.0, reference_frame=world.root
    )


def _standing_in_front_of(grasp: GraspCandidate, world: World) -> Pose:
    """
    :return: A standing pose on the floor :data:`STANDING_DISTANCE` back along the
        direction `grasp` is approached along, so that it is approached straight from
        there.
    """
    world_T_grasp = world.transform(grasp.grasp_pose, world.root).to_np()
    world_P_standing = world_T_grasp[:3, 3] - STANDING_DISTANCE * world_T_grasp[:3, 0]
    return Pose.from_xyz_rpy(
        world_P_standing[0], world_P_standing[1], 0.0, reference_frame=world.root
    )


def _raised(grasp: GraspCandidate, height: float) -> GraspCandidate:
    """
    :param height: How far to move the grasp up along its object's z-axis.
    :return: `grasp`, approached the same way from higher up.
    """
    root_P_grasp = grasp.grasp_pose.to_np()[:3, 3] + np.array([0.0, 0.0, height])
    return GraspCandidate(
        grasp.graspable,
        Pose(
            position=Point3.from_iterable(root_P_grasp),
            orientation=grasp.grasp_pose.quaternion,
            reference_frame=grasp.graspable.root,
        ),
    )


def _turned_around(grasp: GraspCandidate, nearer_by: float = 0.0) -> GraspCandidate:
    """
    :param nearer_by: How far to move the grasp back along the direction `grasp` is
        approached along.
    :return: `grasp`, approached from the opposite side.
    """
    grasp_pose = grasp.grasp_pose
    root_P_grasp = grasp_pose.to_np()[:3, 3] - nearer_by * grasp_pose.to_np()[:3, 0]
    return GraspCandidate(
        grasp.graspable,
        Pose(
            position=Point3.from_iterable(root_P_grasp),
            orientation=(
                grasp_pose.rotation_matrix @ RotationMatrix.from_rpy(yaw=np.pi)
            ).quaternion,
            reference_frame=grasp.graspable.root,
        ),
    )


def _grasp_signature(grasp: GraspCandidate) -> tuple:
    """
    :return: The grasp's transform, rounded, to tell grasps of separately generated
        lists apart by value.
    """
    return tuple(np.round(grasp.grasp_pose.to_np(), 6).ravel())


def _assert_each_standing_pose_keeps_the_closest_grasps(
    pick_ups: List[MoveAndPickUpAction],
    graspable: HasGraspCandidates,
    number_of_grasps: int,
) -> None:
    """
    Assert that each standing pose among `pick_ups` is tried with exactly the
    `number_of_grasps` grasps of `graspable` closest to it.
    """
    pick_ups_by_standing_pose = {}
    for pick_up in pick_ups:
        standing_position = pick_up.navigate.target_location
        pick_ups_by_standing_pose.setdefault(
            tuple(np.round(standing_position.to_np(), 6).ravel()), []
        ).append(pick_up)
    grasps = graspable.grasp_candidates()
    for pick_ups_from_one_pose in pick_ups_by_standing_pose.values():
        standing_position = pick_ups_from_one_pose[0].navigate.target_location
        closest = {
            _grasp_signature(grasp)
            for grasp in grasps
            if IsAmongTheClosestGraspsTo(
                grasp, standing_position, grasps, number_of_grasps
            )()
        }
        kept = [
            _grasp_signature(pick_up.pick_up.grasp)
            for pick_up in pick_ups_from_one_pose
        ]
        assert len(kept) == number_of_grasps
        assert set(kept) == closest


def test_a_grasp_approached_straight_from_the_standing_position_is_the_closest(
    pr2_apartment_context,
):
    world, robot, context = pr2_apartment_context
    grasps = world.get_semantic_annotations_by_type(Milk)[0].grasp_candidates()

    assert IsAmongTheClosestGraspsTo(
        grasps[0], _standing_in_front_of(grasps[0], world), grasps, number_of_grasps=1
    )()


def test_a_grasp_approached_from_the_far_side_is_not_among_the_closest(
    pr2_apartment_context,
):
    world, robot, context = pr2_apartment_context
    grasps = world.get_semantic_annotations_by_type(Milk)[0].grasp_candidates()

    assert not IsAmongTheClosestGraspsTo(
        _turned_around(grasps[0]), _standing_in_front_of(grasps[0], world), grasps
    )()


def test_a_nearer_grasp_is_the_closest_however_it_is_approached(pr2_apartment_context):
    """
    Distance is ranked before the direction a grasp is approached along, which only
    decides between grasps at the same distance.
    """
    world, robot, context = pr2_apartment_context
    grasps = world.get_semantic_annotations_by_type(Milk)[0].grasp_candidates()

    assert IsAmongTheClosestGraspsTo(
        _turned_around(grasps[0], nearer_by=CLOSER_BY),
        _standing_in_front_of(grasps[0], world),
        grasps,
        number_of_grasps=1,
    )()


def test_grasps_tied_for_the_closest_are_still_only_as_many_as_asked_for(
    pr2_apartment_context,
):
    """
    Grasps can be exactly as close as one another, for example mirror images of each
    other seen from a standing pose on the object's axis, and no more of them count as
    the closest than were asked for.
    """
    world, robot, context = pr2_apartment_context
    grasp = world.get_semantic_annotations_by_type(Milk)[0].grasp_candidates()[0]
    tied = [grasp, _raised(grasp, 0.0)]
    standing_position = _standing_in_front_of(grasp, world)

    closest = [
        candidate
        for candidate in tied
        if IsAmongTheClosestGraspsTo(
            candidate, standing_position, tied, number_of_grasps=1
        )()
    ]

    assert closest == [grasp]


def test_grasps_at_the_standing_position_itself_still_rank(pr2_apartment_context):
    """
    A grasp right where the robot stands has no direction it is approached from, and
    grasps tied there still count no more of themselves as the closest than were asked
    for.
    """
    world, robot, context = pr2_apartment_context
    grasp = world.get_semantic_annotations_by_type(Milk)[0].grasp_candidates()[0]
    tied = [grasp, _raised(grasp, 0.0)]
    standing_position = world.transform(grasp.grasp_pose, world.root)

    closest = [
        candidate
        for candidate in tied
        if IsAmongTheClosestGraspsTo(
            candidate, standing_position, tied, number_of_grasps=1
        )()
    ]

    assert closest == [grasp]


def test_a_grasp_higher_up_is_as_close_as_one_below_it(pr2_apartment_context):
    """
    Only the horizontal distance counts, so between a grasp and one at the same spot
    higher up, the direction they are approached along decides.
    """
    world, robot, context = pr2_apartment_context
    grasps = world.get_semantic_annotations_by_type(Milk)[0].grasp_candidates()
    raised_head_on = _raised(grasps[0], RAISED_BY)

    assert IsAmongTheClosestGraspsTo(
        raised_head_on,
        _standing_in_front_of(grasps[0], world),
        [_turned_around(grasps[0]), raised_head_on],
        number_of_grasps=1,
    )()


def test_a_pick_up_of_a_graspable_tries_each_standing_pose_with_its_closest_grasps(
    pr2_apartment_context,
):
    world, robot, context = pr2_apartment_context
    milk = world.get_semantic_annotations_by_type(Milk)[0]
    step = MoveAndPickUpAction.from_graspable_by_closest_grasps(
        milk, context.robot.right_arm, context
    )

    pick_ups = list(
        itertools.islice(
            context.query_backend.evaluate(step),
            2 * IsAmongTheClosestGraspsTo.number_of_grasps,
        )
    )

    assert len({id(pick_up.navigate.target_location) for pick_up in pick_ups}) == 2
    _assert_each_standing_pose_keeps_the_closest_grasps(
        pick_ups, milk, IsAmongTheClosestGraspsTo.number_of_grasps
    )


def test_a_pick_up_of_a_graspable_tries_as_many_closest_grasps_as_asked_for(
    pr2_apartment_context,
):
    world, robot, context = pr2_apartment_context
    milk = world.get_semantic_annotations_by_type(Milk)[0]
    step = MoveAndPickUpAction.from_graspable_by_closest_grasps(
        milk,
        context.robot.right_arm,
        context,
        number_of_grasps=NON_DEFAULT_NUMBER_OF_GRASPS,
    )

    pick_ups = list(
        itertools.islice(
            context.query_backend.evaluate(step), 2 * NON_DEFAULT_NUMBER_OF_GRASPS
        )
    )

    assert len({id(pick_up.navigate.target_location) for pick_up in pick_ups}) == 2
    _assert_each_standing_pose_keeps_the_closest_grasps(
        pick_ups, milk, NON_DEFAULT_NUMBER_OF_GRASPS
    )


def test_a_transport_of_a_graspable_picks_it_up_by_the_closest_grasps(
    pr2_apartment_context,
):
    world, robot, context = pr2_apartment_context
    milk = world.get_semantic_annotations_by_type(Milk)[0]
    transport = TransportAction.from_graspable_by_closest_grasps(
        milk,
        Pose.from_xyz_rpy(4.0, 1.5, 0.9, reference_frame=world.root),
        context.robot.right_arm,
        context,
    )

    pick_ups = list(
        itertools.islice(
            context.query_backend.evaluate(transport.pick_up),
            IsAmongTheClosestGraspsTo.number_of_grasps,
        )
    )

    _assert_each_standing_pose_keeps_the_closest_grasps(
        pick_ups, milk, IsAmongTheClosestGraspsTo.number_of_grasps
    )


def test_a_transport_of_a_graspable_tries_as_many_closest_grasps_as_asked_for(
    pr2_apartment_context,
):
    world, robot, context = pr2_apartment_context
    milk = world.get_semantic_annotations_by_type(Milk)[0]
    transport = TransportAction.from_graspable_by_closest_grasps(
        milk,
        Pose.from_xyz_rpy(4.0, 1.5, 0.9, reference_frame=world.root),
        context.robot.right_arm,
        context,
        number_of_grasps=NON_DEFAULT_NUMBER_OF_GRASPS,
    )

    pick_ups = list(
        itertools.islice(
            context.query_backend.evaluate(transport.pick_up),
            NON_DEFAULT_NUMBER_OF_GRASPS,
        )
    )

    _assert_each_standing_pose_keeps_the_closest_grasps(
        pick_ups, milk, NON_DEFAULT_NUMBER_OF_GRASPS
    )


def test_the_closest_grasps_can_be_required_of_a_pick_up_from_a_fixed_standing_pose(
    pr2_apartment_context,
):
    """
    The condition applies to a pick-up whatever its caller left open, so a fixed
    standing pose is tried with only the grasps closest to it.
    """
    world, robot, context = pr2_apartment_context
    milk = world.get_semantic_annotations_by_type(Milk)[0]
    grasps = milk.grasp_candidates()
    object_pose = Pose(reference_frame=milk.root)
    step = a(MoveAndPickUpAction)(
        navigate=NavigateAction(_standing_behind_the_milk(world)),
        face_and_look_at=FaceAndLookAtAction(
            FaceAtAction(object_pose), LookAtAction(object_pose)
        ),
        pick_up=a(PickUpAction)(
            grasp=variable(GraspCandidate, domain=grasps),
            arm=context.robot.right_arm,
        ),
    )
    step.where(
        IsAmongTheClosestGraspsTo(
            step.pick_up.grasp,
            step.navigate.target_location,
            grasps,
            number_of_grasps=NON_DEFAULT_NUMBER_OF_GRASPS,
        )
    )

    pick_ups = list(context.query_backend.evaluate(step))

    _assert_each_standing_pose_keeps_the_closest_grasps(
        pick_ups, milk, NON_DEFAULT_NUMBER_OF_GRASPS
    )


# %% a step faces what it acts on where that is when the step runs

MOVED_MILK_POSITION = (2.4, 2.1, 0.95)
"""
Where the milk is moved to after a pick-up of it has been built.
"""


def _move_the_milk(world: World) -> Milk:
    """
    Move the milk away from where it stood, as a step running before a pick-up of it
    might.

    :return: The milk.
    """
    milk = world.get_semantic_annotations_by_type(Milk)[0]
    milk.root.parent_connection.origin = HomogeneousTransformationMatrix.from_xyz_rpy(
        *MOVED_MILK_POSITION, reference_frame=world.root
    )
    return milk


def _assert_every_target_is_at(targets: List[Pose], body: Body, world: World) -> None:
    """
    Assert that every target resolves to where `body` is now.
    """
    for target in targets:
        np.testing.assert_allclose(
            world.transform(target, world.root).position.to_np(),
            body.global_pose.position.to_np(),
        )


def test_a_transport_of_a_graspable_faces_it_where_it_is_when_it_picks_it_up(
    pr2_apartment_context,
):
    world, robot, context = pr2_apartment_context
    milk = world.get_semantic_annotations_by_type(Milk)[0]
    transport = TransportAction.from_graspable_by_closest_grasps(
        milk,
        Pose.from_xyz_rpy(4.0, 1.5, 0.9, reference_frame=world.root),
        context.robot.right_arm,
        context,
    )

    _move_the_milk(world)

    facing = transport.pick_up._kwargs_["face_and_look_at"]._kwargs_
    _assert_every_target_is_at(
        [facing["face_at"]._kwargs_["target"], facing["look_at"]._kwargs_["target"]],
        milk.root,
        world,
    )


def test_a_move_and_pick_up_faces_the_object_where_it_is_when_it_picks_it_up(
    pr2_apartment_context,
):
    world, robot, context = pr2_apartment_context
    milk = world.get_semantic_annotations_by_type(Milk)[0]
    move_and_pick_up = MoveAndPickUpAction.from_standing_position(
        standing_position=Pose(reference_frame=world.root),
        grasp=milk.grasp_candidates()[0],
        arm=context.robot.left_arm,
    )

    _move_the_milk(world)

    facing = move_and_pick_up.face_and_look_at
    _assert_every_target_is_at(
        [facing.face_at.target, facing.look_at.target], milk.root, world
    )


DRAWER = "cabinet10_drawer_top"
"""
The apartment drawer the opening tests pull out.
"""

DRAWER_HANDLE = "handle_cab10_t"
"""
The handle of :data:`DRAWER`.
"""

OPENED_DRAWER_POSITION = 0.3
"""
How far :data:`DRAWER` is pulled out after an opening of it has been built.
"""


def test_a_move_and_open_faces_the_handle_where_it_is_when_it_opens_the_container(
    pr2_apartment_context,
):
    world, robot, context = pr2_apartment_context
    handle = Handle(root=world.get_body_by_name(DRAWER_HANDLE))
    move_and_open = MoveAndOpenAction.from_standing_position(
        Pose(reference_frame=world.root), handle, context.robot.left_arm
    )

    world.get_connection_by_name(f"{DRAWER}_joint").position = OPENED_DRAWER_POSITION

    facing = move_and_open.face_and_look_at
    _assert_every_target_is_at(
        [facing.face_at.target, facing.look_at.target], handle.root, world
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
    pr2_apartment_context,
):
    world, robot, context = pr2_apartment_context
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
    pr2_apartment_context,
):
    world, robot, context = pr2_apartment_context
    handle = Handle(root=world.get_body_by_name(DRAWER_HANDLE))
    standing_position = Pose(reference_frame=world.root)

    move_and_open = MoveAndOpenAction.from_standing_position(
        standing_position, handle, context.robot.left_arm
    )

    assert move_and_open.navigate.target_location is standing_position
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
        grasp=world.get_semantic_annotations_by_type(Milk)[0].grasp_candidates()[0],
        arm=context.robot.left_arm,
    ),
    "place": _placing_the_held_milk,
}


@pytest.mark.parametrize("build", MOVE_AND_ACT_STEPS.values(), ids=MOVE_AND_ACT_STEPS)
def test_a_move_and_act_step_only_ever_stands_where_it_was_sent(
    pr2_apartment_context, build: Callable[[World, Context], ActionDescription]
):
    """
    Its plan is built before the robot moves, so turning to face the target has to be
    worked out from where the robot is sent rather than from where it stands at first,
    or the robot is sent back there before it acts.
    """
    world, robot, context = pr2_apartment_context
    step = build(world, context)
    sequential([step], context)

    for target in _navigation_targets(step):
        np.testing.assert_allclose(
            target.position.to_np()[:2].ravel(), STANDING_POSITION
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


def test_facing_after_navigating_turns_where_the_robot_was_sent(pr2_apartment_context):
    world, robot, context = pr2_apartment_context
    target = Pose.from_xyz_rpy(4.0, 2.5, 0.9, reference_frame=world.root)
    plan = sequential(
        [NavigateAction(_standing_pose(world)), FaceAtAction(target)], context
    )

    with simulated_robot:
        plan.perform()

    np.testing.assert_allclose(
        robot.root.global_pose.position.to_np()[:2],
        STANDING_POSITION,
        atol=0.03,
    )
    _assert_base_faces(robot, target.position)


def test_facing_a_target_given_relative_to_a_body_turns_towards_that_body(
    pr2_apartment_context,
):
    world, robot, context = pr2_apartment_context
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

    _assert_base_faces(robot, milk.global_pose.position)


def test_facing_and_looking_at_a_target_turns_the_base_and_the_camera_towards_it(
    pr2_apartment_context,
):
    world, robot, context = pr2_apartment_context
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

    _assert_base_faces(robot, target.position)
    camera = robot.get_default_camera()
    camera_P_target = world.transform(target.position, camera.root).to_np()[:3]
    camera_V_forward = camera.forward_facing_axis.to_np()[:3]
    np.testing.assert_allclose(
        camera_P_target / np.linalg.norm(camera_P_target),
        camera_V_forward / np.linalg.norm(camera_V_forward),
        atol=0.02,
    )
