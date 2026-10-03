import logging
from dataclasses import dataclass

import numpy as np
import pytest
from typing_extensions import List

from coraplex.datastructures.enums import InsertionPosition, ReachFraction
from coraplex.exceptions import CannotMatchOnType
from coraplex.orm.ormatic_interface import *  # type: ignore
from coraplex.language import SequentialNode
from coraplex.plans.factories import execute_single, sequential
from coraplex.plans.plan import logger as plan_logger
from coraplex.plans.plan_node import ActionLike, ActionNode, MotionNode, PlanNode
from coraplex.plans.plan_transformation import (
    InsertionTransformation,
    PlanTransformation,
)
from coraplex.plans.underspecified import UnderspecifiedNode
from coraplex.robot_plans.actions.core.misc import DetectAction
from coraplex.robot_plans.actions.core.navigation import LookAtAction, NavigateAction
from coraplex.robot_plans.actions.core.pick_up import PickUpAction, ReachAction
from coraplex.robot_plans.actions.core.robot_body import MoveTorsoAction, ParkArmsAction
from coraplex.robot_plans.motions.gripper import (
    MoveGripperMotion,
    MoveToolCenterPointMotion,
)
from coraplex.robot_plans.motions.robot_body import MoveJointsMotion
from coraplex.robot_plans.actions.composite.transporting import (
    MoveAndOpenAction,
    MoveAndPickUpAction,
    TransportAction,
)
from coraplex.robot_plans.plan_transformations import (
    DetectBeforeGrasp,
    OpenDrawerBeforeMoveAndPickUp,
    OpenDrawerBeforePickUp,
    OpenDrawerBeforeTransport,
    ParkArmsBeforeFirstAction,
)
from krrood.entity_query_language.factories import a
from krrood.entity_query_language.query.match import Match
from krrood.exceptions import UnboundGenericParameter
from semantic_digital_twin.datastructures.definitions import GripperState, TorsoState
from semantic_digital_twin.robots.robot_parts import Arm, EndEffector
from semantic_digital_twin.semantic_annotations.mixins import (
    GraspCandidate,
    HasRootBody,
)
from semantic_digital_twin.semantic_annotations.semantic_annotations import (
    Drawer,
    Handle,
    Milk,
    Spoon,
)
from semantic_digital_twin.spatial_types.spatial_types import Pose
from semantic_digital_twin.world import World

from .test_graph_parsing import detect_actions_of, reach_action

# %% transformations under test


def left_gripper(plan_node: PlanNode) -> EndEffector:
    """
    :param plan_node: A node of the plan a transformation rewrites.
    :return: The left gripper of the robot that plan is run by.
    """
    return plan_node.plan.context.robot.left_arm.end_effector


def right_gripper(plan_node: PlanNode) -> EndEffector:
    """
    :param plan_node: A node of the plan a transformation rewrites.
    :return: The right gripper of the robot that plan is run by.
    """
    return plan_node.plan.context.robot.right_arm.end_effector


def motion_of(plan_node: ActionNode) -> MotionNode:
    """
    :param plan_node: The node of an action that expands into a single motion.
    :return: That motion's node.
    """
    [motion] = [node for node in plan_node.descendants if isinstance(node, MotionNode)]
    return motion


@dataclass
class MoveGrippersBeforeTorsoMotion(InsertionTransformation[MoveTorsoAction]):
    """
    Puts two distinguishable gripper motions in front of the motion a torso move expands
    into.
    """

    @property
    def position(self) -> InsertionPosition:
        return InsertionPosition.BEFORE

    def is_applicable(self, plan_node: PlanNode) -> bool:
        return True

    def anchor(self, plan_node: ActionNode) -> PlanNode:
        return motion_of(plan_node)

    def nodes_to_insert(self, plan_node: ActionNode) -> List[ActionLike]:
        return [
            MoveGripperMotion(GripperState.OPEN, left_gripper(plan_node)),
            MoveGripperMotion(GripperState.CLOSE, right_gripper(plan_node)),
        ]


@dataclass
class MoveGrippersAfterTorsoMotion(MoveGrippersBeforeTorsoMotion):
    """
    Puts the same gripper motions behind that motion instead.
    """

    @property
    def position(self) -> InsertionPosition:
        return InsertionPosition.AFTER


@dataclass
class ParkArmsBeforeTorsoMotion(InsertionTransformation[MoveTorsoAction]):
    """
    Puts an action, which has a plan of its own, in front of the motion a torso move
    expands into.
    """

    @property
    def position(self) -> InsertionPosition:
        return InsertionPosition.BEFORE

    def is_applicable(self, plan_node: PlanNode) -> bool:
        return True

    def anchor(self, plan_node: ActionNode) -> PlanNode:
        return motion_of(plan_node)

    def nodes_to_insert(self, plan_node: ActionNode) -> List[ActionLike]:
        return [ParkArmsAction(plan_node.plan.context.robot.get_arms())]


@dataclass
class MoveGripperLastInTheReachBody(InsertionTransformation[ReachAction]):
    """
    Puts a gripper motion at the end of the sequence a reach expands into.
    """

    @property
    def position(self) -> InsertionPosition:
        return InsertionPosition.LAST_CHILD

    def is_applicable(self, plan_node: PlanNode) -> bool:
        return True

    def anchor(self, plan_node: ActionNode) -> PlanNode:
        [body] = [
            node for node in plan_node.children if isinstance(node, SequentialNode)
        ]
        return body

    def nodes_to_insert(self, plan_node: ActionNode) -> List[ActionLike]:
        return [MoveGripperMotion(GripperState.CLOSE, right_gripper(plan_node))]


def motions_of(plan_node: PlanNode) -> List[MotionNode]:
    """
    :param plan_node: The node whose children to look at.
    :return: The motions directly below the given node, in their plan order.
    """
    return [node for node in plan_node.children if isinstance(node, MotionNode)]


# %% what makes a transformation


@dataclass
class TransformationWithoutRewrite(PlanTransformation[MoveTorsoAction]):
    """
    Says which nodes it applies to without saying how to rewrite the plan around them.
    """

    def is_applicable(self, plan_node: ActionNode) -> bool:
        return True


@dataclass
class TransformationWithoutApplicability(PlanTransformation[MoveTorsoAction]):
    """
    Rewrites the plan without saying whether the case at hand needs it.
    """

    def apply(self, plan_node: ActionNode) -> None:
        pass


@dataclass
class MoveGripperBeforeEveryAction(InsertionTransformation[ActionNode]):
    """
    Puts a gripper motion in front of every action node, whatever action it holds.
    """

    @property
    def position(self) -> InsertionPosition:
        return InsertionPosition.BEFORE

    def is_applicable(self, plan_node: PlanNode) -> bool:
        return True

    def anchor(self, plan_node: ActionNode) -> PlanNode:
        return plan_node

    def nodes_to_insert(self, plan_node: ActionNode) -> List[ActionLike]:
        return [MoveGripperMotion(GripperState.CLOSE, right_gripper(plan_node))]


@dataclass
class TransformationWithoutPosition(InsertionTransformation[MoveTorsoAction]):
    """
    Inserts nodes without saying where they go.
    """

    def anchor(self, plan_node: ActionNode) -> PlanNode:
        return plan_node

    def nodes_to_insert(self, plan_node: ActionNode) -> List[ActionLike]:
        return [MoveGripperMotion(GripperState.OPEN, left_gripper(plan_node))]


@dataclass
class TransformationWithoutMatchedType(PlanTransformation):
    """
    Rewrites nothing and binds no type, to be asked what it matches.
    """

    def is_applicable(self, plan_node: PlanNode) -> bool:
        return True

    def apply(self, plan_node: PlanNode) -> None:
        pass


@dataclass
class TransformationOnAnUnmatchableType(PlanTransformation[GraspCandidate]):
    """
    Binds a type that is neither a plan node nor a designator.
    """

    def is_applicable(self, plan_node: PlanNode) -> bool:
        return True

    def apply(self, plan_node: PlanNode) -> None:
        pass


def test_a_transformation_that_binds_no_type_cannot_say_what_it_matches():
    """
    Which nodes it applies to is part of what a transformation is, so one that binds no
    type has nothing to match on rather than matching every node.
    """
    with pytest.raises(UnboundGenericParameter):
        TransformationWithoutMatchedType().matched_type


def test_a_transformation_bound_to_an_unmatchable_type_is_rejected(
    pr2_apartment_context,
):
    """
    A transformation selects its nodes either by their type or by the designator they
    carry, so a type that is neither leaves no rule to select by.
    """
    world, view, context = pr2_apartment_context
    node = execute_single(MoveTorsoAction(TorsoState.HIGH), context=context)

    with pytest.raises(CannotMatchOnType):
        TransformationOnAnUnmatchableType().matches_node(node)


def test_a_transformation_that_says_no_rewrite_cannot_be_built():
    """
    How it changes the plan is part of what a transformation is, so one that only says
    which nodes it applies to is incomplete.
    """
    with pytest.raises(TypeError):
        TransformationWithoutRewrite()


def test_a_transformation_that_says_no_applicability_cannot_be_built():
    """
    Whether the case at hand needs it is part of what a transformation is, so one that
    leaves it unsaid is incomplete rather than applying to every case it matches.
    """
    with pytest.raises(TypeError):
        TransformationWithoutApplicability()


def test_a_transformation_that_says_no_position_cannot_be_built():
    """
    Where an insertion goes is part of what the transformation is, so one that leaves it
    unsaid is incomplete rather than placed somewhere by default.
    """
    with pytest.raises(TypeError):
        TransformationWithoutPosition()


@dataclass
class MoveGripperBeforeHighTorso(InsertionTransformation[MoveTorsoAction]):
    """
    Puts a gripper motion in front of a torso move, but only when the torso goes up.
    """

    @property
    def position(self) -> InsertionPosition:
        return InsertionPosition.BEFORE

    def is_applicable(self, plan_node: ActionNode) -> bool:
        return plan_node.action.torso_state is TorsoState.HIGH

    def anchor(self, plan_node: ActionNode) -> PlanNode:
        return motion_of(plan_node)

    def nodes_to_insert(self, plan_node: ActionNode) -> List[ActionLike]:
        return [MoveGripperMotion(GripperState.OPEN, left_gripper(plan_node))]


def test_a_transformation_the_case_needs_is_applied(pr2_apartment_context):
    """
    A node the transformation matches and whose case needs it is rewritten.
    """
    world, view, context = pr2_apartment_context
    context.plan_transformations.append(MoveGripperBeforeHighTorso())

    plan = execute_single(MoveTorsoAction(TorsoState.HIGH), context=context)
    plan.notify()

    assert [type(motion.designator) for motion in motions_of(plan)] == [
        MoveGripperMotion,
        MoveJointsMotion,
    ]


def test_a_transformation_the_case_does_not_need_is_skipped(pr2_apartment_context):
    """
    Matching the node type is not enough: a case that does not need the transformation
    keeps the plan the action describes itself.
    """
    world, view, context = pr2_apartment_context
    context.plan_transformations.append(MoveGripperBeforeHighTorso())

    plan = execute_single(MoveTorsoAction(TorsoState.LOW), context=context)
    plan.notify()

    assert [type(motion.designator) for motion in motions_of(plan)] == [
        MoveJointsMotion
    ]


def test_a_transformation_bound_to_a_node_type_reaches_every_action(
    pr2_apartment_context,
):
    """
    Binding the node type selects the nodes of actions of every type, which a binding to
    one action type cannot express.
    """
    world, view, context = pr2_apartment_context
    context.plan_transformations.append(MoveGripperBeforeEveryAction())

    plan = sequential(
        [MoveTorsoAction(TorsoState.HIGH), ParkArmsAction(view.get_arms())], context
    )
    plan.notify()

    assert [type(node.designator) for node in plan.children] == [
        MoveGripperMotion,
        MoveTorsoAction,
        MoveGripperMotion,
        ParkArmsAction,
    ]


def test_a_transformation_bound_to_a_designator_type_selects_the_nodes_carrying_it(
    pr2_apartment_context,
):
    """
    A transformation bound to a designator type reports that type and selects the nodes
    carrying one, leaving the nodes of every other designator alone.
    """
    world, view, context = pr2_apartment_context
    transformation = MoveGrippersBeforeTorsoMotion()
    plan = sequential(
        [MoveTorsoAction(TorsoState.HIGH), ParkArmsAction(view.get_arms())], context
    )
    torso, parking = plan.children

    assert transformation.matched_type is MoveTorsoAction
    assert transformation.matches_node(torso)
    assert not transformation.matches_node(parking)


def test_a_transformation_bound_to_a_node_type_selects_the_nodes_of_that_type(
    pr2_apartment_context,
):
    """
    A transformation bound to a node type reports that type and selects the nodes of it,
    leaving the nodes of every other type alone.
    """
    world, view, context = pr2_apartment_context
    transformation = MoveGripperBeforeEveryAction()
    plan = sequential([MoveTorsoAction(TorsoState.HIGH)], context)
    [torso] = plan.children

    assert transformation.matched_type is ActionNode
    assert transformation.matches_node(torso)
    assert not transformation.matches_node(plan)


@dataclass
class MoveGripperBeforeJointMotion(InsertionTransformation[MoveJointsMotion]):
    """
    Puts a gripper motion in front of a joint motion.
    """

    @property
    def position(self) -> InsertionPosition:
        return InsertionPosition.BEFORE

    def is_applicable(self, plan_node: MotionNode) -> bool:
        return True

    def anchor(self, plan_node: MotionNode) -> PlanNode:
        return plan_node

    def nodes_to_insert(self, plan_node: MotionNode) -> List[ActionLike]:
        return [MoveGripperMotion(GripperState.CLOSE, right_gripper(plan_node))]


def test_a_transformation_bound_to_a_motion_type_selects_the_motion_node(
    pr2_apartment_context,
):
    """
    A designator binding selects by the designator a node carries rather than by the
    kind of node, so binding a motion type selects that motion's node and not the action
    it belongs to.
    """
    world, view, context = pr2_apartment_context
    node = execute_single(MoveTorsoAction(TorsoState.HIGH), context=context)
    node.notify()
    transformation = MoveGripperBeforeJointMotion()

    assert transformation.matched_type is MoveJointsMotion
    assert transformation.matches_node(motion_of(node))
    assert not transformation.matches_node(node)


# %% inserting


def test_a_transformation_inserts_its_nodes_before_the_anchor(pr2_apartment_context):
    """
    The nodes are placed in front of the anchor, keeping the order the transformation
    gives them.
    """
    world, view, context = pr2_apartment_context
    context.plan_transformations.append(MoveGrippersBeforeTorsoMotion())

    plan = execute_single(MoveTorsoAction(TorsoState.HIGH), context=context)
    plan.notify()

    motions = motions_of(plan)
    assert [type(motion.designator) for motion in motions] == [
        MoveGripperMotion,
        MoveGripperMotion,
        MoveJointsMotion,
    ]
    assert [motion.designator.gripper for motion in motions[:2]] == [
        view.left_arm.end_effector,
        view.right_arm.end_effector,
    ]


def test_a_transformation_inserts_its_nodes_after_the_anchor(pr2_apartment_context):
    """
    Inserting after the anchor keeps the given order too, rather than reversing it by
    pushing every node into the same place behind the anchor.
    """
    world, view, context = pr2_apartment_context
    context.plan_transformations.append(MoveGrippersAfterTorsoMotion())

    plan = execute_single(MoveTorsoAction(TorsoState.HIGH), context=context)
    plan.notify()

    motions = motions_of(plan)
    assert [type(motion.designator) for motion in motions] == [
        MoveJointsMotion,
        MoveGripperMotion,
        MoveGripperMotion,
    ]
    assert [motion.designator.gripper for motion in motions[1:]] == [
        view.left_arm.end_effector,
        view.right_arm.end_effector,
    ]


def test_a_transformation_inserts_its_nodes_as_the_last_child_of_the_anchor(
    pr2_apartment_context,
):
    """
    Inserting as the last child makes the node a child of the anchor instead of its
    sibling.
    """
    world, view, context = pr2_apartment_context
    milk = world.get_semantic_annotations_by_type(Milk)[0]
    context.plan_transformations.append(MoveGripperLastInTheReachBody())

    plan = execute_single(reach_action(milk, view), context=context)
    plan.notify()

    [reach_body] = [node for node in plan.children if isinstance(node, SequentialNode)]
    assert [type(node.designator) for node in reach_body.children] == [
        MoveToolCenterPointMotion,
        MoveToolCenterPointMotion,
        MoveGripperMotion,
    ]


def test_a_transformation_leaves_actions_of_another_type_alone(pr2_apartment_context):
    """
    A transformation bound to one action type must not rewrite the plan of another one.
    """
    world, view, context = pr2_apartment_context
    context.plan_transformations.append(MoveGrippersBeforeTorsoMotion())

    plan = execute_single(ParkArmsAction(view.get_arms()), context=context)
    plan.notify()

    assert [
        node
        for node in plan.descendants
        if isinstance(node, MotionNode)
        and isinstance(node.designator, MoveGripperMotion)
    ] == []


def test_an_inserted_action_is_expanded(pr2_apartment_context):
    """
    Transformations run while the plan is expanded, so an inserted action still gets a
    plan of its own instead of staying an unexpanded leaf.
    """
    world, view, context = pr2_apartment_context
    context.plan_transformations.append(ParkArmsBeforeTorsoMotion())

    plan = execute_single(MoveTorsoAction(TorsoState.HIGH), context=context)
    plan.notify()

    [park] = [
        node
        for node in plan.descendants
        if isinstance(node, ActionNode) and isinstance(node.designator, ParkArmsAction)
    ]
    assert [type(motion.designator) for motion in motions_of(park)] == [
        MoveJointsMotion
    ]


# %% detecting before a grasp


def test_the_detection_asks_for_the_object_being_reached_for(pr2_apartment_context):
    """
    The detection has to ask for the object the reach was given, so that a plan grasping
    something else does not query for the wrong thing.
    """
    world, view, context = pr2_apartment_context
    milk = world.get_semantic_annotations_by_type(Milk)[0]
    context.plan_transformations.append(DetectBeforeGrasp())

    plan = execute_single(reach_action(milk, view), context=context)
    plan.notify()

    [detection] = detect_actions_of(plan)
    assert detection.object_sem_annotation is type(milk)


def test_the_perception_precedes_the_final_approach(pr2_apartment_context):
    """
    Perceiving is only worth anything before the approach it corrects, so the look and
    the detection go in front of the reach's last motion.
    """
    world, view, context = pr2_apartment_context
    milk = world.get_semantic_annotations_by_type(Milk)[0]
    context.plan_transformations.append(DetectBeforeGrasp())

    plan = execute_single(reach_action(milk, view), context=context)
    plan.notify()

    [reach_body] = [node for node in plan.children if isinstance(node, SequentialNode)]
    assert [type(node.designator) for node in reach_body.children] == [
        MoveToolCenterPointMotion,
        LookAtAction,
        DetectAction,
        MoveToolCenterPointMotion,
    ]


def test_a_transformation_on_reaches_also_fires_inside_a_pick_up(pr2_apartment_context):
    """
    The reach a pick-up builds is expanded like any other, so a transformation on
    reaches reaches it without the pick-up having to pass anything down.
    """
    world, view, context = pr2_apartment_context
    milk = world.get_semantic_annotations_by_type(Milk)[0]
    context.plan_transformations.append(DetectBeforeGrasp())

    plan = execute_single(
        PickUpAction(milk.grasp_candidates()[0], view.right_arm),
        context=context,
    )
    plan.notify()

    [detection] = detect_actions_of(plan)
    assert detection.object_sem_annotation is type(milk)


# %% opening the drawer an object lies in

OPENED_DRAWER_POSITION = 0.3
"""
How far the drawer is pulled out after an opening of it has been built.
"""


def motions_below(plan_node: PlanNode) -> List[type]:
    """
    :param plan_node: The node whose expansion to look at.
    :return: The type of every motion under the given node, in their plan order.
    """
    return [
        type(node.designator)
        for node in plan_node.descendants
        if isinstance(node, MotionNode)
    ]


def drawer_holding(annotation: HasRootBody, world: World) -> Drawer:
    """
    :param annotation: The object lying in a drawer.
    :param world: The world both belong to.
    :return: The drawer the object hangs under.
    """
    [drawer] = [
        candidate
        for candidate in world.get_semantic_annotations_by_type(Drawer)
        if candidate.root is annotation.root.parent_connection.parent
    ]
    return drawer


def pick_up_action(annotation, arm: Arm) -> PickUpAction:
    """
    :param annotation: The object to pick up.
    :param arm: The arm to pick it up with.
    :return: A pick-up of the object by the first grasp it offers.
    """
    return PickUpAction(annotation.grasp_candidates()[0], arm)


def handle_opened_by(opening: Match) -> Handle:
    """
    :param opening: The step that opens a drawer.
    :return: The handle it opens the drawer by.
    """
    return opening._kwargs_["open_container"]._kwargs_["handle"]


def arm_opening_with(opening: Match) -> Arm:
    """
    :param opening: The step that opens a drawer.
    :return: The arm it opens the drawer with.
    """
    return opening._kwargs_["open_container"]._kwargs_["arm"]


def test_the_drawer_is_only_opened_for_an_object_that_lies_in_one(
    pr2_apartment_context,
):
    """
    Opening a drawer is worth doing only for an object lying in one, so the pick-up of
    the spoon needs the transformation and the pick-up of the milk does not.
    """
    world, view, context = pr2_apartment_context
    spoon = world.get_semantic_annotations_by_type(Spoon)[0]
    milk = world.get_semantic_annotations_by_type(Milk)[0]
    transformation = OpenDrawerBeforePickUp()

    [in_a_drawer] = sequential(
        [pick_up_action(spoon, view.right_arm)], context
    ).children
    [in_the_open] = sequential([pick_up_action(milk, view.right_arm)], context).children

    assert transformation.is_applicable(in_a_drawer)
    assert not transformation.is_applicable(in_the_open)


def test_a_drawer_reports_how_far_it_stands_open(pr2_apartment_context):
    """
    How far a drawer stands open is read from its own travel, so it stands none of the
    way open at the lower limit of its joint and all of the way at the upper one.
    """
    world, view, context = pr2_apartment_context
    spoon = world.get_semantic_annotations_by_type(Spoon)[0]
    drawer = drawer_holding(spoon, world)
    connection = drawer.root.parent_connection

    connection.position = connection.dof.limits.lower.position
    world.notify_state_change()
    assert drawer.opening_ratio == 0

    connection.position = connection.dof.limits.upper.position
    world.notify_state_change()
    assert drawer.opening_ratio == 1


def test_a_drawer_that_already_stands_open_needs_no_opening(pr2_apartment_context):
    """
    The opening is worth doing only while the drawer is shut, so a drawer that already
    stands open leaves the pick-up as it is.
    """
    world, view, context = pr2_apartment_context
    spoon = world.get_semantic_annotations_by_type(Spoon)[0]
    drawer = drawer_holding(spoon, world)
    transformation = OpenDrawerBeforePickUp()

    [pick_up] = sequential([pick_up_action(spoon, view.right_arm)], context).children
    assert transformation.is_applicable(pick_up)

    connection = drawer.root.parent_connection
    connection.position = connection.dof.limits.upper.position
    world.notify_state_change()

    assert not transformation.is_applicable(pick_up)


def test_opening_a_drawer_tries_its_standing_pose_with_the_opening(
    pr2_apartment_context,
):
    """
    Where the robot stands decides whether the handle can be reached, so the standing
    pose is tried together with the opening rather than chosen before it.
    """
    world, view, context = pr2_apartment_context
    drawer = drawer_holding(world.get_semantic_annotations_by_type(Spoon)[0], world)

    [opening] = OpenDrawerBeforeMoveAndPickUp().opening_nodes(
        drawer, view.right_arm, context
    )

    assert opening._type_ is MoveAndOpenAction
    assert handle_opened_by(opening) is drawer.handle


def test_opening_a_drawer_stands_where_it_is_opened_from(pr2_apartment_context):
    """
    The robot stands back for opening a container the way it does for any container,
    rather than as close as it would to grasp something that stays put.
    """
    world, view, context = pr2_apartment_context
    drawer = drawer_holding(world.get_semantic_annotations_by_type(Spoon)[0], world)

    [opening] = OpenDrawerBeforeMoveAndPickUp().opening_nodes(
        drawer, view.right_arm, context
    )
    standing_positions = (
        opening._kwargs_["navigate"]._kwargs_["target_location"]._domain_
    )
    location = standing_positions.domain

    assert location.reach_fraction == ReachFraction.ACCESSING


def test_opening_a_drawer_faces_the_handle_where_it_is_when_it_opens_it(
    pr2_apartment_context,
):
    """
    The opening runs after whatever came before it in the plan, so it turns to the
    handle where that left it rather than where it was when the opening was built.
    """
    world, view, context = pr2_apartment_context
    drawer = drawer_holding(world.get_semantic_annotations_by_type(Spoon)[0], world)
    [opening] = OpenDrawerBeforeMoveAndPickUp().opening_nodes(
        drawer, view.right_arm, context
    )

    drawer.root.parent_connection.position = OPENED_DRAWER_POSITION
    world.notify_state_change()

    facing = opening._kwargs_["face_and_look_at"]._kwargs_
    for target in [
        facing["face_at"]._kwargs_["target"],
        facing["look_at"]._kwargs_["target"],
    ]:
        np.testing.assert_allclose(
            world.transform(target, world.root).to_position().to_np(),
            drawer.handle.root.global_pose.to_position().to_np(),
        )


def test_the_drawer_is_opened_before_a_move_and_pick_up_rather_than_inside_it(
    pr2_apartment_context,
):
    """
    A move-and-pick-up drives to the object before picking it up, so the opening
    precedes the whole step, whose own drive then positions the robot at the object.
    """
    world, view, context = pr2_apartment_context
    spoon = world.get_semantic_annotations_by_type(Spoon)[0]
    drawer = drawer_holding(spoon, world)
    context.plan_transformations.append(OpenDrawerBeforeMoveAndPickUp())

    move_and_pick_up = MoveAndPickUpAction.from_standing_position(
        standing_position=Pose(reference_frame=world.root),
        grasp=spoon.grasp_candidates()[0],
        arm=view.right_arm,
    )
    plan = sequential([move_and_pick_up], context)
    plan.notify()

    [opening, moved_and_picked_up] = plan.children
    assert isinstance(opening, UnderspecifiedNode)
    assert opening.designator_type is MoveAndOpenAction
    assert handle_opened_by(opening.underspecified_action) is drawer.handle
    assert moved_and_picked_up.designator is move_and_pick_up


def test_the_drawer_the_object_lies_in_is_opened_before_the_pick_up(
    pr2_apartment_context,
):
    """
    A drawer has to stand open before the gripper goes in, so the opening precedes the
    pick-up, followed by the drive back to where the object can be reached from.
    """
    world, view, context = pr2_apartment_context
    spoon = world.get_semantic_annotations_by_type(Spoon)[0]
    drawer = drawer_holding(spoon, world)
    context.plan_transformations.append(OpenDrawerBeforePickUp())

    plan = sequential([pick_up_action(spoon, view.right_arm)], context)
    plan.notify()

    [opening, parking, drive_to_the_spoon, pick_up] = plan.children
    assert opening.designator_type is MoveAndOpenAction
    assert handle_opened_by(opening.underspecified_action) is drawer.handle
    assert isinstance(parking.designator, ParkArmsAction)
    assert drive_to_the_spoon.designator_type is NavigateAction
    assert isinstance(pick_up.designator, PickUpAction)


def test_the_actions_beside_the_pick_up_are_expanded(pr2_apartment_context):
    """
    The rewrite is inserted beside the node being expanded rather than below it, so its
    new parent has to expand the actions in it as well.

    An unexpanded action is a leaf that cannot be parsed, so it would fail only once the
    plan is run.
    """
    world, view, context = pr2_apartment_context
    spoon = world.get_semantic_annotations_by_type(Spoon)[0]
    context.plan_transformations.append(OpenDrawerBeforePickUp())

    plan = sequential([pick_up_action(spoon, view.right_arm)], context)
    plan.notify()

    [_, parking, _, _] = plan.children
    on_its_own = execute_single(
        ParkArmsAction(parking.designator.arms), context=context
    )
    on_its_own.notify()

    assert motions_below(parking) == motions_below(on_its_own)


def test_the_drawer_is_opened_with_the_arm_that_picks_up(pr2_apartment_context):
    """
    Opening with the other arm would leave the robot holding the handle it has to reach
    past, so the opening takes the arm the pick-up was given.
    """
    world, view, context = pr2_apartment_context
    spoon = world.get_semantic_annotations_by_type(Spoon)[0]
    context.plan_transformations.append(OpenDrawerBeforePickUp())

    pick_up = pick_up_action(spoon, view.left_arm)
    plan = sequential([pick_up], context)
    plan.notify()

    [opening] = [
        node
        for node in plan.children
        if isinstance(node, UnderspecifiedNode)
        and node.designator_type is MoveAndOpenAction
    ]
    assert arm_opening_with(opening.underspecified_action) is pick_up.arm


def test_an_object_that_lies_in_no_drawer_is_picked_up_unchanged(pr2_apartment_context):
    """
    An object standing in the open needs no drawer opened for it, so the pick-up keeps
    the plan it describes itself.
    """
    world, view, context = pr2_apartment_context
    milk = world.get_semantic_annotations_by_type(Milk)[0]
    context.plan_transformations.append(OpenDrawerBeforePickUp())

    plan = sequential([pick_up_action(milk, view.right_arm)], context)
    plan.notify()

    [pick_up] = plan.children
    assert isinstance(pick_up.designator, PickUpAction)


def test_the_opening_joins_the_sequence_an_underspecified_pick_up_runs(
    pr2_apartment_context,
):
    """
    A pick-up written as an underspecified statement is grounded into a candidate at
    execution time, and only the sequence around that candidate is run.

    The opening has to land in that sequence, otherwise it is inserted into the plan but
    never performed.
    """
    world, view, context = pr2_apartment_context
    spoon = world.get_semantic_annotations_by_type(Spoon)[0]
    drawer = drawer_holding(spoon, world)
    context.plan_transformations.append(ParkArmsBeforeFirstAction())
    context.plan_transformations.append(OpenDrawerBeforePickUp())

    described = pick_up_action(spoon, view.right_arm)
    plan = sequential(
        [a(PickUpAction)(grasp=described.grasp, arm=described.arm)],
        context,
    )
    plan.notify()

    [underspecified] = plan.children
    assert isinstance(underspecified, UnderspecifiedNode)
    assert underspecified.advance()

    [
        parking,
        opening,
        parking_again,
        drive_to_the_spoon,
        candidate,
    ] = underspecified.current_candidate_sequence.children
    assert candidate is underspecified.current_candidate
    assert isinstance(parking.designator, ParkArmsAction)
    assert opening.designator_type is MoveAndOpenAction
    assert handle_opened_by(opening.underspecified_action) is drawer.handle
    assert isinstance(parking_again.designator, ParkArmsAction)
    assert drive_to_the_spoon.designator_type is NavigateAction


def test_the_drawer_is_opened_before_a_transport_rather_than_inside_its_pick_up(
    pr2_apartment_context,
):
    """
    Every candidate of a transport's pick-up is tried against the world as it stands, so
    the drawer is opened once before the transport instead of with each candidate.
    """
    world, view, context = pr2_apartment_context
    spoon = world.get_semantic_annotations_by_type(Spoon)[0]
    drawer = drawer_holding(spoon, world)
    context.plan_transformations.append(OpenDrawerBeforeTransport())

    transport = TransportAction.from_graspable_by_closest_grasps(
        spoon, Pose(reference_frame=world.root), view.right_arm, context
    )
    plan = sequential([transport], context)
    plan.notify()

    [opening, transported] = plan.children
    assert isinstance(opening, UnderspecifiedNode)
    assert opening.designator_type is MoveAndOpenAction
    assert handle_opened_by(opening.underspecified_action) is drawer.handle
    assert arm_opening_with(opening.underspecified_action) is view.right_arm
    assert transported.designator is transport


# %% transformations that collide on one node


@dataclass
class MoveLeftGripperBeforeTorso(InsertionTransformation[MoveTorsoAction]):
    """
    Puts a left gripper motion in front of a torso move.
    """

    @property
    def position(self) -> InsertionPosition:
        return InsertionPosition.BEFORE

    def is_applicable(self, plan_node: ActionNode) -> bool:
        return True

    def anchor(self, plan_node: ActionNode) -> PlanNode:
        return plan_node

    def nodes_to_insert(self, plan_node: ActionNode) -> List[ActionLike]:
        return [MoveGripperMotion(GripperState.OPEN, left_gripper(plan_node))]


@dataclass
class MoveRightGripperBeforeTorso(MoveLeftGripperBeforeTorso):
    """
    Puts a right gripper motion there instead.
    """

    def nodes_to_insert(self, plan_node: ActionNode) -> List[ActionLike]:
        return [MoveGripperMotion(GripperState.CLOSE, right_gripper(plan_node))]


def warnings_of(caplog) -> List[str]:
    """
    :param caplog: The capture of this test's log records.
    :return: The message of every warning the plan reported.
    """
    return [
        record.getMessage()
        for record in caplog.records
        if record.name == plan_logger.name and record.levelno == logging.WARNING
    ]


def test_two_transformations_applied_to_one_node_are_reported(
    pr2_apartment_context, caplog
):
    """
    Whichever transformation rewrites a node first decides what the next one finds, so a
    node more than one of them is applied to is reported.
    """
    world, view, context = pr2_apartment_context
    context.plan_transformations.extend(
        [MoveLeftGripperBeforeTorso(), MoveRightGripperBeforeTorso()]
    )

    plan = sequential([MoveTorsoAction(TorsoState.HIGH)], context)
    [torso] = [node for node in plan.children if isinstance(node, ActionNode)]
    with caplog.at_level(logging.WARNING, logger=plan_logger.name):
        plan.notify()

    [warning] = warnings_of(caplog)
    assert str(torso) in warning


def test_the_transformations_that_collide_are_still_applied(pr2_apartment_context):
    """
    The report is a warning rather than a refusal, so both of them rewrite the plan, in
    the order the context lists them.
    """
    world, view, context = pr2_apartment_context
    context.plan_transformations.extend(
        [MoveLeftGripperBeforeTorso(), MoveRightGripperBeforeTorso()]
    )

    plan = sequential([MoveTorsoAction(TorsoState.HIGH)], context)
    plan.notify()

    assert [motion.designator.gripper for motion in motions_of(plan)] == [
        view.left_arm.end_effector,
        view.right_arm.end_effector,
    ]


def test_a_transformation_the_case_does_not_need_is_no_collision(
    pr2_apartment_context, caplog
):
    """
    Two transformations matching the same node type collide only where both are needed,
    so the one whose case does not apply leaves the other one alone.
    """
    world, view, context = pr2_apartment_context
    context.plan_transformations.extend(
        [MoveLeftGripperBeforeTorso(), MoveGripperBeforeHighTorso()]
    )

    plan = sequential([MoveTorsoAction(TorsoState.LOW)], context)
    [torso] = [node for node in plan.children if isinstance(node, ActionNode)]
    with caplog.at_level(logging.WARNING, logger=plan_logger.name):
        plan.notify()

    assert warnings_of(caplog) == []
    assert plan.plan.applicable_transformations(torso) == [MoveLeftGripperBeforeTorso()]
