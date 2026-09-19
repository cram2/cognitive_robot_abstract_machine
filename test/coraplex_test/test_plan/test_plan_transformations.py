import logging
from dataclasses import dataclass

import pytest
from typing_extensions import List

from coraplex.datastructures.enums import (
    ApproachDirection,
    Arms,
    InsertionPosition,
    VerticalAlignment,
)
from coraplex.datastructures.grasp import GraspDescription
from coraplex.exceptions import CannotMatchOnType, PerceptionTargetMissing
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
from coraplex.robot_plans.actions.core.container import OpenAction
from coraplex.robot_plans.actions.core.misc import DetectAction
from coraplex.robot_plans.actions.core.navigation import LookAtAction, NavigateAction
from coraplex.robot_plans.actions.core.pick_up import PickUpAction, ReachAction
from coraplex.robot_plans.actions.core.robot_body import MoveTorsoAction, ParkArmsAction
from coraplex.robot_plans.motions.gripper import (
    MoveGripperMotion,
    MoveToolCenterPointMotion,
)
from coraplex.robot_plans.motions.robot_body import MoveJointsMotion
from coraplex.robot_plans.actions.composite.transporting import TransportAction
from coraplex.robot_plans.plan_transformations import (
    DetectBeforeGrasp,
    OpenDrawerBeforePickUp,
    OpenDrawerBeforeTransport,
    ParkArmsBeforeFirstAction,
)
from krrood.entity_query_language.factories import a
from krrood.exceptions import UnboundGenericParameter
from semantic_digital_twin.datastructures.definitions import GripperState, TorsoState
from semantic_digital_twin.semantic_annotations.mixins import HasRootBody
from semantic_digital_twin.semantic_annotations.semantic_annotations import (
    Drawer,
    Milk,
    Spoon,
)
from semantic_digital_twin.spatial_types.spatial_types import Pose
from semantic_digital_twin.world import World

from .test_graph_parsing import detect_actions_of, reach_action

# %% transformations under test


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
            MoveGripperMotion(GripperState.OPEN, Arms.LEFT),
            MoveGripperMotion(GripperState.CLOSE, Arms.RIGHT),
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
        return [ParkArmsAction(Arms.BOTH)]


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
        return [MoveGripperMotion(GripperState.CLOSE, Arms.RIGHT)]


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
        return [MoveGripperMotion(GripperState.CLOSE, Arms.RIGHT)]


@dataclass
class TransformationWithoutPosition(InsertionTransformation[MoveTorsoAction]):
    """
    Inserts nodes without saying where they go.
    """

    def anchor(self, plan_node: ActionNode) -> PlanNode:
        return plan_node

    def nodes_to_insert(self, plan_node: ActionNode) -> List[ActionLike]:
        return [MoveGripperMotion(GripperState.OPEN, Arms.LEFT)]


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
class TransformationOnAnUnmatchableType(PlanTransformation[GraspDescription]):
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
    immutable_model_world,
):
    """
    A transformation selects its nodes either by their type or by the designator they
    carry, so a type that is neither leaves no rule to select by.
    """
    world, view, context = immutable_model_world
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
        return [MoveGripperMotion(GripperState.OPEN, Arms.LEFT)]


def test_a_transformation_the_case_needs_is_applied(immutable_model_world):
    """
    A node the transformation matches and whose case needs it is rewritten.
    """
    world, view, context = immutable_model_world
    context.plan_transformations.append(MoveGripperBeforeHighTorso())

    plan = execute_single(MoveTorsoAction(TorsoState.HIGH), context=context)
    plan.notify()

    assert [type(motion.designator) for motion in motions_of(plan)] == [
        MoveGripperMotion,
        MoveJointsMotion,
    ]


def test_a_transformation_the_case_does_not_need_is_skipped(immutable_model_world):
    """
    Matching the node type is not enough: a case that does not need the transformation
    keeps the plan the action describes itself.
    """
    world, view, context = immutable_model_world
    context.plan_transformations.append(MoveGripperBeforeHighTorso())

    plan = execute_single(MoveTorsoAction(TorsoState.LOW), context=context)
    plan.notify()

    assert [type(motion.designator) for motion in motions_of(plan)] == [
        MoveJointsMotion
    ]


def test_a_transformation_bound_to_a_node_type_reaches_every_action(
    immutable_model_world,
):
    """
    Binding the node type selects the nodes of actions of every type, which a binding to
    one action type cannot express.
    """
    world, view, context = immutable_model_world
    context.plan_transformations.append(MoveGripperBeforeEveryAction())

    plan = sequential(
        [MoveTorsoAction(TorsoState.HIGH), ParkArmsAction(Arms.BOTH)], context
    )
    plan.notify()

    assert [type(node.designator) for node in plan.children] == [
        MoveGripperMotion,
        MoveTorsoAction,
        MoveGripperMotion,
        ParkArmsAction,
    ]


def test_a_transformation_bound_to_a_designator_type_selects_the_nodes_carrying_it(
    immutable_model_world,
):
    """
    A transformation bound to a designator type reports that type and selects the nodes
    carrying one, leaving the nodes of every other designator alone.
    """
    world, view, context = immutable_model_world
    transformation = MoveGrippersBeforeTorsoMotion()
    plan = sequential(
        [MoveTorsoAction(TorsoState.HIGH), ParkArmsAction(Arms.BOTH)], context
    )
    torso, parking = plan.children

    assert transformation.matched_type is MoveTorsoAction
    assert transformation.matches_node(torso)
    assert not transformation.matches_node(parking)


def test_a_transformation_bound_to_a_node_type_selects_the_nodes_of_that_type(
    immutable_model_world,
):
    """
    A transformation bound to a node type reports that type and selects the nodes of it,
    leaving the nodes of every other type alone.
    """
    world, view, context = immutable_model_world
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
        return [MoveGripperMotion(GripperState.CLOSE, Arms.RIGHT)]


def test_a_transformation_bound_to_a_motion_type_selects_the_motion_node(
    immutable_model_world,
):
    """
    A designator binding selects by the designator a node carries rather than by the
    kind of node, so binding a motion type selects that motion's node and not the action
    it belongs to.
    """
    world, view, context = immutable_model_world
    node = execute_single(MoveTorsoAction(TorsoState.HIGH), context=context)
    node.notify()
    transformation = MoveGripperBeforeJointMotion()

    assert transformation.matched_type is MoveJointsMotion
    assert transformation.matches_node(motion_of(node))
    assert not transformation.matches_node(node)


# %% inserting


def test_a_transformation_inserts_its_nodes_before_the_anchor(immutable_model_world):
    """
    The nodes are placed in front of the anchor, keeping the order the transformation
    gives them.
    """
    world, view, context = immutable_model_world
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
        Arms.LEFT,
        Arms.RIGHT,
    ]


def test_a_transformation_inserts_its_nodes_after_the_anchor(immutable_model_world):
    """
    Inserting after the anchor keeps the given order too, rather than reversing it by
    pushing every node into the same place behind the anchor.
    """
    world, view, context = immutable_model_world
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
        Arms.LEFT,
        Arms.RIGHT,
    ]


def test_a_transformation_inserts_its_nodes_as_the_last_child_of_the_anchor(
    immutable_model_world,
):
    """
    Inserting as the last child makes the node a child of the anchor instead of its
    sibling.
    """
    world, view, context = immutable_model_world
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


def test_a_transformation_leaves_actions_of_another_type_alone(immutable_model_world):
    """
    A transformation bound to one action type must not rewrite the plan of another one.
    """
    world, view, context = immutable_model_world
    context.plan_transformations.append(MoveGrippersBeforeTorsoMotion())

    plan = execute_single(ParkArmsAction(Arms.BOTH), context=context)
    plan.notify()

    assert [
        node
        for node in plan.descendants
        if isinstance(node, MotionNode)
        and isinstance(node.designator, MoveGripperMotion)
    ] == []


def test_an_inserted_action_is_expanded(immutable_model_world):
    """
    Transformations run while the plan is expanded, so an inserted action still gets a
    plan of its own instead of staying an unexpanded leaf.
    """
    world, view, context = immutable_model_world
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


def test_the_detection_asks_for_the_object_being_reached_for(immutable_model_world):
    """
    The detection has to ask for the object the reach was given, so that a plan grasping
    something else does not query for the wrong thing.
    """
    world, view, context = immutable_model_world
    milk = world.get_semantic_annotations_by_type(Milk)[0]
    context.plan_transformations.append(DetectBeforeGrasp())

    plan = execute_single(reach_action(milk, view), context=context)
    plan.notify()

    [detection] = detect_actions_of(plan)
    assert detection.object_sem_annotation is type(milk)


def test_the_perception_precedes_the_final_approach(immutable_model_world):
    """
    Perceiving is only worth anything before the approach it corrects, so the look and
    the detection go in front of the reach's last motion.
    """
    world, view, context = immutable_model_world
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


def test_a_transformation_on_reaches_also_fires_inside_a_pick_up(immutable_model_world):
    """
    The reach a pick-up builds is expanded like any other, so a transformation on
    reaches reaches it without the pick-up having to pass anything down.
    """
    world, view, context = immutable_model_world
    milk = world.get_semantic_annotations_by_type(Milk)[0]
    context.plan_transformations.append(DetectBeforeGrasp())

    plan = execute_single(
        PickUpAction(milk, Arms.RIGHT, reach_action(milk, view).grasp_description),
        context=context,
    )
    plan.notify()

    [detection] = detect_actions_of(plan)
    assert detection.object_sem_annotation is type(milk)


def test_perceiving_without_an_object_to_detect_is_rejected(immutable_model_world):
    """
    A reach may be given a pose without an object, but then there is nothing to build
    the detection query from, so the contradiction is reported instead of guessed away.
    """
    world, view, context = immutable_model_world
    milk = world.get_semantic_annotations_by_type(Milk)[0]
    context.plan_transformations.append(DetectBeforeGrasp())

    reach = reach_action(milk, view)
    reach.object_designator = None

    with pytest.raises(PerceptionTargetMissing):
        execute_single(reach, context=context).notify()


# %% opening the drawer an object lies in


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


def pick_up_action(annotation, view, arm: Arms = Arms.RIGHT) -> PickUpAction:
    """
    :param annotation: The object to pick up.
    :param view: The robot picking it up.
    :param arm: The arm to pick it up with.
    :return: A pick-up of the object.
    """
    return PickUpAction(
        annotation,
        arm,
        GraspDescription(
            ApproachDirection.FRONT,
            VerticalAlignment.NoAlignment,
            view.right_arm.end_effector,
        ),
    )


def test_the_drawer_is_only_opened_for_an_object_that_lies_in_one(
    immutable_model_world,
):
    """
    Opening a drawer is worth doing only for an object lying in one, so the pick-up of
    the spoon needs the transformation and the pick-up of the milk does not.
    """
    world, view, context = immutable_model_world
    spoon = world.get_semantic_annotations_by_type(Spoon)[0]
    milk = world.get_semantic_annotations_by_type(Milk)[0]
    transformation = OpenDrawerBeforePickUp()

    [in_a_drawer] = sequential([pick_up_action(spoon, view)], context).children
    [in_the_open] = sequential([pick_up_action(milk, view)], context).children

    assert transformation.is_applicable(in_a_drawer)
    assert not transformation.is_applicable(in_the_open)


def test_a_drawer_reports_how_far_it_stands_open(immutable_model_world):
    """
    How far a drawer stands open is read from its own travel, so it stands none of the
    way open at the lower limit of its joint and all of the way at the upper one.
    """
    world, view, context = immutable_model_world
    spoon = world.get_semantic_annotations_by_type(Spoon)[0]
    drawer = drawer_holding(spoon, world)
    connection = drawer.root.parent_connection

    connection.position = connection.dof.limits.lower.position
    world.notify_state_change()
    assert drawer.opening_ratio == 0

    connection.position = connection.dof.limits.upper.position
    world.notify_state_change()
    assert drawer.opening_ratio == 1


def test_a_drawer_that_already_stands_open_needs_no_opening(immutable_model_world):
    """
    The opening is worth doing only while the drawer is shut, so a drawer that already
    stands open leaves the pick-up as it is.
    """
    world, view, context = immutable_model_world
    spoon = world.get_semantic_annotations_by_type(Spoon)[0]
    drawer = drawer_holding(spoon, world)
    transformation = OpenDrawerBeforePickUp()

    [pick_up] = sequential([pick_up_action(spoon, view)], context).children
    assert transformation.is_applicable(pick_up)

    connection = drawer.root.parent_connection
    connection.position = connection.dof.limits.upper.position
    world.notify_state_change()

    assert not transformation.is_applicable(pick_up)


def test_the_drawer_is_opened_before_a_transport_rather_than_inside_it(
    immutable_model_world,
):
    """
    A transport drives to the object before picking it up, and that drive is grounded
    against the world it finds, so the opening precedes the whole transport.
    """
    world, view, context = immutable_model_world
    spoon = world.get_semantic_annotations_by_type(Spoon)[0]
    drawer = drawer_holding(spoon, world)
    context.plan_transformations.append(OpenDrawerBeforeTransport())

    transport = TransportAction(
        spoon,
        Pose.from_xyz_rpy(5.1, 3.3, 0.75, reference_frame=world.root),
        Arms.RIGHT,
        pick_up_action(spoon, view).grasp_description,
    )
    plan = sequential([transport], context)
    plan.notify()

    [drive_to_the_handle, opening, transported] = plan.children
    assert drive_to_the_handle.designator_type is NavigateAction
    assert isinstance(opening.designator, OpenAction)
    assert opening.designator.object_designator is drawer.handle.root
    assert transported.designator is transport


def test_the_drawer_the_object_lies_in_is_opened_before_the_pick_up(
    immutable_model_world,
):
    """
    A drawer has to stand open before the gripper goes in, so the opening and the drive
    that makes its handle reachable precede the pick-up rather than following it.
    """
    world, view, context = immutable_model_world
    spoon = world.get_semantic_annotations_by_type(Spoon)[0]
    drawer = drawer_holding(spoon, world)
    context.plan_transformations.append(OpenDrawerBeforePickUp())

    plan = sequential([pick_up_action(spoon, view)], context)
    plan.notify()

    [drive_to_the_handle, opening, parking, drive_to_the_spoon, pick_up] = plan.children
    assert drive_to_the_handle.designator_type is NavigateAction
    assert isinstance(opening.designator, OpenAction)
    assert opening.designator.object_designator is drawer.handle.root
    assert isinstance(parking.designator, ParkArmsAction)
    assert drive_to_the_spoon.designator_type is NavigateAction
    assert isinstance(pick_up.designator, PickUpAction)


def test_the_opening_beside_the_pick_up_is_expanded(immutable_model_world):
    """
    The opening is inserted beside the node being expanded rather than below it, so its
    new parent has to expand it as well.

    An unexpanded action is a leaf that cannot be parsed, so it would fail only once the
    plan is run.
    """
    world, view, context = immutable_model_world
    spoon = world.get_semantic_annotations_by_type(Spoon)[0]
    context.plan_transformations.append(OpenDrawerBeforePickUp())

    plan = sequential([pick_up_action(spoon, view)], context)
    plan.notify()

    [_, opening, _, _, _] = plan.children
    on_its_own = execute_single(
        OpenAction(opening.designator.object_designator, opening.designator.arm),
        context=context,
    )
    on_its_own.notify()

    assert motions_below(opening) == motions_below(on_its_own)


def test_the_drawer_is_opened_with_the_arm_that_picks_up(immutable_model_world):
    """
    Opening with the other arm would leave the robot holding the handle it has to reach
    past, so the opening takes the arm the pick-up was given.
    """
    world, view, context = immutable_model_world
    spoon = world.get_semantic_annotations_by_type(Spoon)[0]
    context.plan_transformations.append(OpenDrawerBeforePickUp())

    pick_up = pick_up_action(spoon, view, arm=Arms.LEFT)
    plan = sequential([pick_up], context)
    plan.notify()

    [opening] = [
        node.designator
        for node in plan.descendants
        if isinstance(node, ActionNode) and isinstance(node.designator, OpenAction)
    ]
    assert opening.arm is pick_up.arm


def test_an_object_that_lies_in_no_drawer_is_picked_up_unchanged(immutable_model_world):
    """
    An object standing in the open needs no drawer opened for it, so the pick-up keeps
    the plan it describes itself.
    """
    world, view, context = immutable_model_world
    milk = world.get_semantic_annotations_by_type(Milk)[0]
    context.plan_transformations.append(OpenDrawerBeforePickUp())

    plan = sequential([pick_up_action(milk, view)], context)
    plan.notify()

    [pick_up] = plan.children
    assert isinstance(pick_up.designator, PickUpAction)


def test_the_opening_joins_the_sequence_an_underspecified_pick_up_runs(
    immutable_model_world,
):
    """
    A pick-up written as an underspecified statement is grounded into a candidate at
    execution time, and only the sequence around that candidate is run.

    The opening has to land in that sequence, otherwise it is inserted into the plan but
    never performed.
    """
    world, view, context = immutable_model_world
    spoon = world.get_semantic_annotations_by_type(Spoon)[0]
    drawer = drawer_holding(spoon, world)
    context.plan_transformations.append(ParkArmsBeforeFirstAction())
    context.plan_transformations.append(OpenDrawerBeforePickUp())

    described = pick_up_action(spoon, view)
    plan = sequential(
        [
            a(PickUpAction)(
                object_designator=described.object_designator,
                arm=described.arm,
                grasp_description=described.grasp_description,
            )
        ],
        context,
    )
    plan.notify()

    [underspecified] = plan.children
    assert isinstance(underspecified, UnderspecifiedNode)
    assert underspecified.advance()

    [
        parking,
        drive_to_the_handle,
        opening,
        parking_again,
        drive_to_the_spoon,
        candidate,
    ] = underspecified.current_candidate_sequence.children
    assert candidate is underspecified.current_candidate
    assert isinstance(parking.designator, ParkArmsAction)
    assert drive_to_the_handle.designator_type is NavigateAction
    assert isinstance(opening.designator, OpenAction)
    assert opening.designator.object_designator is drawer.handle.root
    assert isinstance(parking_again.designator, ParkArmsAction)
    assert drive_to_the_spoon.designator_type is NavigateAction


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
        return [MoveGripperMotion(GripperState.OPEN, Arms.LEFT)]


@dataclass
class MoveRightGripperBeforeTorso(MoveLeftGripperBeforeTorso):
    """
    Puts a right gripper motion there instead.
    """

    def nodes_to_insert(self, plan_node: ActionNode) -> List[ActionLike]:
        return [MoveGripperMotion(GripperState.CLOSE, Arms.RIGHT)]


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
    immutable_model_world, caplog
):
    """
    Whichever transformation rewrites a node first decides what the next one finds, so a
    node more than one of them is applied to is reported.
    """
    world, view, context = immutable_model_world
    context.plan_transformations.extend(
        [MoveLeftGripperBeforeTorso(), MoveRightGripperBeforeTorso()]
    )

    plan = sequential([MoveTorsoAction(TorsoState.HIGH)], context)
    [torso] = [node for node in plan.children if isinstance(node, ActionNode)]
    with caplog.at_level(logging.WARNING, logger=plan_logger.name):
        plan.notify()

    [warning] = warnings_of(caplog)
    assert str(torso) in warning


def test_the_transformations_that_collide_are_still_applied(immutable_model_world):
    """
    The report is a warning rather than a refusal, so both of them rewrite the plan, in
    the order the context lists them.
    """
    world, view, context = immutable_model_world
    context.plan_transformations.extend(
        [MoveLeftGripperBeforeTorso(), MoveRightGripperBeforeTorso()]
    )

    plan = sequential([MoveTorsoAction(TorsoState.HIGH)], context)
    plan.notify()

    assert [motion.designator.gripper for motion in motions_of(plan)] == [
        Arms.LEFT,
        Arms.RIGHT,
    ]


def test_a_transformation_the_case_does_not_need_is_no_collision(
    immutable_model_world, caplog
):
    """
    Two transformations matching the same node type collide only where both are needed,
    so the one whose case does not apply leaves the other one alone.
    """
    world, view, context = immutable_model_world
    context.plan_transformations.extend(
        [MoveLeftGripperBeforeTorso(), MoveGripperBeforeHighTorso()]
    )

    plan = sequential([MoveTorsoAction(TorsoState.LOW)], context)
    [torso] = [node for node in plan.children if isinstance(node, ActionNode)]
    with caplog.at_level(logging.WARNING, logger=plan_logger.name):
        plan.notify()

    assert warnings_of(caplog) == []
    assert plan.plan.applicable_transformations(torso) == [MoveLeftGripperBeforeTorso()]
