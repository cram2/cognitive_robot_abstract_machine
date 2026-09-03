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
from coraplex.exceptions import PerceptionTargetMissing
from coraplex.orm.ormatic_interface import *  # type: ignore
from coraplex.language import SequentialNode
from coraplex.plans.factories import execute_single, sequential
from coraplex.plans.plan_node import (
    ActionLike,
    ActionNode,
    MotionNode,
    PlanNode,
    UnderspecifiedNode,
)
from coraplex.plans.plan_transformation import (
    ActionTransformation,
    InsertionTransformation,
)
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
from coraplex.robot_plans.plan_transformations import (
    DetectBeforeGrasp,
    OpenDrawerBeforePickUp,
)
from krrood.entity_query_language.factories import a
from semantic_digital_twin.datastructures.definitions import GripperState, TorsoState
from semantic_digital_twin.semantic_annotations.mixins import HasRootBody
from semantic_digital_twin.semantic_annotations.semantic_annotations import (
    Drawer,
    Milk,
    Spoon,
)
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
class MoveGrippersBesideTorsoMotion(
    InsertionTransformation, ActionTransformation[MoveTorsoAction]
):
    """
    Puts two distinguishable gripper motions next to the motion a torso move expands
    into.
    """

    def anchor(self, plan_node: ActionNode) -> PlanNode:
        return motion_of(plan_node)

    def nodes_to_insert(self, plan_node: ActionNode) -> List[ActionLike]:
        return [
            MoveGripperMotion(GripperState.OPEN, Arms.LEFT),
            MoveGripperMotion(GripperState.CLOSE, Arms.RIGHT),
        ]


@dataclass
class ParkArmsBesideTorsoMotion(
    InsertionTransformation, ActionTransformation[MoveTorsoAction]
):
    """
    Puts an action, which has a plan of its own, next to the motion a torso move expands
    into.
    """

    def anchor(self, plan_node: ActionNode) -> PlanNode:
        return motion_of(plan_node)

    def nodes_to_insert(self, plan_node: ActionNode) -> List[ActionLike]:
        return [ParkArmsAction(Arms.BOTH)]


@dataclass
class MoveGripperBelowTheReachBody(
    InsertionTransformation, ActionTransformation[ReachAction]
):
    """
    Puts a gripper motion below the sequence a reach expands into.
    """

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


# %% inserting


def test_a_transformation_inserts_its_nodes_before_the_anchor(immutable_model_world):
    """
    The nodes are placed in front of the anchor, keeping the order the transformation
    gives them.
    """
    world, view, context = immutable_model_world
    context.plan_transformations.append(MoveGrippersBesideTorsoMotion())

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
    context.plan_transformations.append(
        MoveGrippersBesideTorsoMotion(position=InsertionPosition.AFTER)
    )

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


def test_a_transformation_inserts_its_nodes_below_the_anchor(immutable_model_world):
    """
    Inserting below the anchor makes the node its last child instead of its sibling.
    """
    world, view, context = immutable_model_world
    milk = world.get_semantic_annotations_by_type(Milk)[0]
    context.plan_transformations.append(
        MoveGripperBelowTheReachBody(position=InsertionPosition.BELOW)
    )

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
    context.plan_transformations.append(MoveGrippersBesideTorsoMotion())

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
    context.plan_transformations.append(ParkArmsBesideTorsoMotion())

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

    [navigation, opening, pick_up] = plan.children
    assert navigation.designator_type is NavigateAction
    assert isinstance(pick_up.designator, PickUpAction)
    assert isinstance(opening.designator, OpenAction)
    assert opening.designator.object_designator is drawer.handle.root


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

    [_, opening, _] = plan.children
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

    [navigation, opening, candidate] = underspecified.current_attempt.children
    assert candidate is underspecified.current_candidate
    assert navigation.designator_type is NavigateAction
    assert isinstance(opening.designator, OpenAction)
    assert opening.designator.object_designator is drawer.handle.root
