import pytest
from semantic_digital_twin.spatial_types.spatial_types import Pose

from semantic_digital_twin.datastructures.definitions import StaticJointState

from coraplex.datastructures.dataclasses import Context
from coraplex.datastructures.enums import Arms
from coraplex.plans.failures import ConditionNotSatisfied, MotionDidNotFinish
from coraplex.language import CodeNode
from coraplex.plans.condition_nodes import ConditionNode
from coraplex.plans.executables import ConditionExecutable
from coraplex.plans.factories import code, execute_single, sequential, try_in_order
from coraplex.plans.failures import (
    AllChildrenFailed,
    BodyUnfetchable,
    ConfigurationNotReached,
    EmptyUnderspecified,
    EndEffectorDidNotReachTarget,
    NavigationGoalNotReachedError,
    PlanFailure,
    RobotInCollision,
)
from coraplex.plans.plan_node import ActionNode
from coraplex.robot_plans.actions.core.navigation import NavigateAction
from coraplex.validation.goal_validator import MultiJointPositionGoalValidator

from .conftest import FailingLeaf

# %% refined_from provenance chain


def test_refined_from_defaults_to_none():
    failure = PlanFailure(node=CodeNode(code=lambda: None))

    assert failure.refined_from is None


def test_refined_from_links_to_the_failure_it_was_refined_from():
    original = PlanFailure(node=CodeNode(code=lambda: None))
    refined = PlanFailure(node=CodeNode(code=lambda: None), refined_from=original)

    assert refined.refined_from is original


def test_refined_from_chain_can_be_walked_across_multiple_links():
    first = PlanFailure(node=CodeNode(code=lambda: None))
    second = PlanFailure(node=CodeNode(code=lambda: None), refined_from=first)
    third = PlanFailure(node=CodeNode(code=lambda: None), refined_from=second)

    assert third.refined_from is second
    assert third.refined_from.refined_from is first


# %% resolution bookkeeping


def test_resolution_defaults_to_none():
    failure = PlanFailure(node=CodeNode(code=lambda: None))

    assert failure.resolution is None


# %% action_node resolution


def test_action_node_returns_the_node_itself_when_it_is_an_action_node():
    action_node = execute_single(NavigateAction(target_location=Pose()))

    failure = PlanFailure(node=action_node)

    assert failure.action_node is action_node


def test_action_node_finds_the_nearest_ancestor_action_node():
    action_node = execute_single(NavigateAction(target_location=Pose()))
    child_node = CodeNode(code=lambda: None)
    action_node.add_child(child_node)

    failure = PlanFailure(node=child_node)

    assert isinstance(action_node, ActionNode)
    assert failure.action_node is action_node


def test_action_node_is_none_when_no_action_node_is_in_the_path():
    root_node = code(lambda: None)

    failure = PlanFailure(node=root_node)

    assert failure.action_node is None


# %% context resolution


def test_context_returns_the_failing_plan_context(immutable_model_world):
    world, robot, context = immutable_model_world
    root_node = sequential([NavigateAction(target_location=Pose())], context)

    failure = PlanFailure(node=root_node)

    assert failure.context is context


# %% raise-site construction


def test_a_try_in_order_of_failing_children_raises_a_well_formed_all_children_failed():
    root = try_in_order(
        [FailingLeaf(), FailingLeaf()], Context(world=None, robot=None)
    )

    with pytest.raises(AllChildrenFailed) as raised:
        root.perform()

    assert raised.value.node is root
    assert raised.value.language_node is root


def test_an_unsatisfied_condition_raises_a_well_formed_condition_not_satisfied():
    action_node = execute_single(NavigateAction(target_location=Pose()))
    condition_node = ConditionNode(
        condition=False, pre_condition=True, action_node=action_node
    )
    executable = ConditionExecutable(condition_node=condition_node, context=None)

    with pytest.raises(ConditionNotSatisfied) as raised:
        executable.execute()

    assert raised.value.node is condition_node
    assert raised.value.action is NavigateAction
    assert raised.value.pre_condition is True


# %% every concrete failure constructs with its required kwargs


def test_every_concrete_failure_is_constructible(immutable_model_world):
    world, view, context = immutable_model_world
    node = CodeNode(code=lambda: None)

    failures = [
        PlanFailure(node=node),
        EmptyUnderspecified(node=node),
        AllChildrenFailed(node=node, language_node=node),
        RobotInCollision(node=node),
        ConfigurationNotReached(
            node=node,
            goal_validator=MultiJointPositionGoalValidator(),
            configuration_type=StaticJointState.PARK,
        ),
        NavigationGoalNotReachedError(node=node, current_pose=Pose(), goal_pose=Pose()),
        BodyUnfetchable(node=node, body=world.root, arm=Arms.LEFT),
        EndEffectorDidNotReachTarget(
            node=node, end_effector=view.left_arm.end_effector, target=Pose()
        ),
        MotionDidNotFinish(node=node, failed_motions=[]),
        ConditionNotSatisfied(
            node=node, pre_condition=True, action=NavigateAction, condition=False
        ),
    ]

    for failure in failures:
        assert failure.node is node
        assert failure.error_message()
