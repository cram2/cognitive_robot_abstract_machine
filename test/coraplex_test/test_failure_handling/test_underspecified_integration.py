from dataclasses import dataclass, field

import pytest
from krrood.entity_query_language.factories import a
from semantic_digital_twin.spatial_types.spatial_types import Pose

from coraplex.datastructures.enums import TaskStatus
from coraplex.failure_handling.failure_handler import FailureHandler
from coraplex.failure_handling.failure_handling_strategy import (
    FailureHandlingStrategy,
    FailureResolution,
    Propagate,
)
from coraplex.language import CodeNode, SequentialNode
from coraplex.plans.factories import execute_single, sequential
from coraplex.plans.failures import EmptyUnderspecified, PlanFailure
from coraplex.plans.plan_node import PlanNode, UnderspecifiedNode
from coraplex.robot_plans.actions.base import ActionDescription
from coraplex.robot_plans.actions.core.navigation import NavigateAction
from typing_extensions import Iterator, List

# %% candidate stub


@dataclass
class RecordedCandidateAction(ActionDescription):
    """
    An action candidate whose body records that it ran and optionally fails.

    Standing in for a grounded action, it lets a test drive candidate iteration without
    executing motions: the body is a :class:`~coraplex.language.CodeNode`, so it runs
    while the candidate is expanded.
    """

    label: str = ""
    """
    The name this candidate records when its body runs.
    """

    fails: bool = False
    """
    Whether the body raises instead of completing.
    """

    executions: List[str] = field(default_factory=list)
    """
    The shared log every candidate of one test appends its label to.
    """

    @property
    def _action_plan(self) -> PlanNode:
        body = CodeNode(code=lambda: None)
        execute_single(body)

        def run_body():
            self.executions.append(self.label)
            if self.fails:
                raise PlanFailure(node=body)

        body.code = run_body
        return body


# %% stub strategies


@dataclass
class PropagatingStrategy(FailureHandlingStrategy):
    """
    Gives up on every failure, so candidate iteration is short-circuited.
    """

    def resolve(self, failure: PlanFailure) -> FailureResolution:
        return Propagate(failure=failure)


# %% helpers


def candidate_stream(
    *candidates: RecordedCandidateAction,
) -> Iterator[RecordedCandidateAction]:
    """
    :param candidates: The candidates to hand out in order.
    :return: A generator over the candidates, which the underspecified node can close.
    """
    yield from candidates


def underspecified_node_over(
    *candidates: RecordedCandidateAction,
) -> UnderspecifiedNode:
    """
    :param candidates: The candidates the node resolves to, in order.
    :return: A node whose candidate stream is fixed, bypassing the query backend.
    """
    return UnderspecifiedNode(
        underspecified_action=a(NavigateAction)(target_location=Pose()),
        _action_iterator=candidate_stream(*candidates),
    )


# %% candidate iteration at the plan root


def test_candidates_are_iterated_until_one_succeeds_at_the_plan_root(
    immutable_model_world,
):
    world, robot, context = immutable_model_world
    executions = []
    node = underspecified_node_over(
        RecordedCandidateAction(label="first", fails=True, executions=executions),
        RecordedCandidateAction(label="second", fails=False, executions=executions),
    )
    execute_single(node, context)

    node.perform()

    assert executions == ["first", "second"]
    assert node.status == TaskStatus.SUCCEEDED


def test_the_accepted_candidate_releases_the_action_iterator(immutable_model_world):
    world, robot, context = immutable_model_world
    node = underspecified_node_over(
        RecordedCandidateAction(label="only", fails=False, executions=[])
    )
    execute_single(node, context)

    node.perform()

    assert node._action_iterator is None


def test_exhausted_candidates_raise_empty_underspecified(immutable_model_world):
    world, robot, context = immutable_model_world
    executions = []
    node = underspecified_node_over(
        RecordedCandidateAction(label="first", fails=True, executions=executions),
        RecordedCandidateAction(label="second", fails=True, executions=executions),
    )
    execute_single(node, context)

    with pytest.raises(EmptyUnderspecified) as raised:
        node.perform()

    assert executions == ["first", "second"]
    assert raised.value.node is node
    assert node.status == TaskStatus.FAILED


# %% candidate iteration below an enclosing node

# A language node runs its children inside one merged execution list rather than in their
# own perform frames, so an underspecified child only keeps its retry semantics if it is
# performed in a frame of its own.


def test_candidates_are_iterated_when_nested_in_a_sequence(immutable_model_world):
    world, robot, context = immutable_model_world
    executions = []
    node = underspecified_node_over(
        RecordedCandidateAction(label="first", fails=True, executions=executions),
        RecordedCandidateAction(label="second", fails=False, executions=executions),
    )
    root = sequential([node], context)

    root.perform()

    # The nested node runs inside the sequence's execution list rather than in a perform
    # frame of its own, so only the performed root carries a status.
    assert executions == ["first", "second"]
    assert root.status == TaskStatus.SUCCEEDED


def test_exhaustion_below_a_sequence_propagates_out_of_the_plan(immutable_model_world):
    world, robot, context = immutable_model_world
    executions = []
    node = underspecified_node_over(
        RecordedCandidateAction(label="only", fails=True, executions=executions)
    )
    root = sequential([node], context)

    with pytest.raises(EmptyUnderspecified):
        root.perform()

    assert executions == ["only"]
    assert node.status == TaskStatus.FAILED
    assert root.status == TaskStatus.FAILED


# %% handler control over candidate iteration


def test_a_propagating_handler_short_circuits_candidate_iteration(
    immutable_model_world,
):
    world, robot, context = immutable_model_world
    context.failure_handler = FailureHandler(strategies=[PropagatingStrategy()])
    executions = []
    node = underspecified_node_over(
        RecordedCandidateAction(label="first", fails=True, executions=executions),
        RecordedCandidateAction(label="second", fails=False, executions=executions),
    )
    execute_single(node, context)

    with pytest.raises(PlanFailure):
        node.perform()

    assert executions == ["first"]
    assert node.status == TaskStatus.FAILED
