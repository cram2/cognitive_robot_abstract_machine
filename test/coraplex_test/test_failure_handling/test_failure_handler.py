from dataclasses import dataclass, field

import pytest
from krrood.entity_query_language.factories import a
from semantic_digital_twin.spatial_types.spatial_types import Pose

from coraplex.datastructures.dataclasses import Context
from coraplex.datastructures.enums import TaskStatus
from coraplex.exceptions import AmbiguousFailureHandlingStrategy
from coraplex.failure_handling.failure_handler import FailureHandler
from coraplex.failure_handling.failure_handling_strategy import (
    FailureHandlingStrategy,
    FailureResolution,
    Propagate,
    Reparameterize,
    RetryNode,
)
from coraplex.failure_handling.failure_refiner import FailureDetector, FailureRefiner
from coraplex.failure_handling.strategies.underspecified_reparameterization_strategy import (
    UnderspecifiedReparameterizationStrategy,
)
from coraplex.plans.factories import execute_single
from coraplex.plans.failures import PlanFailure
from coraplex.plans.plan_node import UnderspecifiedNode
from coraplex.robot_plans.actions.core.navigation import NavigateAction

from .conftest import child_of

# %% stub failures


@dataclass
class HandledFailure(PlanFailure):
    """
    The failure the stub strategies are declared for.
    """


@dataclass
class SpecificHandledFailure(HandledFailure):
    """
    A more specific failure, so that strategies declared for the base and for the
    subclass compete.
    """


@dataclass
class RefinedHandledFailure(PlanFailure):
    """
    The failure the stub detector produces, a sibling of :class:`HandledFailure` like
    real refined failure types are.
    """


# %% stub detectors


@dataclass
class RefiningDetector(FailureDetector):
    """
    Refines the base stub failure into the refined one, proving that the handler refines
    before it selects a strategy.
    """

    input_failure_type = HandledFailure
    output_failure_type = RefinedHandledFailure

    def detect(self, failure: PlanFailure) -> PlanFailure:
        return RefinedHandledFailure(node=failure.node)


# %% stub strategies


@dataclass
class RetryingStrategy(FailureHandlingStrategy):
    """
    Handles the base stub failure by retrying the failing frame.
    """

    handled_failure_type = HandledFailure

    def resolve(self, failure: PlanFailure) -> FailureResolution:
        return RetryNode(failure=failure, target_node=failure.node)


@dataclass
class AlternativeRetryingStrategy(FailureHandlingStrategy):
    """
    Handles the base stub failure just as specifically as :class:`RetryingStrategy`,
    which makes the two of them ambiguous.
    """

    handled_failure_type = HandledFailure

    def resolve(self, failure: PlanFailure) -> FailureResolution:
        return RetryNode(failure=failure, target_node=failure.node)


@dataclass
class PropagatingSubclassStrategy(FailureHandlingStrategy):
    """
    Declared for a subclass of :class:`RetryingStrategy`'s handled type and therefore
    more specific than it.
    """

    handled_failure_type = SpecificHandledFailure

    def resolve(self, failure: PlanFailure) -> FailureResolution:
        return Propagate(failure=failure)


@dataclass
class RefinedFailureOnlyStrategy(FailureHandlingStrategy):
    """
    Handles only the detector's output type, so it is selected exactly when the handler
    refined the failure first.
    """

    handled_failure_type = RefinedHandledFailure

    def resolve(self, failure: PlanFailure) -> FailureResolution:
        return Propagate(failure=failure)


@dataclass
class RefinedRetryingStrategy(FailureHandlingStrategy):
    """
    Handles the detector's output type by retrying the failing node, so that the
    original failure and the refined one part ways after refinement.
    """

    handled_failure_type = RefinedHandledFailure

    def resolve(self, failure: PlanFailure) -> FailureResolution:
        return RetryNode(failure=failure, target_node=failure.node)


@dataclass
class ExhaustibleRetryStrategy(FailureHandlingStrategy):
    """
    Retries a bounded number of times and propagates once its attempts are exhausted,
    keeping the attempt bookkeeping inside the strategy instance.
    """

    handled_failure_type = HandledFailure

    maximum_attempts: int = 2
    """
    How many retries this strategy grants before it propagates.
    """

    attempts: int = field(default=0, init=False)
    """
    How many retries this strategy has granted so far.
    """

    def resolve(self, failure: PlanFailure) -> FailureResolution:
        if self.attempts >= self.maximum_attempts:
            return Propagate(failure=failure)
        self.attempts += 1
        return RetryNode(failure=failure, target_node=failure.node)


# %% resolution apply contract


def test_propagate_records_the_failure_and_raises_it_at_the_root(code_node):
    failure = HandledFailure(node=code_node)
    resolution = Propagate(failure=failure)

    with pytest.raises(HandledFailure) as raised:
        resolution.apply(code_node)

    assert raised.value is failure
    assert code_node.status == TaskStatus.FAILED
    assert code_node.reason is failure
    assert failure.resolution is resolution


def test_a_targeted_resolution_returns_at_its_target(code_node):
    failure = HandledFailure(node=code_node)
    resolution = RetryNode(failure=failure, target_node=code_node)
    failure.resolution = resolution

    resolution.apply(code_node)

    assert failure.resolution is None
    assert code_node.status == TaskStatus.CREATED


def test_a_targeted_resolution_escalates_below_its_target(code_node):
    """
    Applied below its target the resolution records the failure and escalates along the
    plan tree, so the target deals with it without needing a perform frame of its own.
    """
    child = child_of(code_node)
    failure = HandledFailure(node=child)
    resolution = RetryNode(failure=failure, target_node=code_node)

    resolution.apply(child)

    assert child.status == TaskStatus.FAILED
    assert child.reason is failure
    assert failure.resolution is None


def test_the_original_failure_carries_no_resolution_after_refinement():
    """
    Only the failure that escalates carries the resolution; a resolution decided for a
    refined failure must not stick to the original one, which would otherwise never be
    re-consulted for.
    """
    handler = FailureHandler(
        refiner=FailureRefiner(failure_detectors=[RefiningDetector()]),
        strategies=[RefinedRetryingStrategy()],
    )
    action_node = execute_single(
        NavigateAction(target_location=Pose()),
        context=Context(world=None, robot=None, failure_handler=handler),
    )
    original = HandledFailure(node=action_node)

    action_node.handle_failure(original)

    assert original.resolution is None


# %% strategy selection


def test_the_strategy_declared_for_the_failure_subclass_wins(code_node):
    handler = FailureHandler(
        strategies=[RetryingStrategy(), PropagatingSubclassStrategy()]
    )
    failure = SpecificHandledFailure(node=code_node)

    assert isinstance(handler.handle(failure), Propagate)


def test_no_applicable_strategy_propagates_the_refined_failure(code_node):
    handler = FailureHandler()
    failure = HandledFailure(node=code_node)

    resolution = handler.handle(failure)

    assert isinstance(resolution, Propagate)
    assert resolution.failure is failure


def test_the_handler_refines_before_selecting_a_strategy():
    action_node = execute_single(NavigateAction(target_location=Pose()))
    handler = FailureHandler(
        refiner=FailureRefiner(failure_detectors=[RefiningDetector()]),
        strategies=[RefinedFailureOnlyStrategy()],
    )
    failure = HandledFailure(node=action_node)

    resolution = handler.handle(failure)

    assert isinstance(resolution, Propagate)
    assert isinstance(resolution.failure, RefinedHandledFailure)
    assert resolution.failure.refined_from is failure


def test_equally_specific_strategies_are_ambiguous(code_node):
    handler = FailureHandler(
        strategies=[RetryingStrategy(), AlternativeRetryingStrategy()]
    )
    failure = HandledFailure(node=code_node)

    with pytest.raises(AmbiguousFailureHandlingStrategy):
        handler.handle(failure)


# %% attempt exhaustion


def test_an_exhausted_strategy_propagates(code_node):
    handler = FailureHandler(strategies=[ExhaustibleRetryStrategy(maximum_attempts=2)])

    first = handler.handle(HandledFailure(node=code_node))
    second = handler.handle(HandledFailure(node=code_node))
    third = handler.handle(HandledFailure(node=code_node))

    assert isinstance(first, RetryNode)
    assert isinstance(second, RetryNode)
    assert isinstance(third, Propagate)


# %% baseline strategy


def test_the_baseline_strategy_targets_the_nearest_underspecified_ancestor(
    underspecified_node,
):
    nested_underspecified = UnderspecifiedNode(
        underspecified_action=a(NavigateAction)(target_location=Pose())
    )
    underspecified_node.add_child(nested_underspecified)
    failing_leaf = child_of(nested_underspecified)
    failure = HandledFailure(node=failing_leaf)

    resolution = UnderspecifiedReparameterizationStrategy().resolve(failure)

    assert isinstance(resolution, Reparameterize)
    assert resolution.target_node is nested_underspecified


def test_the_baseline_strategy_propagates_without_an_underspecified_ancestor(code_node):
    failure = HandledFailure(node=code_node)

    resolution = UnderspecifiedReparameterizationStrategy().resolve(failure)

    assert isinstance(resolution, Propagate)
    assert resolution.failure is failure


def test_the_baseline_strategy_does_not_target_the_failing_node_itself(
    underspecified_node,
):
    failure = HandledFailure(node=underspecified_node)

    resolution = UnderspecifiedReparameterizationStrategy().resolve(failure)

    assert isinstance(resolution, Propagate)


# %% baseline handler factory


def test_the_baseline_handler_reparameterizes_under_an_underspecified_node(
    underspecified_node,
):
    failing_leaf = child_of(underspecified_node)
    failure = HandledFailure(node=failing_leaf)

    resolution = FailureHandler.baseline().handle(failure)

    assert isinstance(resolution, Reparameterize)
    assert resolution.target_node is underspecified_node
