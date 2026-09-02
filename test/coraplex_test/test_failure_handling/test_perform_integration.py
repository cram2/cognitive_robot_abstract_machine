from dataclasses import dataclass

import pytest

from coraplex.datastructures.dataclasses import Context
from coraplex.datastructures.enums import TaskStatus
from coraplex.failure_handling.failure_handler import FailureHandler
from coraplex.failure_handling.failure_handling_strategy import Propagate
from coraplex.failure_handling.failure_refiner import FailureRefiner
from coraplex.plans.factories import execute_single
from coraplex.plans.failures import PlanFailure

from .conftest import FailingLeaf, context_with
from .test_failure_handler import (
    ExhaustibleRetryStrategy,
    HandledFailure,
    RefinedHandledFailure,
    RefiningDetector,
    RetryingStrategy,
)

# %% stub detectors


@dataclass
class ActionFreeRefiningDetector(RefiningDetector):
    """
    Refines the base stub failure without requiring an action node, so refinement can be
    exercised in world-free plans.
    """

    def applies(self, failure: PlanFailure) -> bool:
        return isinstance(failure, self.input_failure_type)


# %% baseline handler on the context


def test_a_context_carries_a_baseline_handler_by_default():
    context = Context(world=None, robot=None)

    assert isinstance(context.failure_handler, FailureHandler)


def test_the_baseline_handler_propagates_a_plain_failure_like_today():
    leaf = FailingLeaf()
    execute_single(leaf, context=Context(world=None, robot=None))

    with pytest.raises(PlanFailure) as raised:
        leaf.perform()

    failure = raised.value
    assert failure.node is leaf
    assert leaf.status == TaskStatus.FAILED
    assert leaf.reason is failure
    assert isinstance(failure.resolution, Propagate)
    assert leaf.end_time is not None


# %% refinement inside perform


def test_a_configured_detector_refines_the_raised_failure():
    handler = FailureHandler(
        refiner=FailureRefiner(failure_detectors=[ActionFreeRefiningDetector()])
    )
    leaf = FailingLeaf(failure_type=HandledFailure)
    execute_single(leaf, context=context_with(handler))

    with pytest.raises(RefinedHandledFailure) as raised:
        leaf.perform()

    assert isinstance(raised.value.refined_from, HandledFailure)
    assert leaf.status == TaskStatus.FAILED
    assert leaf.reason is raised.value


# %% retrying nodes


def test_a_retrying_strategy_reruns_the_node_until_success():
    handler = FailureHandler(strategies=[RetryingStrategy()])
    leaf = FailingLeaf(failure_type=HandledFailure, remaining_failures=2)
    execute_single(leaf, context=context_with(handler))

    leaf.perform()

    assert leaf.executions == 3
    assert leaf.status == TaskStatus.SUCCEEDED


def test_exhausted_retries_propagate():
    handler = FailureHandler(strategies=[ExhaustibleRetryStrategy(maximum_attempts=2)])
    leaf = FailingLeaf(failure_type=HandledFailure)
    execute_single(leaf, context=context_with(handler))

    with pytest.raises(HandledFailure):
        leaf.perform()

    assert leaf.executions == 3
    assert leaf.status == TaskStatus.FAILED
