from dataclasses import dataclass, field

import pytest

from coraplex.datastructures.dataclasses import Context
from coraplex.failure_handling.failure_handler import FailureHandler
from coraplex.failure_handling.failure_handling_strategy import (
    FailureResolution,
    Propagate,
    RecoveryPlanStrategy,
    RetryNode,
)
from coraplex.language import CodeNode
from coraplex.plans.factories import code
from coraplex.plans.failures import PlanFailure
from coraplex.plans.plan_node import ActionLike

from .test_failure_handler import HandledFailure

# %% stub strategies


@dataclass
class ContextRecordingRecoveryStrategy(RecoveryPlanStrategy):
    """
    Recovers with a plan that records the context it performed in.
    """

    handled_failure_type = HandledFailure

    recovery_contexts: list[Context] = field(default_factory=list, init=False)
    """
    The contexts the recovery plan performed in, one per recovery.
    """

    def recovery_plan(self, failure: PlanFailure) -> ActionLike:
        recovery_node = CodeNode()
        recovery_node.code = lambda: self.recovery_contexts.append(
            recovery_node.plan.context
        )
        return recovery_node

    def resolution_after_recovery(self, failure: PlanFailure) -> FailureResolution:
        return RetryNode(failure=failure, target_node=failure.node)


@dataclass
class FailingRecoveryStrategy(RecoveryPlanStrategy):
    """
    Recovers with a plan that itself fails, counting how often recovery was attempted.
    """

    handled_failure_type = HandledFailure

    recovery_attempts: int = field(default=0, init=False)
    """
    How often a recovery plan was built.
    """

    def recovery_plan(self, failure: PlanFailure) -> ActionLike:
        self.recovery_attempts += 1
        recovery_node = CodeNode()

        def fail_recovery():
            raise HandledFailure(node=recovery_node)

        recovery_node.code = fail_recovery
        return recovery_node

    def resolution_after_recovery(self, failure: PlanFailure) -> FailureResolution:
        return RetryNode(failure=failure, target_node=failure.node)


@dataclass
class RecoverylessStrategy(RecoveryPlanStrategy):
    """
    Never has a recovery plan to offer.
    """

    handled_failure_type = HandledFailure

    def recovery_plan(self, failure: PlanFailure) -> None:
        return None

    def resolution_after_recovery(self, failure: PlanFailure) -> FailureResolution:
        return RetryNode(failure=failure, target_node=failure.node)


# %% recovery outcomes


def test_no_recovery_plan_propagates():
    failing_node = code(lambda: None, context=Context(world=None, robot=None))
    failure = HandledFailure(node=failing_node)

    resolution = RecoverylessStrategy().resolve(failure)

    assert isinstance(resolution, Propagate)
    assert resolution.failure is failure


def test_the_recovery_plan_performs_in_the_failing_plans_context():
    context = Context(world=None, robot=None)
    failing_node = code(lambda: None, context=context)
    strategy = ContextRecordingRecoveryStrategy()
    failure = HandledFailure(node=failing_node)

    resolution = strategy.resolve(failure)

    assert strategy.recovery_contexts == [context]
    assert isinstance(resolution, RetryNode)
    assert resolution.target_node is failing_node


def test_a_failing_recovery_propagates_with_the_recovery_failure_linked():
    failing_node = code(lambda: None, context=Context(world=None, robot=None))
    failure = HandledFailure(node=failing_node)

    resolution = FailingRecoveryStrategy().resolve(failure)

    assert isinstance(resolution, Propagate)
    assert resolution.failure is failure
    assert isinstance(failure.__cause__, HandledFailure)
    assert failure.__cause__ is not failure


def test_the_failing_plans_context_keeps_its_plan_after_recovery():
    context = Context(world=None, robot=None)
    failing_node = code(lambda: None, context=context)
    failing_plan = context.plan
    failure = HandledFailure(node=failing_node)

    ContextRecordingRecoveryStrategy().resolve(failure)

    assert context.plan is failing_plan


# %% re-entrant failures during recovery


def test_a_reentrant_failure_during_recovery_propagates_without_a_second_recovery():
    """
    The recovery plan shares the failing plan's context and therefore its handler, so a
    recovery failing like the original failure re-enters the same strategy; that re-
    entry must give up instead of recovering from the recovery.
    """
    strategy = FailingRecoveryStrategy()
    context = Context(
        world=None, robot=None, failure_handler=FailureHandler(strategies=[strategy])
    )
    failing_node = code(lambda: None, context=context)

    def fail():
        raise HandledFailure(node=failing_node)

    failing_node.code = fail

    with pytest.raises(HandledFailure):
        failing_node.perform()

    assert strategy.recovery_attempts == 1
