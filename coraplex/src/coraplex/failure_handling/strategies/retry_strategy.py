from __future__ import annotations

from dataclasses import dataclass, field

from coraplex.failure_handling.attempt_budget import AttemptBudget
from coraplex.failure_handling.failure_handling_strategy import (
    FailureHandlingStrategy,
    FailureResolution,
    Propagate,
    RetryNode,
)
from coraplex.plans.failures import (
    EndEffectorDidNotReachTarget,
    MotionDidNotFinish,
    PlanFailure,
)

# %% bounded retries


@dataclass
class RetryStrategy(FailureHandlingStrategy):
    """
    Runs the failing work again without repairing anything, for failures that are worth
    a second attempt on their own.

    Subclasses declare which failure they retry; the base is declared for
    :class:`~coraplex.plans.failures.PlanFailure` and would therefore be as specific as
    the baseline re-parameterization strategy.
    """

    attempt_budget: AttemptBudget = field(default_factory=AttemptBudget)
    """
    How often each node may be run again before the failure propagates.
    """

    def resolve(self, failure: PlanFailure) -> FailureResolution:
        retried_node = self.retried_node(failure)
        if not self.attempt_budget.grant(retried_node):
            return Propagate(failure=failure)
        return RetryNode(failure=failure, target_node=retried_node)


@dataclass
class MotionRetryStrategy(RetryStrategy):
    """
    Retries an action whose motion stopped short for a reason no detector recognised.
    """

    handled_failure_type = MotionDidNotFinish


@dataclass
class EndEffectorRetryStrategy(RetryStrategy):
    """
    Retries an action whose end effector did not reach the pose it was sent to.
    """

    handled_failure_type = EndEffectorDidNotReachTarget
