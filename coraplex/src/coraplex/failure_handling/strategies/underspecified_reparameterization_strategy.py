from __future__ import annotations

from dataclasses import dataclass

from coraplex.failure_handling.failure_handling_strategy import (
    FailureHandlingStrategy,
    FailureResolution,
    Propagate,
    Reparameterize,
)
from coraplex.plans.failures import PlanFailure
from coraplex.plans.plan_node import UnderspecifiedNode

# %% baseline re-parameterization


@dataclass
class UnderspecifiedReparameterizationStrategy(FailureHandlingStrategy):
    """
    Resolve a failure by advancing the nearest enclosing underspecified node to its next
    action candidate, and propagate when there is none.

    This reproduces the blind candidate iteration that used to live inside
    :class:`~coraplex.plans.executables.UnderspecifiedExecutable`.

    ..note:: Only strict ancestors are considered, so a failure raised *at* an
        underspecified node (for example an exhausted candidate iterator) propagates
        instead of re-running the node that just failed to produce a candidate.
    """

    def resolve(self, failure: PlanFailure) -> FailureResolution:
        for ancestor in failure.node.path:
            if isinstance(ancestor, UnderspecifiedNode):
                return Reparameterize(failure=failure, target_node=ancestor)
        return Propagate(failure=failure)
