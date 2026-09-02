from __future__ import annotations

from dataclasses import dataclass, field

from typing_extensions import List, Optional

from krrood.patterns.specificity_ranking import mro_depth, sole_maximum

from coraplex.exceptions import AmbiguousFailureHandlingStrategy
from coraplex.failure_handling.failure_handling_strategy import (
    FailureHandlingStrategy,
    FailureResolution,
    Propagate,
)
from coraplex.failure_handling.failure_refiner import FailureRefiner
from coraplex.failure_handling.strategies.underspecified_reparameterization_strategy import (
    UnderspecifiedReparameterizationStrategy,
)
from coraplex.plans.failures import PlanFailure

# %% handler


@dataclass
class FailureHandler:
    """
    Handles a failure raised during plan execution by first refining it and then letting
    the most specific applicable strategy decide how execution continues.
    """

    refiner: FailureRefiner = field(default_factory=FailureRefiner)
    """
    The refiner that narrows a failure down before a strategy is selected.
    """

    strategies: List[FailureHandlingStrategy] = field(default_factory=list)
    """
    The strategies that resolve refined failures.
    """

    @classmethod
    def baseline(cls) -> FailureHandler:
        """
        :return: The handler every plan context starts with: no detectors and only the
            baseline re-parameterization strategy, which reproduces the
            pre-failure-handling execution semantics.

        ..note:: This construction lives here rather than in
            :mod:`coraplex.failure_handling.factories`, because
            :class:`~coraplex.datastructures.dataclasses.Context` defaults to it and the
            full ensemble in that module imports actions and locations, which in turn
            import the context.
        """
        return cls(
            refiner=FailureRefiner(),
            strategies=[UnderspecifiedReparameterizationStrategy()],
        )

    def most_specific_strategy(
        self, failure: PlanFailure
    ) -> Optional[FailureHandlingStrategy]:
        """
        :param failure: The refined failure to find a strategy for.
        :return: The single most specific applicable strategy, or None if no strategy
            applies.
        :raises AmbiguousFailureHandlingStrategy: If several strategies are equally
            specific.
        """
        applicable = [
            strategy for strategy in self.strategies if strategy.applies(failure)
        ]
        return sole_maximum(
            applicable,
            key=lambda strategy: mro_depth(strategy.handled_failure_type),
            collision_error=lambda strategies: AmbiguousFailureHandlingStrategy(
                failure=failure, strategies=strategies
            ),
        )

    def handle(self, failure: PlanFailure) -> FailureResolution:
        """
        Refine the failure and resolve it with the most specific applicable strategy.

        :param failure: The failure that was raised during plan execution.
        :return: The resolution the handling nodes apply along the plan tree; a
            :class:`~coraplex.failure_handling.failure_handling_strategy.Propagate`
            carrying the refined failure when no strategy applies.
        """
        refined = self.refiner.refine(failure)
        strategy = self.most_specific_strategy(refined)
        if strategy is None:
            return Propagate(failure=refined)
        return strategy.resolve(refined)
