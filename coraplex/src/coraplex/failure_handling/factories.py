from __future__ import annotations

from coraplex.failure_handling.detectors import (
    BodyUnfetchableDetector,
    EndEffectorTargetDetector,
    NavigationGoalDetector,
)
from coraplex.failure_handling.failure_handler import FailureHandler
from coraplex.failure_handling.failure_refiner import FailureRefiner
from coraplex.failure_handling.strategies.navigation_recovery_strategy import (
    NavigationRecoveryStrategy,
)
from coraplex.failure_handling.strategies.retry_strategy import (
    EndEffectorRetryStrategy,
    MotionRetryStrategy,
)
from coraplex.failure_handling.strategies.underspecified_reparameterization_strategy import (
    UnderspecifiedReparameterizationStrategy,
)

# %% shipped ensemble


def default_failure_handler() -> FailureHandler:
    """
    :return: A handler that refines motion failures with every shipped detector and
        answers each refined failure with the strategy written for it, falling back to
        the baseline re-parameterization.

    ..note:: This is not the default of
        :class:`~coraplex.datastructures.dataclasses.Context`, which starts from
        :meth:`~coraplex.failure_handling.failure_handler.FailureHandler.baseline`; a
        plan opts into recovery by assigning this handler to its context.
    """
    return FailureHandler(
        refiner=FailureRefiner(
            failure_detectors=[
                NavigationGoalDetector(),
                EndEffectorTargetDetector(),
                BodyUnfetchableDetector(),
            ]
        ),
        strategies=[
            NavigationRecoveryStrategy(),
            MotionRetryStrategy(),
            EndEffectorRetryStrategy(),
            UnderspecifiedReparameterizationStrategy(),
        ],
    )
