from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from typing_extensions import Optional

from coraplex.failure_handling.attempt_budget import AttemptBudget
from coraplex.failure_handling.failure_handling_strategy import (
    FailureResolution,
    RecoveryPlanStrategy,
    RetryNode,
)
from coraplex.locations.factories import occupancy_location
from coraplex.plans.failures import NavigationGoalNotReachedError
from coraplex.plans.plan_node import ActionLike
from coraplex.robot_plans.actions.core.navigation import NavigateAction
from semantic_digital_twin.spatial_types.spatial_types import Pose

# %% navigation recovery


@dataclass
class NavigationRecoveryStrategy(RecoveryPlanStrategy):
    """
    Recovers from a robot that stopped short of where it was driving by sending it to a
    freshly generated standing pose near that destination, and then running the failing
    action again from there.
    """

    handled_failure_type = NavigationGoalNotReachedError

    attempt_budget: AttemptBudget = field(default_factory=AttemptBudget)
    """
    How often each node may be recovered before the failure propagates.
    """

    arrival_tolerance: float = 0.03
    """
    How close a candidate may be to the missed destination before it counts as the same
    pose.

    Matches the tolerance
    :meth:`~coraplex.robot_plans.actions.core.navigation.NavigateAction.post_condition`
    accepts, so the recovery agrees with what navigating considers success.
    """

    def recovery_plan(
        self, failure: NavigationGoalNotReachedError
    ) -> Optional[ActionLike]:
        if not self.attempt_budget.grant(self.retried_node(failure)):
            return None
        standing_pose = self.fresh_standing_pose(failure)
        if standing_pose is None:
            return None
        return NavigateAction(target_location=standing_pose)

    def fresh_standing_pose(
        self, failure: NavigationGoalNotReachedError
    ) -> Optional[Pose]:
        """
        Generate a standing pose to recover to.

        Driving to the very pose the robot just failed to reach is no recovery, so that
        candidate is skipped.

        :param failure: The failure naming the destination the robot did not reach.
        :return: A free standing pose near that destination, or None if there is none.
        """
        for candidate in occupancy_location(failure.goal_pose, failure.context):
            if not np.allclose(
                candidate.to_np(),
                failure.goal_pose.to_np(),
                atol=self.arrival_tolerance,
            ):
                return candidate
        return None

    def resolution_after_recovery(
        self, failure: NavigationGoalNotReachedError
    ) -> FailureResolution:
        return RetryNode(failure=failure, target_node=self.retried_node(failure))
