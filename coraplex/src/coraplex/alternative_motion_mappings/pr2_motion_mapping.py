"""
PR2 Motion Mapping for CoraPlex
==============================

Registers PR2-specific AlternativeMotion handlers that optimize base navigation
and execution goals for the PR2 robot within CoraPlex.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from giskardpy.motion_statechart.goals.cartesian_goals import DifferentialDriveBaseGoal
from semantic_digital_twin.robots.pr2 import PR2

from coraplex.datastructures.enums import ExecutionType
from coraplex.robot_plans.motions.base import AlternativeMotion
from coraplex.robot_plans.motions.navigation import MoveMotion

logger = logging.getLogger(__name__)


@dataclass
class PR2MoveSim(MoveMotion, AlternativeMotion[PR2]):
    """
    Alternative base motion for the PR2 in simulation using a base goal.
    """

    execution_type = ExecutionType.SIMULATED

    def perform(self) -> None:
        return

    @property
    def _motion_chart(self) -> DifferentialDriveBaseGoal:
        return DifferentialDriveBaseGoal(
            goal_pose=self.target,
            threshold=0.02,
        )


@dataclass
class PR2MoveReal(MoveMotion, AlternativeMotion[PR2]):
    """
    Alternative base motion for the real PR2 using a base goal.
    """

    execution_type = ExecutionType.REAL

    def perform(self) -> None:
        return

    @property
    def _motion_chart(self) -> DifferentialDriveBaseGoal:
        return DifferentialDriveBaseGoal(
            goal_pose=self.target,
            threshold=0.05,
        )
