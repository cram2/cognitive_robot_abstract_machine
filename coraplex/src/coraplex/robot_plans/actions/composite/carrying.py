"""
Driving the robot's base with what it holds kept upright.
"""

from __future__ import annotations

from dataclasses import dataclass

from coraplex.robot_plans.actions.base import ParallelAction
from coraplex.robot_plans.actions.core.navigation import NavigateAction
from coraplex.robot_plans.actions.core.robot_body import ParkArmsAction
from cramph.node import StatechartNode
from giskardpy.motion_statechart.tasks.align_planes import (
    AlignPlanes,
    KeepHeldBodyUpright,
)
from semantic_digital_twin.robots.robot_parts import Arm

# %% driving with held objects upright


@dataclass(eq=False, repr=False, kw_only=True)
class CarryAction(ParallelAction):
    """
    Drives the robot's base to a pose while every object :attr:`holding_arms` hold keeps
    its z-axis pointing up and every other arm is parked.
    """

    navigate: NavigateAction
    """
    The drive.
    """

    holding_arms: list[Arm]
    """
    The arms holding an object during the drive, even if they pick it up only just
    before it.
    """

    upright_threshold: float = AlignPlanes.threshold
    """
    Angle in radians between a held object's z-axis and the world's z-axis within which
    the object counts as upright.
    """

    def create_steps(self) -> list[StatechartNode]:
        free_arms = [arm for arm in self.robot.all_arms if arm not in self.holding_arms]
        park_free_arms = [ParkArmsAction(arms=free_arms)] if free_arms else []
        keep_upright = [
            KeepHeldBodyUpright(
                root_link=self.world.root,
                end_effector=arm.end_effector,
                threshold=self.upright_threshold,
            )
            for arm in self.holding_arms
        ]
        return [self.navigate, *park_free_arms, *keep_upright]
