"""
Physics model for articulated container manipulation.

Simulates the rigid-body kinematics of articulated containers (drawers, doors)
by executing a MotionStatechart against the world and recording the resulting
actuator trajectory.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

from giskardpy.motion_statechart.motion_statechart import MotionStatechart

from pycram.body_motion_problem.types import (
    Effect,
    PhysicsModel,
)
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.world_entity import Connection


class MissingMotionStatechartError(Exception):
    """
    Raised when RunMSCModel is asked to run without a configured MotionStatechart.
    """


@dataclass
class RunMSCModel(PhysicsModel):
    """
    Physics model that executes a MotionStatechart against a World.

    The MotionStatechart must be fully parameterized before being passed in.
    This model compiles it against the provided World, steps it tick-by-tick
    while recording the actuator position after each tick, and resets the World
    state before returning.
    """

    msc: MotionStatechart
    """The fully parameterized (but uncompiled) MotionStatechart to execute."""

    actuator: Connection
    """The connection whose position is recorded as the trajectory."""

    timeout: int = 500
    """Maximum number of ticks before aborting the rollout."""

    def run(self, effect: Effect, world: World) -> Tuple[Optional[List[float]], bool]:
        """
        Simulate the MotionStatechart and return the recorded actuator trajectory.

        :param effect: The desired effect used to check whether the simulation achieved it.
        :param world: The world to simulate in (state is reset before returning).
        :return: (trajectory, achieved) where trajectory is the list of actuator positions
                 recorded at each tick, and achieved indicates whether effect.is_achieved().
        """
        if self.msc is None:
            raise MissingMotionStatechartError(
                "RunMSCModel requires a MotionStatechart instance to run."
            )
        trajectory: List[float] = []
        achieved = self._run_msc(
            self.msc,
            effect,
            world,
            self.timeout,
            on_tick=lambda: trajectory.append(float(self.actuator.position)),
        )
        return trajectory, achieved
