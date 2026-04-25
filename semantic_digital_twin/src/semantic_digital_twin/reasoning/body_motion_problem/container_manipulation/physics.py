"""
Physics model implementation for articulated container manipulation (D_artic).

Implements Φ_artic using a MotionStatechart (MSC) to simulate rigid-body
kinematics of articulated containers (drawers, doors) and record the
resulting actuator trajectory.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

from giskardpy.executor import Executor, SimulationPacer
from giskardpy.motion_statechart.context import MotionStatechartContext
from giskardpy.motion_statechart.motion_statechart import MotionStatechart

from semantic_digital_twin.reasoning.body_motion_problem.types import (
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
    Concrete physics model Φ_artic: execute a MotionStatechart against a World.

    The MotionStatechart must be fully parameterized before being passed in.
    This model binds it to the provided World via Executor.compile, rolls it
    out introspectively (recording the actuator trajectory after each tick),
    and resets the World state before returning.
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

        context = MotionStatechartContext(world=world)
        executor = Executor(
            context=context,
            # pacer=SimulationPacer(real_time_factor=1.0),
        )
        executor.compile(motion_statechart=self.msc)

        trajectory: List[float] = []
        with world.reset_state_context():
            for _ in range(self.timeout):
                executor.tick()
                trajectory.append(float(self.actuator.position))
                if self.msc.is_end_motion():
                    break

            achieved = effect.is_achieved()

        return trajectory, achieved
