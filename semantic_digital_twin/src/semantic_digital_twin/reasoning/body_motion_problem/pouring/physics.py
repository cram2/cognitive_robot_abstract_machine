"""
Physics model for the pouring domain (D_pour).

Implements Φ_pour using a MotionStatechart with a PouringTask that integrates
the Torricelli ODE and controls the tilt joint via QP.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from giskardpy.executor import Executor, SimulationPacer
from giskardpy.motion_statechart.context import MotionStatechartContext
from giskardpy.motion_statechart.graph_node import EndMotion
from giskardpy.motion_statechart.motion_statechart import MotionStatechart
from giskardpy.motion_statechart.tasks.pouring import PouringTask

from semantic_digital_twin.reasoning.body_motion_problem.types import (
    Effect,
    PhysicsModel,
)
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import (
    PrismaticConnection,
    RevoluteConnection,
)


@dataclass
class PouringMSCModel(PhysicsModel):
    """
    Concrete physics model Φ_pour: execute a PouringTask MotionStatechart against a World.

    The MotionStatechart contains a single :class:`PouringTask` that integrates the
    Torricelli ODE and drives the tilt joint via QP. This model compiles the MSC,
    ticks it until EndMotion, records both the tilt and fill trajectories, and
    resets the World state before returning.

    ODE: ``d(fill)/dt = -k * max(0, sin(θ(t) − θ_threshold)) * sqrt(fill)``
    """

    tilt_connection: RevoluteConnection
    """Revolute DOF representing the container's tilt angle (the actuated joint)."""

    fill_connection: PrismaticConnection
    """Virtual DOF whose position encodes the current fill level."""

    theta_max: float = math.pi / 3
    """Maximum tilt angle in radians."""

    n_ramp: int = 20
    """Steps used in the predictive lookahead to estimate the ramp-down trajectory."""

    k: float = 1.0
    """Outflow rate constant."""

    theta_threshold: float = 0.3
    """Tilt onset angle in radians; pouring begins above this."""

    dt: float = 0.1
    """ODE integration timestep in seconds."""

    timeout: int = 500
    """Maximum number of ticks before aborting the rollout."""

    last_fill_trajectory: List[float] = field(default_factory=list, init=False)
    """Fill-level trajectory from the most recent :meth:`run` call."""

    def run(self, effect: Effect, world: World) -> Tuple[Optional[List[float]], bool]:
        """
        Simulate pouring via the PouringTask MotionStatechart and return the tilt trajectory.

        :param effect: Desired fill-level effect; provides goal_value and tolerance.
        :param world: World whose state is simulated and reset on return.
        :return: ``(tilt_trajectory, achieved)`` — tilt samples at each tick and success flag.
        """
        msc = self._build_msc(effect)
        context = MotionStatechartContext(world=world)
        executor = Executor(
            context=context,
            pacer=SimulationPacer(real_time_factor=1.0),
        )
        executor.compile(motion_statechart=msc)

        initial_state = world.state._data.copy()
        tilt_trajectory: List[float] = []
        fill_trajectory: List[float] = []
        try:
            for _ in range(self.timeout):
                executor.tick()
                tilt_trajectory.append(float(self.tilt_connection.position))
                fill_trajectory.append(float(self.fill_connection.position))
                if msc.is_end_motion():
                    break

            achieved = effect.is_achieved()
        finally:
            world.state._data[:] = initial_state
            world.notify_state_change()

        self.last_fill_trajectory = fill_trajectory
        return tilt_trajectory, achieved

    def _build_msc(self, effect: Effect) -> MotionStatechart:
        """
        Build the MotionStatechart for a single pouring manoeuvre.

        :param effect: Effect providing goal_value and tolerance for PouringTask.
        :return: Compiled-ready MotionStatechart with PouringTask and EndMotion.
        """
        msc = MotionStatechart()
        task = PouringTask(
            tilt_connection=self.tilt_connection,
            fill_connection=self.fill_connection,
            goal_value=effect.goal_value,
            tolerance=effect.tolerance,
            theta_max=self.theta_max,
            n_ramp=self.n_ramp,
            theta_threshold=self.theta_threshold,
            k=self.k,
            dt=self.dt,
        )
        msc.add_node(task)
        msc.add_node(EndMotion.when_true(task))
        return msc
