"""
Physics models for the liquid pouring domain.

Simulates pouring by integrating a fill-level differential equation coupled
with QP-based tilt control via a MotionStatechart.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from giskardpy.motion_statechart.graph_node import EndMotion
from giskardpy.motion_statechart.monitors.monitors import LocalMinimumReached
from giskardpy.motion_statechart.motion_statechart import MotionStatechart
from giskardpy.motion_statechart.tasks.pouring import PouringTask

from pycram.body_motion_problem.types import (
    Effect,
    PhysicsModel,
)
from semantic_digital_twin.physics.pouring_equations import PouringEquation
from semantic_digital_twin.semantic_annotations.mixins import HasFillLevel
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import Connection
from semantic_digital_twin.world_description.world_entity import Body


@dataclass
class PouringMSCModel(PhysicsModel):
    """
    Physics model for single-source pouring.

    Simulates tilting a cup to pour its contents by running a MotionStatechart
    that integrates the fill-level ODE and drives tilt via QP. Records the
    tilt-angle and fill-level trajectories, and resets the world state on return.
    """

    fill_equation: PouringEquation
    """Pouring ODE — owns the fill connection and k."""

    tilt_connection: Connection = field(kw_only=True)
    """Revolute connection whose angle drives pouring; used for trajectory recording and initial setup."""

    root_link: Body = field(kw_only=True)
    """Root of the kinematic chain for the FK-derived tilt expression in PouringTask."""

    tip_link: Body = field(kw_only=True)
    """Tip of the kinematic chain (the cup body)."""

    reference_velocity: float = field(default=0.05)
    """Desired rate of decrease for the normalized fill level."""

    initial_tilt: float = field(default=0.1)
    """Initial tilt angle in radians to help the gradient-driven task."""

    timeout: int = field(default=900)
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
        fill_connection = self.fill_equation.fill_connection
        tilt_trajectory: List[float] = []
        fill_trajectory: List[float] = []

        def on_tick() -> None:
            tilt_trajectory.append(float(self.tilt_connection.position))
            fill_trajectory.append(float(fill_connection.position))

        achieved = self._run_msc(
            msc,
            effect,
            world,
            self.timeout,
            on_tick=on_tick,
            setup=lambda: world.set_positions_1DOF_connection(
                {self.tilt_connection: self.initial_tilt}
            ),
        )
        self.last_fill_trajectory = fill_trajectory
        return tilt_trajectory, achieved

    def build_secondary_trajectories(
        self, effect: Effect
    ) -> List[Tuple[Connection, List[float]]]:
        """
        Return the fill-level trajectory recorded by the most recent run().

        :param effect: Effect whose target_object provides the fill connection.
        :return: Single-entry list of (fill_connection, fill_level_positions).
        """
        return [(effect.target_object.fill_connection, self.last_fill_trajectory)]

    def _build_msc(self, effect: Effect) -> MotionStatechart:
        """
        Build the MotionStatechart for a single pouring manoeuvre.

        :param effect: Effect providing goal_value and tolerance for PouringTask.
        :return: Compiled-ready MotionStatechart with PouringTask and EndMotion.
        """
        msc = MotionStatechart()
        task = PouringTask(
            fill_equation=self.fill_equation,
            root_link=self.root_link,
            tip_link=self.tip_link,
            goal_value=effect.goal_value,
            tolerance=effect.tolerance,
            reference_velocity=self.reference_velocity,
        )
        msc.add_node(task)
        msc.add_node(local_min := LocalMinimumReached())
        msc.add_node(EndMotion.when_true(local_min))
        return msc
