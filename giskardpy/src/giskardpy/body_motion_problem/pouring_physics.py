from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from giskardpy.body_motion_problem.giskard_physics_model import GiskardPhysicsModel
from giskardpy.motion_statechart.graph_node import EndMotion
from giskardpy.motion_statechart.motion_statechart import MotionStatechart
from giskardpy.motion_statechart.tasks.pouring import PouringTask
from semantic_digital_twin.physics.equations.pouring_equations import PouringEquation
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import (
    ActiveConnection1DOF,
    LiquidConnection,
)
from semantic_digital_twin.world_description.effects import Effect
from semantic_digital_twin.world_description.motion import MotionTrajectory
from semantic_digital_twin.world_description.world_entity import Body

_DEFAULT_FILL_LEVEL_TOLERANCE: float = 0.05


@dataclass
class PouringMSCModel(GiskardPhysicsModel):
    """
    Physics model that drives a :class:`~giskardpy.motion_statechart.tasks.pouring.PouringTask`
    MSC to generate a tilt trajectory.

    The primary trajectory records the tilt joint positions at each control step.
    The fill-level trajectory is recorded alongside it by
    :meth:`_build_motion_trajectory`.
    """

    fill_equation: PouringEquation
    """Pouring ODE that couples tilt angle to fill-level dynamics."""

    fill_connection: LiquidConnection
    """Virtual DOF whose position encodes fill level in [0, 1]."""

    tilt_connection: ActiveConnection1DOF
    """The revolute joint that tilts the container."""

    root_link: Body
    """Root of the kinematic chain used to derive the cup tilt expression."""

    tip_link: Body
    """Tip of the kinematic chain (the cup body)."""

    initial_tilt: Optional[float] = field(default=None)
    """If set, the tilt connection is moved to this angle before the simulation begins."""

    def build_motion_statechart(self, effect: Effect, world: World) -> MotionStatechart:
        """
        Build an MSC with a :class:`~giskardpy.motion_statechart.tasks.pouring.PouringTask`
        targeting ``effect.goal_value`` as the fill level.
        """
        if self.initial_tilt is not None:
            world.set_positions_1DOF_connection(
                {self.tilt_connection: self.initial_tilt}
            )
        msc = MotionStatechart()
        pouring_task = PouringTask(
            fill_equation=self.fill_equation,
            fill_connection=self.fill_connection,
            root_link=self.root_link,
            tip_link=self.tip_link,
            goal_value=effect.goal_value,
            fill_level_tolerance=_DEFAULT_FILL_LEVEL_TOLERANCE,
        )
        msc.add_node(pouring_task)
        msc.add_node(EndMotion.when_true(pouring_task))
        return msc

    def _build_motion_trajectory(self, effect: Effect) -> MotionTrajectory:
        """
        :return: Trajectories for both the tilt and fill-level connections recorded
                 during the most recent :meth:`run` call.
        """
        tilt_positions = self._extract_dof_positions(
            self._recorded_trajectory, self.tilt_connection
        )
        fill_positions = self._extract_dof_positions(
            self._recorded_trajectory, self.fill_connection
        )
        return MotionTrajectory(
            {self.tilt_connection: tilt_positions, self.fill_connection: fill_positions}
        )

    def interaction_body(self):
        """
        :return: The cup body that the robot physically interacts with during pouring.
        """
        return self.tip_link
