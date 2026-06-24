from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from giskardpy.executor import Executor, SimulationPacer
from giskardpy.motion_statechart.context import MotionStatechartContext
from giskardpy.motion_statechart.graph_node import EndMotion
from giskardpy.motion_statechart.motion_statechart import MotionStatechart
from giskardpy.motion_statechart.tasks.pouring import PouringTask
from semantic_digital_twin.physics.physics_model import PhysicsModel
from semantic_digital_twin.physics.pouring_equations import PouringEquation
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import (
    ActiveConnection1DOF,
    LiquidConnection,
)
from semantic_digital_twin.world_description.effects import Effect
from semantic_digital_twin.world_description.world_entity import Body, Connection

_DEFAULT_FILL_LEVEL_TOLERANCE: float = 0.05
_DEFAULT_TIMEOUT: int = 500


@dataclass
class PouringMSCModel(PhysicsModel):
    """
    Physics model that drives a PouringTask MotionStatechart to generate a tilt trajectory.

    Runs the statechart in simulation, recording both the tilt-joint position and the
    fill-level at each control step. The world state is restored via
    ``world.reset_state_context()`` after each call to :meth:`run`.
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

    _tilt_trajectory: list[float] = field(default_factory=list, init=False, repr=False)
    _fill_trajectory: list[float] = field(default_factory=list, init=False, repr=False)

    def run(self, effect: Effect, world: World) -> tuple[Optional[list[float]], bool]:
        """
        Simulate the pouring motion and record tilt and fill-level trajectories.

        :param effect: Desired pouring effect; its ``goal_value`` is used as the fill target.
        :param world: World to simulate in; state is restored on exit.
        :return: (tilt_trajectory, achieved) where tilt_trajectory holds the tilt-joint
                 position recorded at each control step.
        """
        with world.reset_state_context():
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

            executor = Executor(
                context=MotionStatechartContext(world=world),
                pacer=SimulationPacer(real_time_factor=None),
            )
            executor.compile(motion_statechart=msc)

            tilt_trajectory: list[float] = []
            fill_trajectory: list[float] = []

            try:
                for _ in range(_DEFAULT_TIMEOUT):
                    executor.tick()
                    tilt_trajectory.append(float(self.tilt_connection.position))
                    fill_trajectory.append(float(self.fill_connection.position))
                    if msc.is_end_motion():
                        break
            finally:
                executor._set_velocity_acceleration_jerk_to_zero()
                msc.cleanup_nodes(context=executor.context)
                executor.context.cleanup()

            self._tilt_trajectory = tilt_trajectory
            self._fill_trajectory = fill_trajectory

        return tilt_trajectory, bool(tilt_trajectory)

    def build_secondary_trajectories(
        self, effect: Effect
    ) -> list[tuple[Connection, list[float]]]:
        """
        :return: The fill-level trajectory recorded during the most recent :meth:`run` call.
        """
        return [(self.fill_connection, self._fill_trajectory)]

    def interaction_body(self) -> Optional[Body]:
        """
        :return: The cup body that the robot physically interacts with during pouring.
        """
        return self.tip_link
