from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass, field
from typing import Optional

from giskardpy.executor import Executor, SimulationPacer
from giskardpy.motion_statechart.context import MotionStatechartContext
from giskardpy.motion_statechart.motion_statechart import MotionStatechart
from semantic_digital_twin.physics.physics_model import PhysicsModel
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import ActiveConnection1DOF
from semantic_digital_twin.world_description.effects import Effect
from semantic_digital_twin.world_description.motion import MotionTrajectory
from semantic_digital_twin.world_description.world_state import WorldStateTrajectory
from semantic_digital_twin.world_description.world_state_trajectory_plotter import (
    WorldStateTrajectoryPlotter,
)


@dataclass
class GiskardPhysicsModel(PhysicsModel):
    """
    Abstract base for physics models that simulate a MotionStatechart via Giskard's Executor.

    Subclasses define the MSC to run (:meth:`build_motion_statechart`) and which
    connections to record (:meth:`_build_motion_trajectory`).

    The executor runs at maximum speed (no real-time pacing) inside a
    ``world.reset_state_context()``, recording the full ``WorldStateTrajectory`` for
    later DOF extraction.
    """

    timeout: int = field(default=500, kw_only=True)
    """Maximum number of control ticks before stopping the simulation."""

    _recorded_trajectory: Optional[WorldStateTrajectory] = field(
        init=False, repr=False, default=None
    )

    @abstractmethod
    def build_motion_statechart(self, effect: Effect, world: World) -> MotionStatechart:
        """
        Build the MotionStatechart that drives the simulation for the given effect.

        :param effect: The desired effect this model should achieve.
        :param world: The world in which the simulation runs.
        :return: A compiled-ready MotionStatechart.
        """

    def run(self, effect: Effect, world: World) -> MotionTrajectory:
        """
        Simulate the MSC and return trajectories for all tracked connections.

        Runs inside ``world.reset_state_context()``, so all world state changes are
        discarded on exit. The recorded world-state trajectory persists for use by
        :meth:`_build_motion_trajectory`.

        :param effect: Desired effect passed to :meth:`build_motion_statechart`.
        :param world: World to simulate in.
        :return: Recorded position sequences for all connections involved in the motion.
        """
        with world.reset_state_context():
            msc = self.build_motion_statechart(effect, world)
            plotter = WorldStateTrajectoryPlotter()
            executor = Executor(
                context=MotionStatechartContext(world=world),
                pacer=SimulationPacer(real_time_factor=None),
                trajectory_plotter=plotter,
            )
            executor.compile(motion_statechart=msc)
            try:
                executor.tick_until_end(timeout=self.timeout)
            except TimeoutError:
                pass
            self._recorded_trajectory = plotter.world_state_trajectory

        return self._build_motion_trajectory(effect)

    @abstractmethod
    def _build_motion_trajectory(self, effect: Effect) -> MotionTrajectory:
        """
        Construct the :class:`~semantic_digital_twin.world_description.motion.MotionTrajectory`
        from the most recently recorded world-state trajectory.

        :param effect: The effect that was passed to the most recent :meth:`run` call.
        :return: Trajectory covering all connections relevant to this model.
        """

    def _extract_dof_positions(
        self, trajectory: WorldStateTrajectory, connection: ActiveConnection1DOF
    ) -> list[float]:
        """
        Extract and convert positions for a specific DOF from the recorded trajectory.

        Applies the connection's multiplier and offset to convert raw DOF state values
        to the same scale as ``connection.position``, ensuring compatibility with
        :func:`~semantic_digital_twin.world.World.set_positions_1DOF_connection`.

        The first entry in the trajectory (recorded during :meth:`Executor.compile` before
        any control commands have been applied) is excluded so the result only contains
        post-tick positions, matching the behaviour of manual per-tick recording.

        :param trajectory: Recorded world state trajectory.
        :param connection: The connection whose positions to extract.
        :return: List of connection-space positions at each recorded timestep.
        """
        raw = trajectory.get_dof_positions(connection.raw_dof.id)
        if raw is None:
            return []
        return (raw[1:] * connection.multiplier + connection.offset).tolist()
