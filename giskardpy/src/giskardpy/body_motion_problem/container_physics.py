from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from giskardpy.executor import Executor, SimulationPacer
from giskardpy.motion_statechart.context import MotionStatechartContext
from giskardpy.motion_statechart.motion_statechart import MotionStatechart
from semantic_digital_twin.physics.physics_model import PhysicsModel
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import ActiveConnection1DOF
from semantic_digital_twin.world_description.effects import Effect


@dataclass
class RunMSCModel(PhysicsModel):
    """
    Physics model that executes a pre-built MotionStatechart to generate an actuator trajectory.

    Runs the statechart in simulation without real-time pacing, records the actuator
    position after each control step, and restores the world state via
    ``world.reset_state_context()`` afterwards.
    """

    motion_statechart: MotionStatechart
    """The statechart to execute during simulation."""

    actuator: ActiveConnection1DOF
    """The connection whose position is sampled after each control tick."""

    timeout: int = field(default=500)
    """Maximum number of control ticks before stopping the simulation."""

    def run(self, effect: Effect, world: World) -> tuple[Optional[list[float]], bool]:
        """
        Execute the statechart and record the actuator-position trajectory.

        The world state is restored to its pre-simulation values after this call.

        :param effect: Desired effect; passed for interface consistency but not directly used.
        :param world: World in which to simulate.
        :return: (trajectory, achieved) where trajectory is the list of recorded actuator positions.
        """
        with world.reset_state_context():
            trajectory: list[float] = []
            executor = Executor(
                context=MotionStatechartContext(world=world),
                pacer=SimulationPacer(real_time_factor=None),
            )
            executor.compile(motion_statechart=self.motion_statechart)
            try:
                for _ in range(self.timeout):
                    executor.tick()
                    trajectory.append(float(self.actuator.position))
                    if self.motion_statechart.is_end_motion():
                        break
            finally:
                executor._set_velocity_acceleration_jerk_to_zero()
                self.motion_statechart.cleanup_nodes(context=executor.context)
                executor.context.cleanup()
        return trajectory, bool(trajectory)
