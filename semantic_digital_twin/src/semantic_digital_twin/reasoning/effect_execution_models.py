from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, List

from giskardpy.executor import Executor
from giskardpy.motion_statechart.motion_statechart import MotionStatechart

from ..semantic_annotations.task_effect_motion import Effect, Motion
from ..world import World
from ..world_description.connections import Connection


class MissingMotionStatechartError(Exception):
    """
    Raised when a model is asked to run without a configured MotionStatechart.
    """


# TODO: Abstract parent class to define other executors than for motion statecharts
@dataclass
class RunMSCModel:
    """
    Execute an already-constructed (but uncompiled) MotionStatechart against a World.

    The MotionStatechart must be fully parameterized (nodes and edges added) before being passed
    into this model. This model only binds it to the provided World via Executor.compile(...),
    rolls it out introspectively (recording the Effect's observed value after each tick), and resets
    the World's state before returning.
    """

    msc: MotionStatechart
    actuator: Connection
    timeout: int = 500

    def run(self, effect: Effect, world: World) -> Tuple[Optional[Motion], bool]:
        if self.msc is None:
            raise MissingMotionStatechartError(
                "RunMSCModel requires a MotionStatechart instance to run."
            )

        executor = Executor(world=world)
        executor.compile(motion_statechart=self.msc)

        # Introspective rollout: mutate world during rollout but always reset before returning
        initial_state_data = world.state.data.copy()
        try:
            trajectory: List[float] = []
            for _ in range(self.timeout):
                executor.tick()
                trajectory.append(effect.current_value)
                if self.msc.is_end_motion():
                    break

            motion = Motion(trajectory=trajectory, actuator=self.actuator)
            return motion, effect.is_achieved()
        finally:
            world.state.data = initial_state_data
            world.notify_state_change()
