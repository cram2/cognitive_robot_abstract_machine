"""
Giskardpy-backed abstract physics model with MotionStatechart execution support.
"""

from __future__ import annotations

from typing import Callable, Optional

from giskardpy.executor import Executor
from giskardpy.motion_statechart.context import MotionStatechartContext
from giskardpy.motion_statechart.motion_statechart import MotionStatechart

from semantic_digital_twin.physics.physics_model import PhysicsModel
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.effects import Effect


class MotionStatechartPhysicsModel(PhysicsModel):
    """
    Abstract physics model that executes MotionStatecharts against a World.

    Extends the semdt :class:`PhysicsModel` interface with a protected helper
    that drives a MotionStatechart tick-by-tick inside a world reset context.
    Concrete subclasses implement :meth:`run` using :meth:`_run_motion_statechart`.
    """

    def _run_motion_statechart(
        self,
        motion_statechart: MotionStatechart,
        effect: Effect,
        world: World,
        timeout: int,
        on_tick: Callable[[], None],
        setup: Optional[Callable[[], None]] = None,
    ) -> bool:
        """
        Execute a MotionStatechart tick-by-tick and return whether the effect was achieved.

        :param motion_statechart: The compiled-ready MotionStatechart to execute.
        :param effect: The desired effect; checked after the loop exits.
        :param world: The world to simulate in (state is reset before returning).
        :param timeout: Maximum number of ticks before aborting.
        :param on_tick: Called after each tick to record trajectory samples.
        :param setup: Optional callable run once inside the reset context before ticking.
        :return: Whether effect.is_achieved() after the simulation.
        """
        context = MotionStatechartContext(world=world)
        executor = Executor(context=context)
        executor.compile(motion_statechart=motion_statechart)
        with world.reset_state_context():
            if setup is not None:
                setup()
            for _ in range(timeout):
                executor.tick()
                on_tick()
                if motion_statechart.is_end_motion():
                    break
            achieved = effect.is_achieved()
        return achieved
