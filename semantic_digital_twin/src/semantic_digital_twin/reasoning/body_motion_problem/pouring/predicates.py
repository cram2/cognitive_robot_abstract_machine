"""
Concrete BMP predicate implementations for the pouring domain (D_pour).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import List

from semantic_digital_twin.reasoning.body_motion_problem.predicates import (
    Causes,
    SatisfiesRequest,
)
from semantic_digital_twin.reasoning.body_motion_problem.pouring.effects import (
    PouringEffect,
)
from semantic_digital_twin.world_description.connections import PrismaticConnection


@dataclass
class PouringSatisfiesRequest(SatisfiesRequest):
    """
    Semantic correctness check for the D_pour TEE class.

    Returns ``True`` when the task type is ``"pour"`` and the effect is a
    :class:`PouringEffect`.
    """

    def __call__(self, *args, **kwargs) -> bool:
        if self.task.task_type != "pour":
            return False
        return isinstance(self.effect, PouringEffect)


@dataclass
class PouringCauses(Causes):
    """
    Causal sufficiency predicate for D_pour.

    Extends :class:`Causes` to handle the two coupled DOFs of the pouring domain:
    the tilt joint (``motion.actuator``) and the fill-level joint (``fill_connection``).
    Both trajectories are replayed during :meth:`_map_motion_to_effect` so that the
    fill-level effect can be verified against the world state.
    """

    fill_connection: PrismaticConnection
    """The virtual fill-level DOF coupled to the tilt actuator."""

    _fill_trajectory: List[float] = field(default_factory=list, init=False)

    def __call__(self, *args, **kwargs) -> bool:
        if self.effect.is_achieved():
            return False

        if (
            self.motion
            and self.motion.motion_model
            and len(self.motion.trajectory) == 0
        ):
            trajectory, _ = self.motion.motion_model.run(self.effect, self.environment)
            if trajectory:
                self.motion.trajectory = trajectory
                self._fill_trajectory = self.motion.motion_model.last_fill_trajectory

        return self._map_motion_to_effect()

    def _map_motion_to_effect(self) -> bool:
        initial_state = self.environment.state._data.copy()
        tilt_actuator = self.motion.actuator

        is_achieved_pre = self.effect.is_achieved()

        for tilt_pos, fill_pos in zip(self.motion.trajectory, self._fill_trajectory):
            self.environment.set_positions_1DOF_connection(
                {
                    tilt_actuator: float(tilt_pos),
                    self.fill_connection: float(fill_pos),
                }
            )

        is_achieved_post = self.effect.is_achieved()

        self.environment.state._data[:] = initial_state
        self.environment.notify_state_change()

        return (not is_achieved_pre) and is_achieved_post

    def replay(self, step_delay: float = 0.05) -> None:
        """
        Re-apply the computed trajectory to the world with a per-step delay.

        Leaves the world in the final state of the trajectory so that the result
        is visible in RViz. Call this after a successful :meth:`__call__`.

        :param step_delay: Seconds to sleep between steps (default 50 ms ≈ 20 fps).
        """
        tilt_actuator = self.motion.actuator
        for tilt_pos, fill_pos in zip(self.motion.trajectory, self._fill_trajectory):
            self.environment.set_positions_1DOF_connection(
                {
                    tilt_actuator: float(tilt_pos),
                    self.fill_connection: float(fill_pos),
                }
            )
            time.sleep(step_delay)
