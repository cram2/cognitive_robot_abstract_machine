from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

from semantic_digital_twin.world_description.effects import Effect
from semantic_digital_twin.world_description.motion import MotionTrajectory
from semantic_digital_twin.world_description.world_entity import Body
from semantic_digital_twin.world import World


class PhysicsModel(ABC):
    """
    Abstract interface for simulating the causal effect of a motion.

    Implementations define how a motion trajectory changes the world state
    within a specific physical regime (e.g., rigid-body kinematics, fluid-flow
    dynamics). Concrete subclasses live in the package that owns the simulation
    machinery (e.g., giskardpy).
    """

    @abstractmethod
    def run(self, effect: Effect, world: World) -> MotionTrajectory:
        """
        Simulate the motion and return trajectories for all tracked connections.

        :param effect: The desired effect to simulate towards.
        :param world: The world to simulate in (state is reset after simulation).
        :return: Recorded position sequences for all connections involved in the motion.
        """

    def interaction_body(self) -> Optional[Body]:
        """
        :return: The body the robot physically interacts with during this motion,
                 or ``None`` to fall back to ``motion.actuator.child``.
        """
        return None
