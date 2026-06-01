from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

from semantic_digital_twin.world_description.effects import Effect
from semantic_digital_twin.world_description.world_entity import Body, Connection
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
    def run(self, effect: Effect, world: World) -> tuple[Optional[list[float]], bool]:
        """
        Simulate the motion and return the recorded actuator trajectory and whether
        the effect was achieved.

        :param effect: The desired effect to check against.
        :param world: The world to simulate in (state is reset after simulation).
        :return: (trajectory, achieved) — actuator positions and success flag.
        """

    def build_secondary_trajectories(
        self, effect: Effect
    ) -> list[tuple[Connection, list[float]]]:
        """
        Return secondary actuator trajectories recorded by the most recent run().

        :param effect: The effect passed to the most recent run() call.
        :return: List of (connection, positions) pairs parallel to the primary trajectory.
        """
        return []

    def interaction_body(self) -> Optional[Body]:
        """
        :return: The body the robot physically interacts with during this motion,
                 or ``None`` to fall back to ``motion.actuator.child``.
        """
        return None
