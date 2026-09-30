from __future__ import annotations

from dataclasses import dataclass
from uuid import UUID

import numpy as np
from typing_extensions import Dict, Self

from semantic_digital_twin.world import World


@dataclass
class WorldSnapshot:
    """
    The model version and degree of freedom states of a world at one point in time, to
    which the world can be returned.
    """

    world: World
    """
    The world whose state is captured.
    """

    model_version: int
    """
    The model version of :attr:`world` when the snapshot was captured.
    """

    degree_of_freedom_states: Dict[UUID, np.ndarray]
    """
    The position, velocity, acceleration and jerk of every degree of freedom of
    :attr:`world` when the snapshot was captured.
    """

    @classmethod
    def capture(cls, world: World) -> Self:
        """
        :param world: The world to capture.
        :return: A snapshot of the current model version and state of ``world``.
        """
        return cls(
            world=world,
            model_version=world.get_world_model_manager().version,
            degree_of_freedom_states=dict(world.state.items()),
        )

    def restore(self) -> None:
        """
        Roll back every model modification made since the snapshot was captured, then
        restore the captured degree of freedom states.
        """
        self.world.rollback_to_version(self.model_version)
        for degree_of_freedom_id, values in self.degree_of_freedom_states.items():
            self.world.state[degree_of_freedom_id] = values
        self.world.notify_state_change()
