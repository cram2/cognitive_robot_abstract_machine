from __future__ import annotations

import time
from dataclasses import dataclass
from time import sleep
from typing import Any, Dict, Optional

from krrood.adapters.json_serializer import from_json
from semantic_digital_twin.adapters.ros.messages import StreamPosition
from semantic_digital_twin.adapters.ros.world_synchronizer import (
    ModelReloadSynchronizer,
    WorldSynchronizer,
)

from giskardpy.middleware.ros2.exceptions import GiskardWorldUpdateNotReceivedError

# %% the world of Giskard


@dataclass
class IncomingWorldUpdates:
    """
    The world updates other processes sent to Giskard.

    The updates arrive on a ros thread but are applied on the thread that owns the
    world, so that the world cannot change in the middle of a control cycle. This class
    answers the questions the loops of Giskard have about them: what may be applied right
    now, whether the structure of the world is about to change, and whether the world
    caught up with a given publisher.
    """

    world_synchronizer: WorldSynchronizer
    """
    Delivers the model and state updates of other processes.
    """

    model_reload_synchronizer: ModelReloadSynchronizer | None = None
    """
    Delivers requests to replace the whole world model.

    ``None`` when no database is configured, in which case a reload cannot be received.
    """

    @property
    def has_pending_model_change(self) -> bool:
        """
        Whether a change of the structure of the world is waiting to be applied.

        Anything compiled against the current structure becomes invalid once it is, so a
        running motion has to end before it can be applied.
        """
        if self.world_synchronizer.has_buffered_model_modification:
            return True
        if self.model_reload_synchronizer is None:
            return False
        return self.model_reload_synchronizer.has_pending_reload

    def apply_state_updates(self) -> None:
        """
        Apply the state that arrived before the next model change.
        """
        self.world_synchronizer.apply_missed_state_updates()

    def apply_all(self) -> None:
        """
        Apply everything that was received.

        Only safe while nothing is compiled against the structure of the world.
        """
        self.world_synchronizer.apply_missed_messages()
        if self.model_reload_synchronizer is None:
            return
        self.model_reload_synchronizer.apply_pending_reload()

    def has_applied(self, position: StreamPosition) -> bool:
        """
        Whether everything the publisher of ``position`` sent up to it was applied to
        the world.
        """
        return self.world_synchronizer.has_applied(position)


# %% the world of a client


@dataclass
class ClientWorldUpdates:
    """
    Keeps the world of a client in step with Giskard around a goal.

    A goal is built against a world the client may just have changed, and executing it
    changes that world again, so both sides have to be told what to catch up with.
    """

    world_synchronizer: WorldSynchronizer
    """
    Publishes the changes of the client's world and receives those of Giskard.
    """

    timeout: float = 30.0
    """
    Seconds to wait for the changes Giskard made during a goal.
    """

    poll_interval: float = 0.01
    """
    Seconds between two looks at what the world caught up with.
    """

    def required_position(self) -> Optional[StreamPosition]:
        """
        The position a goal built on this world requires, or ``None`` if this world
        never published a change.
        """
        if self.world_synchronizer.published_sequence_number == 0:
            return None
        return self.world_synchronizer.latest_published_position

    def wait_for_the_changes_of_a_goal(self, result: Dict[str, Any]) -> None:
        """
        Wait until the changes Giskard made while executing a goal reached this world.

        :raises GiskardWorldUpdateNotReceivedError: If they do not arrive within
            ``timeout``.
        """
        published_position = result.get("published_position")
        if published_position is None:
            return
        position = from_json(published_position)
        deadline = time.monotonic() + self.timeout
        while not self.world_synchronizer.has_applied(position):
            if time.monotonic() >= deadline:
                raise GiskardWorldUpdateNotReceivedError(
                    awaited_sequence_number=position.sequence_number,
                    timeout=self.timeout,
                )
            sleep(self.poll_interval)
