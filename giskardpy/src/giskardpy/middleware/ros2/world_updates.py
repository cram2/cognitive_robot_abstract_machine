from __future__ import annotations

import time
from dataclasses import dataclass, field
from time import sleep
from typing_extensions import Any

from krrood.adapters.json_serializer import from_json
from semantic_digital_twin.adapters.ros.messages import StreamPosition
from semantic_digital_twin.adapters.ros.world_synchronizer import (
    ModelReloadSynchronizer,
    PublicationProgress,
    WorldSynchronizer,
)

from giskardpy.middleware.ros2.exceptions import GiskardWorldUpdateNotReceivedError
from giskardpy.middleware.ros2.feedback_publisher import MotionStatechartPayloadKey

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


@dataclass
class OutgoingWorldUpdates:
    """
    The changes of Giskard's world that the running goal published to other processes.

    A client reading its world while or after Giskard runs its goal first has to catch
    up with them.
    """

    world_synchronizer: PublicationProgress
    """
    Publishes the changes of Giskard's world.
    """

    _published_sequence_number_before_goal: int = field(init=False, default=0)
    """
    How far Giskard's world had published when the running goal was accepted.
    """

    def start_goal(self) -> None:
        """
        Count only what is published from now on as a change of the running goal.
        """
        self._published_sequence_number_before_goal = (
            self.world_synchronizer.published_sequence_number
        )

    def published_position_of_goal(self) -> StreamPosition | None:
        """
        The position Giskard's world published up to while the goal was running, or
        ``None`` if the goal published nothing.
        """
        if (
            self.world_synchronizer.published_sequence_number
            == self._published_sequence_number_before_goal
        ):
            return None
        return self.world_synchronizer.latest_published_position


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

    def required_position(self) -> StreamPosition | None:
        """
        The position a goal built on this world requires, or ``None`` if this world
        never published a change.
        """
        if self.world_synchronizer.published_sequence_number == 0:
            return None
        return self.world_synchronizer.latest_published_position

    def wait_for_the_changes_of_a_goal(self, result: dict[str, Any]) -> None:
        """
        Wait until the changes Giskard made while executing a goal reached this world.

        :param result: The result of the goal.
        :raises GiskardWorldUpdateNotReceivedError: If they do not arrive within
            ``timeout``.
        """
        waiting_since = time.monotonic()
        while not self.has_applied_the_changes_reported_in(result, waiting_since):
            sleep(self.poll_interval)

    def has_applied_the_changes_reported_in(
        self, report: dict[str, Any], waiting_since: float
    ) -> bool:
        """
        :param report: A feedback or the result of a goal.
        :param waiting_since: The :func:`time.monotonic` time this world started waiting
            for the changes.
        :return: Whether the changes Giskard reports to have published while executing
            the goal reached this world.
        :raises GiskardWorldUpdateNotReceivedError: If they did not, and ``timeout``
            passed since `waiting_since`.
        """
        published_position = report.get(MotionStatechartPayloadKey.PUBLISHED_POSITION)
        if published_position is None:
            return True
        position = from_json(published_position)
        if self.world_synchronizer.has_applied(position):
            return True
        if time.monotonic() - waiting_since >= self.timeout:
            raise GiskardWorldUpdateNotReceivedError(
                awaited_sequence_number=position.sequence_number,
                timeout=self.timeout,
            )
        return False
