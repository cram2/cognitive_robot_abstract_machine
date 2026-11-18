"""
The messages the synchronizers of worlds exchange over ROS 2 topics.
"""

from dataclasses import dataclass, fields
from enum import StrEnum
from uuid import UUID

from typing_extensions import Any, Self, get_args, get_type_hints

from krrood.adapters.json_serializer import (
    JSONData,
    SubclassJSONSerializer,
    to_json,
    from_json,
)
from krrood_synchronization.messages import Message, MessageField, StreamPosition
from krrood.patterns.classproperty import classproperty
from semantic_digital_twin.adapters.ros.exceptions import (
    WorldUpdateReferencesUnknownEntityError,
)
from semantic_digital_twin.adapters.world_entity_kwargs_tracker import (
    WorldEntityWithIDKwargsTracker,
)
from semantic_digital_twin.exceptions import WorldEntityWithIDNotInKwargs
from semantic_digital_twin.world import World

from semantic_digital_twin.world_description.world_modification import (
    WorldModelModificationBlock,
)


@dataclass
class WorldStateUpdate(Message):
    """
    Class describing the updates to the free variables of a world state.
    """

    ids: list[UUID]
    """The ids of the changed free variables."""

    states: list[float]
    """The states of the changed free variables."""


@dataclass
class ModificationBlock(Message):
    """
    Message describing the modifications done to a world.
    """

    modifications: WorldModelModificationBlock
    """The modifications done to a world."""


@dataclass
class WorldUpdate(Message):
    """
    Combined model and state update published on a single ordered ROS topic.

    Sending both types of change on the same topic gives FIFO delivery guarantees:
    a model update published before a state update is always received before that
    state update, eliminating the cross-topic ordering race present when model and
    state travel on separate topics.

    Either field may be ``None`` when only one type of change is being communicated.
    """

    modification_block: ModificationBlock | None = None
    """The model modification to apply, if any."""

    state_update: WorldStateUpdate | None = None
    """The state values to apply, if any."""

    @classproperty
    def modification_block_json_key(cls) -> str:
        """
        The key under which a serialized world update holds its model modification: the
        name of the one field that holds a :class:`ModificationBlock`.
        """
        type_hints = get_type_hints(cls)
        [key] = [
            field_.name
            for field_ in fields(cls)
            if ModificationBlock in get_args(type_hints[field_.name])
        ]
        return key


@dataclass
class LoadModel(Message):
    """
    Message for requesting the loading of a model identified by its primary key.
    """

    primary_key: int
    """The primary key identifying the model to be loaded."""


@dataclass(kw_only=True)
class ReceivedMessage:
    """
    A message as it arrived: its position read, everything else still json.

    The rest refers to entities of the world it is applied to, so it is built only when
    the message is applied, against the world as it is then.
    """

    content: JSONData
    """
    The serialized message.
    """

    position: StreamPosition
    """
    The position of the message in the stream of its publisher.
    """

    @classmethod
    def from_json(cls, *, content: JSONData) -> Self:
        """
        :param content: A serialized message.
        :return: The message with its position read.
        """
        return cls(
            content=content,
            position=from_json(content[MessageField.POSITION]),
        )

    @property
    def carries_model_modification(self) -> bool:
        """
        Whether the message is a world update that modifies the model.
        """
        return self.content.get(WorldUpdate.modification_block_json_key) is not None

    def to_message(self, *, world: World) -> Message:
        """
        :param world: The world whose entities the message refers to.
        :return: The message built against the world as it is now.
        :raises WorldUpdateReferencesUnknownEntityError: If the message refers to an
            entity the world does not have.
        """
        with world._world_lock:
            tracker = WorldEntityWithIDKwargsTracker.from_world(world)
            try:
                return from_json(self.content, **tracker.create_kwargs())
            except WorldEntityWithIDNotInKwargs as unknown_entity:
                raise WorldUpdateReferencesUnknownEntityError(
                    publisher=self.position.origin,
                    entity_id=unknown_entity.key,
                    entity_name=unknown_entity.world_entity_name,
                ) from unknown_entity


class SnapshotField(StrEnum):
    """
    The parts a serialized world snapshot is made of.
    """

    MODIFICATIONS = "modifications"
    """The modification blocks the world was built from."""

    STATE = "state"
    """The free variables and their values."""

    IDS = "ids"
    """The ids of the free variables, inside the state."""

    STATES = "states"
    """The values of the free variables, inside the state."""


@dataclass
class WorldModelSnapshot(SubclassJSONSerializer):
    """
    Snapshot containing the complete modification history and the latest world state.
    """

    modifications: list[WorldModelModificationBlock]
    """The ordered list of world model modification blocks."""

    ids: list[UUID]
    """The names of the free variables contained in the state snapshot."""

    states: list[float]
    """The values of the free variables contained in the state snapshot."""

    def to_json(self, **kwargs) -> dict[str, Any]:
        return {
            **super().to_json(**kwargs),
            SnapshotField.MODIFICATIONS: to_json(self.modifications, **kwargs),
            SnapshotField.STATE: {
                SnapshotField.IDS: to_json(self.ids, **kwargs),
                SnapshotField.STATES: list(self.states),
            },
        }

    @classmethod
    def _from_json(cls, *, data: dict[str, Any], **kwargs) -> Self:
        state = data.get(SnapshotField.STATE, {})
        return cls(
            modifications=[
                from_json(m, **kwargs)
                for m in data.get(SnapshotField.MODIFICATIONS, [])
            ],
            ids=from_json(state[SnapshotField.IDS]),
            states=state.get(SnapshotField.STATES, []),
        )

    @staticmethod
    def apply_to_json_snapshot_to_world(
        world: World, json_data: dict[str, Any], **kwargs
    ):
        """
        Deserialize modifications and state from JSON and apply them to the world.

        1. Deserialize modifications from JSON and apply them to the world, block by block.
        2. Deserialize state from JSON and apply it to the world.

        :param world: The world to apply the snapshot to.
        :param json_data: The JSON data containing the snapshot.
        """
        with world.modify_world():
            for modification in json_data.get(SnapshotField.MODIFICATIONS, []):
                from_json(modification, **kwargs).apply(world)

        state = json_data.get(SnapshotField.STATE, {})
        ids = from_json(state[SnapshotField.IDS])
        states = state.get(SnapshotField.STATES, [])
        WorldModelSnapshot._apply_json_state(world, ids, states)

    @staticmethod
    def _apply_json_state(world: World, ids: list[float], states: list[UUID]):
        """
        Apply the state contained in the JSON snapshot to the world.

        :param world: The world whose state to update.
        :param ids: The ids of the free variables.
        :param states: The values of the free variables.
        """
        if not (ids or states):
            return
        indices = [world.state._index.get(_id) for _id in ids]
        assign_pairs = [(i, float(s)) for i, s in zip(indices, states) if i is not None]
        if not assign_pairs:
            return
        for i, s in assign_pairs:
            world.state._data[0, i] = s
        world.notify_state_change()
