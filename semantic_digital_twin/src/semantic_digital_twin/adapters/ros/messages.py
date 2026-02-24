import uuid
from abc import ABC
from dataclasses import dataclass, field
from functools import lru_cache
from uuid import UUID

from typing_extensions import Dict, Any, Self, List

from krrood.adapters.json_serializer import SubclassJSONSerializer, to_json, from_json
from ...world import World

from ...world_description.world_modification import (
    WorldModelModificationBlock,
)


@dataclass
class MetaData(SubclassJSONSerializer):
    """
    Class for data describing the origin of a message.
    """

    node_name: str
    """The name of the node that published this message."""

    process_id: int
    """The id of the process that published this message."""

    world_id: UUID = field(default_factory=uuid.uuid4)
    """The id of the origin world. This is used to identify messages that were published by the same publisher."""

    @lru_cache(maxsize=None)
    def to_json(self) -> Dict[str, Any]:
        return {
            **super().to_json(),
            "node_name": self.node_name,
            "process_id": self.process_id,
            "world_id": to_json(self.world_id),
        }

    @classmethod
    def _from_json(cls, data: Dict[str, Any], **kwargs) -> Self:
        return cls(
            node_name=data["node_name"],
            process_id=data["process_id"],
            world_id=from_json(data["world_id"]),
        )

    def __hash__(self):
        return hash((self.node_name, self.process_id, self.world_id))


@dataclass
class Message(ABC):
    """
    Abstract base class for all messages.
    """

    meta_data: MetaData
    """Message origin meta data."""

    publication_event_id: UUID = field(default_factory=uuid.uuid4, kw_only=True)
    """UUID uniquely identifying the event (world update / state update / ...) that originated this message.

    Recipients can use this UUID in responses to refer to this event.
    Allows the publication/subscription mechanism to track what messages have been received and acknowledged.
    """


@dataclass
class Acknowledgment:
    """
    Message acknowledging receipt of a published event.

    :param publication_event_id: The UUID of the publication event being acknowledged.
    :param node_meta_data: The metadata identifying the acknowledging node.
    """

    publication_event_id: UUID
    """The UUID of the publication event being acknowledged."""

    node_meta_data: MetaData
    """The metadata identifying the acknowledging node."""


@dataclass
class WorldStateUpdate(Message):
    """
    Class describing the updates to the free variables of a world state.
    """

    ids: List[UUID]
    """The ids of the changed free variables."""

    states: List[float]
    """The states of the changed free variables."""


@dataclass
class ModificationBlock(Message):
    """
    Message describing the modifications done to a world.
    """

    modifications: WorldModelModificationBlock
    """The modifications done to a world."""


@dataclass
class LoadModel(Message):
    """
    Message for requesting the loading of a model identified by its primary key.
    """

    primary_key: int
    """The primary key identifying the model to be loaded."""


@dataclass
class WorldModelSnapshot(SubclassJSONSerializer):
    """
    Snapshot containing the complete modification history and the latest world state.
    """

    modifications: List[WorldModelModificationBlock]
    """The ordered list of world model modification blocks."""

    ids: List[UUID]
    """The names of the free variables contained in the state snapshot."""

    states: List[float]
    """The values of the free variables contained in the state snapshot."""

    def to_json(self) -> Dict[str, Any]:
        return {
            **super().to_json(),
            "modifications": to_json(self.modifications),
            "state": {
                "ids": to_json(self.ids),
                "states": list(self.states),
            },
        }

    @classmethod
    def _from_json(cls, data: Dict[str, Any], **kwargs) -> Self:
        state = data.get("state", {})
        return cls(
            modifications=[
                WorldModelModificationBlock.from_json(m, **kwargs)
                for m in data.get("modifications", [])
            ],
            ids=from_json(state["ids"]),
            states=state.get("states", []),
        )

    @staticmethod
    def apply_to_json_snapshot_to_world(
        world: World, json_data: Dict[str, Any], **kwargs
    ):
        """
        Deserialize modifications and state from JSON and apply them to the world.

        1. Deserialize modifications from JSON and apply them to the world, block by block.
        2. Deserialize state from JSON and apply it to the world.

        :param world: The world to apply the snapshot to.
        :param json_data: The JSON data containing the snapshot.
        """
        with world.modify_world():
            for modification in json_data.get("modifications", []):
                WorldModelModificationBlock.apply_from_json(
                    world, modification, **kwargs
                )

        state = json_data.get("state", {})
        ids = from_json(state["ids"])
        states = state.get("states", [])
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
            world.state.data[0, i] = s
        world.notify_state_change()
