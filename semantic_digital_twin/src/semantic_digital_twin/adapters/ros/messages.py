from abc import ABC, abstractmethod
from dataclasses import dataclass
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
    """
    The name of the node that published this message
    """

    process_id: int
    """
    The id of the process that published this message
    """

    object_id: int
    """
    The id of the object in the process that issues this publishing call
    """

    @lru_cache(maxsize=None)
    def to_json(self) -> Dict[str, Any]:
        return {
            **super().to_json(),
            "node_name": self.node_name,
            "process_id": self.process_id,
            "object_id": self.object_id,
        }

    @classmethod
    def _from_json(cls, data: Dict[str, Any], **kwargs) -> Self:
        return cls(
            node_name=data["node_name"],
            process_id=data["process_id"],
            object_id=data["object_id"],
        )

    def __hash__(self):
        return hash((self.node_name, self.process_id, self.object_id))


@dataclass
class Message(SubclassJSONSerializer, ABC):

    meta_data: MetaData
    """
    Message origin meta data.
    """

    def to_json(self) -> Dict[str, Any]:
        return {
            **super().to_json(),
            "meta_data": self.meta_data.to_json(),
        }


@dataclass
class WorldStateUpdate(Message):
    """
    Class describing the updates to the free variables of a world state.
    """

    ids: List[UUID]
    """
    The ids of the changed free variables.
    """

    states: List[float]
    """
    The states of the changed free variables.
    """

    def to_json(self) -> Dict[str, Any]:
        return {
            **super().to_json(),
            "ids": to_json(self.ids),
            "states": list(self.states),
        }

    @classmethod
    def _from_json(cls, data: Dict[str, Any], **kwargs) -> Self:
        return cls(
            meta_data=MetaData.from_json(data["meta_data"], **kwargs),
            ids=from_json(data["ids"]),
            states=data["states"],
        )


@dataclass
class ModificationBlock(Message):
    """
    Message describing the modifications done to a world.
    """

    modifications: WorldModelModificationBlock
    """
    The modifications done to a world.
    """

    def to_json(self) -> Dict[str, Any]:
        return {
            **super().to_json(),
            "modifications": self.modifications.to_json(),
        }

    @classmethod
    def _from_json(cls, data: Dict[str, Any], **kwargs) -> Self:
        return cls(
            meta_data=MetaData.from_json(data["meta_data"], **kwargs),
            modifications=WorldModelModificationBlock.from_json(
                data["modifications"], **kwargs
            ),
        )


@dataclass
class LoadModel(Message):
    """
    Message for requesting the loading of a model identified by its primary key.
    """

    primary_key: int
    """
    The primary key identifying the model to be loaded.
    """

    def to_json(self) -> Dict[str, Any]:
        return {
            **super().to_json(),
            "primary_key": self.primary_key,
        }

    @classmethod
    def _from_json(cls, data: Dict[str, Any], **kwargs) -> Self:
        return cls(
            meta_data=MetaData.from_json(data["meta_data"], **kwargs),
            primary_key=data["primary_key"],
        )


@dataclass
class WorldModelSnapshot(SubclassJSONSerializer):
    """
    Snapshot containing the complete modification history and the latest world state.
    """

    modifications: List[WorldModelModificationBlock]
    """
    The ordered list of world model modification blocks.
    """

    ids: List[UUID]
    """
    The names of the free variables contained in the state snapshot.
    """

    states: List[float]
    """
    The values of the free variables contained in the state snapshot.
    """

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
        1. deserialize modifications from json and apply them to the world, block by block
        2. deserialize state from json and apply it to the world
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
        Apply the state contained in the json snapshot to the world.
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
