from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Type, Self

from typing_extensions import Dict, List, TYPE_CHECKING

from krrood.adapters.json_serializer import (
    DataclassJSONSerializer,
    from_json,
    SubclassJSONSerializer,
    to_json,
)
from semantic_digital_twin.adapters.world_entity_kwargs_tracker import (
    WorldEntityWithIDKwargsTracker,
)
from semantic_digital_twin.datastructures.definitions import JointStateType
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.world_description.connections import ActiveConnection1DOF

if TYPE_CHECKING:
    from semantic_digital_twin.robots.abstract_robot import AbstractRobot
    from semantic_digital_twin.world import World


@dataclass
class JointState(SubclassJSONSerializer):
    """
    Represents a specific joint state by mapping a set of connections to target values.
    """

    connections: List[ActiveConnection1DOF] = field(default_factory=list)
    """
    All connections in this state
    """

    target_values: List[float] = field(default_factory=list)
    """
    All target values in this state, order has to correspond to the order of connections
    """

    state_type: JointStateType = field(default=None)
    """
    A type to better describe this state
    """

    name: PrefixedName = field(default=PrefixedName("JointState"))
    """
    A Name for this JointState
    """

    _robot: AbstractRobot = field(init=False, default=None)

    def __len__(self):
        return len(self.connections)

    def __hash__(self):
        """
        Returns the hash of the joint state, which is based on the joint state name.
        This allows for proper comparison and storage in sets or dictionaries.
        """
        return hash((self.connections, self.target_values))

    def assign_to_robot(self, robot: AbstractRobot):
        """
        Assigns the joint state to the given robot. This method ensures that the joint state is only assigned
        to one robot at a time, and raises an error if it is already assigned to another robot.
        """
        if self._robot is not None and self._robot != robot:
            raise ValueError(
                f"Joint State {self.name} is already assigned to another robot: {self._robot.name}."
            )
        if self._robot is not None:
            return
        self._robot = robot

    def items(self):
        return zip(self.connections, self.target_values)

    @classmethod
    def from_str_dict(cls, mapping: Dict[str, float], world: World):
        connections = [world.get_connection_by_name(name) for name in mapping.keys()]
        return cls(connections, list(mapping.values()))

    @classmethod
    def from_mapping(
        cls,
        mapping: Dict[ActiveConnection1DOF, float],
        state_type: JointStateType = None,
        name: PrefixedName = PrefixedName("JointState"),
    ):
        return cls(
            state_type=state_type,
            name=name,
            connections=list(mapping.keys()),
            target_values=list(mapping.values()),
        )

    @classmethod
    def from_lists(cls, connections: List[ActiveConnection1DOF], targets: List[float]):
        return cls(connections, targets)

    def to_json(self) -> Dict[str, Any]:
        return {
            **super().to_json(),
            "connections": [
                to_json(connection.name) for connection in self.connections
            ],
            "target_values": self.target_values,
            "joint_state_type": to_json(self.state_type),
            "name": to_json(self.name),
        }

    @classmethod
    def _from_json(cls, data: Dict[str, Any], **kwargs) -> Self:
        tracker = WorldEntityWithIDKwargsTracker.from_kwargs(kwargs)
        world = tracker._world
        if world:
            connections = [
                world.get_connection_by_name(from_json(name, **kwargs))
                for name in data["connections"]
            ]
        else:
            connections = from_json(data["connections"])
        target_values = from_json(data["target_values"])
        state_type = from_json(data["joint_state_type"])
        name = from_json(data["name"])
        return cls(connections, target_values, state_type=state_type, name=name)


GripperState = JointState
ArmState = JointState
TorsoState = JointState
