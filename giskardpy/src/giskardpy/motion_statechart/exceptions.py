from __future__ import annotations
from dataclasses import dataclass, field

from krrood.adapters.json_serializer import JSON_TYPE_NAME, JSONSerializableTypeRegistry
from krrood.utils import get_full_class_name
from typing_extensions import Any, Dict, TYPE_CHECKING

from semantic_digital_twin.spatial_types import FloatVariable, Expression

if TYPE_CHECKING:
    from giskardpy.motion_statechart.graph_node import (
        MotionStatechartNode,
        TrinaryCondition,
    )


@dataclass
class MotionStatechartError(Exception):
    reason: str

    def __post_init__(self):
        super().__init__(self.reason)


@dataclass
class NodeInitializationError(MotionStatechartError):
    node: MotionStatechartNode
    reason: str = field(init=False)

    def __post_init__(self):
        self.reason = f'Failed to initialize Goal "{self.node.unique_name}". Reason: {self.reason}'
        super().__post_init__()


class EmptyMotionStatechartError(MotionStatechartError):
    reason: str = field(default="MotionStatechart is empty.", init=False)


@dataclass
class NodeAlreadyBelongsToDifferentNodeError(NodeInitializationError):
    new_node: MotionStatechartNode
    reason: str = field(init=False)

    def __post_init__(self):
        if self.new_node.parent_node is not None:
            parent_name = self.new_node.parent_node.unique_name
        else:
            parent_name = "top level of motion statechart"
        self.reason = (
            f'Node "{self.new_node.unique_name}" already belongs to "{parent_name}".'
        )
        super().__post_init__()


@dataclass
class EndMotionInGoalError(NodeInitializationError):
    reason: str = field(
        default="Goals are not allowed to have EndMotion as a child.", init=False
    )


@dataclass
class NodeNotFoundError(MotionStatechartError):
    name: str
    reason: str = field(init=False)

    def __post_init__(self):
        self.reason = f"Node '{self.name}' not found in MotionStatechart."
        super().__post_init__()


@dataclass
class NotInMotionStatechartError(MotionStatechartError):
    name: str
    reason: str = field(init=False)

    def __post_init__(self):
        self.reason = f"Operation can't be performed because node '{self.name}' does not belong to a MotionStatechart."
        super().__post_init__()


@dataclass
class InvalidConditionError(MotionStatechartError):
    condition: TrinaryCondition
    new_expression: Expression
    reason: str = field(init=False)

    def __post_init__(self):
        self.reason = f'Invalid {self.condition.kind.name} condition of node "{self.condition.owner.unique_name}": "{self.new_expression}". Reason: "{self.reason}"'
        super().__post_init__()


@dataclass
class InputNotExpressionError(InvalidConditionError):
    reason: str = field(
        default="Input is not an expression. Did you forget '.observation_variable'?",
        init=False,
    )


@dataclass
class SelfInStartConditionError(InvalidConditionError):
    reason: str = field(
        default=f"Start condition cannot contain the node itself.", init=False
    )


@dataclass
class NonObservationVariableError(InvalidConditionError):
    reason: str = field(init=False)
    non_observation_variable: FloatVariable

    def __post_init__(self):
        self.reason = f'Contains "{self.non_observation_variable}", which is not an observation variable.'
        super().__post_init__()


def serialize_exception(obj: Exception) -> Dict[str, Any]:
    return {
        JSON_TYPE_NAME: get_full_class_name(type(obj)),
        "value": str(obj),
    }


def deserialize_exception(data: Dict[str, Any], **kwargs) -> Exception:
    return Exception(data["value"])


JSONSerializableTypeRegistry().register(
    Exception, serialize_exception, deserialize_exception
)
