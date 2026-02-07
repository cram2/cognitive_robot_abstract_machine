from __future__ import annotations

from dataclasses import dataclass, field

from typing_extensions import TYPE_CHECKING

from krrood.symbolic_math.symbolic_math import FloatVariable, Scalar
from krrood.utils import DataclassException

if TYPE_CHECKING:
    from giskardpy.motion_statechart.graph_node import (
        MotionStatechartNode,
        TrinaryCondition,
    )


@dataclass
class MotionStatechartError(DataclassException):
    message: str = field(init=False)


@dataclass
class NodeInitializationError(MotionStatechartError):
    node: MotionStatechartNode
    reason: str

    def __post_init__(self):
        self.message = f'Failed to initialize Goal "{self.node.unique_name}". Reason: {self.reason}'


@dataclass
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


@dataclass
class EndMotionInGoalError(NodeInitializationError):
    reason: str = field(
        default="Goals are not allowed to have EndMotion as a child.", init=False
    )


@dataclass
class NodeNotFoundError(MotionStatechartError):
    name: str

    def __post_init__(self):
        self.message = f"Node '{self.name}' not found in MotionStatechart."


@dataclass
class NotInMotionStatechartError(MotionStatechartError):
    name: str

    def __post_init__(self):
        self.message = f"Operation can't be performed because node '{self.name}' does not belong to a MotionStatechart."


@dataclass
class InvalidConditionError(MotionStatechartError):
    condition: TrinaryCondition
    new_expression: Scalar

    def __post_init__(self):
        self.message = f'Invalid {self.condition.kind.name} condition of node "{self.condition.owner.unique_name}": "{self.new_expression}". Reason: "{self.message}"'


@dataclass
class InputNotExpressionError(InvalidConditionError):
    message: str = field(
        default="Input is not an expression. Did you forget '.observation_variable'?",
        init=False,
    )


@dataclass
class SelfInStartConditionError(InvalidConditionError):
    message: str = field(
        default=f"Start condition cannot contain the node itself.", init=False
    )


@dataclass
class NonObservationVariableError(InvalidConditionError):
    non_observation_variable: FloatVariable

    def __post_init__(self):
        self.message = f'Contains "{self.non_observation_variable}", which is not an observation variable.'


@dataclass
class MissingContextExtensionError(MotionStatechartError):
    expected_extension: type

    def __post_init__(self):
        self.message = (
            f'Missing context extension "{self.expected_extension.__name__}".'
        )
