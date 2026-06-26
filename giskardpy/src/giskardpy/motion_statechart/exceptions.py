from __future__ import annotations

from dataclasses import dataclass, field

from krrood.symbolic_math.symbolic_math import FloatVariable, Scalar
from krrood.exceptions import DataclassException
from semantic_digital_twin.collision_checking.collision_detector import ClosestPoints
from typing_extensions import TYPE_CHECKING, Type

if TYPE_CHECKING:
    from giskardpy.motion_statechart.graph_node import (
        MotionStatechartNode,
        TrinaryCondition,
    )


@dataclass
class CollisionViolatedError(DataclassException):
    violated_collisions: list[ClosestPoints]
    thresholds: list[float]

    def error_message(self) -> str:
        violations = "".join(
            f"{str(collision.body_a.name), str(collision.body_b.name)}: {collision.distance} < {threshold}\n"
            for collision, threshold in zip(self.violated_collisions, self.thresholds)
        )
        return f"Violated collision constraints: \n{violations}"

    def suggest_correction(self) -> str:
        return ""


@dataclass
class MotionStatechartError(DataclassException):
    """
    Base class for errors in the motion statechart.
    """


@dataclass
class NodeInitializationError(MotionStatechartError):
    node: MotionStatechartNode
    reason: str

    def error_message(self) -> str:
        return f'Failed to initialize Goal "{self.node.unique_name}". Reason: {self.reason}'

    def suggest_correction(self) -> str:
        return ""


@dataclass
class EmptyMotionStatechartError(MotionStatechartError):
    def error_message(self) -> str:
        return "MotionStatechart is empty."

    def suggest_correction(self) -> str:
        return ""


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
class InvalidConstraintExpressionShapeError(MotionStatechartError):
    actual_shape: list[int]

    def error_message(self) -> str:
        shape_str = " ".join(map(str, self.actual_shape))
        return f"Constraint expression must have shape (1, 1), has ({shape_str})."

    def suggest_correction(self) -> str:
        return ""


@dataclass
class NodeNotFoundError(MotionStatechartError):
    name: str

    def error_message(self) -> str:
        return f"Node '{self.name}' not found in MotionStatechart."

    def suggest_correction(self) -> str:
        return ""


@dataclass
class NotInMotionStatechartError(MotionStatechartError):
    name: str

    def error_message(self) -> str:
        return f"Operation can't be performed because node '{self.name}' does not belong to a MotionStatechart."

    def suggest_correction(self) -> str:
        return ""


@dataclass
class InvalidConditionError(MotionStatechartError):
    condition: TrinaryCondition
    new_expression: Scalar

    def reason(self) -> str:
        raise NotImplementedError

    def error_message(self) -> str:
        return f'Invalid {self.condition.kind.name} condition of node "{self.condition.owner.unique_name}": "{self.new_expression}". Reason: "{self.reason()}"'

    def suggest_correction(self) -> str:
        return ""


@dataclass
class InputNotExpressionError(InvalidConditionError):
    def reason(self) -> str:
        return "Input is not an expression."

    def suggest_correction(self) -> str:
        return "did you forget '.observation_variable'?"


@dataclass
class SelfInStartConditionError(InvalidConditionError):
    def reason(self) -> str:
        return "Start condition cannot contain the node itself."


@dataclass
class NonObservationVariableError(InvalidConditionError):
    non_observation_variable: FloatVariable

    def reason(self) -> str:
        return f'Contains "{self.non_observation_variable}", which is not an observation variable.'


@dataclass
class MissingContextExtensionError(MotionStatechartError):
    expected_extension: Type

    def error_message(self) -> str:
        return f'Missing context extension "{self.expected_extension.__name__}".'

    def suggest_correction(self) -> str:
        return ""


@dataclass
class DuplicateContextExtensionError(MotionStatechartError):
    extension_type: Type

    def error_message(self) -> str:
        return f"Extension of type {self.extension_type.__name__} already exists. You cannot add it twice."

    def suggest_correction(self) -> str:
        return ""
