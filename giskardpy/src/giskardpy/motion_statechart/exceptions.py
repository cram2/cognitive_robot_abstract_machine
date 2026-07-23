from __future__ import annotations

from abc import ABC
from dataclasses import dataclass

from typing_extensions import TYPE_CHECKING, Type

from krrood.exceptions import DataclassException
from krrood.symbolic_math.symbolic_math import FloatVariable, Scalar
from semantic_digital_twin.collision_checking.collision_detector import ClosestPoints

if TYPE_CHECKING:
    from giskardpy.motion_statechart.graph_node import (
        MotionStatechartNode,
        TrinaryCondition,
    )
    from semantic_digital_twin.world_description.world_entity import (
        KinematicStructureEntity,
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
class MotionStatechartError(DataclassException, ABC):
    """
    Base class for errors in the motion statechart.
    """


@dataclass
class NodeInitializationError(MotionStatechartError, ABC):
    node: MotionStatechartNode


@dataclass
class EmptyMotionStatechartError(MotionStatechartError):
    def error_message(self) -> str:
        return "MotionStatechart is empty."

    def suggest_correction(self) -> str:
        return ""


@dataclass
class NodeAlreadyBelongsToDifferentNodeError(NodeInitializationError):
    new_node: MotionStatechartNode

    def error_message(self) -> str:
        if self.new_node.parent_node is not None:
            parent_name = self.new_node.parent_node.unique_name
        else:
            parent_name = "top level of motion statechart"
        return f'Node "{self.new_node.unique_name}" already belongs to "{parent_name}".'

    def suggest_correction(self) -> str:
        return "Create a copy of the node or remove it from its current parent first."


@dataclass
class EndMotionInGoalError(NodeInitializationError):

    def error_message(self) -> str:
        return "Goals are not allowed to have EndMotion as a child."

    def suggest_correction(self) -> str:
        return "Use a different node type or move the EndMotion node outside the Goal."


@dataclass
class UnexpectedWorldEntityCountError(NodeInitializationError):
    expected_count: int | str
    actual_count: int
    entity_type: Type | str | tuple[Type, ...]

    def error_message(self) -> str:
        return f"Expected {self.expected_count} entities of type {self.entity_type}, but found {self.actual_count}."

    def suggest_correction(self) -> str:
        return ""


@dataclass
class EmptyGoalStateError(NodeInitializationError):
    def error_message(self) -> str:
        return "Goal state is empty."

    def suggest_correction(self) -> str:
        return ""


@dataclass
class GoalPointsReferenceFrameMismatchError(NodeInitializationError):
    reference_frame_a: KinematicStructureEntity
    reference_frame_b: KinematicStructureEntity

    def error_message(self) -> str:
        return f"All goal points must have the same reference frame, but got {self.reference_frame_a} and {self.reference_frame_b}."

    def suggest_correction(self) -> str:
        return "Make sure all goal points have the same reference frame."


@dataclass
class InvalidConstraintExpressionShapeError(MotionStatechartError):
    actual_shape: list[int]

    def error_message(self) -> str:
        shape_str = " ".join(map(str, self.actual_shape))
        return f"Constraint expression must have shape (1, 1), has ({shape_str})."

    def suggest_correction(self) -> str:
        return "Ensure the expression evaluates to a (1, 1) scalar."


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

    def suggest_correction(self) -> str:
        return "Use an observation variable from a node instead, e.g. 'node.observation_variable'."


@dataclass
class ConditionScopeError(InvalidConditionError):
    """
    Raised when a condition references a node from a different scope level.

    A condition may only reference the owning node itself or nodes sharing the same
    parent.
    """

    dependency: MotionStatechartNode
    """
    The referenced node that lives in a different scope than the condition's owner.
    """

    def reason(self) -> str:
        owner_scope = self._scope_name(self.condition.owner)
        dependency_scope = self._scope_name(self.dependency)
        return (
            f'References "{self.dependency.unique_name}" from scope "{dependency_scope}", '
            f'but the condition\'s owner lives in scope "{owner_scope}". '
            f"Conditions may only reference the node itself or its siblings."
        )

    def suggest_correction(self) -> str:
        return "Reference a sibling of the owning node instead, e.g. the template node that contains the dependency."

    @staticmethod
    def _scope_name(node: MotionStatechartNode) -> str:
        parent_node = node.parent_node
        if parent_node is None:
            return "top level"
        return parent_node.unique_name


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


@dataclass
class PlotterNotConfiguredError(MotionStatechartError):
    """
    Raised when a plot is requested but the corresponding plotter was never configured.
    """

    plotter_name: str
    """
    The human-readable name of the plotter that is missing.
    """

    def error_message(self) -> str:
        return (
            f"Cannot plot: the {self.plotter_name} was not configured on the executor."
        )

    def suggest_correction(self) -> str:
        return f"Pass a {self.plotter_name} when constructing the executor."


@dataclass
class EmptyDebugExpressionTrajectoryError(MotionStatechartError):
    """
    Raised when a plot is requested but no debug expression samples were recorded.
    """

    def error_message(self) -> str:
        return "Cannot plot: no debug expression samples were recorded."

    def suggest_correction(self) -> str:
        return "Call tick() at least once before plotting, or configure debug expressions to record."
