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
    from giskardpy.motion_statechart.monitors.progress_monitors import ProgressStalled
    from semantic_digital_twin.world_description.world_entity import (
        KinematicStructureEntity,
    )


@dataclass
class CollisionViolatedError(DataclassException):
    """
    Raised when bodies came closer to each other than their collision threshold allows.
    """

    violated_collisions: list[ClosestPoints]
    """
    The closest points of every body pair that violated its threshold.
    """

    thresholds: list[float]
    """
    The minimum allowed distance of each violated collision, in the same order as
    :attr:`violated_collisions`.
    """

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
    """
    Base class for errors that a single node raises while it is set up or built.
    """

    node: MotionStatechartNode
    """
    The node that could not be initialized.
    """


@dataclass
class EmptyMotionStatechartError(MotionStatechartError):
    """
    Raised when a motion statechart without any node is executed.
    """

    def error_message(self) -> str:
        return "MotionStatechart is empty."

    def suggest_correction(self) -> str:
        return ""


@dataclass
class NodeAlreadyBelongsToDifferentNodeError(NodeInitializationError):
    """
    Raised when a node that is already part of the statechart is added a second time.
    """

    new_node: MotionStatechartNode
    """
    The node that was about to be added again.
    """

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
    """
    Raised when a node that ends the motion is added as a child of a goal.
    """

    def error_message(self) -> str:
        return "Goals are not allowed to have EndMotion as a child."

    def suggest_correction(self) -> str:
        return "Use a different node type or move the EndMotion node outside the Goal."


@dataclass
class UnexpectedWorldEntityCountError(NodeInitializationError):
    """
    Raised when a node searches the world for entities and finds a different number than
    it can work with.
    """

    expected_count: int | str
    """
    The number of entities the node needs, either as a number or as a textual
    description of the accepted range.
    """

    actual_count: int
    """
    The number of matching entities that were found in the world.
    """

    entity_type: Type | str | tuple[Type, ...]
    """
    The type of entity that was searched for.
    """

    def error_message(self) -> str:
        return f"Expected {self.expected_count} entities of type {self.entity_type}, but found {self.actual_count}."

    def suggest_correction(self) -> str:
        return ""


@dataclass
class EmptyGoalStateError(NodeInitializationError):
    """
    Raised when a node is given a goal state that names no degree of freedom.
    """

    def error_message(self) -> str:
        return "Goal state is empty."

    def suggest_correction(self) -> str:
        return ""


@dataclass
class EmptyDegreesOfFreedomError(NodeInitializationError):
    """
    Raised when a node is explicitly given an empty list of degrees of freedom.
    """

    def error_message(self) -> str:
        return "Degrees of freedom list is empty."

    def suggest_correction(self) -> str:
        return "Pass at least one degree of freedom, or leave it None to use every active degree of freedom in the world."


@dataclass
class GoalPointsReferenceFrameMismatchError(NodeInitializationError):
    """
    Raised when the goal points of a node are expressed in more than one reference
    frame.
    """

    reference_frame_a: KinematicStructureEntity
    """
    The reference frame of the first goal point.
    """

    reference_frame_b: KinematicStructureEntity
    """
    The reference frame that differs from :attr:`reference_frame_a`.
    """

    def error_message(self) -> str:
        return f"All goal points must have the same reference frame, but got {self.reference_frame_a} and {self.reference_frame_b}."

    def suggest_correction(self) -> str:
        return "Make sure all goal points have the same reference frame."


@dataclass
class NodeNotBuiltError(NodeInitializationError):
    """
    Raised when the build artifacts of a node are read before it has been built.
    """

    def error_message(self) -> str:
        return f'Node "{self.node.unique_name}" has not been built yet.'

    def suggest_correction(self) -> str:
        return "Compile the motion statechart before reading a node's build artifacts."


@dataclass
class MissingErrorSignalError(NodeInitializationError):
    """
    Raised when a converging task builds artifacts that carry no error signal.
    """

    def error_message(self) -> str:
        return (
            f'Converging task "{self.node.unique_name}" built artifacts without an error '
            f"signal, so there is nothing to compare against its threshold."
        )

    def suggest_correction(self) -> str:
        return (
            "Set NodeArtifacts.error in build_artifacts to the error the task's constraints "
            "drive to zero."
        )


@dataclass
class NoConvergingTaskError(NodeInitializationError):
    """
    Raised when a node is asked to watch the goal error of a node that contains no
    converging task.
    """

    monitored_node: MotionStatechartNode
    """
    The node that was supposed to be watched.
    """

    def error_message(self) -> str:
        return (
            f'"{self.monitored_node.unique_name}" contains no ConvergingTask, so it has '
            f"no goal error whose progress could be watched."
        )

    def suggest_correction(self) -> str:
        return (
            "Watch a task that converges towards a goal. Tasks that enforce an invariant, "
            "such as a velocity limit or a collision predicate, never converge."
        )


@dataclass
class CyclicNodeDependencyError(NodeInitializationError):
    """
    Raised when nodes depend on each other in a cycle, so no build order exists.
    """

    cycle: list[MotionStatechartNode]
    """
    The nodes forming the cycle, in the order in which they depend on each other.
    """

    def error_message(self) -> str:
        cycle_str = " -> ".join(node.unique_name for node in self.cycle)
        return f"Nodes depend on each other in a cycle: {cycle_str}."

    def suggest_correction(self) -> str:
        return "Break the cycle so the nodes can be expanded and built in some order."


@dataclass
class NoProgressError(MotionStatechartError):
    """
    Raised when the watched tasks stopped approaching their goal for too long.
    """

    progress_monitor: ProgressStalled
    """
    The monitor that detected the stall and knows which tasks are affected.
    """

    def error_message(self) -> str:
        stalled_tasks = self.progress_monitor.stalled_tasks
        names = ", ".join(task.unique_name for task in stalled_tasks)
        return (
            f"{names or self.progress_monitor.monitored_node.unique_name} stopped "
            f"approaching a goal for {self.progress_monitor.timeout} seconds."
        )

    def suggest_correction(self) -> str:
        return (
            "Check whether the goal is reachable, whether another task of equal or higher "
            "weight is opposing it, or whether the robot is at a joint limit."
        )


@dataclass
class InvalidConstraintExpressionShapeError(MotionStatechartError):
    """
    Raised when a constraint expression is not a scalar.
    """

    actual_shape: list[int]
    """
    The shape of the offending expression.
    """

    def error_message(self) -> str:
        shape_str = " ".join(map(str, self.actual_shape))
        return f"Constraint expression must have shape (1, 1), has ({shape_str})."

    def suggest_correction(self) -> str:
        return "Ensure the expression evaluates to a (1, 1) scalar."


@dataclass
class NodeNotFoundError(MotionStatechartError):
    """
    Raised when a node is looked up by name and the statechart has no such node.
    """

    name: str
    """
    The name that was looked up.
    """

    def error_message(self) -> str:
        return f"Node '{self.name}' not found in MotionStatechart."

    def suggest_correction(self) -> str:
        return ""


@dataclass
class NotInMotionStatechartError(MotionStatechartError):
    """
    Raised when an operation that requires a surrounding statechart is performed on a
    node that does not belong to one.
    """

    name: str
    """
    The name of the node that does not belong to a statechart.
    """

    def error_message(self) -> str:
        return f"Operation can't be performed because node '{self.name}' does not belong to a MotionStatechart."

    def suggest_correction(self) -> str:
        return ""


@dataclass
class InvalidConditionError(MotionStatechartError):
    """
    Base class for errors raised when a condition is set to an unusable expression.
    """

    condition: TrinaryCondition
    """
    The condition that was about to be set.
    """

    new_expression: Scalar
    """
    The rejected expression.
    """

    def reason(self) -> str:
        """
        Returns why the expression is not a valid condition.
        """
        raise NotImplementedError

    def error_message(self) -> str:
        return f'Invalid {self.condition.kind.name} condition of node "{self.condition.owner.unique_name}": "{self.new_expression}". Reason: "{self.reason()}"'

    def suggest_correction(self) -> str:
        return ""


@dataclass
class InputNotExpressionError(InvalidConditionError):
    """
    Raised when a condition is set to something that is not a symbolic expression.
    """

    def reason(self) -> str:
        return "Input is not an expression."

    def suggest_correction(self) -> str:
        return "did you forget '.observation_variable'?"


@dataclass
class SelfInStartConditionError(InvalidConditionError):
    """
    Raised when the start condition of a node references the node itself.
    """

    def reason(self) -> str:
        return "Start condition cannot contain the node itself."


@dataclass
class NonObservationVariableError(InvalidConditionError):
    """
    Raised when a condition contains a variable that is not the observation variable of
    a node.
    """

    non_observation_variable: FloatVariable
    """
    The variable in the condition that does not observe a node.
    """

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
        """
        Returns the name of the scope level that a node belongs to.

        Top-level nodes are called "top level".
        """
        parent_node = node.parent_node
        if parent_node is None:
            return "top level"
        return parent_node.unique_name


@dataclass
class TerminalNodeInConditionError(InvalidConditionError):
    """
    Raised when a condition references a node that ends the motion.

    Such a condition can never take effect, because the motion is already over by the
    time the referenced node is true.
    """

    terminal_node: MotionStatechartNode
    """
    The referenced node that ends the motion when its observation state turns true.
    """

    def reason(self) -> str:
        return (
            f'References "{self.terminal_node.unique_name}", which ends the motion when '
            "it turns true, so no transition can depend on it."
        )

    def suggest_correction(self) -> str:
        return "Reference the node that makes it true instead."


@dataclass
class MissingContextExtensionError(MotionStatechartError):
    """
    Raised when a context extension is requested that was never added to the context.
    """

    expected_extension: Type
    """
    The type of the requested extension.
    """

    def error_message(self) -> str:
        return f'Missing context extension "{self.expected_extension.__name__}".'

    def suggest_correction(self) -> str:
        return ""


@dataclass
class DuplicateContextExtensionError(MotionStatechartError):
    """
    Raised when an extension is added to a context that already holds one of that type.
    """

    extension_type: Type
    """
    The type of the extension that is already present.
    """

    def error_message(self) -> str:
        return f"Extension of type {self.extension_type.__name__} already exists. You cannot add it twice."

    def suggest_correction(self) -> str:
        return ""


@dataclass
class ActionClientTypeMismatchError(MotionStatechartError):
    """
    Raised when an action topic is requested with a different message type than the
    one its cached action client was created with.
    """

    action_topic: str
    """
    The action topic that was requested with two different message types.
    """

    existing_message_type: Type
    """
    The message type the cached action client for this topic was created with.
    """

    requested_message_type: Type
    """
    The message type that was requested for this topic instead.
    """

    def error_message(self) -> str:
        return (
            f'Action topic "{self.action_topic}" was already used with message type '
            f'"{self.existing_message_type.__name__}", but is now requested with '
            f'"{self.requested_message_type.__name__}".'
        )

    def suggest_correction(self) -> str:
        return "Use a unique action_topic per message type."


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
