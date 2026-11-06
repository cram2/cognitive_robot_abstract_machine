from __future__ import annotations

from abc import ABC
from dataclasses import dataclass

from typing_extensions import TYPE_CHECKING

from krrood.adapters.exceptions import JSONSerializationError
from krrood.exceptions import DataclassException
from krrood.symbolic_math.symbolic_math import FloatVariable, Scalar

if TYPE_CHECKING:
    from cramph.composites import ChildChooser, CompositeNodeChoosingItsChild
    from cramph.context import ContextExtension
    from cramph.data_types import TransitionKind
    from cramph.node import NodeStateVariable, StatechartNode, TransitionCondition


@dataclass
class StatechartError(DataclassException, ABC):
    """
    Base class for errors in a statechart.
    """


@dataclass
class NodeInitializationError(StatechartError, ABC):
    """
    Base class for errors that a single node raises while it is set up or built.
    """

    node: StatechartNode
    """
    The node that could not be initialized.
    """


@dataclass
class EmptyStatechartError(StatechartError):
    """
    Raised when a statechart without any node is executed.
    """

    def error_message(self) -> str:
        return "Statechart is empty."

    def suggest_correction(self) -> str:
        return ""


@dataclass
class NodeAlreadyBelongsToDifferentNodeError(NodeInitializationError):
    """
    Raised when a node that is already part of the statechart is added a second time.
    """

    new_node: StatechartNode
    """
    The node that was about to be added again.
    """

    def error_message(self) -> str:
        if self.new_node.parent_node is not None:
            parent_name = self.new_node.parent_node.unique_name
        else:
            parent_name = "top level of statechart"
        return f'Node "{self.new_node.unique_name}" already belongs to "{parent_name}".'

    def suggest_correction(self) -> str:
        return "Create a copy of the node or remove it from its current parent first."


@dataclass
class EndInCompositeNodeError(NodeInitializationError):
    """
    Raised when a node that ends the statechart is added as a child of a composite
    statechart node.
    """

    def error_message(self) -> str:
        return "Composite nodes are not allowed to have EndStatechart as a child."

    def suggest_correction(self) -> str:
        return "Use a different node type or move the EndStatechart node outside the CompositeNode."


@dataclass
class ChildCannotFailError(NodeInitializationError):
    """
    Raised when a template that only moves on once a child failed is handed a child that
    cannot fail.
    """

    child: StatechartNode
    """
    The child that has no way to fail on its own.
    """

    def error_message(self) -> str:
        return (
            f'"{self.child.unique_name}" cannot fail, but "{self.node.unique_name}" '
            f"only moves on once it failed."
        )

    def suggest_correction(self) -> str:
        return (
            "Give the child a fail condition, for example one reading a monitor that "
            "decides when to give up on it."
        )


@dataclass
class ChildTransitionAlreadyWiredError(NodeInitializationError):
    """
    Raised when a child handed to a template already has one of its life cycle
    transitions wired, which is the template's to decide.
    """

    child: StatechartNode
    """
    The child whose transition was already wired.
    """

    transition_kind: TransitionKind
    """
    The transition that was already wired.
    """

    def error_message(self) -> str:
        return (
            f"The {self.transition_kind.name.lower()} condition of "
            f'"{self.child.unique_name}" was wired before it was passed to '
            f'"{self.node.unique_name}", which decides it.'
        )

    def suggest_correction(self) -> str:
        return (
            "Leave the child's life cycle to the template, or express the condition "
            "where the template cannot: a fail condition stays with the node itself."
        )


@dataclass
class TransitionHasNoOutcomeError(StatechartError):
    """
    Raised when the outcome of a transition that does not end a node is asked for.
    """

    transition_kind: TransitionKind
    """
    The transition whose outcome was asked for.
    """

    def error_message(self) -> str:
        return (
            f"The {self.transition_kind.name.lower()} transition does not end a node, "
            f"so it has no outcome."
        )

    def suggest_correction(self) -> str:
        return "Only ask the transitions that end a node for their outcome."


@dataclass
class UnknownTransitionKindError(StatechartError):
    """
    Raised when a node is asked for a transition of a kind it does not have.
    """

    transition_kind: TransitionKind
    """
    The kind that was asked for.
    """

    def error_message(self) -> str:
        return f"{self.transition_kind!r} is not a kind of transition."

    def suggest_correction(self) -> str:
        return "Pass a member of TransitionKind."


@dataclass
class CompositeNodeWithoutChildrenError(NodeInitializationError):
    """
    Raised when a composite node that exists to hold child nodes is built without any.
    """

    def error_message(self) -> str:
        return f'CompositeNode "{self.node.unique_name}" was given no child nodes.'

    def suggest_correction(self) -> str:
        return "Pass at least one node to it, or leave it out entirely."


@dataclass
class NodeNotBuiltError(NodeInitializationError):
    """
    Raised when the build artifacts of a node are read before it has been built.
    """

    def error_message(self) -> str:
        return f'Node "{self.node.unique_name}" has not been built yet.'

    def suggest_correction(self) -> str:
        return "Compile the statechart before reading a node's build artifacts."


@dataclass
class CyclicNodeDependencyError(NodeInitializationError):
    """
    Raised when nodes depend on each other in a cycle, so no build order exists.
    """

    cycle: list[StatechartNode]
    """
    The nodes forming the cycle, in the order in which they depend on each other.
    """

    def error_message(self) -> str:
        cycle = " -> ".join(node.unique_name for node in self.cycle)
        return f"Nodes depend on each other in a cycle: {cycle}."

    def suggest_correction(self) -> str:
        return "Break the cycle so the nodes can be expanded and built in some order."


@dataclass
class TickDoesNotSettleError(StatechartError):
    """
    Raised when passes through a statechart return it to a state it already had within
    one tick, or still change it after the most passes one tick may take.
    """

    pass_limit: int
    """
    The most passes that may change the statechart within one tick.
    """

    passes_taken: int
    """
    The passes that changed the statechart before the tick was stopped.
    """

    unsettled_nodes: list[StatechartNode]
    """
    The nodes whose state the last pass changed.
    """

    def error_message(self) -> str:
        names = ", ".join(node.unique_name for node in self.unsettled_nodes)
        return (
            f"The statechart did not settle within one tick after "
            f"{self.passes_taken} of at most {self.pass_limit} passes, at {names}."
        )

    def suggest_correction(self) -> str:
        return (
            "Check whether these nodes read each other's observations in a way no state "
            "satisfies, for example each observing True while the other does not."
        )


@dataclass
class NodeNotFoundError(StatechartError):
    """
    Raised when a node is looked up by name and the statechart has no such node.
    """

    name: str
    """
    The name that was looked up.
    """

    def error_message(self) -> str:
        return f"Node '{self.name}' not found in Statechart."

    def suggest_correction(self) -> str:
        return ""


@dataclass
class UnknownConditionVariableError(StatechartError):
    """
    Raised when a rendered condition names a variable that no node offers.
    """

    variable_name: str
    """
    The name the rendered condition uses for the variable.
    """

    def error_message(self) -> str:
        return f'The condition names "{self.variable_name}", which no node offers.'

    def suggest_correction(self) -> str:
        return (
            "Name a predicate of a node, e.g. 'observes_true', 'last_observed_true' or "
            "'is_succeeded'."
        )


@dataclass
class UnsupportedConditionSyntaxError(StatechartError):
    """
    Raised when a rendered condition contains syntax that has no meaning as a condition.
    """

    unsupported_part: str
    """
    The part of the rendered condition that is not supported.
    """

    def error_message(self) -> str:
        return (
            f'The condition contains "{self.unsupported_part}", which is not supported.'
        )

    def suggest_correction(self) -> str:
        return (
            "Write the condition from quoted node predicates, True and False, combined "
            "with 'and', 'or' and 'not'."
        )


@dataclass
class NotInStatechartError(StatechartError):
    """
    Raised when an operation that requires a surrounding statechart is performed on a
    node that does not belong to one.
    """

    name: str
    """
    The name of the node that does not belong to a statechart.
    """

    def error_message(self) -> str:
        return f"Operation can't be performed because node '{self.name}' does not belong to a Statechart."

    def suggest_correction(self) -> str:
        return ""


@dataclass
class InvalidConditionError(StatechartError):
    """
    Base class for errors raised when a condition is set to an unusable expression.
    """

    condition: TransitionCondition
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
        return "did you forget '.observes_true'?"


@dataclass
class SelfInStartConditionError(InvalidConditionError):
    """
    Raised when the start condition of a node references the node itself.
    """

    def reason(self) -> str:
        return "Start condition cannot contain the node itself."


@dataclass
class UnsupportedConditionVariableError(InvalidConditionError):
    """
    Raised when a condition contains a variable that is not a two-valued predicate of a
    node, such as a node's observation, which may be unknown.
    """

    unsupported_variable: FloatVariable
    """
    The variable in the condition that a transition may not read.
    """

    def reason(self) -> str:
        return (
            f'Contains "{self.unsupported_variable}", which a transition may not read.'
        )

    def suggest_correction(self) -> str:
        return (
            "Use a predicate of a node, e.g. 'node.observes_true', "
            "'node.last_observed_true' or 'node.is_failed'."
        )


@dataclass
class ConditionScopeError(InvalidConditionError):
    """
    Raised when a condition references a node from a different scope level.

    A condition may only reference the owning node itself or nodes sharing the same
    parent.
    """

    dependency: StatechartNode
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
    def _scope_name(node: StatechartNode) -> str:
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
    Raised when a condition references a node that ends the statechart.

    Such a condition can never take effect, because the statechart is already over by
    the time the referenced node is true.
    """

    terminal_node: StatechartNode
    """
    The referenced node that ends the statechart when its observation state turns true.
    """

    def reason(self) -> str:
        return (
            f'References "{self.terminal_node.unique_name}", which ends the statechart when '
            "it turns true, so no transition can depend on it."
        )

    def suggest_correction(self) -> str:
        return "Reference the node that makes it true instead."


@dataclass
class MissingContextExtensionError(StatechartError):
    """
    Raised when a context extension is requested that was never added to the context.
    """

    expected_extension: type
    """
    The type of the requested extension.
    """

    def error_message(self) -> str:
        return f'Missing context extension "{self.expected_extension.__name__}".'

    def suggest_correction(self) -> str:
        return ""


@dataclass
class DuplicateContextExtensionError(StatechartError):
    """
    Raised when an extension is added to a context that already holds one of that type.
    """

    extension_type: type
    """
    The type of the extension that is already present.
    """

    def error_message(self) -> str:
        return f"Extension of type {self.extension_type.__name__} already exists. You cannot add it twice."

    def suggest_correction(self) -> str:
        return ""


@dataclass
class AmbiguousContextExtensionError(StatechartError):
    """
    Raised when a context extension is requested by a type that several extensions of
    the context are instances of.
    """

    requested_type: type[ContextExtension]
    """
    The type of the requested extension.
    """

    matching_extensions: list[ContextExtension]
    """
    The extensions that are instances of :attr:`requested_type`.
    """

    def error_message(self) -> str:
        matching_types = ", ".join(
            type(extension).__name__ for extension in self.matching_extensions
        )
        return (
            f'Several context extensions are a "{self.requested_type.__name__}": '
            f"{matching_types}."
        )

    def suggest_correction(self) -> str:
        return "Request one of them by its own type, or add only one of them."


@dataclass
class NodesMissingContextExtensionsError(StatechartError):
    """
    Raised when a statechart compiles nodes that require context extensions its context
    does not hold.
    """

    nodes_by_missing_extension: dict[type[ContextExtension], list[StatechartNode]]
    """
    The nodes requiring each missing extension type.
    """

    def error_message(self) -> str:
        lines = [
            f'"{extension_type.__name__}", required by '
            + ", ".join(f'"{node.unique_name}"' for node in nodes)
            for extension_type, nodes in self.nodes_by_missing_extension.items()
        ]
        return "Missing context extensions: " + "; ".join(lines) + "."

    def suggest_correction(self) -> str:
        return (
            "Add the extensions to the context, or the executor extensions providing "
            "them to the executor, before compiling."
        )


@dataclass
class NonPositiveRealTimeFactorError(StatechartError):
    """
    Raised when a simulation is configured to run at a non positive speed.
    """

    real_time_factor: float
    """
    The rejected factor.
    """

    def error_message(self) -> str:
        return f"A real time factor of {self.real_time_factor} would never advance the simulation."

    def suggest_correction(self) -> str:
        return "Use a positive factor, or NoPacing to run as fast as possible."


@dataclass
class TickDurationUnknownError(StatechartError):
    """
    Raised when something needs to know how long a tick lasts, but the context does not
    say.
    """

    def error_message(self) -> str:
        return "The statechart context does not know how many seconds one tick lasts."

    def suggest_correction(self) -> str:
        return "Pass a tick_duration to the StatechartContext."


@dataclass
class ConflictingTickDurationError(StatechartError):
    """
    Raised when a context is told a tick lasts a different time than it already knows.
    """

    tick_duration: float
    """
    How many seconds one tick lasts according to the context.
    """

    requested_tick_duration: float
    """
    How many seconds one tick was requested to last.
    """

    def error_message(self) -> str:
        return (
            f"One tick already lasts {self.tick_duration} seconds, it cannot also last "
            f"{self.requested_tick_duration} seconds."
        )

    def suggest_correction(self) -> str:
        return "Configure everything that sets the tick duration with the same value."


@dataclass
class TickLimitReachedError(StatechartError):
    """
    Raised when a statechart has not ended after the most ticks it was given.
    """

    tick_limit: int
    """
    How many ticks the statechart was given to end.
    """

    def error_message(self) -> str:
        return f"The statechart did not end within {self.tick_limit} ticks."

    def suggest_correction(self) -> str:
        return (
            "Check that an EndStatechart node can start, or allow more ticks if the "
            "statechart needs them."
        )


@dataclass
class MissingExecutorExtensionError(StatechartError):
    """
    Raised when an executor extension is requested that the executor was not given.
    """

    expected_extension: type
    """
    The type of the requested extension.
    """

    def error_message(self) -> str:
        return f'Missing executor extension "{self.expected_extension.__name__}".'

    def suggest_correction(self) -> str:
        return "Pass an instance of it in the extensions of the StatechartExecutor."


@dataclass
class StatechartOfDifferentContextError(StatechartError):
    """
    Raised when an executor is handed a statechart that was built in a context other
    than its own.
    """

    def error_message(self) -> str:
        return "The statechart was built in a context other than the executor's."

    def suggest_correction(self) -> str:
        return "Create the statechart with the context of the executor that runs it."


@dataclass
class NoChildChooserForNodeError(StatechartError):
    """
    Raised when a node choosing its child finds no chooser of its kind of node in its
    context.
    """

    node: CompositeNodeChoosingItsChild
    """
    The node that asked for its child.
    """

    choosers: list[ChildChooser]
    """
    The choosers the context holds.
    """

    def error_message(self) -> str:
        node_types = [chooser.node_type.__name__ for chooser in self.choosers]
        return (
            f"{self.node} is no {' or '.join(node_types) or 'node any chooser chooses for'}"
            f", so no chooser of the context chooses its child."
        )

    def suggest_correction(self) -> str:
        return (
            f"add a chooser for {type(self.node).__name__} to the ChildChooserAccess of "
            f"the context."
        )


@dataclass
class SeveralChildChoosersForNodeError(StatechartError):
    """
    Raised when a node choosing its child finds more than one chooser for the most
    specific kind of node it is of.
    """

    node: CompositeNodeChoosingItsChild
    """
    The node that asked for its child.
    """

    choosers: list[ChildChooser]
    """
    The equally specific choosers for the node.
    """

    def error_message(self) -> str:
        return (
            f"{len(self.choosers)} choosers are equally specific for {self.node}: "
            f"{', '.join(type(chooser).__name__ for chooser in self.choosers)}."
        )

    def suggest_correction(self) -> str:
        return (
            f"keep one chooser for {type(self.node).__name__} in the ChildChooserAccess "
            f"of the context."
        )


@dataclass
class StatechartAlreadyCompiledError(StatechartError):
    """
    Raised when a compiled part of a statechart is changed: a node is removed, or a node
    is added as the child of a composite node that is compiled already.
    """

    def error_message(self) -> str:
        return "That part of the statechart is compiled, so it can no longer change."

    def suggest_correction(self) -> str:
        return (
            "Add new nodes at the top level or inside a composite node that joins with "
            "them; batch several additions in Statechart.modify()."
        )


@dataclass
class PrerequisiteNotExpandedError(NodeInitializationError):
    """
    Raised when a composite node joins a statechart before a composite node it reads
    while expanding.
    """

    prerequisite: StatechartNode
    """
    The composite node that has not joined the statechart yet.
    """

    def error_message(self) -> str:
        return (
            f"{self.node.name} reads {self.prerequisite.name} while it expands, but "
            f"{self.prerequisite.name} has not joined the statechart yet."
        )

    def suggest_correction(self) -> str:
        return (
            f"Add {self.prerequisite.name} to the statechart before {self.node.name}."
        )


@dataclass
class NodeIsNotAChildError(NodeInitializationError):
    """
    Raised when a composite node is asked about a node that is not its child.
    """

    child: StatechartNode
    """
    The node that is not a child of the composite node.
    """

    def error_message(self) -> str:
        return f"{self.child.name} is not a child of {self.node.name}."

    def suggest_correction(self) -> str:
        return f"Refer to one of the children of {self.node.name}."


@dataclass
class NodeAlreadyAChildError(NodeInitializationError):
    """
    Raised when a composite node is handed a node that is its child already.
    """

    child: StatechartNode
    """
    The node that is a child of the composite node already.
    """

    def error_message(self) -> str:
        return f"{self.child.name} is a child of {self.node.name} already."

    def suggest_correction(self) -> str:
        return "Hand the composite node a node that is not its child yet."


@dataclass
class NoControlStructureNodeAboveError(NodeInitializationError):
    """
    Raised when the closest control structure node above a node is asked for, but there
    is none, which leaves nothing that could hold a neighbour of it.
    """

    def error_message(self) -> str:
        return f"No control structure node is above {self.node.name}."

    def suggest_correction(self) -> str:
        return (
            f"Put {self.node.name} below a control structure node, or add the new node as "
            f"the last child of one."
        )


@dataclass
class NotHeldByControlStructureNodeError(NodeInitializationError):
    """
    Raised when a node is to be replaced whose parent is not a control structure node,
    which leaves nothing that could run another node in its place.
    """

    parent: StatechartNode
    """
    The parent of the node.
    """

    def error_message(self) -> str:
        return (
            f"{self.parent.name} holds {self.node.name}, and is no control structure node "
            f"that could run another node in its place."
        )

    def suggest_correction(self) -> str:
        return (
            f"Replace the node above {self.node.name} that a control structure node or the "
            f"top level holds."
        )


@dataclass
class RemovedNodeStillReferencedError(StatechartError):
    """
    Raised when a node is removed from a statechart while a node that stays still refers
    to it.
    """

    removed_node: StatechartNode
    """
    The node that was about to be removed.
    """

    referencing_node: StatechartNode
    """
    The node that stays and still refers to :attr:`removed_node`.
    """

    def error_message(self) -> str:
        return (
            f"{self.removed_node.name} cannot be removed, because "
            f"{self.referencing_node.name} still refers to it."
        )

    def suggest_correction(self) -> str:
        return (
            f"Rewire {self.referencing_node.name} so it no longer refers to "
            f"{self.removed_node.name} before removing it."
        )


@dataclass
class NodeStateVariableNotSerializableError(JSONSerializationError):
    """
    Raised when a node state variable is serialized to JSON, which has no way to refer
    to the node the variable belongs to.
    """

    variable: NodeStateVariable
    """
    The variable that was serialized.
    """

    def error_message(self) -> str:
        return (
            f"Cannot serialize {self.variable}, since JSON cannot refer to the node it "
            f"belongs to."
        )

    def suggest_correction(self) -> str:
        return ""


# %% a statechart that ran but did not reach its goal


@dataclass
class ExecutionFailure(DataclassException):
    """
    Raised when a statechart was run but did not reach its goal, as opposed to an error
    in how it was built.

    Whoever runs a statechart can recover from it, for example by trying something else.
    """

    def error_message(self) -> str:
        return "The statechart did not reach its goal."

    def suggest_correction(self) -> str:
        return ""


@dataclass
class RepetitionsExhausted(ExecutionFailure):
    """
    Raised when a repeating node ran out of attempts.
    """

    repeated_node: StatechartNode
    """
    The node that never succeeded.
    """

    maximum_repetitions: int
    """
    How many attempts were allowed.
    """

    def error_message(self) -> str:
        return (
            f"{self.repeated_node} was attempted {self.maximum_repetitions} "
            f"times without succeeding."
        )

    def suggest_correction(self) -> str:
        return (
            "Allow more repetitions, or check whether the children can succeed at all "
            "from the state each attempt starts in."
        )


@dataclass
class CancelledByNode(ExecutionFailure, ABC):
    """
    Raised when a :class:`~cramph.node.CancelStatechart` given no exception of its own
    ends the statechart because of the node it watches.
    """

    node: StatechartNode
    """
    The node whose outcome cancelled the statechart.
    """

    def suggest_correction(self) -> str:
        return "Give the cancelling node an exception that says why it cancels."


@dataclass
class CancelledByNodeReachingItsGoal(CancelledByNode):
    """
    Raised when the watched node observed True or succeeded.
    """

    def error_message(self) -> str:
        return f"Cancelled because {self.node.unique_name} reached its goal."


@dataclass
class CancelledByNodeFailing(CancelledByNode):
    """
    Raised when the watched node failed.
    """

    def error_message(self) -> str:
        return f"Cancelled because {self.node.unique_name} failed."


@dataclass
class PlanCancelled(ExecutionFailure):
    """
    Raised when a monitor cancelled the part of a plan it was watching.
    """

    monitor: StatechartNode
    """
    The monitor that cancelled the plan.
    """

    def error_message(self) -> str:
        return f"{self.monitor} cancelled the plan."

    def suggest_correction(self) -> str:
        return (
            "The world is no longer in the state the rest of the plan assumed, so plan "
            "again from the state the robot is in now."
        )


# %% visualization


@dataclass
class RvizUnavailableError(DataclassException):
    """
    Raised when the world is to be shown in RViz without ROS installed.
    """

    def error_message(self) -> str:
        return "RViz cannot show the world, because ROS is not installed."

    def suggest_correction(self) -> str:
        return (
            "Source a ROS installation, or show the world with another visualization."
        )
