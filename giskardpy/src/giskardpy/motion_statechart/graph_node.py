from __future__ import annotations

import ast
import logging
import threading
import uuid
from abc import ABC, abstractmethod
from dataclasses import field, dataclass, fields
from functools import cached_property

import numpy as np
from typing_extensions import (
    ClassVar,
    Dict,
    Any,
    Self,
    Optional,
    TYPE_CHECKING,
    List,
    TypeVar,
    Tuple,
)

import krrood.symbolic_math.symbolic_math as sm
from krrood.adapters.json_serializer import (
    SubclassJSONSerializer,
    JSON_TYPE_NAME,
    to_json,
    from_json,
)
from krrood.patterns.field_metadata import JSONMetadata
from krrood.symbolic_math.symbolic_math import FloatVariable, Scalar, trinary_logic_not
from krrood.exceptions import DataclassException
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.world_description.degree_of_freedom import DegreeOfFreedom
from semantic_digital_twin.spatial_types import (
    Point3,
    Vector3,
    Quaternion,
    RotationMatrix,
    HomogeneousTransformationMatrix,
    Pose,
)
from semantic_digital_twin.world_description.geometry import Color
from giskardpy.motion_statechart.context import MotionStatechartContext
from giskardpy.motion_statechart.data_types import (
    LifeCycleValues,
    LifeCyclePredicate,
    ObservationStateValues,
    TransitionKind,
    DefaultWeights,
)
from giskardpy.motion_statechart.exceptions import (
    NotInMotionStatechartError,
    EndMotionInGoalError,
    GoalWithoutChildrenError,
    InputNotExpressionError,
    SelfInStartConditionError,
    UnsupportedConditionVariableError,
    NodeAlreadyBelongsToDifferentNodeError,
    NodeNotBuiltError,
    TerminalNodeInConditionError,
    MissingErrorSignalError,
)
from giskardpy.motion_statechart.error_signals import ErrorSignal
from giskardpy.motion_statechart.plotters.plot_specs import (
    NodePlotSpec,
    plot_specification_field,
)
from giskardpy.motion_statechart.constraint_builders import GeometricConstraintBuilder
from giskardpy.qp.constraint_collection import ConstraintCollection
from giskardpy.utils.utils import string_shortener

if TYPE_CHECKING:
    from giskardpy.motion_statechart.motion_statechart import MotionStatechart

logger = logging.getLogger(__name__)


@dataclass(eq=False, repr=False)
class TrinaryCondition(SubclassJSONSerializer):
    """
    Represents a trinary condition used to define transitions in a motion statechart model.

    This class serves as a representation of a logical trinary condition with three possible states: true, false, and
    unknown. It is used as part of a motion statechart system to define transitions between nodes. The condition is
    evaluated using a logical expression and connects nodes via parent-child relationships. It includes methods to
    create predefined trinary values, update the expression of the condition, and format the condition for display.
    """

    kind: TransitionKind
    """
    The type of transition associated with this condition.
    """
    expression: Scalar = field(default_factory=Scalar.const_trinary_unknown)
    """
    The logical trinary condition to be evaluated.
    """

    owner: Optional[MotionStatechartNode] = field(default=None)
    """
    The node this transition belongs to.
    """

    def __hash__(self) -> int:
        return hash((str(self), self.kind, self.owner.index))

    def __eq__(self, other):
        return hash(self) == hash(other)

    @classmethod
    def create_true(
        cls, kind: TransitionKind, owner: Optional[MotionStatechartNode] = None
    ) -> Self:
        """
        Creates a condition that always evaluates to true.

        :param kind: The type of transition this condition controls.
        :param owner: The node this condition belongs to.
        :return: The new condition.
        """
        return cls(expression=Scalar.const_true(), kind=kind, owner=owner)

    @classmethod
    def create_false(
        cls, kind: TransitionKind, owner: Optional[MotionStatechartNode] = None
    ) -> Self:
        """
        Creates a condition that always evaluates to false.

        :param kind: The type of transition this condition controls.
        :param owner: The node this condition belongs to.
        :return: The new condition.
        """
        return cls(expression=Scalar.const_false(), kind=kind, owner=owner)

    @classmethod
    def create_unknown(
        cls, kind: TransitionKind, owner: Optional[MotionStatechartNode] = None
    ) -> Self:
        """
        Creates a condition that always evaluates to unknown.

        :param kind: The type of transition this condition controls.
        :param owner: The node this condition belongs to.
        :return: The new condition.
        """
        return cls(
            expression=Scalar.const_trinary_unknown(),
            kind=kind,
            owner=owner,
        )

    def update_expression(
        self, new_expression: Scalar, child: MotionStatechartNode
    ) -> None:
        """
        Replaces the expression of this condition, rejecting invalid expressions.

        :param new_expression: The expression to evaluate for this transition.
        :param child: The node the new expression is set on.
        """
        self._sanity_check(new_expression)
        self.expression = new_expression
        self._child = child

    def _sanity_check(self, new_expression: Scalar) -> None:
        """
        Rejects expressions that may not be used as a transition condition.

        :param new_expression: The expression to validate.
        """
        self._check_condition_is_variable_or_expression(new_expression)
        self._check_only_condition_variables(new_expression)
        self._check_owner_not_in_start_condition(new_expression)
        self._check_no_terminal_node(new_expression)

    def _check_condition_is_variable_or_expression(self, new_expression: Scalar):
        """
        Rejects values that are not symbolic expressions.

        :param new_expression: The expression to validate.
        """
        if not isinstance(new_expression, Scalar):
            raise InputNotExpressionError(condition=self, new_expression=new_expression)

    def _check_only_condition_variables(self, new_expression: Scalar):
        """
        Rejects expressions that reference state a transition may not read.

        :param new_expression: The expression to validate.
        """
        for variable in new_expression.free_variables():
            if not isinstance(variable, ConditionVariable):
                raise UnsupportedConditionVariableError(
                    condition=self,
                    unsupported_variable=variable,
                    new_expression=new_expression,
                )

    def _check_no_terminal_node(self, new_expression: Scalar):
        """
        Rejects references to nodes that end the motion.

        .. note:: Runs after :meth:`_check_only_condition_variables`, so every free
            variable is known to refer to a node.

        :param new_expression: The expression to validate.
        """
        for variable in new_expression.free_variables():
            if isinstance(variable.motion_statechart_node, TerminalNode):
                raise TerminalNodeInConditionError(
                    condition=self,
                    new_expression=new_expression,
                    terminal_node=variable.motion_statechart_node,
                )

    def _check_owner_not_in_start_condition(self, new_expression: Scalar):
        """
        Rejects start conditions that reference the state of their own node.

        .. note:: Runs after :meth:`_check_only_condition_variables`, so every free
            variable is known to refer to a node.

        :param new_expression: The expression to validate.
        """
        if self.kind != TransitionKind.START:
            return
        for variable in new_expression.free_variables():
            if variable.motion_statechart_node is self.owner:
                raise SelfInStartConditionError(
                    condition=self, new_expression=new_expression
                )

    @property
    def variables(self) -> List[ConditionVariable]:
        """
        :return: The terms of this condition, each knowing how it is written and what it
            currently evaluates to.
        """
        return [
            variable
            for variable in self.expression.free_variables()
            if isinstance(variable, ConditionVariable)
        ]

    @property
    def node_dependencies(self) -> List[MotionStatechartNode]:
        """
        :return: The nodes this condition reads.
        """
        return [variable.motion_statechart_node for variable in self.variables]

    def __str__(self):
        """
        Renders the condition, replacing each variable with its
        :attr:`~NodeStateVariable.display_name` so the result is readable and reproducible
        across processes (the variable's own name uses a process-local id).

        :return: The rendered condition.
        """
        free_symbols = self.expression.free_variables()
        if not free_symbols:
            return str(self.expression.is_const_true())
        str_representation = sm.trinary_logic_to_str(self.expression)
        for variable in free_symbols:
            str_representation = str_representation.replace(
                variable.name, variable.display_name
            )
        return str_representation

    def __repr__(self):
        return str(self)

    def to_json(self) -> Dict[str, Any]:
        json_data = super().to_json()
        json_data["kind"] = self.kind.name
        json_data["expression"] = str(self)
        json_data["owner"] = self.owner.index if self.owner else None
        return json_data

    @classmethod
    def create_from_trinary_logic_str(
        cls,
        kind: TransitionKind,
        trinary_logic_str: str,
        state_variables: List[NodeStateVariable],
        owner: Optional[MotionStatechartNode] = None,
    ):
        """
        Creates a condition from the string representation produced by :meth:`__str__`.

        :param kind: The type of transition this condition controls.
        :param trinary_logic_str: The condition, with nodes referenced by their unique name.
        :param state_variables: The variables the referenced display names are resolved against.
        :param owner: The node this condition belongs to.
        :return: The new condition.
        """
        tree = ast.parse(trinary_logic_str, mode="eval")
        return cls(
            kind=kind,
            expression=cls._parse_ast_expression(tree.body, state_variables),
            owner=owner,
        )

    @staticmethod
    def _parse_ast_expression(
        node: ast.expr, state_variables: List[NodeStateVariable]
    ) -> Scalar:
        """
        Translates a parsed trinary logic expression into a symbolic expression.

        :param node: The syntax tree node to translate.
        :param state_variables: The variables the referenced display names are resolved against.
        :return: The symbolic expression.
        """
        match node:
            case ast.BoolOp(op=ast.And()):
                return TrinaryCondition._parse_ast_and(node, state_variables)
            case ast.BoolOp(op=ast.Or()):
                return TrinaryCondition._parse_ast_or(node, state_variables)
            case ast.UnaryOp():
                return TrinaryCondition._parse_ast_not(node, state_variables)
            case ast.Constant(value=str(val)):
                for state_variable in state_variables:
                    if val == state_variable.display_name:
                        return state_variable
                raise KeyError(f"unknown state variable: {val!r}")
            case ast.Constant(value=True):
                return Scalar.const_true()
            case ast.Constant(value=False):
                return Scalar.const_false()
            case _:
                raise TypeError(f"failed to parse {type(node).__name__}")

    @staticmethod
    def _parse_ast_and(node, state_variables: List[NodeStateVariable]):
        """
        Translates a parsed conjunction into a symbolic expression.

        :param node: The syntax tree node to translate.
        :param state_variables: The variables the referenced display names are resolved against.
        :return: The symbolic expression.
        """
        return sm.trinary_logic_and(
            *[
                TrinaryCondition._parse_ast_expression(x, state_variables)
                for x in node.values
            ]
        )

    @staticmethod
    def _parse_ast_or(node, state_variables: List[NodeStateVariable]):
        """
        Translates a parsed disjunction into a symbolic expression.

        :param node: The syntax tree node to translate.
        :param state_variables: The variables the referenced display names are resolved against.
        :return: The symbolic expression.
        """
        return sm.trinary_logic_or(
            *[
                TrinaryCondition._parse_ast_expression(x, state_variables)
                for x in node.values
            ]
        )

    @staticmethod
    def _parse_ast_not(node, state_variables: List[NodeStateVariable]):
        """
        Translates a parsed negation into a symbolic expression.

        :param node: The syntax tree node to translate.
        :param state_variables: The variables the referenced display names are resolved against.
        :return: The symbolic expression, or None if the unary operator is not a negation.
        """
        if isinstance(node.op, ast.Not):
            return sm.trinary_logic_not(
                TrinaryCondition._parse_ast_expression(node.operand, state_variables)
            )

    @classmethod
    def _from_json(
        cls, data: Dict[str, Any], motion_statechart: MotionStatechart, **kwargs
    ) -> Self:
        return cls.create_from_trinary_logic_str(
            kind=TransitionKind[data["kind"]],
            trinary_logic_str=data["expression"],
            state_variables=motion_statechart.condition_variables(),
            owner=motion_statechart.get_node_by_index(data["owner"]),
        )


@dataclass(repr=False, eq=False, init=False)
class NodeStateVariable(FloatVariable):
    """
    A symbol standing for part of the state of one node.
    """

    motion_statechart_node: MotionStatechartNode = field(kw_only=True)
    """
    The node this variable refers to.
    """

    def __init__(self, name: str, motion_statechart_node: MotionStatechartNode):
        super().__init__(name)
        self.motion_statechart_node = motion_statechart_node

    @property
    def display_name(self) -> str:
        """
        :return: How this variable is written in a rendered condition, using the node's
            :attr:`~MotionStatechartNode.unique_name` so it reproduces across processes.
        """
        return self.motion_statechart_node.unique_name

    def as_expression(self) -> Scalar:
        """
        A variable the compiled updaters read directly stands for itself; one that is
        only a name for something else overrides this.

        :return: What this variable stands for.
        """
        return self

    @classmethod
    def replace_in(cls, expression: Scalar) -> Scalar:
        """
        Replaces every variable of this type in `expression` by what it stands for, see
        :meth:`as_expression`.

        :param expression: The expression to replace them in.
        :return: `expression` with every variable of this type replaced.
        """
        variables = [
            variable
            for variable in expression.free_variables()
            if isinstance(variable, cls)
        ]
        return expression.substitute(
            variables,
            [variable.as_expression() for variable in variables],
        )


@dataclass(repr=False, eq=False, init=False)
class ConditionVariable(NodeStateVariable):
    """
    A node state variable a transition condition may read.

    Every one of them stands for a trinary value, which is what makes a condition built
    from them renderable and evaluable.
    """

    def resolve(self) -> ObservationStateValues:
        """
        :return: The trinary value this variable currently stands for.
        """
        raise NotImplementedError


@dataclass(repr=False, eq=False, init=False)
class ObservationVariable(ConditionVariable):
    """
    A symbol representing the observation state of a node.
    """

    def resolve(self) -> ObservationStateValues:
        return self.motion_statechart_node.observation_state


@dataclass(repr=False, eq=False, init=False)
class LifeCycleVariable(NodeStateVariable):
    """
    A symbol representing the life cycle state of a node.

    .. warning:: Legal in observation expressions, but not in transition conditions.
        Use a :class:`LifeCyclePredicateVariable` there, so the condition stays renderable.
    """

    def resolve(self) -> LifeCycleValues:
        return self.motion_statechart_node.life_cycle_state


@dataclass(repr=False, eq=False, init=False)
class LifeCyclePredicateVariable(ConditionVariable):
    """
    A symbol representing a trinary test on the life cycle state of a node.

    Unlike the other node state variables this one cannot be expanded on its own, because
    it is read in the state its node reaches this control cycle rather than the one it
    entered with. See
    :meth:`~giskardpy.motion_statechart.motion_statechart.NextLifeCycle.of`.
    """

    predicate: LifeCyclePredicate = field(kw_only=True)
    """
    The test this variable holds the value of.
    """

    def __init__(
        self,
        name: str,
        motion_statechart_node: MotionStatechartNode,
        predicate: LifeCyclePredicate,
    ):
        super().__init__(name, motion_statechart_node)
        self.predicate = predicate

    @property
    def display_name(self) -> str:
        return (
            f"{self.motion_statechart_node.unique_name}.{self.predicate.attribute_name}"
        )

    def resolve(self) -> ObservationStateValues:
        return self.predicate.value.truth_value(
            self.motion_statechart_node.life_cycle_state
        )


@dataclass(repr=False, eq=False, init=False)
class GoalReachedVariable(ConditionVariable):
    """
    A symbol representing whether a node reached its goal, whether it is still running
    or has already ended.
    """

    attribute_name: ClassVar[str] = "goal_reached"
    """
    The name this variable is reached under on a node, also used to render it inside a
    condition.
    """

    @property
    def display_name(self) -> str:
        return f"{self.motion_statechart_node.unique_name}.{self.attribute_name}"

    def resolve(self) -> ObservationStateValues:
        return self.motion_statechart_node.goal_reached_state

    def as_expression(self) -> Scalar:
        """
        :return: The same value as :meth:`resolve`, read off the life cycle and
            observation variables of the node rather than off their current states.
        """
        return self.motion_statechart_node._create_goal_reached()


@dataclass
class DebugExpression:
    """
    Symbolic expressions used for debugging only.
    Allows you to keep track of any expression and evaluate them later in debug mode.
    """

    name: str
    """
    Name used for this expression in some debugging tools.
    """

    expression: (
        Scalar
        | Point3
        | Vector3
        | Quaternion
        | RotationMatrix
        | HomogeneousTransformationMatrix
        | Pose
    )
    """The tracked expression; spatial types are additionally rendered as RViz markers."""

    color: Color = field(default_factory=lambda: Color(1, 0, 0, 1))
    """
    The color used when this expression is rendered in visualization tools.
    """

    def __repr__(self) -> str:
        return self.name

    @property
    def evaluated(self) -> np.ndarray:
        """
        :return: The current value of the tracked expression.
        """
        return self.expression.evaluate()


@dataclass
class NodeArtifacts:
    """
    Represents the artifacts produced by the `build_artifacts` method of a node.
    It makes explicit what artifacts are produced by a node.
    """

    constraints: ConstraintCollection = field(default_factory=ConstraintCollection)
    """
    A collection of constraints that describe a motion task. 
    """
    observation: Optional[Scalar] = field(default=None)
    """
    A symbolic expression that describes the observation state of this node.
    Instead of setting this attribute directly, you may also implement the `on_tick` method of a node.
    The advantage of using observation is that you can reuse the expressions used in constraints.
    .. warning:: the result of `on_tick` takes precedence over the observation expression.
    """
    error: Optional[ErrorSignal] = field(default=None)
    """
    How far this node is from its goal. Set by :class:`ConvergingTask`, which derives
    :attr:`observation` from it, and used to watch whether the node is still converging.
    """
    debug_expressions: List[DebugExpression] = field(default_factory=list)
    """
    A list of symbolic expressions used for debugging only.
    While in debug mode, you can call .evaluate() on them to get their current value.
    """

    @cached_property
    def geometry(self) -> GeometricConstraintBuilder:
        """
        Builder for high-level geometric constraints (point, vector, and rotation goals, and
        Cartesian velocity limits) that writes into :attr:`constraints`.

        :return: The builder for this collection of artifacts.
        """
        return GeometricConstraintBuilder(self.constraints)


@dataclass
class LifeCycleTransitions:
    """
    The next life cycle state of one node, as an expression per state it can be in.
    """

    not_started: sm.Scalar
    """
    Where the node goes while it has not started.
    """
    running: sm.Scalar
    """
    Where the node goes while it is running.
    """
    paused: sm.Scalar
    """
    Where the node goes while it is paused.
    """
    terminal: sm.Scalar
    """
    Where the node goes while it has ended. Shared by every terminal state, because a
    verdict is only left by a reset.
    """

    def as_cases(self) -> List[Tuple[LifeCycleValues, sm.Scalar]]:
        """
        :return: (current state, next state) pairs covering every life cycle state.
        """
        return [
            (LifeCycleValues.NOT_STARTED, self.not_started),
            (LifeCycleValues.RUNNING, self.running),
            (LifeCycleValues.PAUSED, self.paused),
            *(
                (state, self.terminal)
                for state in sorted(LifeCycleValues.terminal_states())
            ),
        ]


@dataclass(repr=False, eq=False)
class MotionStatechartNode:
    name: str = field(default=None, kw_only=True)
    """
    A name for the node within a motion statechart.
    The name is not unique, use `.unique_name`, if you need a unique identifier.
    """

    _motion_statechart: MotionStatechart = field(init=False, default=None)
    """
    Back reference to the motion statechart that owns this node.
    """
    index: Optional[int] = field(default=None, init=False)
    """
    The index of this node in the motion statechart.
    """

    _node_id: str = field(init=False, default=None)
    """
    Process-unique identifier assigned at construction and used to name this node's state
    variables. Unlike :attr:`index` it exists before the node is added to a motion statechart,
    so variable names are unique from construction time. It is not serialized: conditions
    reference nodes by :attr:`unique_name`, which is reproduced deterministically on load.
    """

    parent_node_index: Optional[int] = field(
        default=None, init=False, metadata=JSONMetadata(serialize=True).as_dict()
    )
    """
    The index of the parent node in the motion statechart, if None, it is on the top layer of a motion statechart.
    """

    _life_cycle_variable: LifeCycleVariable = field(init=False, default=None)
    """
    A variable referring to the life cycle state of this node.
    """
    _observation_variable: ObservationVariable = field(init=False, default=None)
    """
    A variable referring to the observation state of this node.
    """
    _goal_reached_variable: GoalReachedVariable = field(init=False, default=None)
    """
    A variable referring to whether this node reached its goal.
    """

    _constraint_collection: ConstraintCollection = field(init=False, repr=False)
    """The parameter is set after build() using its NodeArtifacts."""
    _observation_expression: Scalar = field(init=False, repr=False)
    """The parameter is set after build() using its NodeArtifacts."""
    _error_signal: Optional[ErrorSignal] = field(init=False, repr=False, default=None)
    """The parameter is set after build() using its NodeArtifacts."""
    _debug_expressions: List[DebugExpression] = field(default_factory=list, init=False)
    """The parameter is set after build() using its NodeArtifacts."""

    _start_condition: TrinaryCondition = field(init=False, default=None)
    """
    Decides when this node transitions from life cycle state NOT_STARTED to RUNNING.
    """
    _pause_condition: TrinaryCondition = field(init=False, default=None)
    """
    Decides when this node transitions from RUNNING to PAUSED or back.
    """
    _end_condition: TrinaryCondition = field(init=False, default=None)
    """
    Decides when this node transitions from RUNNING or PAUSED to a terminal state.
    """
    _reset_condition: TrinaryCondition = field(init=False, default=None)
    """
    Decides when this transitions to NOT_STARTED.
    """
    _life_cycle_predicate_variables: Dict[
        LifeCyclePredicate, LifeCyclePredicateVariable
    ] = field(init=False, default_factory=dict, repr=False)
    """
    The predicate variables handed out so far, so each one is created only once.
    """

    plot_specifications: NodePlotSpec = plot_specification_field(
        NodePlotSpec.create_monitor_style
    )
    """
    Describes how this node is plotted during a MotionStatechart.draw call or in the MotionStatechartInspector.
    """

    def __post_init__(self):
        if self.name is None:
            self.name = self.__class__.__name__
        self._node_id = str(uuid.uuid4())
        self._create_state_variables()
        self._start_condition = TrinaryCondition.create_true(
            kind=TransitionKind.START, owner=self
        )
        self._pause_condition = TrinaryCondition.create_false(
            kind=TransitionKind.PAUSE, owner=self
        )
        self._end_condition = TrinaryCondition.create_false(
            kind=TransitionKind.END, owner=self
        )
        self._reset_condition = TrinaryCondition.create_false(
            kind=TransitionKind.RESET, owner=self
        )

    def _create_state_variables(self):
        """
        Creates the observation, life cycle and goal reached variables for this node,
        named from :attr:`_node_id` so they are available before the node is added to a
        motion statechart.
        """
        name = f"{self.name}#{self._node_id}"
        self._observation_variable = ObservationVariable(
            name=str(PrefixedName("observation", name)),
            motion_statechart_node=self,
        )
        self._life_cycle_variable = LifeCycleVariable(
            name=str(PrefixedName("life_cycle", name)),
            motion_statechart_node=self,
        )
        self._goal_reached_variable = GoalReachedVariable(
            name=str(PrefixedName(GoalReachedVariable.attribute_name, name)),
            motion_statechart_node=self,
        )

    @property
    def parent_node(self) -> Optional[MotionStatechartNode]:
        """
        :return: Reference to the parent node of this node.
        """
        if self.parent_node_index is None:
            return None
        return self._motion_statechart.get_node_by_index(self.parent_node_index)

    @property
    def debug_expressions(self) -> List[DebugExpression]:
        """
        :return: The debug expressions registered by this node during build.
        """
        return self._debug_expressions

    @property
    def prerequisite_nodes(self) -> List[MotionStatechartNode]:
        """
        Nodes that must be expanded and built before this one, because this node reads
        artifacts they only produce during expansion or build.

        :return: The nodes this node depends on, empty unless a subclass declares any.
        """
        return []

    @property
    def depth(self) -> int:
        """
        Distance (in edges) from this node to the root of the motion statechart.

        The root node (no parent) has depth 0, its children depth 1, and so on.

        :return: The number of edges between this node and the root.
        """
        depth = 0
        current = self
        # Walk up the parent chain until there is no parent
        while current.parent_node is not None:
            depth += 1
            current = current.parent_node
        return depth

    @parent_node.setter
    def parent_node(self, parent_node: Optional[MotionStatechartNode]) -> None:
        """
        :param parent_node: The node this node becomes a child of, or None to put it on the top layer.
        """
        if parent_node is None:
            self.parent_node_index = None
        else:
            self.parent_node_index = parent_node.index

    def _set_transition(self, transition: TrinaryCondition) -> None:
        """
        Sets the transition condition for this node, depending on its kind.
        Used in json parsing.

        :param transition: The condition to set, whose kind decides which transition it replaces.
        """
        match transition.kind:
            case TransitionKind.START:
                self._start_condition = transition
            case TransitionKind.PAUSE:
                self._pause_condition = transition
            case TransitionKind.END:
                self._end_condition = transition
            case TransitionKind.RESET:
                self._reset_condition = transition
            case _:
                raise ValueError(f"Unknown transition kind: {transition.kind}")

    def create_lifecycle_transitions(self) -> LifeCycleTransitions:
        """
        Builds the state machine of this node.

        :return: The next life cycle state of this node, per state it can currently be in.
        """
        any_reset_condition_true = self._create_any_ancestor_condition_true(
            TransitionKind.RESET
        )

        return LifeCycleTransitions(
            not_started=self._create_not_started_transitions(
                any_reset_condition_true=any_reset_condition_true
            ),
            running=self._create_running_transitions(
                any_reset_condition_true=any_reset_condition_true
            ),
            paused=self._create_pause_transitions(
                any_reset_condition_true=any_reset_condition_true
            ),
            terminal=self._create_terminal_transitions(
                any_reset_condition_true=any_reset_condition_true
            ),
        )

    def _create_condition_holds(self, transition_kind: TransitionKind) -> sm.Scalar:
        """
        Coerces a trinary condition into a binary one, because the ``if`` builders treat
        the trinary unknown value as true.

        :param transition_kind: The transition whose condition to read.
        :return: 1 while this node's own condition of that kind is true, 0 otherwise.
        """
        return sm.Scalar(self.get_condition(transition_kind) == sm.Scalar.const_true())

    def _create_verdict(self) -> sm.Scalar:
        """
        The terminal state this node reaches when it is ended, read off what it observes
        at that moment.

        An observation with no answer is no basis for a judgement, so it interrupts the
        node instead.

        :return: The life cycle state to transition to.
        """
        return sm.if_cases(
            cases=[
                (
                    sm.Scalar(self.observation_variable == float(observation)),
                    sm.Scalar(LifeCycleValues.verdict_for(observation)),
                )
                for observation in ObservationStateValues
            ],
            else_result=sm.Scalar(LifeCycleValues.INTERRUPTED),
        )

    def _create_end_case(self) -> Tuple[sm.Scalar, sm.Scalar]:
        """
        An ancestor ending ends this node on the same terms as its own end condition:
        what ends a node decides only *when* it ends, never what the ending is worth.

        :return: The (condition, resulting life cycle state) pair for being ended.
        """
        return (
            self._create_any_ancestor_condition_true(TransitionKind.END),
            self._create_verdict(),
        )

    def _create_any_ancestor_condition_true(
        self,
        transition_kind: TransitionKind,
    ) -> sm.Scalar:
        """
        Builds a combined condition by OR-ing the 'true' conditions of this node and its ancestors.
        Traverses from the current node up to the root, combining conditions using trinary OR logic.

        :param transition_kind: Transition type to check (e.g., RESET for reset_condition)
        :return: Combined condition where True = any ancestor condition is Scalar.const_true()
        """
        current_node = self
        condition = current_node._create_condition_holds(transition_kind)
        while current_node.parent_node is not None:
            current_node = current_node.parent_node
            condition = sm.trinary_logic_or(
                condition, current_node._create_condition_holds(transition_kind)
            )
        return condition

    def get_condition(self, transition_kind: TransitionKind) -> Scalar:
        """
        Get the condition for the given transition kind.
        :param transition_kind: The kind of transition whose condition to get.
        :return: The condition for the given transition kind.
        """
        match transition_kind:
            case TransitionKind.START:
                return self.start_condition
            case TransitionKind.PAUSE:
                return self.pause_condition
            case TransitionKind.END:
                return self.end_condition
            case TransitionKind.RESET:
                return self.reset_condition
            case _:
                raise ValueError(f"Unknown transition kind: {transition_kind}")

    def _create_terminal_transitions(
        self, any_reset_condition_true: sm.Scalar
    ) -> sm.Scalar:
        """
        Create the transitions out of a terminal state for this node. A terminal state is
        only left by a reset, which is why the verdict is kept as it is otherwise.

        :param any_reset_condition_true: The combined reset condition for this node and its parents. Combined using trinary_logic_or.
        :return: The LifeCycleState transitions for every terminal state.
        """
        return sm.if_else(
            condition=any_reset_condition_true,
            if_result=sm.Scalar(LifeCycleValues.NOT_STARTED),
            else_result=sm.Scalar(self.life_cycle_variable),
        )

    def _create_pause_transitions(
        self, any_reset_condition_true: sm.Scalar
    ) -> sm.Scalar:
        """
        Create the pause transitions of the LifeCycleState for this node.
        :param any_reset_condition_true: The combined reset condition for this node and its parents. Combined using trinary_logic_or.
        :return: The LifeCycleState transitions for the PAUSED state.
        """
        unpause_condition = sm.Scalar(self.pause_condition != sm.Scalar.const_true())
        current = self
        while current.parent_node is not None:
            parent = current.parent_node
            unpause_condition = sm.trinary_logic_and(
                unpause_condition,
                sm.Scalar(parent.pause_condition != sm.Scalar.const_true()),
            )
            current = parent

        return sm.if_cases(
            cases=[
                (
                    any_reset_condition_true,
                    sm.Scalar(LifeCycleValues.NOT_STARTED),
                ),
                self._create_end_case(),
                (
                    unpause_condition,
                    sm.Scalar(LifeCycleValues.RUNNING),
                ),
            ],
            else_result=sm.Scalar(LifeCycleValues.PAUSED),
        )

    def _create_running_transitions(
        self, any_reset_condition_true: sm.Scalar
    ) -> sm.Scalar:
        """
        Create the running transitions of the LifeCycleState for this node.
        :param any_reset_condition_true: The combined reset condition for this node and its parents. Combined using trinary_logic_or.
        :return: The LifeCycleState transitions for the RUNNING state.
        """
        any_pause_condition = self._create_any_ancestor_condition_true(
            TransitionKind.PAUSE
        )
        return sm.if_cases(
            cases=[
                (
                    any_reset_condition_true,
                    sm.Scalar(LifeCycleValues.NOT_STARTED),
                ),
                self._create_end_case(),
                (any_pause_condition, sm.Scalar(LifeCycleValues.PAUSED)),
            ],
            else_result=sm.Scalar(LifeCycleValues.RUNNING),
        )

    def _create_not_started_transitions(
        self, any_reset_condition_true: sm.Scalar
    ) -> sm.Scalar:
        """
        A node starts once it is asked to and every ancestor is being started too, but
        never underneath an ancestor that is ending on the same cycle, and never while
        it or an ancestor is being reset, because a reset outranks a start.

        :param any_reset_condition_true: The combined reset condition for this node and its parents. Combined using trinary_logic_or.
        :return: The life cycle state this node reaches while it has not started.
        """
        start_condition = self._create_condition_holds(TransitionKind.START)
        current = self
        while current.parent_node is not None:
            parent = current.parent_node
            start_condition = sm.trinary_logic_and(
                start_condition,
                sm.trinary_logic_not(
                    parent._create_condition_holds(TransitionKind.END)
                ),
                parent._create_condition_holds(TransitionKind.START),
            )
            current = parent

        return sm.if_cases(
            cases=[
                (
                    any_reset_condition_true,
                    sm.Scalar(LifeCycleValues.NOT_STARTED),
                ),
                (start_condition, sm.Scalar(LifeCycleValues.RUNNING)),
            ],
            else_result=sm.Scalar(LifeCycleValues.NOT_STARTED),
        )

    @property
    def life_cycle_variable(self) -> LifeCycleVariable:
        """
        :return: The variable representing the life cycle state of this node.
        """
        return self._life_cycle_variable

    def belongs_to_motion_statechart(self) -> bool:
        """
        :return: Whether this node has been added to a motion statechart.
        """
        return self._motion_statechart is not None

    @property
    def observation_variable(self) -> ObservationVariable:
        """
        :return: The variable representing the observation state of this node.
        """
        return self._observation_variable

    @property
    def motion_statechart(self) -> MotionStatechart:
        """
        :return: The motion statechart this node belongs to.
        """
        if self._motion_statechart is None:
            raise NotInMotionStatechartError(self.name)
        return self._motion_statechart

    @motion_statechart.setter
    def motion_statechart(self, motion_statechart: MotionStatechart) -> None:
        """
        :param motion_statechart: The motion statechart this node now belongs to.
        """
        self._motion_statechart = motion_statechart

    def build(self, context: MotionStatechartContext) -> NodeArtifacts:
        """
        Called exactly once during motion statechart compilation.
        Override this method for setup steps that produce no artifacts.
        .. warning:: Don't create other nodes within this function.
        .. warning:: An override must return ``super().build(context)``, otherwise
            :meth:`build_artifacts` never runs.
        :param context: The context that contains data that can be used to build this node.
        :return: A NodeArtifacts instance that describes this node.
        """
        return self.build_artifacts(context)

    def build_artifacts(self, context: MotionStatechartContext) -> NodeArtifacts:
        """
        Describe this node in terms of constraints, observation and debug expressions.
        :param context: The context that contains data that can be used to build this node.
        :return: A NodeArtifacts instance that describes this node. It is normal for nodes that don't directly affect the motion to return empty NodeArtifacts.
        """
        return NodeArtifacts()

    def on_tick(
        self, context: MotionStatechartContext
    ) -> Optional[ObservationStateValues]:
        """
        Triggered when the node is ticked.
        .. warning:: This method is called inside a control loop, make sure it is fast.
        .. warning:: Only happens while the node is in state RUNNING.
        .. warning:: The result of this method takes precedence over the observation expression created in build().
        :param context: The context that contains data that can be used while ticking this node.
        :return: An optional observation state overwrite
        """

    def on_start(self, context: MotionStatechartContext):
        """
        Triggered when the node transitions from NOT_STARTED to RUNNING.
        .. warning:: This method is called inside a control loop, make sure it is fast.
        :param context: The context that contains data that can be used by this node.
        """

    def on_pause(self, context: MotionStatechartContext):
        """
        Triggered when the node transitions from RUNNING to PAUSED.
        .. warning:: This method is called inside a control loop, make sure it is fast.
        :param context: The context that contains data that can be used by this node.
        """

    def on_unpause(self, context: MotionStatechartContext):
        """
        Triggered when the node transitions from PAUSED to RUNNING.
        .. warning:: This method is called inside a control loop, make sure it is fast.
        :param context: The context that contains data that can be used by this node.
        """

    def on_end(self, context: MotionStatechartContext):
        """
        Triggered when the node transitions from RUNNING or PAUSED into any terminal
        state. Read :attr:`life_cycle_state` for the verdict.
        .. warning:: This method is called inside a control loop, make sure it is fast.
        :param context: The context that contains data that can be used by this node.
        """

    def on_reset(self, context: MotionStatechartContext):
        """
        Triggered when the node transitions from any state to NOT_STARTED.
        .. warning:: This method is called inside a control loop, make sure it is fast.
        :param context: The context that contains data that can be used by this node.
        """

    def cleanup(self, context: MotionStatechartContext):
        """
        Triggered after an EndMotion or CancelMotion was triggered.
        Place code here to clean up after execution.
        :param context: The context that contains data that can be used by this node.
        """

    def __hash__(self):
        return hash(self.name)

    @property
    def life_cycle_state(self) -> LifeCycleValues:
        """
        :return: The current life cycle state of this node.
        """
        return LifeCycleValues(self.motion_statechart.life_cycle_state[self])

    @property
    def observation_state(self) -> float:
        """
        :return: The current observation state of this node.
        """
        return self.motion_statechart.observation_state[self]

    @property
    def start_condition(self) -> Scalar:
        """
        :return: The expression deciding when this node transitions from NOT_STARTED to RUNNING.
        """
        return self._start_condition.expression

    @start_condition.setter
    def start_condition(self, expression: Scalar) -> None:
        """
        :param expression: The expression deciding when this node transitions from NOT_STARTED to RUNNING.
        """
        if self._start_condition is None:
            raise NotInMotionStatechartError(self.name)
        self._start_condition.update_expression(expression, self)

    @property
    def pause_condition(self) -> Scalar:
        """
        :return: The expression deciding when this node transitions from RUNNING to PAUSED or back.
        """
        return self._pause_condition.expression

    @pause_condition.setter
    def pause_condition(self, expression: Scalar) -> None:
        """
        :param expression: The expression deciding when this node transitions from RUNNING to PAUSED or back.
        """
        if self._pause_condition is None:
            raise NotInMotionStatechartError(self.name)
        self._pause_condition.update_expression(expression, self)

    @property
    def end_condition(self) -> Scalar:
        """
        :return: The expression deciding when this node transitions from RUNNING or PAUSED to a terminal state.
        """
        return self._end_condition.expression

    @end_condition.setter
    def end_condition(self, expression: Scalar) -> None:
        """
        :param expression: The expression deciding when this node transitions from RUNNING or PAUSED to a terminal state.
        """
        if self._end_condition is None:
            raise NotInMotionStatechartError(self.name)
        self._end_condition.update_expression(expression, self)

    @property
    def goal_reached(self) -> GoalReachedVariable:
        """
        :return: A variable holding whether this node reached its goal.
        """
        return self._goal_reached_variable

    @property
    def goal_reached_state(self) -> ObservationStateValues:
        """
        :return: Whether this node has reached its goal: what it observes while it runs,
            and the verdict it earned once it has ended.
        """
        if self.life_cycle_state.is_terminal:
            return LifeCyclePredicate.IS_SUCCEEDED.value.truth_value(
                self.life_cycle_state
            )
        return ObservationStateValues(self.observation_state)

    def _create_goal_reached(self) -> sm.Scalar:
        """
        The same value as :attr:`goal_reached_state`, but read off the life cycle and
        observation variables rather than off the states they stand for.

        :return: The trinary value :attr:`goal_reached` stands for.
        """
        return sm.if_eq_cases(
            a=self.life_cycle_variable,
            b_result_cases=[
                (
                    int(state),
                    sm.Scalar(
                        float(LifeCyclePredicate.IS_SUCCEEDED.value.truth_value(state))
                    ),
                )
                for state in sorted(LifeCycleValues.terminal_states())
            ],
            else_result=sm.Scalar(self.observation_variable),
        )

    @property
    def reset_condition(self) -> Scalar:
        """
        :return: The expression deciding when this node transitions to NOT_STARTED.
        """
        return self._reset_condition.expression

    @reset_condition.setter
    def reset_condition(self, expression: Scalar) -> None:
        """
        :param expression: The expression deciding when this node transitions to NOT_STARTED.
        """
        if self._reset_condition is None:
            raise NotInMotionStatechartError(self.name)
        self._reset_condition.update_expression(expression, self)

    def _life_cycle_predicate(
        self, predicate: LifeCyclePredicate
    ) -> LifeCyclePredicateVariable:
        """
        Hands out the variable for one test on this node's life cycle state, creating it
        on first use so an unused predicate costs nothing.

        :param predicate: The test to read.
        :return: The variable holding that test's value for this node.
        """
        if predicate not in self._life_cycle_predicate_variables:
            self._life_cycle_predicate_variables[predicate] = (
                LifeCyclePredicateVariable(
                    name=str(
                        PrefixedName(
                            predicate.attribute_name, f"{self.name}#{self._node_id}"
                        )
                    ),
                    motion_statechart_node=self,
                    predicate=predicate,
                )
            )
        return self._life_cycle_predicate_variables[predicate]

    @property
    def conditions(self) -> List[TrinaryCondition]:
        """
        :return: Every transition condition of this node.
        """
        return [
            self._start_condition,
            self._pause_condition,
            self._end_condition,
            self._reset_condition,
        ]

    @property
    def is_not_started(self) -> LifeCyclePredicateVariable:
        """
        :return: True while this node has not started, false otherwise.
        """
        return self._life_cycle_predicate(LifeCyclePredicate.IS_NOT_STARTED)

    @property
    def is_running(self) -> LifeCyclePredicateVariable:
        """
        :return: True while this node is running, false otherwise.
        """
        return self._life_cycle_predicate(LifeCyclePredicate.IS_RUNNING)

    @property
    def is_paused(self) -> LifeCyclePredicateVariable:
        """
        :return: True while this node is paused, false otherwise.
        """
        return self._life_cycle_predicate(LifeCyclePredicate.IS_PAUSED)

    @property
    def is_terminated(self) -> LifeCyclePredicateVariable:
        """
        :return: True once this node has ended, whatever its verdict, false before that.
        """
        return self._life_cycle_predicate(LifeCyclePredicate.IS_TERMINATED)

    @property
    def is_succeeded(self) -> LifeCyclePredicateVariable:
        """
        :return: True if this node ended by succeeding, false if it ended any other way,
            and unknown until it ends.
        """
        return self._life_cycle_predicate(LifeCyclePredicate.IS_SUCCEEDED)

    @property
    def is_failed(self) -> LifeCyclePredicateVariable:
        """
        :return: True if this node ended by failing, false if it ended any other way, and
            unknown until it ends.
        """
        return self._life_cycle_predicate(LifeCyclePredicate.IS_FAILED)

    @property
    def is_interrupted(self) -> LifeCyclePredicateVariable:
        """
        :return: True if this node ended without earning a verdict, false if it ended
            any other way, and unknown until it ends.
        """
        return self._life_cycle_predicate(LifeCyclePredicate.IS_INTERRUPTED)

    def formatted_name(self, quoted: bool = False) -> str:
        """
        Renders the name of this node together with all of its transition conditions.

        :param quoted: Whether to wrap the result in double quotes.
        :return: The multi line representation of this node.
        """
        formatted_name = string_shortener(
            original_str=str(self.name), max_lines=4, max_line_length=25
        )
        result = (
            f"{formatted_name}\n"
            f"----start_condition----\n"
            f"{str(self._start_condition)}\n"
            f"----pause_condition----\n"
            f"{str(self._pause_condition)}\n"
            f"----end_condition----\n"
            f"{str(self._end_condition)}\n"
            f"----reset_condition----\n"
            f"{str(self._reset_condition)}"
        )
        if quoted:
            return '"' + result + '"'
        return result

    @property
    def unique_name(self) -> str:
        """
        :return: The name of this node, made unique by appending its index.
        """
        return f"{self.name}#{self.index}"

    def __repr__(self) -> str:
        return self.unique_name


GenericMotionStatechartNode = TypeVar(
    "GenericMotionStatechartNode", bound=MotionStatechartNode
)


def velocity_convergence_expression(
    context: MotionStatechartContext,
    joint_convergence_threshold: float,
    minimum_threshold: float,
    maximum_threshold: float,
    degrees_of_freedom: Optional[List[DegreeOfFreedom]] = None,
    minimum_time: float = 1.0,
    reference_cycle_variable: Optional[FloatVariable] = None,
) -> Scalar:
    """
    Builds a trinary expression that is true once every given degree of freedom's
    velocity has dropped below a threshold derived from its own maximum velocity, and
    at least ``minimum_time`` simulated seconds of trajectory time have elapsed.

    :param context: Supplies the world's active degrees of freedom and control cycle
        timing.
    :param joint_convergence_threshold: Fraction of a degree of freedom's maximum
        velocity below which it is considered settled.
    :param minimum_threshold: Lower bound for the per-degree-of-freedom velocity
        threshold.
    :param maximum_threshold: Upper bound for the per-degree-of-freedom velocity
        threshold.
    :param degrees_of_freedom: Degrees of freedom to check for convergence. Defaults to
        every active degree of freedom in the world when ``None``. Those without an
        upper velocity limit are skipped, since no threshold can be derived for them.
    :param minimum_time: Minimum elapsed control time before the expression can become
        true.
    :param reference_cycle_variable: Cycle count elapsed time is measured from, instead
        of the start of the whole motion chart. Pass a variable a caller updates in its
        own ``on_start`` so ``minimum_time`` gates on how long that caller has been
        active, not on how many cycles the entire chart has already ticked through.
        ``None`` keeps the chart-wide behaviour.
    :return: A trinary :class:`~krrood.symbolic_math.symbolic_math.Scalar` expression,
        true once the given degrees of freedom have settled.
    """
    degrees_of_freedom = (
        degrees_of_freedom
        if degrees_of_freedom is not None
        else context.world.active_degrees_of_freedom
    )
    ref = []
    symbols = []
    for dof in degrees_of_freedom:
        if dof.limits.upper.velocity is None:
            # nothing to derive a threshold from, so this degree of freedom cannot
            # converge by this measure; environment joints are routinely parsed
            # without a velocity limit
            continue
        velocity_limit = dof.limits.upper.velocity * joint_convergence_threshold
        velocity_limit = min(max(minimum_threshold, velocity_limit), maximum_threshold)
        ref.append(velocity_limit)
        symbols.append(dof.variables.velocity)

    dt = (
        context.qp_controller_config.control_dt
        or context.qp_controller_config.model_predictive_control_time_step
    )
    elapsed_cycles = context.control_cycle_variable
    if reference_cycle_variable is not None:
        elapsed_cycles = elapsed_cycles - reference_cycle_variable
    trajectory_longer_than_minimum_time = elapsed_cycles * dt > minimum_time
    return sm.trinary_logic_and(
        trajectory_longer_than_minimum_time,
        sm.logic_all(sm.abs(sm.Vector(symbols)) < sm.Vector(ref)),
    )


@dataclass(eq=False, repr=False)
class Task(MotionStatechartNode):
    """
    Tasks are MotionStatechartNodes that add motion constraints.
    """

    weight: float = field(
        default=DefaultWeights.WEIGHT_BELOW_COLLISION_AVOIDANCE.value, kw_only=True
    )
    """Task priority relative to other tasks."""

    plot_specs: NodePlotSpec = plot_specification_field(NodePlotSpec.create_task_style)


@dataclass(eq=False, repr=False)
class ConvergingTask(ABC, Task):
    """
    A task that drives a single scalar error towards zero and counts as having reached
    its goal once that error is within :attr:`threshold`.

    Reaching the goal is not by itself a reason to end: the same task is a milestone in
    a sequence and an invariant to hold inside a goal that grasps something. Whatever
    ends it reads what the task observes to decide whether it succeeded.

    Subclasses declare the error rather than the observation, so that "reached the goal"
    is defined in one place, and so that how fast the goal is being approached can be
    measured. Tasks that enforce an invariant instead of converging, such as a velocity
    limit or a collision predicate, are plain :class:`Task` and write their own
    observation.
    """

    threshold: float = field(default=0.01, kw_only=True)
    """Error at or below which the goal counts as reached, in the task's own units."""

    def build(self, context: MotionStatechartContext) -> NodeArtifacts:
        """
        Build the task and derive its observation from its error.

        Being within :attr:`threshold` is what this task observes about the world, and
        succeeding means reaching it, which is what any node is judged by.
        """
        artifacts = super().build(context)
        if artifacts.error is None:
            raise MissingErrorSignalError(node=self)
        artifacts.observation = artifacts.error.expression <= self.threshold
        return artifacts

    @abstractmethod
    def build_artifacts(self, context: MotionStatechartContext) -> NodeArtifacts:
        """
        Add the motion constraints of this task and set :attr:`NodeArtifacts.error` to
        the error they drive to zero.

        :param context: The context that contains data that can be used to build this
            task.
        :return: The artifacts describing this task.
        """

    @property
    def error_signal(self) -> ErrorSignal:
        """
        :return: The error signal produced during build.
        """
        if self._error_signal is None:
            raise NodeNotBuiltError(node=self)
        return self._error_signal

    @property
    def normalized_error(self) -> Scalar:
        """
        The error divided by :attr:`threshold`, so that a value of at most 1 means the
        goal is reached.

        Dividing out the threshold makes errors of different tasks, and of different
        units, comparable against a single convergence rate.

        :return: The threshold relative error of this task.
        """
        return self.error_signal.expression / self.threshold


@dataclass(eq=False, repr=False)
class Goal(MotionStatechartNode):
    nodes: List[MotionStatechartNode] = field(default_factory=list, init=False)
    plot_specifications: NodePlotSpec = plot_specification_field(
        NodePlotSpec.create_goal_style
    )

    def expand(self, context: MotionStatechartContext) -> None:
        """
        Instantiate child nodes, add them to this goal, and wire their life cycle transition conditions.
        ..warning:: Nodes have not been built yet.
        :param context: The context that contains data that can be used to expand this goal.
        """

    def add_node(self, node: MotionStatechartNode) -> None:
        """
        Adds a node to this goal and the motion statechart this goal belongs to.

        :param node: The node to add as a child of this goal.
        """
        self._add_node_sanity_check(node)
        if node not in self.nodes:
            self.nodes.append(node)
        if node._motion_statechart is self.motion_statechart:
            return
        node.parent_node = self
        self.motion_statechart.add_node(node)

    def _add_node_sanity_check(self, node: MotionStatechartNode) -> None:
        """
        Rejects nodes that may not become a child of this goal.

        :param node: The node to validate.
        """
        self._check_node_has_no_end_motion(node)
        self._check_node_doesnt_belong_to_different_parent(node)

    def _check_has_children(self) -> None:
        """
        Rejects a goal that was built without the child nodes it exists to run.

        Call this at the start of :meth:`expand`, while :attr:`nodes` still holds only
        what the caller passed.

        :raises GoalWithoutChildrenError: If this goal has no child nodes.
        """
        if not self.nodes:
            raise GoalWithoutChildrenError(node=self)

    def _check_node_has_no_end_motion(self, node: MotionStatechartNode) -> None:
        """
        Rejects nodes that end the whole motion.

        :param node: The node to validate.
        """
        if isinstance(node, EndMotion):
            raise EndMotionInGoalError(node=self)

    def _check_node_doesnt_belong_to_different_parent(self, node: MotionStatechartNode):
        """
        .. note:: A node held by a *different* motion statechart is allowed, because it is
            moved into this goal's statechart; only two parents within one statechart are
            an error.
        """
        if node.belongs_to_motion_statechart() and node.parent_node != self:
            raise NodeAlreadyBelongsToDifferentNodeError(node=self, new_node=node)

    def add_nodes(self, nodes: List[MotionStatechartNode]) -> None:
        """
        Adds multiple nodes to this goal and the motion statechart this goal belongs to.

        :param nodes: The nodes to add as children of this goal.
        """
        for node in nodes:
            self.add_node(node)


@dataclass(eq=False, repr=False)
class ThreadPayloadMonitor(ABC, MotionStatechartNode):
    """
    Payload monitor that evaluates _compute_observation in a background thread.

    - compute_observation triggers an async evaluation and immediately returns.
    - Until the first successful completion, returns TrinaryUnknown.
    - Afterwards, returns the last successfully computed value.
    """

    # Internal threading primitives
    _request_event: threading.Event = field(
        default_factory=threading.Event, init=False, repr=False
    )
    _stop_event: threading.Event = field(
        default_factory=threading.Event, init=False, repr=False
    )
    _thread: threading.Thread = field(init=False, repr=False)

    # Cache of last successful result from _compute_observation
    _has_result: bool = field(default=False, init=False, repr=False)
    _last_result: float = field(
        default=ObservationStateValues.UNKNOWN, init=False, repr=False
    )

    def __post_init__(self):
        super().__post_init__()
        # Start a daemon worker thread that computes observations when requested
        self._thread = threading.Thread(
            target=self._worker_loop,
            name=f"{self.__class__.__name__}-worker",
            daemon=True,
        )
        self._thread.start()

    def compute_observation(
        self,
    ) -> float:
        """
        Requests a fresh observation from the worker thread without waiting for it.

        :return: The last successfully computed observation, unknown until the first one finished.
        """
        # Signal the worker to compute a fresh value if it is not already signaled.
        self._request_event.set()
        # Return the last known result (initialized to Unknown until first success)
        return self._last_result

    def cleanup(self, context: MotionStatechartContext):
        """
        Stops the background worker thread.
        """
        self._stop_event.set()
        self._thread.join(timeout=1.0)

    def _worker_loop(self):
        while not self._stop_event.is_set():
            # Wait until a request is made (wake periodically to check for stop)
            triggered = self._request_event.wait(timeout=0.1)
            if not triggered:
                continue
            # Clear early to allow new requests while we compute
            self._request_event.clear()
            try:
                result = self._compute_observation()
                self._last_result = result
                self._has_result = True
            except Exception:
                # Keep the previous result, but surface the failure instead of hiding it.
                logger.exception(
                    "%s failed to compute its observation.", self.__class__.__name__
                )


@dataclass(eq=False, repr=False)
class TerminalNode(ABC, MotionStatechartNode):
    """
    A node that ends the whole motion once its observation state turns true.

    No transition can happen afterwards, so conditions may not reference such a node.
    """


@dataclass(eq=False, repr=False)
class EndMotion(TerminalNode):

    plot_specs: NodePlotSpec = plot_specification_field(NodePlotSpec.create_end_style)

    joint_convergence_threshold: float = field(default=0.01, kw_only=True)
    """
    Fraction of a degree of freedom's maximum velocity below which it is considered
    settled. Only used while at least one active degree of freedom exists; see
    :meth:`build`.
    """

    minimum_threshold: float = field(default=0.01, kw_only=True)
    """
    Lower bound for the per-degree-of-freedom velocity threshold.
    """

    maximum_threshold: float = field(default=0.06, kw_only=True)
    """
    Upper bound for the per-degree-of-freedom velocity threshold.
    """

    def build_artifacts(self, context: MotionStatechartContext) -> NodeArtifacts:
        """
        Reports "done" only once the world has actually settled, so the motion isn't
        cut short while the controller is still commanding nonzero velocity.
        .. note:: If the world has no active degrees of freedom, there is nothing to
            converge, so this reports done immediately once running, same as before.
        """
        if not context.world.active_degrees_of_freedom:
            return NodeArtifacts(observation=Scalar.const_true())
        observation = velocity_convergence_expression(
            context=context,
            joint_convergence_threshold=self.joint_convergence_threshold,
            minimum_threshold=self.minimum_threshold,
            maximum_threshold=self.maximum_threshold,
        )
        return NodeArtifacts(observation=observation)

    @classmethod
    def when_true(cls, node: MotionStatechartNode) -> Self:
        """
        Factory method for creating an EndMotion node that activates once the given node
        reached its goal.

        :param node: The node whose goal ends the motion.
        :return: The new EndMotion node.
        """
        end = cls()
        end.start_condition = node.goal_reached
        return end

    @classmethod
    def when_failed(cls, node: MotionStatechartNode) -> Self:
        """
        Factory method for creating an EndMotion node that activates once the given node
        ended by failing.

        :param node: The node whose failure ends the motion.
        :return: The new EndMotion node.
        """
        end = cls()
        end.start_condition = node.is_failed
        return end

    @classmethod
    def when_false(cls, node: MotionStatechartNode) -> Self:
        """
        Factory method for creating an EndMotion node that activates while the given node
        has a false observation state.

        Unlike its counterparts this asks only what the node observes now, so it stops
        mattering once that node ends rather than latching onto the verdict it earned.

        .. note:: Use :meth:`when_failed` to wait for a node to end short of its goal.

        :param node: The node whose observation state activates the created node.
        :return: The new EndMotion node.
        """
        end = cls()
        end.start_condition = trinary_logic_not(node.observation_variable)
        return end

    @classmethod
    def when_all_true(cls, nodes: List[MotionStatechartNode]) -> Self:
        """
        Factory method for creating an EndMotion node that activates once *all* of the
        given nodes reached their goals.

        :param nodes: The nodes whose goals end the motion.
        :return: The new EndMotion node.
        """
        end = cls()
        end.start_condition = sm.trinary_logic_and(
            *[node.goal_reached for node in nodes]
        )
        return end

    @classmethod
    def when_any_true(cls, nodes: List[MotionStatechartNode]) -> Self:
        """
        Factory method for creating an EndMotion node that activates once *any* of the
        given nodes reached its goal.

        :param nodes: The nodes whose goals end the motion.
        :return: The new EndMotion node.
        """
        end = cls()
        end.start_condition = sm.trinary_logic_or(
            *[node.goal_reached for node in nodes]
        )
        return end


@dataclass(eq=False, repr=False)
class CancelMotion(TerminalNode):
    """
    Ends the motion by raising :attr:`exception`.

    Its factory methods mirror :class:`EndMotion`'s: they read whether a node reached its
    goal, which keeps answering once that node has ended, rather than the observation
    behind it, which is gone by then.
    """

    exception: DataclassException = field(kw_only=True)
    observation_expression: Scalar = field(
        default_factory=Scalar.const_true, init=False
    )

    plot_specs: NodePlotSpec = plot_specification_field(
        NodePlotSpec.create_cancel_style
    )

    def build_artifacts(self, context: MotionStatechartContext) -> NodeArtifacts:
        return NodeArtifacts(observation=Scalar.const_true())

    def on_tick(self, context: MotionStatechartContext) -> Optional[float]:
        raise self.exception

    @classmethod
    def when_true(
        cls, node: MotionStatechartNode, exception: Optional[Exception] = None
    ) -> Self:
        """
        Factory method for creating a CancelMotion node that activates once the given
        node reached its goal.

        :param node: The node whose goal activates the created node.
        :param exception: The exception raised on activation, defaults to one naming the given node.
        :return: The new CancelMotion node.
        """
        exception = exception or Exception(
            f"Cancelled because {node.unique_name} reached its goal"
        )
        end = cls(exception=exception)
        end.start_condition = node.goal_reached
        return end

    @classmethod
    def when_failed(
        cls, node: MotionStatechartNode, exception: Optional[Exception] = None
    ) -> Self:
        """
        Factory method for creating a CancelMotion node that activates once the given
        node ended by failing.

        :param node: The node whose failure activates the created node.
        :param exception: The exception raised on activation, defaults to one naming the given node.
        :return: The new CancelMotion node.
        """
        exception = exception or Exception(
            f"Cancelled because {node.unique_name} failed"
        )
        end = cls(exception=exception)
        end.start_condition = node.is_failed
        return end

    @classmethod
    def when_all_true(
        cls, nodes: List[MotionStatechartNode], exception: Exception
    ) -> Self:
        """
        Factory method for creating a CancelMotion node that activates once *all* of the
        given nodes reached their goals.

        :param nodes: The nodes whose goals activate the created node.
        :param exception: The exception raised on activation.
        :return: The new CancelMotion node.
        """
        end = cls(exception=exception)
        end.start_condition = sm.trinary_logic_and(
            *[node.goal_reached for node in nodes]
        )
        return end

    @classmethod
    def when_any_true(
        cls, nodes: List[MotionStatechartNode], exception: Exception
    ) -> Self:
        """
        Factory method for creating a CancelMotion node that activates once *any* of the
        given nodes reached its goal.

        :param nodes: The nodes whose goals activate the created node.
        :param exception: The exception raised on activation.
        :return: The new CancelMotion node.
        """
        end = cls(exception=exception)
        end.start_condition = sm.trinary_logic_or(
            *[node.goal_reached for node in nodes]
        )
        return end
