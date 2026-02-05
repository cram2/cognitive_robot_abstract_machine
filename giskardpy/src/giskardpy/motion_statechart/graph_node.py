from __future__ import annotations

import ast
import re
import threading
from abc import ABC
from dataclasses import field, dataclass, fields

from typing_extensions import (
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
from giskardpy.motion_statechart.context import BuildContext, ExecutionContext
from giskardpy.motion_statechart.data_types import (
    LifeCycleValues,
    ObservationStateValues,
    TransitionKind,
)
from giskardpy.motion_statechart.exceptions import (
    NotInMotionStatechartError,
    EndMotionInGoalError,
    InputNotExpressionError,
    SelfInStartConditionError,
    NonObservationVariableError,
    NodeAlreadyBelongsToDifferentNodeError,
)
from giskardpy.motion_statechart.plotters.plot_specs import NodePlotSpec
from giskardpy.qp.constraint_collection import ConstraintCollection
from giskardpy.utils.utils import string_shortener
from krrood.adapters.json_serializer import (
    SubclassJSONSerializer,
    JSON_TYPE_NAME,
    to_json,
    DataclassJSONSerializer,
)
from krrood.symbolic_math.symbolic_math import FloatVariable, Scalar, trinary_logic_not
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.spatial_types import (
    Point3,
    Vector3,
    Quaternion,
    RotationMatrix,
    HomogeneousTransformationMatrix,
)
from semantic_digital_twin.world_description.geometry import Color

if TYPE_CHECKING:
    from giskardpy.motion_statechart.motion_statechart import (
        MotionStatechart,
    )


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
    expression: Scalar = field(default=lambda: Scalar.const_trinary_unknown())
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
        return cls(expression=Scalar.const_true(), kind=kind, owner=owner)

    @classmethod
    def create_false(
        cls, kind: TransitionKind, owner: Optional[MotionStatechartNode] = None
    ) -> Self:
        return cls(expression=Scalar.const_false(), kind=kind, owner=owner)

    @classmethod
    def create_unknown(
        cls, kind: TransitionKind, owner: Optional[MotionStatechartNode] = None
    ) -> Self:
        return cls(
            expression=Scalar.const_trinary_unknown(),
            kind=kind,
            owner=owner,
        )

    def update_expression(
        self, new_expression: Scalar, child: MotionStatechartNode
    ) -> None:
        self._sanity_check(new_expression)
        self.expression = new_expression
        self._child = child

    def _sanity_check(self, new_expression: Scalar) -> None:
        self._check_condition_is_variable_or_expression(new_expression)
        self._check_owner_not_in_start_condition(new_expression)
        self._check_only_observation_variables(new_expression)

    def _check_condition_is_variable_or_expression(self, new_expression: Scalar):
        if not isinstance(new_expression, Scalar):
            raise InputNotExpressionError(condition=self, new_expression=new_expression)

    def _check_only_observation_variables(self, new_expression: Scalar):
        free_variables = new_expression.free_variables()
        for variable in free_variables:
            if not isinstance(variable, ObservationVariable):
                raise NonObservationVariableError(
                    condition=self,
                    non_observation_variable=variable,
                    new_expression=new_expression,
                )

    def _check_owner_not_in_start_condition(self, new_expression: Scalar):
        if (
            self.kind == TransitionKind.START
            and self.owner.belongs_to_motion_statechart()
            and self.owner.observation_variable in new_expression.free_variables()
        ):
            raise SelfInStartConditionError(
                condition=self, new_expression=new_expression
            )

    @property
    def node_dependencies(self) -> List[MotionStatechartNode]:
        """
        List of parent nodes involved in the condition, derived from the free symbols in the expression.
        """
        return [
            x.motion_statechart_node
            for x in self.expression.free_variables()
            if isinstance(x, ObservationVariable)
        ]

    def __str__(self):
        """
        Replaces the state symbols with motion statechart node names and formats it nicely.
        """
        free_symbols = self.expression.free_variables()
        if not free_symbols:
            str_representation = str(self.expression.is_const_true())
        else:
            str_representation = sm.trinary_logic_to_str(self.expression)
        str_representation = re.sub(
            r'"([^"]*?)/observation"', r'"\1"', str_representation
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
        observation_variables: List[ObservationVariable],
        owner: Optional[MotionStatechartNode] = None,
    ):
        tree = ast.parse(trinary_logic_str, mode="eval")
        return cls(
            kind=kind,
            expression=cls._parse_ast_expression(tree.body, observation_variables),
            owner=owner,
        )

    @staticmethod
    def _parse_ast_expression(
        node: ast.expr, observation_variables: List[ObservationVariable]
    ) -> Scalar:
        match node:
            case ast.BoolOp(op=ast.And()):
                return TrinaryCondition._parse_ast_and(node, observation_variables)
            case ast.BoolOp(op=ast.Or()):
                return TrinaryCondition._parse_ast_or(node, observation_variables)
            case ast.UnaryOp():
                return TrinaryCondition._parse_ast_not(node, observation_variables)
            case ast.Constant(value=str(val)):
                variable_name = str(PrefixedName("observation", val))
                for v in observation_variables:
                    if variable_name == v.name:
                        return v
                raise KeyError(f"unknown observation variable: {val!r}")
            case ast.Constant(value=True):
                return Scalar.const_true()
            case ast.Constant(value=False):
                return Scalar.const_false()
            case _:
                raise TypeError(f"failed to parse {type(node).__name__}")

    @staticmethod
    def _parse_ast_and(node, observation_variables: List[ObservationVariable]):
        return sm.trinary_logic_and(
            *[
                TrinaryCondition._parse_ast_expression(x, observation_variables)
                for x in node.values
            ]
        )

    @staticmethod
    def _parse_ast_or(node, observation_variables: List[ObservationVariable]):
        return sm.trinary_logic_or(
            *[
                TrinaryCondition._parse_ast_expression(x, observation_variables)
                for x in node.values
            ]
        )

    @staticmethod
    def _parse_ast_not(node, observation_variables: List[ObservationVariable]):
        if isinstance(node.op, ast.Not):
            return sm.trinary_logic_not(
                TrinaryCondition._parse_ast_expression(
                    node.operand, observation_variables
                )
            )

    @classmethod
    def _from_json(
        cls, data: Dict[str, Any], motion_statechart: MotionStatechart, **kwargs
    ) -> Self:
        return cls.create_from_trinary_logic_str(
            kind=TransitionKind[data["kind"]],
            trinary_logic_str=data["expression"],
            observation_variables=motion_statechart.observation_state.observation_symbols(),
            owner=motion_statechart.get_node_by_index(data["owner"]),
        )


@dataclass(repr=False, eq=False, init=False)
class ObservationVariable(FloatVariable):
    """
    A symbol representing the observation state of a node.
    """

    motion_statechart_node: MotionStatechartNode = field(kw_only=True)
    """
    The node this variable is the observation state of.
    """

    def __init__(self, name: str, motion_statechart_node: MotionStatechartNode):
        super().__init__(name)
        self.motion_statechart_node = motion_statechart_node

    def resolve(self) -> float:
        return self.motion_statechart_node.observation_state


@dataclass(repr=False, eq=False, init=False)
class LifeCycleVariable(FloatVariable):
    """
    A symbol representing the life cycle state of a node.
    """

    motion_statechart_node: MotionStatechartNode = field(kw_only=True)
    """
    The node this variable is the life cycle state of.
    """

    def __init__(self, name: str, motion_statechart_node: MotionStatechartNode):
        super().__init__(name)
        self.motion_statechart_node = motion_statechart_node

    def resolve(self) -> LifeCycleValues:
        return self.motion_statechart_node.life_cycle_state


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
    )

    color: Color = field(default_factory=lambda: Color(1, 0, 0, 1))
    """
    The color used when this expression is rendered in visualization tools.
    """


@dataclass
class NodeArtifacts:
    """
    Represents the artifacts produced by the `build` method of a node.
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
    debug_expressions: List[DebugExpression] = field(default_factory=list)
    """
    A list of symbolic expressions used for debugging only.
    While in debug mode, you can call .evaluate() on them to get their current value.
    """


@dataclass(repr=False, eq=False)
class MotionStatechartNode(SubclassJSONSerializer):
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

    parent_node_index: Optional[int] = field(default=None, init=False)
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

    _constraint_collection: ConstraintCollection = field(init=False, repr=False)
    """The parameter is set after build() using its NodeArtifacts."""
    _observation_expression: Scalar = field(init=False, repr=False)
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
    Decides when this node transitions from RUNNING or PAUSED to DONE.
    """
    _reset_condition: TrinaryCondition = field(init=False, default=None)
    """
    Decides when this transitions to NOT_STARTED.
    """

    plot_specs: NodePlotSpec = field(
        default_factory=NodePlotSpec.create_monitor_style, kw_only=True, init=False
    )

    def __post_init__(self):
        if self.name is None:
            self.name = self.__class__.__name__
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

    def _post_add_to_motion_statechart(self):
        """
        Called after this node is added to a motion statechart.
        Finalizes the initialization parts that require the motion statechart to be set.
        """
        self._observation_variable = ObservationVariable(
            name=str(PrefixedName("observation", self.unique_name)),
            motion_statechart_node=self,
        )
        self._life_cycle_variable = LifeCycleVariable(
            name=str(PrefixedName("life_cycle", self.unique_name)),
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
    def depth(self) -> int:
        """
        Distance (in edges) from this node to the root of the motion statechart.

        The root node (no parent) has depth 0, its children depth 1, and so on.
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
        if parent_node is None:
            self.parent_node_index = None
        else:
            self.parent_node_index = parent_node.index

    def _set_transition(self, transition: TrinaryCondition) -> None:
        """
        Sets the transition condition for this node, depending on its kind.
        Used in json parsing.
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

    def create_lifecycle_transitions(
        self,
    ) -> Tuple[
        sm.Scalar,
        sm.Scalar,
        sm.Scalar,
        sm.Scalar,
    ]:
        """
        Create the life cycle transitions for this node.
        :return: A tuple of (not_started_transitions, running_transitions, pause_transitions, ended_transitions)
        """
        any_end_condition_true = self._create_any_ancestor_condition_true(
            TransitionKind.END
        )
        any_reset_condition_true = self._create_any_ancestor_condition_true(
            TransitionKind.RESET
        )

        not_started_transitions = self._create_not_started_transitions()
        running_transitions = self._create_running_transitions(
            any_end_condition_true=any_end_condition_true,
            any_reset_condition_true=any_reset_condition_true,
        )
        pause_transitions = self._create_pause_transitions(
            any_end_condition_true=any_end_condition_true,
            any_reset_condition_true=any_reset_condition_true,
        )
        ended_transitions = self._create_ended_transitions(
            any_reset_condition_true=any_reset_condition_true
        )

        return (
            not_started_transitions,
            running_transitions,
            pause_transitions,
            ended_transitions,
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
        condition = sm.Scalar(
            current_node.get_condition(transition_kind) == sm.Scalar.const_true()
        )
        while current_node.parent_node is not None:
            current_node = current_node.parent_node
            cond_expr = sm.Scalar(
                current_node.get_condition(transition_kind) == sm.Scalar.const_true()
            )
            condition = sm.trinary_logic_or(condition, cond_expr)
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

    def _create_ended_transitions(
        self, any_reset_condition_true: sm.Scalar
    ) -> sm.Scalar:
        """
        Create the ended transitions of the LifeCycleState for this node.
        :param any_reset_condition_true: The combined reset condition for this node and its parents. Combined using trinary_logic_or.
        :return: The LifeCycleState transitions for the DONE state.
        """
        return sm.if_else(
            condition=any_reset_condition_true,
            if_result=sm.Scalar(LifeCycleValues.NOT_STARTED),
            else_result=sm.Scalar(LifeCycleValues.DONE),
        )

    def _create_pause_transitions(
        self,
        any_end_condition_true: sm.Scalar,
        any_reset_condition_true: sm.Scalar,
    ) -> sm.Scalar:
        """
        Create the pause transitions of the LifeCycleState for this node.
        :param any_end_condition_true: The combined end condition for this node and its parents. Combined using trinary_logic_or.
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
                (any_end_condition_true, sm.Scalar(LifeCycleValues.DONE)),
                (
                    unpause_condition,
                    sm.Scalar(LifeCycleValues.RUNNING),
                ),
            ],
            else_result=sm.Scalar(LifeCycleValues.PAUSED),
        )

    def _create_running_transitions(
        self,
        any_end_condition_true: sm.Scalar,
        any_reset_condition_true: sm.Scalar,
    ) -> sm.Scalar:
        """
        Create the running transitions of the LifeCycleState for this node.
        :param any_end_condition_true: The combined end condition for this node and its parents. Combined using trinary_logic_or.
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
                (any_end_condition_true, sm.Scalar(LifeCycleValues.DONE)),
                (any_pause_condition, sm.Scalar(LifeCycleValues.PAUSED)),
            ],
            else_result=sm.Scalar(LifeCycleValues.RUNNING),
        )

    def _create_not_started_transitions(self) -> sm.Scalar:
        """
        Create the not started transitions of the LifeCycleState for this node.
        :return: The LifeCycleState transitions for the NOT_STARTED state.
        """
        start_condition = sm.Scalar(self.start_condition == sm.Scalar.const_true())
        current = self
        while current.parent_node is not None:
            parent = current.parent_node
            start_condition = sm.trinary_logic_and(
                start_condition,
                sm.trinary_logic_not(parent.end_condition),
                sm.Scalar(parent.start_condition == sm.Scalar.const_true()),
            )
            current = parent

        return sm.if_else(
            condition=start_condition,
            if_result=sm.Scalar(LifeCycleValues.RUNNING),
            else_result=sm.Scalar(LifeCycleValues.NOT_STARTED),
        )

    @property
    def life_cycle_variable(self) -> LifeCycleVariable:
        """
        The variable representing the life cycle state of this node.
        :return:
        """
        if self._life_cycle_variable is None:
            raise NotInMotionStatechartError(self.name)
        return self._life_cycle_variable

    def belongs_to_motion_statechart(self) -> bool:
        return self._motion_statechart is not None

    @property
    def observation_variable(self) -> ObservationVariable:
        if not self.belongs_to_motion_statechart():
            raise NotInMotionStatechartError(self.name)
        return self._observation_variable

    @property
    def motion_statechart(self) -> MotionStatechart:
        if self._motion_statechart is None:
            raise NotInMotionStatechartError(self.name)
        return self._motion_statechart

    @motion_statechart.setter
    def motion_statechart(self, motion_statechart: MotionStatechart) -> None:
        self._motion_statechart = motion_statechart

    def build(self, context: BuildContext) -> NodeArtifacts:
        """
        Called exactly once during motion statechart compilation.
        Use this method for any setup steps.
        .. warning:: Don't create other nodes within this function.
        :param context: The context that contains data that can be used to build this node.
        :return: A NodeArtifacts instance that describes this node. It is normal for nodes that don't directly affect the motion to return empty NodeArtifacts.
        """
        return NodeArtifacts()

    def on_tick(self, context: ExecutionContext) -> Optional[ObservationStateValues]:
        """
        Triggered when the node is ticked.
        .. warning:: This method is called inside a control loop, make sure it is fast.
        .. warning:: Only happens while the node is in state RUNNING.
        .. warning:: The result of this method takes precedence over the observation expression created in build().
        :return: An optional observation state overwrite
        """

    def on_start(self, context: ExecutionContext):
        """
        Triggered when the node transitions from NOT_STARTED to RUNNING.
        .. warning:: This method is called inside a control loop, make sure it is fast.
        """

    def on_pause(self, context: ExecutionContext):
        """
        Triggered when the node transitions from RUNNING to PAUSED.
        .. warning:: This method is called inside a control loop, make sure it is fast.
        """

    def on_unpause(self, context: ExecutionContext):
        """
        Triggered when the node transitions from PAUSED to RUNNING.
        .. warning:: This method is called inside a control loop, make sure it is fast.
        """

    def on_end(self, context: ExecutionContext):
        """
        Triggered when the node transitions from RUNNING to DONE.
        .. warning:: This method is called inside a control loop, make sure it is fast.
        """

    def on_reset(self, context: ExecutionContext):
        """
        Triggered when the node transitions from any state to NOT_STARTED.
        .. warning:: This method is called inside a control loop, make sure it is fast.
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
        return self._start_condition.expression

    @start_condition.setter
    def start_condition(self, expression: Scalar) -> None:
        if self._start_condition is None:
            raise NotInMotionStatechartError(self.name)
        self._start_condition.update_expression(expression, self)

    @property
    def pause_condition(self) -> Scalar:
        return self._pause_condition.expression

    @pause_condition.setter
    def pause_condition(self, expression: Scalar) -> None:
        if self._pause_condition is None:
            raise NotInMotionStatechartError(self.name)
        self._pause_condition.update_expression(expression, self)

    @property
    def end_condition(self) -> Scalar:
        return self._end_condition.expression

    @end_condition.setter
    def end_condition(self, expression: Scalar) -> None:
        if self._end_condition is None:
            raise NotInMotionStatechartError(self.name)
        self._end_condition.update_expression(expression, self)

    @property
    def reset_condition(self) -> Scalar:
        return self._reset_condition.expression

    @reset_condition.setter
    def reset_condition(self, expression: Scalar) -> None:
        if self._reset_condition is None:
            raise NotInMotionStatechartError(self.name)
        self._reset_condition.update_expression(expression, self)

    def to_json(self) -> Dict[str, Any]:
        json_data = super().to_json()
        for field_ in fields(self):
            if not field_.name.startswith("_") and field_.init:
                value = getattr(self, field_.name)
                json_data[field_.name] = to_json(value)
        if self.parent_node_index is not None:
            json_data["parent_node_index"] = self.parent_node_index
        return json_data

    @classmethod
    def _from_json(cls, data: Dict[str, Any], **kwargs) -> Self:
        node_kwargs = {}
        for field_name, field_data in data.items():
            if field_name == JSON_TYPE_NAME:
                continue
            if isinstance(field_data, dict) and JSON_TYPE_NAME in field_data:
                field_data = SubclassJSONSerializer.from_json(field_data, **kwargs)
            if isinstance(field_data, list):
                field_data = [
                    SubclassJSONSerializer.from_json(element_data, **kwargs)
                    for element_data in field_data
                ]
            if isinstance(field_data, dict):
                raise NotImplementedError(
                    "dict parameters of MotionStatechartNode are not supported yet. Use a list instead."
                )
            node_kwargs[field_name] = field_data
        parent_node_index = node_kwargs.pop("parent_node_index", None)
        result = cls(**node_kwargs)
        result.parent_node_index = parent_node_index
        return result

    def formatted_name(self, quoted: bool = False) -> str:
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
        return f"{self.name}#{self.index}"

    def __repr__(self) -> str:
        return self.unique_name


GenericMotionStatechartNode = TypeVar(
    "GenericMotionStatechartNode", bound=MotionStatechartNode
)


@dataclass(eq=False, repr=False)
class Task(MotionStatechartNode):
    """
    Tasks are MotionStatechartNodes that add motion constraints.
    """

    plot_specs: NodePlotSpec = field(
        default_factory=NodePlotSpec.create_task_style, kw_only=True, init=False
    )


@dataclass(eq=False, repr=False)
class Goal(MotionStatechartNode):
    nodes: List[MotionStatechartNode] = field(default_factory=list, init=False)
    plot_specs: NodePlotSpec = field(
        default_factory=NodePlotSpec.create_goal_style, kw_only=True, init=False
    )

    def expand(self, context: BuildContext) -> None:
        """
        Instantiate child nodes, add them to this goal, and wire their life cycle transition conditions.
        ..warning:: Nodes have not been built yet.
        """

    def add_node(self, node: MotionStatechartNode) -> None:
        """
        Adds a node to this goal and the motion statechart this goal belongs to.
        Should be used in expand().
        """
        self._add_node_sanity_check(node)
        if node not in self.nodes:
            self.nodes.append(node)
        node.parent_node = self
        self.motion_statechart.add_node(node)

    def _add_node_sanity_check(self, node: MotionStatechartNode) -> None:
        self._check_node_has_no_end_motion(node)
        self._check_node_doesnt_belong_to_different_parent(node)

    def _check_node_has_no_end_motion(self, node: MotionStatechartNode) -> None:
        if isinstance(node, EndMotion):
            raise EndMotionInGoalError(node=self)

    def _check_node_doesnt_belong_to_different_parent(self, node: MotionStatechartNode):
        if node.belongs_to_motion_statechart() and node.parent_node != self:
            raise NodeAlreadyBelongsToDifferentNodeError(node=self, new_node=node)

    def add_nodes(self, nodes: List[MotionStatechartNode]) -> None:
        for node in nodes:
            self.add_node(node)


@dataclass(eq=False, repr=False)
class ThreadPayloadMonitor(MotionStatechartNode, ABC):
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
        # Signal the worker to compute a fresh value if it is not already signaled.
        self._request_event.set()
        # Return the last known result (initialized to Unknown until first success)
        return self._last_result

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
                # Accept only valid trinary values (floats expected)
                self._last_result = result
                self._has_result = True
            except Exception:
                # On failure, keep previous result and mark as having no new value
                pass


@dataclass(eq=False, repr=False)
class EndMotion(MotionStatechartNode):

    plot_specs: NodePlotSpec = field(
        default_factory=NodePlotSpec.create_end_style, kw_only=True, init=False
    )

    def build(self, context: BuildContext) -> NodeArtifacts:
        return NodeArtifacts(observation=Scalar.const_true())

    @classmethod
    def when_true(cls, node: MotionStatechartNode) -> Self:
        """
        Factory method for creating an EndMotion node that activates when the given node has a true observation state.
        """
        end = cls()
        end.start_condition = node.observation_variable
        return end

    @classmethod
    def when_false(cls, node: MotionStatechartNode) -> Self:
        """
        Factory method for creating an EndMotion node that activates when the given node has a false observation state.
        """
        end = cls()
        end.start_condition = trinary_logic_not(node.observation_variable)
        return end

    @classmethod
    def when_all_true(cls, nodes: List[MotionStatechartNode]) -> Self:
        """
        Factory method for creating an EndMotion node that activates when ALL of the given nodes have a true observation state.
        """
        end = cls()
        end.start_condition = sm.trinary_logic_and(
            *[node.observation_variable for node in nodes]
        )
        return end

    @classmethod
    def when_any_true(cls, nodes: List[MotionStatechartNode]) -> Self:
        """
        Factory method for creating an EndMotion node that activates when ANY of the given nodes have a true observation state.
        """
        end = cls()
        end.start_condition = sm.trinary_logic_or(
            *[node.observation_variable for node in nodes]
        )
        return end


@dataclass(eq=False, repr=False)
class CancelMotion(MotionStatechartNode):
    exception: Exception = field(kw_only=True)
    observation_expression: Scalar = field(
        default_factory=Scalar.const_true, init=False
    )

    plot_specs: NodePlotSpec = field(
        default_factory=NodePlotSpec.create_cancel_style, kw_only=True, init=False
    )

    def build(self, context: BuildContext) -> NodeArtifacts:
        return NodeArtifacts(observation=Scalar.const_true())

    def on_tick(self, context: ExecutionContext) -> Optional[float]:
        raise self.exception

    def to_json(self) -> Dict[str, Any]:
        exception_field = next(f for f in fields(self) if f.name == "exception")
        # set init to False to prevent superclass from calling to_json on it
        exception_field.init = False
        json_data = super().to_json()
        # cast to general exception, because it can be json serialized
        json_data["exception"] = to_json(Exception(str(self.exception)))
        return json_data

    @classmethod
    def when_true(
        cls, node: MotionStatechartNode, exception: Optional[Exception] = None
    ) -> Self:
        """
        Factory method for creating an EndMotion node that activates when the given node has a true observation state.
        """
        exception = exception or Exception(
            f"Cancelled because {node.unique_name} is true"
        )
        end = cls(exception=exception)
        end.start_condition = node.observation_variable
        return end

    @classmethod
    def when_all_true(
        cls, nodes: List[MotionStatechartNode], exception: Exception
    ) -> Self:
        """
        Factory method for creating an EndMotion node that activates when ALL of the given nodes have a true observation state.
        """
        end = cls(exception=exception)
        end.start_condition = sm.trinary_logic_and(
            *[node.observation_variable for node in nodes]
        )
        return end

    @classmethod
    def when_any_true(
        cls, nodes: List[MotionStatechartNode], exception: Exception
    ) -> Self:
        """
        Factory method for creating an EndMotion node that activates when ANY of the given nodes have a true observation state.
        """
        end = cls(exception=exception)
        end.start_condition = sm.trinary_logic_or(
            *[node.observation_variable for node in nodes]
        )
        return end
