from __future__ import annotations

import ast
import logging
import threading
import uuid
from abc import ABC, ABCMeta, abstractmethod
from dataclasses import MISSING, field, dataclass

from typing_extensions import (
    Any,
    Self,
    TYPE_CHECKING,
    TypeVar,
    Callable,
)

import krrood.symbolic_math.symbolic_math as sm
from cramph.context import ContextExtension, StatechartContext
from cramph.data_types import (
    LifeCycleValues,
    LifeCyclePredicate,
    ObservationPredicate,
    ObservationReading,
    ObservationStateValues,
    TransitionKind,
    TransitionConditionJSONKey,
    NodeJSONKey,
)
from cramph.exceptions import (
    CancelledByNodeFailing,
    CancelledByNodeReachingItsGoal,
    ChildTransitionAlreadyWiredError,
    NotInStatechartError,
    EndInCompositeNodeError,
    CompositeNodeWithoutChildrenError,
    InputNotExpressionError,
    SelfInStartConditionError,
    UnsupportedConditionVariableError,
    NodeAlreadyBelongsToDifferentNodeError,
    NodeNotBuiltError,
    TerminalNodeInConditionError,
    UnknownConditionVariableError,
    UnsupportedConditionSyntaxError,
    NodeStateVariableNotSerializableError,
    StatechartAlreadyCompiledError,
    UnknownTransitionKindError,
)
from cramph.plotters.plot_specs import NodePlotSpec, plot_specification_field
from krrood.adapters.deserialized_object_tracker import DeserializedObjectTracker
from krrood.adapters.exceptions import UntrackedObjectError
from krrood.adapters.json_serializer import (
    DataclassJSONSerializer,
    SubclassJSONSerializer,
)
from krrood.exceptions import DataclassException
from krrood.ormatic.utils import classproperty
from krrood.patterns.field_metadata import JSONMetadata
from krrood.symbolic_math.symbolic_math import (
    FloatVariable,
    GenericSymbolicType,
    Scalar,
)
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName

if TYPE_CHECKING:
    from cramph.statechart import Statechart
    from semantic_digital_twin.world import World

logger = logging.getLogger(__name__)


@dataclass(eq=False, repr=False)
class TransitionCondition(SubclassJSONSerializer):
    """
    The condition deciding when one transition of a node's life cycle happens.

    A condition is two-valued: it combines two-valued node variables, such as
    :attr:`~StatechartNode.observes_true` or
    :attr:`~StatechartNode.is_succeeded`, with ``and``, ``or`` and ``not``.
    """

    kind: TransitionKind
    """
    The type of transition associated with this condition.
    """
    expression: Scalar = field(default_factory=Scalar.const_false)
    """
    The two-valued expression deciding when the transition happens.
    """

    owner: StatechartNode | None = field(default=None)
    """
    The node this transition belongs to.
    """

    def __hash__(self) -> int:
        return hash((str(self), self.kind, self.owner.index))

    def __eq__(self, other: object) -> bool:
        return hash(self) == hash(other)

    @classmethod
    def create_true(
        cls, kind: TransitionKind, owner: StatechartNode | None = None
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
        cls, kind: TransitionKind, owner: StatechartNode | None = None
    ) -> Self:
        """
        Creates a condition that always evaluates to false.

        :param kind: The type of transition this condition controls.
        :param owner: The node this condition belongs to.
        :return: The new condition.
        """
        return cls(expression=Scalar.const_false(), kind=kind, owner=owner)

    def update_expression(self, new_expression: Scalar, child: StatechartNode) -> None:
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
        self._check_is_two_valued_logic(new_expression)
        self._check_owner_not_in_start_condition(new_expression)
        self._check_no_terminal_node(new_expression)

    def _check_condition_is_variable_or_expression(
        self, new_expression: Scalar
    ) -> None:
        """
        Rejects values that are not symbolic expressions.

        :param new_expression: The expression to validate.
        """
        if not isinstance(new_expression, Scalar):
            raise InputNotExpressionError(condition=self, new_expression=new_expression)

    def _check_only_condition_variables(self, new_expression: Scalar) -> None:
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

    @staticmethod
    def _check_is_two_valued_logic(new_expression: Scalar) -> None:
        """
        Rejects expressions built from anything but the two-valued logic operators, which
        could not be rendered and parsed back.

        :param new_expression: The expression to validate.
        :raises CannotConvertToStringError: If `new_expression` has no rendered form.
        """
        sm.boolean_expression_to_str(new_expression)

    def _check_no_terminal_node(self, new_expression: Scalar) -> None:
        """
        Rejects references to nodes that end the statechart.

        .. note:: Runs after :meth:`_check_only_condition_variables`, so every free
            variable is known to refer to a node.

        :param new_expression: The expression to validate.
        """
        for variable in new_expression.free_variables():
            if isinstance(variable.statechart_node, TerminalNode):
                raise TerminalNodeInConditionError(
                    condition=self,
                    new_expression=new_expression,
                    terminal_node=variable.statechart_node,
                )

    def _check_owner_not_in_start_condition(self, new_expression: Scalar) -> None:
        """
        Rejects start conditions that reference the state of their own node.

        .. note:: Runs after :meth:`_check_only_condition_variables`, so every free
            variable is known to refer to a node.

        :param new_expression: The expression to validate.
        """
        if self.kind != TransitionKind.START:
            return
        for variable in new_expression.free_variables():
            if variable.statechart_node is self.owner:
                raise SelfInStartConditionError(
                    condition=self, new_expression=new_expression
                )

    @property
    def variables(self) -> list[ConditionVariable]:
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
    def node_dependencies(self) -> list[StatechartNode]:
        """
        :return: The nodes this condition reads.
        """
        return [variable.statechart_node for variable in self.variables]

    def __str__(self) -> str:
        """
        Renders the condition, naming each variable by its
        :attr:`~DerivedConditionVariable.display_name` so the result is readable.

        :return: The rendered condition.
        """
        return self._render(lambda variable: variable.display_name)

    def __repr__(self) -> str:
        return str(self)

    def _render(self, name_variable: Callable[[DerivedConditionVariable], str]) -> str:
        """
        Renders the condition with ``and``, ``or``, ``not``, ``True`` and ``False``.

        :param name_variable: Gives the name each variable is written as.
        :return: The rendered condition.
        """
        free_symbols = self.expression.free_variables()
        if not free_symbols:
            return str(self.expression.is_constant_true())
        rendered_condition = sm.boolean_expression_to_str(self.expression)
        for variable in free_symbols:
            rendered_condition = rendered_condition.replace(
                variable.name, name_variable(variable)
            )
        return rendered_condition

    def to_json(self, **kwargs: Any) -> dict[str, Any]:
        """
        Names the owner and every variable by the id of their node, which, unlike the
        index of a node, exists before the node joins a statechart.
        """
        json_data = super().to_json(**kwargs)
        json_data[TransitionConditionJSONKey.KIND] = self.kind.name
        json_data[TransitionConditionJSONKey.EXPRESSION] = self._render(
            lambda variable: str(variable.reference)
        )
        json_data[TransitionConditionJSONKey.OWNER] = self.owner._node_id
        return json_data

    @staticmethod
    def _parse_ast_expression(
        node: ast.expr, resolve_variable: Callable[[str], DerivedConditionVariable]
    ) -> Scalar:
        """
        Translates a parsed condition into a symbolic expression.

        :param node: The syntax tree node to translate.
        :param resolve_variable: Gives the variable a quoted name stands for.
        :return: The symbolic expression.
        :raises UnsupportedConditionSyntaxError: If `node` is neither a quoted name,
            ``True``, ``False``, nor an ``and``, ``or`` or ``not``.
        """
        match node:
            case ast.BoolOp(op=ast.And()):
                return TransitionCondition._parse_ast_and(node, resolve_variable)
            case ast.BoolOp(op=ast.Or()):
                return TransitionCondition._parse_ast_or(node, resolve_variable)
            case ast.UnaryOp(op=ast.Not()):
                return TransitionCondition._parse_ast_not(node, resolve_variable)
            case ast.Constant(value=str(variable_name)):
                return resolve_variable(variable_name)
            case ast.Constant(value=True):
                return Scalar.const_true()
            case ast.Constant(value=False):
                return Scalar.const_false()
            case _:
                raise UnsupportedConditionSyntaxError(
                    unsupported_part=ast.unparse(node)
                )

    @staticmethod
    def _parse_ast_and(
        node: ast.BoolOp, resolve_variable: Callable[[str], DerivedConditionVariable]
    ) -> Scalar:
        """
        Translates a parsed conjunction into a symbolic expression.

        :param node: The syntax tree node to translate.
        :param resolve_variable: Gives the variable a quoted name stands for.
        :return: The symbolic expression.
        """
        return sm.logic_and(
            *[
                TransitionCondition._parse_ast_expression(x, resolve_variable)
                for x in node.values
            ]
        )

    @staticmethod
    def _parse_ast_or(
        node: ast.BoolOp, resolve_variable: Callable[[str], DerivedConditionVariable]
    ) -> Scalar:
        """
        Translates a parsed disjunction into a symbolic expression.

        :param node: The syntax tree node to translate.
        :param resolve_variable: Gives the variable a quoted name stands for.
        :return: The symbolic expression.
        """
        return sm.logic_or(
            *[
                TransitionCondition._parse_ast_expression(x, resolve_variable)
                for x in node.values
            ]
        )

    @staticmethod
    def _parse_ast_not(
        node: ast.UnaryOp, resolve_variable: Callable[[str], DerivedConditionVariable]
    ) -> Scalar:
        """
        Translates a parsed negation into a symbolic expression.

        :param node: The syntax tree node to translate, whose operator is ``not``.
        :param resolve_variable: Gives the variable a quoted name stands for.
        :return: The symbolic expression.
        """
        return sm.logic_not(
            TransitionCondition._parse_ast_expression(node.operand, resolve_variable)
        )

    @classmethod
    def _from_json(cls, data: dict[str, Any], **kwargs: Any) -> Self:
        """
        Resolves the nodes the condition names through the
        :class:`DeserializedNodeTracker` in `kwargs`.

        :raises UnknownConditionVariableError: If the condition names a variable its
            node does not offer.
        :raises UnsupportedConditionSyntaxError: If the condition contains syntax that
            has no meaning as a condition.
        """
        nodes = DeserializedNodeTracker.from_kwargs(kwargs)
        tree = ast.parse(data[TransitionConditionJSONKey.EXPRESSION], mode="eval")
        return cls(
            kind=TransitionKind[data[TransitionConditionJSONKey.KIND]],
            expression=cls._parse_ast_expression(
                tree.body,
                lambda variable_name: ConditionVariableReference.parse(
                    variable_name
                ).resolve(nodes),
            ),
            owner=nodes.get(data[TransitionConditionJSONKey.OWNER]),
        )


@dataclass(repr=False, eq=False, init=False)
class NodeStateVariable(FloatVariable, ABC):
    """
    A symbol standing for part of the state of one node.
    """

    statechart_node: StatechartNode = field(kw_only=True)
    """
    The node this variable refers to.
    """

    def __init__(self, name: str, statechart_node: StatechartNode) -> None:
        super().__init__(name)
        self.statechart_node = statechart_node

    def _value_to_json(self, **kwargs: Any) -> dict[str, Any]:
        """
        :raises NodeStateVariableNotSerializableError: Always, since JSON cannot refer to
            the node this variable belongs to.
        """
        raise NodeStateVariableNotSerializableError(variable=self)

    @property
    def display_name(self) -> str:
        """
        :return: How this variable is written in a rendered condition, using the node's
            :attr:`~StatechartNode.unique_name` so it reproduces across processes.
        """
        return self.statechart_node.unique_name

    @abstractmethod
    def resolve(self) -> float:
        """
        :return: The part of its node's state this variable currently stands for.
        """


@dataclass(repr=False, eq=False, init=False)
class ConditionVariable(NodeStateVariable, ABC):
    """
    A node state variable a transition condition may read.

    Every one of them stands for a two-valued answer, which is what keeps a condition
    built from them two-valued.
    """

    @abstractmethod
    def resolve(self) -> ObservationStateValues:
        """
        Narrows :meth:`NodeStateVariable.resolve` to a two-valued answer.

        :return: True or false, whichever this variable currently stands for.
        """


@dataclass(repr=False, eq=False, init=False)
class ObservationVariable(NodeStateVariable):
    """
    A symbol representing the observation state of a node.

    .. warning:: Legal in observation expressions, but not in transition conditions,
        because it may be unknown. Use an :class:`ObservationPredicateVariable` there.
    """

    def resolve(self) -> ObservationStateValues:
        return self.statechart_node.observation_state


@dataclass(repr=False, eq=False, init=False)
class LifeCycleVariable(NodeStateVariable):
    """
    A symbol representing the life cycle state of a node.

    .. warning:: Legal in observation expressions, but not in transition conditions.
        Use a :class:`LifeCyclePredicateVariable` there, so the condition stays renderable.
    """

    def resolve(self) -> LifeCycleValues:
        return self.statechart_node.life_cycle_state


@dataclass(repr=False, eq=False, init=False)
class LastObservationVariable(NodeStateVariable):
    """
    A symbol representing the observation a node took most recently, which it keeps once
    it has ended, whatever its outcome.

    .. warning:: Legal in observation expressions, but not in transition conditions,
        because it may be unknown. Use an :class:`ObservationPredicateVariable` there.
    """

    @classproperty
    def attribute_name(cls) -> str:
        """
        :return: The name this variable is reached under on a node, also used to render
            it inside a condition.
        """
        return "last_observation"

    @property
    def display_name(self) -> str:
        return f"{self.statechart_node.unique_name}.{self.attribute_name}"

    def resolve(self) -> ObservationStateValues:
        return self.statechart_node.last_observation_state


@dataclass(repr=False, eq=False, init=False)
class DerivedConditionVariable(ConditionVariable, ABC):
    """
    A condition variable standing for an expression over its node's state variables,
    which replaces it before anything is compiled.
    """

    predicate: LifeCyclePredicate | ObservationPredicate = field(kw_only=True)
    """
    The test this variable holds the value of.
    """

    @property
    def display_name(self) -> str:
        return f"{self.statechart_node.unique_name}.{self.predicate.attribute_name}"

    @property
    def reference(self) -> ConditionVariableReference:
        """
        :return: How a serialized condition names this variable.
        """
        return ConditionVariableReference(
            node_id=self.statechart_node._node_id,
            attribute_name=self.predicate.attribute_name,
        )

    @abstractmethod
    def expression(self) -> Scalar:
        """
        :return: The expression over this variable's node's state variables that this
            variable stands for.
        """

    @abstractmethod
    def for_node(self, node: StatechartNode) -> Self:
        """
        :param node: The node to apply this variable's test to.
        :return: The variable holding the same test on `node`.
        """

    @classmethod
    def substitute_in(cls, expression: GenericSymbolicType) -> GenericSymbolicType:
        """
        Replaces every variable of this kind in `expression` by the expression it stands
        for, which is what the compiled statechart evaluates.

        :param expression: The expression to replace them in.
        :return: `expression` with every such variable replaced.
        """
        variables = [
            variable
            for variable in expression.free_variables()
            if isinstance(variable, cls)
        ]
        if not variables:
            return expression
        if isinstance(expression, FloatVariable):
            # A node may hand back one of these unwrapped, which cannot be substituted into.
            expression = Scalar(expression)
        return expression.substitute(
            variables, [variable.expression() for variable in variables]
        )


@dataclass(repr=False, eq=False, init=False)
class LifeCyclePredicateVariable(DerivedConditionVariable):
    """
    A symbol representing a binary test on the life cycle state of a node.
    """

    predicate: LifeCyclePredicate = field(kw_only=True)
    """
    The test this variable holds the value of.
    """

    def __init__(
        self,
        name: str,
        statechart_node: StatechartNode,
        predicate: LifeCyclePredicate,
    ) -> None:
        super().__init__(name, statechart_node)
        self.predicate = predicate

    def resolve(self) -> ObservationStateValues:
        return self.predicate.truth_value(self.statechart_node.life_cycle_state)

    def for_node(self, node: StatechartNode) -> Self:
        return node._life_cycle_predicate(self.predicate)

    def expression(self) -> Scalar:
        """
        :return: The truth table of this predicate over the life cycle state of its node.
        """
        return self.statechart_node.life_cycle_predicate_expression(self.predicate)


@dataclass(repr=False, eq=False, init=False)
class ObservationPredicateVariable(DerivedConditionVariable):
    """
    A symbol representing a binary test whether one of a node's observations is a
    particular value.

    It reads the same observation as :class:`ObservationVariable` or
    :class:`LastObservationVariable`, but answers false where they would be unknown.
    """

    predicate: ObservationPredicate = field(kw_only=True)
    """
    The test this variable holds the value of.
    """

    def __init__(
        self,
        name: str,
        statechart_node: StatechartNode,
        predicate: ObservationPredicate,
    ) -> None:
        super().__init__(name, statechart_node)
        self.predicate = predicate

    @property
    def observation(self) -> NodeStateVariable:
        """
        :return: The variable holding the observation this test reads.
        """
        match self.predicate.reading:
            case ObservationReading.CURRENT:
                return self.statechart_node.observation_variable
            case ObservationReading.LAST:
                return self.statechart_node.last_observation

    def resolve(self) -> ObservationStateValues:
        return self.predicate.truth_value(
            ObservationStateValues(self.observation.resolve())
        )

    def for_node(self, node: StatechartNode) -> Self:
        return node._observation_predicate(self.predicate)

    def expression(self) -> Scalar:
        """
        :return: The test of this predicate on the observation it reads.
        """
        return self.predicate.expression(self.observation)


@dataclass(frozen=True)
class ConditionVariableReference:
    """
    How a serialized condition names a variable: by the id of its node in the JSON
    document and by the attribute the variable is reached under on that node.
    """

    node_id: str
    """
    The id the node of the variable was serialized with.
    """

    attribute_name: str
    """
    The name of the variable on its node, see
    :attr:`~cramph.data_types.LifeCyclePredicate.attribute_name`.
    """

    @classproperty
    def separator(cls) -> str:
        """
        :return: What separates the node id from the attribute name in the written
            reference.
        """
        return "."

    @classmethod
    def parse(cls, written_reference: str) -> Self:
        """
        :param written_reference: A reference as :meth:`__str__` writes it.
        :return: The reference.
        :raises UnknownConditionVariableError: If the name is not a reference at all.
        """
        if cls.separator not in written_reference:
            raise UnknownConditionVariableError(variable_name=written_reference)
        node_id, attribute_name = written_reference.split(cls.separator, maxsplit=1)
        return cls(node_id=node_id, attribute_name=attribute_name)

    def resolve(self, nodes: DeserializedNodeTracker) -> DerivedConditionVariable:
        """
        :param nodes: The nodes of the JSON document.
        :return: The variable this reference names.
        :raises UnknownConditionVariableError: If the node offers no variable of that name.
        """
        node = nodes.get(self.node_id)
        for variable in node.condition_variables:
            if variable.predicate.attribute_name == self.attribute_name:
                return variable
        raise UnknownConditionVariableError(variable_name=str(self))

    def __str__(self) -> str:
        return f"{self.node_id}{self.separator}{self.attribute_name}"


@dataclass
class NodeArtifacts:
    """
    Represents the artifacts produced by the `build_artifacts` method of a node.
    It makes explicit what artifacts are produced by a node.
    """

    observation: Scalar | None = field(default=None)
    """
    A symbolic expression that describes the observation state of this node.
    Instead of setting this attribute directly, you may also implement the `on_tick` method of a node.
    The advantage of using observation is that you can reuse expressions the node builds
    anyway.
    .. warning:: the result of `on_tick` takes precedence over the observation expression.
    """


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
    outcome is only left by a reset.
    """

    def as_cases(self) -> list[tuple[LifeCycleValues, sm.Scalar]]:
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


@dataclass(repr=False, eq=False, kw_only=True)
class StatechartNode(SubclassJSONSerializer, metaclass=ABCMeta):
    """
    A node of a statechart.

    Every node class says how it succeeds by inheriting :class:`EndedByOwner` or
    :class:`SucceedsOnObservingTrue`, and may inherit :class:`FailsOnObservingFalse`.
    """

    name: str = None
    """
    A name for the node within a statechart.
    The name is not unique, use :attr:`unique_name`, if you need a unique identifier.
    """

    _statechart: Statechart = field(init=False, default=None)
    """
    Back reference to the statechart that owns this node.
    """
    index: int | None = field(default=None, init=False)
    """
    The index of this node in the statechart.
    """

    _node_id: str = field(init=False, default=None)
    """
    Process-unique identifier assigned at construction and used to name this node's state
    variables. Unlike :attr:`index` it exists before the node is added to a statechart,
    so variable names are unique from construction time. A deserialized node gets a new one:
    the serialized identifier only tells apart the nodes of one JSON document, which is
    also how a serialized condition names its nodes.
    """

    parent_node_index: int | None = field(
        default=None, init=False, metadata=JSONMetadata(serialize=True).as_dict()
    )
    """
    The index of the parent node in the statechart, if None, it is on the top layer of a statechart.
    """

    _life_cycle_variable: LifeCycleVariable = field(init=False, default=None)
    """
    A variable referring to the life cycle state of this node.
    """
    _observation_variable: ObservationVariable = field(init=False, default=None)
    """
    A variable referring to the observation state of this node.
    """
    _last_observation_variable: LastObservationVariable = field(
        init=False, default=None
    )
    """
    A variable referring to the observation this node took most recently.
    """

    _artifacts: NodeArtifacts | None = field(init=False, repr=False, default=None)
    """
    What :meth:`build` returned, set by :meth:`apply_artifacts`.
    """
    _set_up_context: StatechartContext | None = field(
        init=False, repr=False, default=None
    )
    """
    The context :meth:`set_up` last ran in, which it does once per context, on the
    first :meth:`build` in it.
    """
    _observation_expression: Scalar = field(init=False, repr=False)
    """The parameter is set after build() using its NodeArtifacts."""

    _start_condition: TransitionCondition = field(init=False, default=None)
    """
    Decides when this node transitions from life cycle state NOT_STARTED to RUNNING.
    """
    _pause_condition: TransitionCondition = field(init=False, default=None)
    """
    Decides when this node transitions from RUNNING to PAUSED or back.
    """
    _success_condition: TransitionCondition = field(init=False, default=None)
    """
    Decides when this node ends as SUCCEEDED.
    """
    _interrupt_condition: TransitionCondition = field(init=False, default=None)
    """
    Decides when this node ends as INTERRUPTED.
    """
    _reset_condition: TransitionCondition = field(init=False, default=None)
    """
    Decides when this transitions to NOT_STARTED.
    """
    _fail_condition: TransitionCondition = field(init=False, default=None)
    """
    Decides when this node declares that it cannot continue and transitions to FAILED.
    """
    _life_cycle_predicate_variables: dict[
        LifeCyclePredicate, LifeCyclePredicateVariable
    ] = field(init=False, default_factory=dict, repr=False)
    """
    The predicate variables handed out so far, so each one is created only once.
    """
    _life_cycle_predicate_expression_cache: dict[LifeCyclePredicate, Scalar] = field(
        init=False, default_factory=dict, repr=False
    )
    """
    The expressions handed out by :meth:`life_cycle_predicate_expression` so far.
    """
    _observation_predicate_variables: dict[
        ObservationPredicate, ObservationPredicateVariable
    ] = field(init=False, default_factory=dict, repr=False)
    """
    The observation predicate variables handed out so far, so each one is created only
    once.
    """

    plot_specifications: NodePlotSpec = plot_specification_field(
        NodePlotSpec.create_monitor_style
    )
    """
    Describes how this node is plotted during a Statechart.draw call or in a statechart inspector.
    """

    def __post_init__(self) -> None:
        if self.name is None:
            self.name = self.__class__.__name__
        self._node_id = str(uuid.uuid4())
        self._create_state_variables()
        self._start_condition = TransitionCondition.create_true(
            kind=TransitionKind.START, owner=self
        )
        self._pause_condition = TransitionCondition.create_false(
            kind=TransitionKind.PAUSE, owner=self
        )
        self._success_condition = TransitionCondition.create_false(
            kind=TransitionKind.SUCCEED, owner=self
        )
        self._interrupt_condition = TransitionCondition.create_false(
            kind=TransitionKind.INTERRUPT, owner=self
        )
        self._reset_condition = TransitionCondition.create_false(
            kind=TransitionKind.RESET, owner=self
        )
        self._fail_condition = TransitionCondition.create_false(
            kind=TransitionKind.FAIL, owner=self
        )

    def _create_state_variables(self) -> None:
        """
        Creates the observation, life cycle and last observation variables for this node,
        named from :attr:`_node_id` so they are available before the node is added to a
        statechart.
        """
        name = f"{self.name}#{self._node_id}"
        self._life_cycle_predicate_expression_cache = {}
        self._observation_variable = ObservationVariable(
            name=str(PrefixedName("observation", name)),
            statechart_node=self,
        )
        self._life_cycle_variable = LifeCycleVariable(
            name=str(PrefixedName("life_cycle", name)),
            statechart_node=self,
        )
        self._last_observation_variable = LastObservationVariable(
            name=str(PrefixedName(LastObservationVariable.attribute_name, name)),
            statechart_node=self,
        )

    @property
    def parent_node(self) -> StatechartNode | None:
        """
        :return: Reference to the parent node of this node.
        """
        if self.parent_node_index is None:
            return None
        return self._statechart.get_node_by_index(self.parent_node_index)

    @property
    def artifacts(self) -> NodeArtifacts:
        """
        :return: What this node built.
        :raises NodeNotBuiltError: If this node has not been built yet.
        """
        if self._artifacts is None:
            raise NodeNotBuiltError(node=self)
        return self._artifacts

    def apply_artifacts(self, artifacts: NodeArtifacts) -> None:
        """
        Keeps what :meth:`build` returned, reading the observation from it.

        A node without an observation expression keeps the observation of its last
        tick, unless :meth:`on_tick` overwrites it.

        :param artifacts: What :meth:`build` returned.
        """
        self._artifacts = artifacts
        if artifacts.observation is None:
            self._observation_expression = self.observation_variable
        else:
            self._observation_expression = artifacts.observation

    @property
    def prerequisite_nodes(self) -> list[StatechartNode]:
        """
        Nodes that must be expanded and built before this one, because this node reads
        artifacts they only produce during expansion or build.

        :return: The nodes this node depends on, empty unless a subclass declares any.
        """
        return []

    @parent_node.setter
    def parent_node(self, parent_node: StatechartNode | None) -> None:
        """
        :param parent_node: The node this node becomes a child of, or None to put it on the top layer.
        """
        if parent_node is None:
            self.parent_node_index = None
        else:
            self.parent_node_index = parent_node.index

    # %% tree navigation

    @property
    def children(self) -> list[StatechartNode]:
        """
        :return: The children of this node, in order, empty for a node without any.
        """
        return []

    @property
    def descendants(self) -> list[StatechartNode]:
        """
        :return: Every node below this one, each child followed by its own subtree.
        """
        return [node for child in self.children for node in [child, *child.descendants]]

    @property
    def path(self) -> list[StatechartNode]:
        """
        :return: The ancestors of this node, from its parent up to and including the
            root, empty for a node on the top layer.
        """
        ancestors = []
        ancestor = self.parent_node
        while ancestor is not None:
            ancestors.append(ancestor)
            ancestor = ancestor.parent_node
        return ancestors

    @property
    def depth(self) -> int:
        """
        :return: The number of edges between this node and the root, 0 for a node on the
            top layer.
        """
        return len(self.path)

    @property
    def is_leaf(self) -> bool:
        """
        :return: Whether this node has no children.
        """
        return not self.children

    @property
    def siblings(self) -> list[StatechartNode]:
        """
        :return: The other nodes on the same layer, in order.
        """
        return [node for node in self._siblings_including_self if node is not self]

    @property
    def left_siblings(self) -> list[StatechartNode]:
        """
        :return: The siblings before this node, in order.
        """
        siblings = self._siblings_including_self
        return siblings[: siblings.index(self)]

    @property
    def right_siblings(self) -> list[StatechartNode]:
        """
        :return: The siblings after this node, in order.
        """
        siblings = self._siblings_including_self
        return siblings[siblings.index(self) + 1 :]

    @property
    def left_neighbour(self) -> StatechartNode | None:
        """
        :return: The closest sibling before this node, or None if it is the first on its
            layer.
        """
        left_siblings = self.left_siblings
        return left_siblings[-1] if left_siblings else None

    @property
    def right_neighbour(self) -> StatechartNode | None:
        """
        :return: The closest sibling after this node, or None if it is the last on its
            layer.
        """
        right_siblings = self.right_siblings
        return right_siblings[0] if right_siblings else None

    @property
    def _siblings_including_self(self) -> list[StatechartNode]:
        """
        The nodes this one shares a parent with, itself among them. A node on the top
        layer shares it with the other top level nodes, which is the same scope a
        transition condition may read.

        .. note:: Membership is decided by identity, because a name is not unique.

        :return: The nodes on the same layer as this one, in order.
        """
        if self.parent_node is not None:
            return self.parent_node.children
        return self.statechart.top_level_nodes

    def to_json(self, **kwargs: Any) -> dict[str, Any]:
        return {
            **DataclassJSONSerializer.to_json(self, **kwargs),
            NodeJSONKey.NODE_ID: self._node_id,
        }

    @classmethod
    def _from_json(cls, data: dict[str, Any], **kwargs: Any) -> Self:
        """
        Deserializes the node, or returns the instance already deserialized for the same
        node of the document, so that nodes referring to it share it.
        """
        tracker = DeserializedNodeTracker.from_kwargs(kwargs)
        node_id = data[NodeJSONKey.NODE_ID]
        if tracker.has(node_id):
            return tracker.get(node_id)
        node = DataclassJSONSerializer.from_json(data, clazz=cls, **kwargs)
        node._take_over_node_id(node_id)
        tracker.add(node_id, node)
        return node

    def _take_over_node_id(self, node_id: str) -> None:
        """
        Gives this node the id of the node it was deserialized from, so that a later
        document referring to that node finds this one.

        :param node_id: The id the node was serialized with.
        """
        self._node_id = node_id
        self._create_state_variables()

    def _set_transition(self, transition: TransitionCondition) -> None:
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
            case TransitionKind.SUCCEED:
                self._success_condition = transition
            case TransitionKind.RESET:
                self._reset_condition = transition
            case TransitionKind.FAIL:
                self._fail_condition = transition
            case TransitionKind.INTERRUPT:
                self._interrupt_condition = transition
            case _:
                raise UnknownTransitionKindError(transition_kind=transition.kind)

    def create_lifecycle_transitions(
        self, own_transitions_allowed: sm.Scalar
    ) -> LifeCycleTransitions:
        """
        Builds the state machine of this node for one pass through the statechart.

        A transition triggered by this node's own conditions only happens while
        `own_transitions_allowed` is true. A transition its parent forces on it always
        happens: the node is reset while its parent has not started, interrupted once its
        parent has ended and paused while its parent is paused. It only starts or
        unpauses while its parent is running, and a node whose pause condition holds
        when it starts starts paused. The parent is read through its
        :attr:`life_cycle_variable`.

        If several transitions are possible, a reset comes first, then this node's own
        ending conditions in the order of
        :meth:`~cramph.data_types.TransitionKind.ending_kinds`, then
        its parent having ended, then pausing, then starting.

        :param own_transitions_allowed: Whether this node may still take a transition
            triggered by its own conditions.
        :return: The next life cycle state of this node, per state it can currently be in.
        """
        reset = sm.logic_or(
            self._own_trigger(
                self.get_condition(TransitionKind.RESET), own_transitions_allowed
            ),
            self._parent_is(LifeCyclePredicate.IS_NOT_STARTED),
        )
        start = sm.logic_and(
            self._own_trigger(
                self.get_condition(TransitionKind.START), own_transitions_allowed
            ),
            self._is_top_level_or_parent_running(),
        )
        end_cases = self._create_end_cases(own_transitions_allowed)
        return LifeCycleTransitions(
            not_started=sm.if_cases(
                cases=[
                    (reset, sm.Scalar(LifeCycleValues.NOT_STARTED)),
                    (
                        sm.logic_and(start, self.get_condition(TransitionKind.PAUSE)),
                        sm.Scalar(LifeCycleValues.PAUSED),
                    ),
                    (start, sm.Scalar(LifeCycleValues.RUNNING)),
                ],
                else_result=sm.Scalar(LifeCycleValues.NOT_STARTED),
            ),
            running=sm.if_cases(
                cases=[
                    (reset, sm.Scalar(LifeCycleValues.NOT_STARTED)),
                    *end_cases,
                    (
                        sm.logic_or(
                            self._own_trigger(
                                self.get_condition(TransitionKind.PAUSE),
                                own_transitions_allowed,
                            ),
                            self._parent_is(LifeCyclePredicate.IS_PAUSED),
                        ),
                        sm.Scalar(LifeCycleValues.PAUSED),
                    ),
                ],
                else_result=sm.Scalar(LifeCycleValues.RUNNING),
            ),
            paused=sm.if_cases(
                cases=[
                    (reset, sm.Scalar(LifeCycleValues.NOT_STARTED)),
                    *end_cases,
                    (
                        sm.logic_and(
                            self._own_trigger(
                                sm.logic_not(self.get_condition(TransitionKind.PAUSE)),
                                own_transitions_allowed,
                            ),
                            self._is_top_level_or_parent_running(),
                        ),
                        sm.Scalar(LifeCycleValues.RUNNING),
                    ),
                ],
                else_result=sm.Scalar(LifeCycleValues.PAUSED),
            ),
            terminal=sm.if_else(
                condition=reset,
                if_result=sm.Scalar(LifeCycleValues.NOT_STARTED),
                else_result=self.life_cycle_variable,
            ),
        )

    def create_life_cycle_forced_by_parent(self) -> sm.Scalar:
        """
        :return: The life cycle state this node reaches in a pass through the
            transitions its parent forces on it alone, the same as
            :meth:`create_lifecycle_transitions` reaches while no transition triggered
            by this node's own conditions may happen. The parent is read through its
            :attr:`life_cycle_variable`.
        """
        return sm.if_cases(
            cases=[
                (
                    self._parent_is(LifeCyclePredicate.IS_NOT_STARTED),
                    sm.Scalar(LifeCycleValues.NOT_STARTED),
                ),
                (
                    sm.logic_and(
                        self._parent_is(LifeCyclePredicate.IS_TERMINATED),
                        sm.logic_or(
                            self.life_cycle_predicate_expression(
                                LifeCyclePredicate.IS_RUNNING
                            ),
                            self.life_cycle_predicate_expression(
                                LifeCyclePredicate.IS_PAUSED
                            ),
                        ),
                    ),
                    sm.Scalar(LifeCycleValues.INTERRUPTED),
                ),
                (
                    sm.logic_and(
                        self._parent_is(LifeCyclePredicate.IS_PAUSED),
                        self.life_cycle_predicate_expression(
                            LifeCyclePredicate.IS_RUNNING
                        ),
                    ),
                    sm.Scalar(LifeCycleValues.PAUSED),
                ),
            ],
            else_result=self.life_cycle_variable,
        )

    def _create_end_cases(
        self, own_transitions_allowed: sm.Scalar
    ) -> list[tuple[sm.Scalar, sm.Scalar]]:
        """
        Every way this node leaves RUNNING or PAUSED, and the outcome each yields.

        Each of this node's own ending conditions yields its own outcome. A parent that
        has ended takes this node down with it, which interrupts it however the parent
        ended.

        :param own_transitions_allowed: Whether this node may still take a transition
            triggered by its own conditions.
        :return: The (condition, resulting life cycle state) pairs, most decisive first.
        """
        return [
            *[
                (
                    self._own_trigger(
                        self.effective_condition(transition_kind),
                        own_transitions_allowed,
                    ),
                    sm.Scalar(transition_kind.outcome),
                )
                for transition_kind in TransitionKind.ending_kinds()
            ],
            (
                self._parent_is(LifeCyclePredicate.IS_TERMINATED),
                sm.Scalar(LifeCycleValues.INTERRUPTED),
            ),
        ]

    @staticmethod
    def _own_trigger(
        trigger: sm.Scalar, own_transitions_allowed: sm.Scalar
    ) -> sm.Scalar:
        """
        :param trigger: When one of this node's own transitions would happen.
        :param own_transitions_allowed: Whether this node may still take such a
            transition.
        :return: When that transition happens.
        """
        return sm.logic_and(trigger, own_transitions_allowed)

    def _parent_is(self, predicate: LifeCyclePredicate) -> sm.Scalar:
        """
        :param predicate: The test on the parent's life cycle state.
        :return: The test on the life cycle state of this node's parent, false for a
            top level node.
        """
        if self.parent_node is None:
            return sm.Scalar.const_false()
        return self.parent_node.life_cycle_predicate_expression(predicate)

    def _is_top_level_or_parent_running(self) -> sm.Scalar:
        """
        :return: True for a top level node, otherwise whether its parent is running.
        """
        if self.parent_node is None:
            return sm.Scalar.const_true()
        return self.parent_node.life_cycle_predicate_expression(
            LifeCyclePredicate.IS_RUNNING
        )

    def get_condition(self, transition_kind: TransitionKind) -> Scalar:
        """
        Get the condition for the given transition kind.
        :param transition_kind: The kind of transition whose condition to get.
        :return: The condition for the given transition kind.
        """
        return self._get_transition(transition_kind).expression

    def set_condition(
        self, transition_kind: TransitionKind, expression: Scalar
    ) -> None:
        """
        Set the condition for the given transition kind.

        :param transition_kind: The kind of transition whose condition to set.
        :param expression: The expression deciding when that transition happens.
        """
        transition = self._get_transition(transition_kind)
        if transition is None:
            raise NotInStatechartError(self.name)
        transition.update_expression(expression, self)

    def _clear_owner_wiring(self) -> None:
        """
        Gives every transition the owner of this node wires back the condition it was
        created with, so that another owner can wire it.
        """
        for transition_kind in TransitionKind.owner_wired_kinds():
            self.set_condition(transition_kind, transition_kind.unwired_condition)

    def _get_transition(self, transition_kind: TransitionKind) -> TransitionCondition:
        """
        :param transition_kind: The kind of transition to look up.
        :return: This node's transition of that kind.
        """
        match transition_kind:
            case TransitionKind.START:
                return self._start_condition
            case TransitionKind.PAUSE:
                return self._pause_condition
            case TransitionKind.SUCCEED:
                return self._success_condition
            case TransitionKind.RESET:
                return self._reset_condition
            case TransitionKind.FAIL:
                return self._fail_condition
            case TransitionKind.INTERRUPT:
                return self._interrupt_condition
            case _:
                raise UnknownTransitionKindError(transition_kind=transition_kind)

    @property
    def life_cycle_variable(self) -> LifeCycleVariable:
        """
        :return: The variable representing the life cycle state of this node.
        """
        return self._life_cycle_variable

    def belongs_to_statechart(self) -> bool:
        """
        :return: Whether this node has been added to a statechart.
        """
        return self._statechart is not None

    @property
    def observation_variable(self) -> ObservationVariable:
        """
        :return: The variable representing the observation state of this node.
        """
        return self._observation_variable

    @property
    def statechart(self) -> Statechart:
        """
        :return: The statechart this node belongs to.
        """
        if self._statechart is None:
            raise NotInStatechartError(self.name)
        return self._statechart

    @statechart.setter
    def statechart(self, statechart: Statechart) -> None:
        """
        :param statechart: The statechart this node now belongs to.
        """
        self._statechart = statechart

    @property
    def context(self) -> StatechartContext:
        """
        :return: The context of the statechart this node belongs to.
        :raises NotInStatechartError: If this node belongs to no statechart yet.
        """
        return self.statechart.context

    @property
    def world(self) -> World:
        """
        :return: The world the statechart this node belongs to runs in.
        :raises NotInStatechartError: If this node belongs to no statechart yet.
        """
        return self.context.world

    @classproperty
    def required_context_extensions(cls) -> tuple[type[ContextExtension], ...]:
        """
        :return: The types of the context extensions a node of this class reads once
            it is in a statechart; a statechart compiles such a node only in a context
            that holds them all.
        """
        return ()

    def create_structure_copy(self) -> StatechartNode:
        """
        :return: A node of the base class of this node's kind, with the same name, see
            :meth:`~cramph.statechart.Statechart.create_structure_copy`.
        """
        return StructureCopyNode(name=self.name)

    def set_up(self) -> None:
        """
        Called once per context, when this node is first built in it, before
        :meth:`build_artifacts`. Register what this node reads from :attr:`context`
        here, such as float variables, because :meth:`build` may run more than once.
        """
        return None

    def build(self) -> NodeArtifacts:
        """
        Called when the statechart compiles this node, and again every time the
        kinematic structure of the world changes, so that expressions reading the
        structure follow it. Anything registered in :attr:`context` belongs in
        :meth:`set_up`.
        .. warning:: Don't create other nodes within this function.
        .. warning:: An override must return ``super().build()``, otherwise
            :meth:`set_up` and :meth:`build_artifacts` never run.
        :return: A NodeArtifacts instance that describes this node.
        """
        self._set_up_once()
        return self.build_artifacts()

    def _set_up_once(self) -> None:
        """
        Calls :meth:`set_up` unless it already ran in :attr:`context`.
        """
        if self._set_up_context is self.context:
            return
        self.set_up()
        self._set_up_context = self.context

    def build_artifacts(self) -> NodeArtifacts:
        """
        Describe this node in terms of its observation.
        :return: A NodeArtifacts instance that describes this node.
        """
        return NodeArtifacts()

    def on_tick(self) -> ObservationStateValues | None:
        """
        Triggered when the node is ticked.
        .. warning:: This method is called inside a control loop, make sure it is fast.
        .. warning:: Only happens while the node is in state RUNNING.
        .. warning:: The result of this method takes precedence over the observation expression created in build().
        :return: An optional observation state overwrite
        """
        return None

    def on_start(self) -> None:
        """
        Triggered when the node transitions from NOT_STARTED to RUNNING.
        .. warning:: This method is called inside a control loop, make sure it is fast.
        """
        return None

    def on_pause(self) -> None:
        """
        Triggered when the node transitions from RUNNING to PAUSED.
        .. warning:: This method is called inside a control loop, make sure it is fast.
        """
        return None

    def on_unpause(self) -> None:
        """
        Triggered when the node transitions from PAUSED to RUNNING.
        .. warning:: This method is called inside a control loop, make sure it is fast.
        """
        return None

    def on_end(self) -> None:
        """
        Triggered when the node transitions from RUNNING or PAUSED into any terminal
        state. Read :attr:`life_cycle_state` for the outcome.
        .. warning:: This method is called inside a control loop, make sure it is fast.
        """
        return None

    def on_reset(self) -> None:
        """
        Triggered when the node transitions from any state to NOT_STARTED.
        .. warning:: This method is called inside a control loop, make sure it is fast.
        """
        return None

    def cleanup(self) -> None:
        """
        Triggered after an EndStatechart or CancelStatechart was triggered.
        Place code here to clean up after execution.
        """
        return None

    def __hash__(self) -> int:
        return hash(self.name)

    def __eq__(self, other: object) -> bool:
        """
        :return: Whether `other` is this very node, whatever parameters the two share.
        """
        return self is other

    @property
    def life_cycle_state(self) -> LifeCycleValues:
        """
        :return: The current life cycle state of this node.
        """
        return LifeCycleValues(self.statechart.life_cycle_state[self])

    @property
    def observation_state(self) -> float:
        """
        :return: The current observation state of this node.
        """
        return self.statechart.observation_state[self]

    @property
    def last_observation(self) -> LastObservationVariable:
        """
        Unlike :attr:`observation_variable`, which turns unknown on the tick after
        this node ended, this keeps the reading the transition that ended it saw, until
        the tick after a reset clears it.

        :return: A variable holding the observation this node took most recently.
        """
        return self._last_observation_variable

    @property
    def last_observation_state(self) -> ObservationStateValues:
        """
        :return: The observation this node took most recently.
        """
        return self.statechart.last_observation_state[self]

    @property
    def start_time(self) -> float | None:
        """
        :return: Seconds since the statechart started at which this node most
            recently started running, None if it has not started since its last
            reset.
        """
        run = self.statechart.state_history.get_current_run_ticks_of_node(self)
        if run is None:
            return None
        return run.start_tick * self.statechart.context.require_tick_duration()

    @property
    def end_time(self) -> float | None:
        """
        :return: Seconds since the statechart started at which this node most
            recently ended, None if it has not started since its last reset or has
            not ended yet.
        """
        run = self.statechart.state_history.get_current_run_ticks_of_node(self)
        if run is None or run.end_tick is None:
            return None
        return run.end_tick * self.statechart.context.require_tick_duration()

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
            raise NotInStatechartError(self.name)
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
            raise NotInStatechartError(self.name)
        self._pause_condition.update_expression(expression, self)

    @property
    def success_condition(self) -> Scalar:
        """
        :return: The expression deciding when this node ends as SUCCEEDED.
        """
        return self._success_condition.expression

    @success_condition.setter
    def success_condition(self, expression: Scalar) -> None:
        """
        :param expression: The expression deciding when this node ends as SUCCEEDED.
        """
        if self._success_condition is None:
            raise NotInStatechartError(self.name)
        self._success_condition.update_expression(expression, self)

    @property
    def interrupt_condition(self) -> Scalar:
        """
        :return: The expression deciding when this node ends as INTERRUPTED.
        """
        return self._interrupt_condition.expression

    @interrupt_condition.setter
    def interrupt_condition(self, expression: Scalar) -> None:
        """
        :param expression: The expression deciding when this node ends as INTERRUPTED.
        """
        if self._interrupt_condition is None:
            raise NotInStatechartError(self.name)
        self._interrupt_condition.update_expression(expression, self)

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
            raise NotInStatechartError(self.name)
        self._reset_condition.update_expression(expression, self)

    @property
    def fail_condition(self) -> Scalar:
        """
        :return: The expression by which this node declares that it cannot continue.
        """
        return self._fail_condition.expression

    @fail_condition.setter
    def fail_condition(self, expression: Scalar) -> None:
        """
        :param expression: The expression by which this node declares that it cannot
            continue.
        """
        if self._fail_condition is None:
            raise NotInStatechartError(self.name)
        self._fail_condition.update_expression(expression, self)

    @property
    def can_fail_on_its_own(self) -> bool:
        """
        Whether this node has a way to fail, written as its :attr:`fail_condition` or
        brought by itself as its :attr:`inherent_fail_condition`.

        .. note:: Complete only once every goal of the statechart has expanded,
            because a template may write a fail condition while expanding.
        """
        return not self.effective_condition(TransitionKind.FAIL).is_constant_false()

    @property
    @abstractmethod
    def inherent_success_condition(self) -> Scalar:
        """
        :return: The expression by which this node succeeds whatever is written as its
            :attr:`success_condition`.
        """

    @property
    def decides_its_own_success(self) -> bool:
        """
        Whether this node succeeds without anything written as its
        :attr:`success_condition`.
        """
        return not self.inherent_success_condition.is_constant_false()

    @property
    def inherent_fail_condition(self) -> Scalar:
        """
        :return: The expression by which this node fails whatever is written as its
            :attr:`fail_condition`.
        """
        return Scalar.const_false()

    def effective_condition(self, transition_kind: TransitionKind) -> Scalar:
        """
        :param transition_kind: The kind of transition whose condition to get.
        :return: The expression deciding when that transition happens: the written
            condition, see :meth:`get_condition`, or for succeeding and failing either
            that or the inherent one.
        """
        written_condition = self.get_condition(transition_kind)
        match transition_kind:
            case TransitionKind.SUCCEED:
                inherent_condition = self.inherent_success_condition
            case TransitionKind.FAIL:
                inherent_condition = self.inherent_fail_condition
            case _:
                return written_condition
        if inherent_condition.is_constant_false():
            return written_condition
        return sm.logic_or(written_condition, inherent_condition)

    def life_cycle_predicate_expression(self, predicate: LifeCyclePredicate) -> Scalar:
        """
        :param predicate: The test to read.
        :return: The truth table of `predicate` over :attr:`life_cycle_variable`.
        """
        if predicate not in self._life_cycle_predicate_expression_cache:
            self._life_cycle_predicate_expression_cache[predicate] = (
                predicate.expression(self.life_cycle_variable)
            )
        return self._life_cycle_predicate_expression_cache[predicate]

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
                    statechart_node=self,
                    predicate=predicate,
                )
            )
        return self._life_cycle_predicate_variables[predicate]

    @property
    def condition_variables(self) -> list[DerivedConditionVariable]:
        """
        Creates the variables not handed out yet, so that a condition read from a string
        can refer to any of them.

        :return: Every life cycle predicate and every observation predicate of this node.
        """
        return [
            *(
                self._life_cycle_predicate(predicate)
                for predicate in LifeCyclePredicate
            ),
            *(
                self._observation_predicate(predicate)
                for predicate in ObservationPredicate
            ),
        ]

    @property
    def conditions(self) -> list[TransitionCondition]:
        """
        :return: Every transition condition of this node, as written.
        """
        return [
            self._start_condition,
            self._pause_condition,
            self._success_condition,
            self._fail_condition,
            self._interrupt_condition,
            self._reset_condition,
        ]

    @property
    def effective_conditions(self) -> list[TransitionCondition]:
        """
        :return: Every transition condition of this node as it takes effect, see
            :meth:`effective_condition`, in the order of :attr:`conditions`.
        """
        conditions = []
        for transition_kind in [written.kind for written in self.conditions]:
            condition = TransitionCondition(kind=transition_kind, owner=self)
            condition.update_expression(self.effective_condition(transition_kind), self)
            conditions.append(condition)
        return conditions

    def _observation_predicate(
        self, predicate: ObservationPredicate
    ) -> ObservationPredicateVariable:
        """
        Hands out the variable for one test on this node's observations, creating it on
        first use so an unused predicate costs nothing.

        :param predicate: The test to read.
        :return: The variable holding that test's value for this node.
        """
        if predicate not in self._observation_predicate_variables:
            self._observation_predicate_variables[predicate] = (
                ObservationPredicateVariable(
                    name=str(
                        PrefixedName(
                            predicate.attribute_name, f"{self.name}#{self._node_id}"
                        )
                    ),
                    statechart_node=self,
                    predicate=predicate,
                )
            )
        return self._observation_predicate_variables[predicate]

    @property
    def observes_true(self) -> ObservationPredicateVariable:
        """
        :return: True while this node observes True, false while it observes anything
            else or is not running.
        """
        return self._observation_predicate(ObservationPredicate.OBSERVES_TRUE)

    @property
    def observes_false(self) -> ObservationPredicateVariable:
        """
        :return: True while this node observes False, false while it observes anything
            else or is not running.
        """
        return self._observation_predicate(ObservationPredicate.OBSERVES_FALSE)

    @property
    def last_observed_true(self) -> ObservationPredicateVariable:
        """
        Unlike :attr:`observes_true`, this keeps its answer once this node has ended,
        until the tick after a reset.

        :return: True if the observation this node took most recently is True, false
            otherwise.
        """
        return self._observation_predicate(ObservationPredicate.LAST_OBSERVED_TRUE)

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
        :return: True once this node has ended, whatever its outcome, false before that.
        """
        return self._life_cycle_predicate(LifeCyclePredicate.IS_TERMINATED)

    @property
    def is_succeeded(self) -> LifeCyclePredicateVariable:
        """
        :return: True once this node ended by succeeding, false otherwise.
        """
        return self._life_cycle_predicate(LifeCyclePredicate.IS_SUCCEEDED)

    @property
    def is_failed(self) -> LifeCyclePredicateVariable:
        """
        :return: True once this node ended by failing, false otherwise.
        """
        return self._life_cycle_predicate(LifeCyclePredicate.IS_FAILED)

    @property
    def is_interrupted(self) -> LifeCyclePredicateVariable:
        """
        :return: True once this node ended because its interrupt condition held or an
            ancestor ended, false otherwise.
        """
        return self._life_cycle_predicate(LifeCyclePredicate.IS_INTERRUPTED)

    @property
    def is_failed_or_interrupted(self) -> sm.Scalar:
        """
        Whether this node ended any way but by succeeding. A node that was interrupted is
        of no more use than one that failed.

        .. note:: This is not a :class:`~cramph.data_types.LifeCyclePredicate`
            but the disjunction of :attr:`is_failed` and :attr:`is_interrupted`.

        ================  =====
        life cycle state  this
        ================  =====
        before it ends    false
        succeeded         false
        failed            true
        interrupted       true
        ================  =====

        :return: True once this node ended without succeeding, false before that.
        """
        return sm.logic_or(self.is_failed, self.is_interrupted)

    def formatted_name(self, quoted: bool = False) -> str:
        """
        Renders the name of this node together with all of its transition conditions.

        :param quoted: Whether to wrap the result in double quotes.
        :return: The multi line representation of this node.
        """
        formatted_name = self._wrap_text(
            text=str(self.name), max_lines=4, max_line_length=25
        )
        result = (
            f"{formatted_name}\n"
            f"----start_condition----\n"
            f"{str(self._start_condition)}\n"
            f"----pause_condition----\n"
            f"{str(self._pause_condition)}\n"
            f"----success_condition----\n"
            f"{str(self._success_condition)}\n"
            f"----fail_condition----\n"
            f"{str(self._fail_condition)}\n"
            f"----interrupt_condition----\n"
            f"{str(self._interrupt_condition)}\n"
            f"----reset_condition----\n"
            f"{str(self._reset_condition)}"
        )
        if quoted:
            return '"' + result + '"'
        return result

    @staticmethod
    def _wrap_text(text: str, max_lines: int, max_line_length: int) -> str:
        """
        :param text: The text to wrap.
        :param max_lines: The most lines the result may have.
        :param max_line_length: The most characters a line may have.
        :return: `text` split into lines of at most `max_line_length` characters, cut
            off with "..." after `max_lines` lines.
        """
        if len(text) < max_line_length:
            return text
        lines = []
        start = 0
        for _ in range(max_lines):
            end = start + max_line_length
            lines.append(text[start:end])
            if end >= len(text):
                break
            start = end
        result = "\n".join(lines)
        if len(text) > start:
            result = result + "..."
        return result

    @property
    def unique_name(self) -> str:
        """
        :return: The name of this node, made unique by appending its index.
        """
        return f"{self.name}#{self.index}"

    def __repr__(self) -> str:
        return self.unique_name


GenericStatechartNode = TypeVar("GenericStatechartNode", bound=StatechartNode)


# %% how a node ends


@dataclass(eq=False, repr=False, kw_only=True)
class EndedByOwner(StatechartNode):
    """
    A node whose observation says whether it reached its goal, but which only its owner
    ends, because releasing it may undo what it reached, as with a node that keeps
    holding a state only while it runs.
    """

    @property
    def inherent_success_condition(self) -> Scalar:
        return Scalar.const_false()


@dataclass(eq=False, repr=False, kw_only=True)
class SucceedsOnObservingTrue(StatechartNode):
    """
    A node that succeeds once it observes True, because ending it undoes nothing it did.
    """

    @property
    def inherent_success_condition(self) -> Scalar:
        return self.observes_true


@dataclass(eq=False, repr=False, kw_only=True)
class FailsOnObservingFalse(StatechartNode):
    """
    A node that fails once it observes False, because it can then no longer reach its
    goal.
    """

    @property
    def inherent_fail_condition(self) -> Scalar:
        return self.observes_false


@dataclass(eq=False, repr=False, kw_only=True)
class StructureCopyNode(EndedByOwner, StatechartNode):
    """
    Stands in for a node in a structure copy, see
    :meth:`~cramph.statechart.Statechart.create_structure_copy`.
    """


# %% deserialization


@dataclass
class DeserializedNodeTracker(DeserializedObjectTracker[str, StatechartNode]):
    """
    The nodes deserialized from one JSON document, by the node id they were serialized
    with.

    A document holds a node once for every place that refers to it, for example as a node
    of a statechart and as the node a monitor watches. A node the document does not
    contain is looked up in the statechart the tracker was created with through
    :meth:`from_statechart`.
    """

    _statechart: Statechart | None = field(init=False, default=None)
    """
    The statechart to look up the nodes in that were not deserialized from the
    document.
    """

    @classmethod
    def from_statechart(cls, statechart: Statechart) -> Self:
        """
        :param statechart: The statechart whose nodes are found by the id
            they carry.
        :return: A new tracker.
        """
        tracker = cls()
        tracker._statechart = statechart
        return tracker

    def _has_untracked(self, key: str) -> bool:
        return self._find_node_of_statechart(key) is not None

    def _get_untracked(self, key: str) -> StatechartNode:
        node = self._find_node_of_statechart(key)
        if node is None:
            raise UntrackedObjectError(key=key)
        return node

    def _find_node_of_statechart(self, node_id: str) -> StatechartNode | None:
        """
        :param node_id: The id of the node to find.
        :return: The node of :attr:`_statechart` with that id, or None if there is
            none or no statechart.
        """
        if self._statechart is None:
            return None
        for node in self._statechart.nodes:
            if node._node_id == node_id:
                return node
        return None


def expanded_child_field(default: Any = MISSING) -> Any:
    """
    Declares a field of a composite node that holds a child the node creates in
    :meth:`~CompositeNode.expand`.

    A statechart is sent through JSON after its composite nodes expanded, and the
    receiver does not expand them again, so such a field is serialized explicitly.

    :param default: The value of the field before the node expanded; without one, the
        field does not exist until then.
    """
    return field(
        init=False, default=default, metadata=JSONMetadata(serialize=True).as_dict()
    )


@dataclass(eq=False, repr=False, kw_only=True)
class CompositeNode(StatechartNode):
    """
    A node that has other nodes as its children.
    """

    accepts_children_after_compile: bool = field(init=False, default=False)
    """
    Whether children may join this node once the statechart compiled it, rather than
    only while it expands.
    """

    nodes: list[StatechartNode] = field(default_factory=list, init=False)
    """
    The children of this node.
    """

    plot_specifications: NodePlotSpec = plot_specification_field(
        NodePlotSpec.create_composite_node_style
    )

    @property
    def children(self) -> list[StatechartNode]:
        """
        :return: The children of this node, which is what :attr:`nodes` holds.
        """
        return list(self.nodes)

    @classproperty
    def context_extensions_required_to_expand(
        cls,
    ) -> tuple[type[ContextExtension], ...]:
        """
        :return: The types of the context extensions a node of this class reads only
            while it expands; a statechart expands such a node only in a context that
            holds them all, and compiles it without them once it is expanded.
        """
        return ()

    def create_structure_copy(self) -> CompositeNode:
        return StructureCopyCompositeNode(name=self.name)

    def adopt_handed_children(self) -> None:
        """
        Puts the children this node was handed below it in its statechart. Puts none
        unless overridden.

        Called once, when this node joins a statechart, right before :meth:`expand`.
        """
        return None

    def expand(self) -> None:
        """
        Instantiate child nodes, add them to this node, and wire their life cycle transition conditions.

        Called once, when this node joins a statechart, after
        :meth:`adopt_handed_children`.

        .. warning:: Nodes have not been built yet.
        """
        return None

    def check_children(self) -> None:
        """
        Rejects children this node cannot have, once the statechart is compiled and the
        children's conditions are complete.
        """
        return None

    def _add_child_to_statechart(self, node: StatechartNode) -> None:
        """
        Adds a node to this node and to the statechart this node belongs to.

        :param node: The node to add as a child of this node.
        :raises StatechartAlreadyCompiledError: If this node accepts no children any
            more, see :meth:`_check_accepts_children`.
        """
        self._add_node_sanity_check(node)
        self._check_accepts_children()
        if node not in self.nodes:
            self.nodes.append(node)
        self._place_child_in_statechart(node)

    def _place_child_in_statechart(self, node: StatechartNode) -> None:
        """
        Makes `node` a child of this node in the statechart this node belongs to,
        without touching :attr:`nodes`.

        .. note:: It does not check :meth:`_check_accepts_children`; check that before
            changing :attr:`nodes`, so a rejected node leaves this node as it was.

        :param node: The node that becomes a child of this node.
        """
        self._add_node_sanity_check(node)
        if node._statechart is self.statechart:
            return
        node.parent_node = self
        self.statechart.add_node(node)

    def _check_accepts_children(self) -> None:
        """
        :raises StatechartAlreadyCompiledError: If the latest compile covered this
            node, which fixed its wiring over its children, and it accepts no children
            after that.
        """
        if self.accepts_children_after_compile:
            return
        if self.statechart.compiler.latest_compile_covers(self):
            raise StatechartAlreadyCompiledError()

    def _add_node_sanity_check(self, node: StatechartNode) -> None:
        """
        Rejects nodes that may not become a child of this node.

        :param node: The node to validate.
        """
        self._check_node_does_not_end_statechart(node)
        self._check_node_doesnt_belong_to_different_parent(node)

    def _check_caller_wired_no_transitions(self, node: StatechartNode) -> None:
        """
        Rejects a child whose life cycle the caller already decided, which is this
        node's to decide.

        The fail condition is exempt: a node declares its own failure, and no owner
        supplies one for it.

        :param node: The child to validate.
        :raises ChildTransitionAlreadyWiredError: If one of the child's transitions is
            no longer the one it was constructed with.
        """
        for transition_kind in TransitionKind.owner_wired_kinds():
            condition = node.get_condition(transition_kind)
            if str(condition) != str(transition_kind.unwired_condition):
                raise ChildTransitionAlreadyWiredError(
                    node=self, child=node, transition_kind=transition_kind
                )

    def list_in_place_of(
        self, child: StatechartNode, substitute: StatechartNode
    ) -> None:
        """
        Lists `substitute` among :attr:`nodes` where `child` itself is listed, leaving
        the statechart as it is.

        :param child: One of :attr:`nodes`.
        :param substitute: The node to list in its place.
        """
        position = next(
            index for index, listed in enumerate(self.nodes) if listed is child
        )
        self.nodes[position] = substitute

    def take_over(self, node: StatechartNode) -> None:
        """
        Becomes the parent of `node`, which forgets the transitions its former parent
        decided for it, so that this node decides them anew.

        :param node: A node of the statechart this node belongs to.
        """
        node._clear_owner_wiring()
        node.parent_node = self

    def _succeed_at_its_goal(self, child: StatechartNode) -> None:
        """
        Ends a child whose owner decides its success once it observes True, which is
        this node deciding it as its owner. A child that fails on the same tick fails,
        and a child that decides its own success is left as it is.

        :param child: A child of this node that already joined the statechart.
        """
        if child.decides_its_own_success:
            return
        child.success_condition = sm.logic_and(
            child.observes_true,
            sm.logic_not(child.effective_condition(TransitionKind.FAIL)),
        )

    def _check_has_children(self) -> None:
        """
        Rejects a node that was built without the child nodes it exists to hold.

        Call this at the start of :meth:`expand`, while :attr:`nodes` still holds only
        what the caller passed.

        :raises CompositeNodeWithoutChildrenError: If this node has no child nodes.
        """
        if not self.nodes:
            raise CompositeNodeWithoutChildrenError(node=self)

    def _check_node_does_not_end_statechart(self, node: StatechartNode) -> None:
        """
        Rejects nodes that end the whole statechart.

        :param node: The node to validate.
        """
        if isinstance(node, EndStatechart):
            raise EndInCompositeNodeError(node=self)

    def _check_node_doesnt_belong_to_different_parent(
        self, node: StatechartNode
    ) -> None:
        """
        Rejects a node that is already a child of another node of this statechart.

        .. note:: A node held by a *different* statechart is allowed, because it is
            moved into this node's statechart; only two parents within one statechart are
            an error.

        :param node: The node about to become a child of this node.
        :raises NodeAlreadyBelongsToDifferentNodeError: If another node of this
            statechart holds `node`.
        """
        if node.belongs_to_statechart() and node.parent_node != self:
            raise NodeAlreadyBelongsToDifferentNodeError(node=self, new_node=node)

    def _add_children_to_statechart(self, nodes: list[StatechartNode]) -> None:
        """
        Adds multiple nodes to this node and to the statechart this node belongs
        to, see :meth:`_add_child_to_statechart`.

        :param nodes: The nodes to add as children of this node.
        """
        for node in nodes:
            self._add_child_to_statechart(node)


@dataclass(eq=False, repr=False, kw_only=True)
class StructureCopyCompositeNode(EndedByOwner, CompositeNode):
    """
    Stands in for a composite node in a structure copy, see
    :meth:`~cramph.statechart.Statechart.create_structure_copy`.
    """


@dataclass(eq=False, repr=False, kw_only=True)
class ThreadPayloadMonitor(EndedByOwner, StatechartNode, ABC):
    """
    A monitor whose observation is computed by :meth:`_compute_observation` in a
    background thread, so that a slow computation never blocks a tick.

    :meth:`compute_observation` returns at once with the latest finished observation,
    which is unknown until the first computation finished.
    """

    _request_event: threading.Event = field(
        default_factory=threading.Event, init=False, repr=False
    )
    """
    Set to ask the worker thread for a fresh observation.
    """

    _stop_event: threading.Event = field(
        default_factory=threading.Event, init=False, repr=False
    )
    """
    Set to end the worker thread.
    """

    _thread: threading.Thread = field(init=False, repr=False)
    """
    The worker thread computing the observations.
    """

    _has_result: bool = field(default=False, init=False, repr=False)
    """
    Whether a computation has finished since this node was created.
    """

    _last_result: float = field(
        default=ObservationStateValues.UNKNOWN, init=False, repr=False
    )
    """
    The observation the latest finished computation returned.
    """

    def __post_init__(self) -> None:
        super().__post_init__()
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
        self._request_event.set()
        return self._last_result

    def cleanup(self) -> None:
        """
        Stops the background worker thread.
        """
        self._stop_event.set()
        self._thread.join(timeout=1.0)

    def _worker_loop(self) -> None:
        """
        Computes an observation whenever one is requested, until the node is cleaned up.

        A failing computation is logged and keeps the previous observation.
        """
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


@dataclass(eq=False, repr=False, kw_only=True)
class TerminalNode(EndedByOwner, StatechartNode, ABC):
    """
    A node that ends the whole statechart once its observation state turns true.

    No transition can happen afterwards, so conditions may not reference such a node.
    """

    @staticmethod
    def _observing_true_or_succeeded(node: StatechartNode) -> Scalar:
        """
        :param node: The node to read.
        :return: A condition that is True while `node` observes True and once it
            succeeded, and never True for a node that failed or was interrupted.
        """
        return sm.logic_or(node.observes_true, node.is_succeeded)


@dataclass(eq=False, repr=False, kw_only=True)
class EndStatechart(TerminalNode):
    """
    Ends the statechart once it observes True, which it does as soon as it runs.
    """

    plot_specifications: NodePlotSpec = plot_specification_field(
        NodePlotSpec.create_end_style
    )
    """
    Draws this node in the style of a node ending the statechart.
    """

    def create_structure_copy(self) -> EndStatechart:
        return EndStatechart(name=self.name)

    def build_artifacts(self) -> NodeArtifacts:
        return NodeArtifacts(observation=Scalar.const_true())

    @classmethod
    def when_true(cls, node: StatechartNode) -> Self:
        """
        Factory method for creating an EndStatechart node that activates once the given node
        observes True or succeeded.

        :param node: The node whose goal ends the statechart.
        :return: The new EndStatechart node.
        """
        end = cls()
        end.start_condition = cls._observing_true_or_succeeded(node)
        return end

    @classmethod
    def when_failed(cls, node: StatechartNode) -> Self:
        """
        Factory method for creating an EndStatechart node that activates once the given node
        ended by failing.

        :param node: The node whose failure ends the statechart.
        :return: The new EndStatechart node.
        """
        end = cls()
        end.start_condition = node.is_failed
        return end

    @classmethod
    def when_false(cls, node: StatechartNode) -> Self:
        """
        Factory method for creating an EndStatechart node that activates while the given node
        has a false observation state.

        Unlike :meth:`when_true` this asks only what the node observes now, so it stops
        mattering once that node ends rather than latching onto the outcome it earned.

        .. note:: Use :meth:`when_failed` to wait for a node to end short of its goal.

        :param node: The node whose observation state activates the created node.
        :return: The new EndStatechart node.
        """
        end = cls()
        end.start_condition = node.observes_false
        return end

    @classmethod
    def when_all_true(cls, nodes: list[StatechartNode]) -> Self:
        """
        Factory method for creating an EndStatechart node that activates once *all* of the
        given nodes observe True or succeeded.

        :param nodes: The nodes whose goals end the statechart.
        :return: The new EndStatechart node.
        """
        end = cls()
        end.start_condition = sm.logic_and(
            *[cls._observing_true_or_succeeded(node) for node in nodes]
        )
        return end

    @classmethod
    def when_any_true(cls, nodes: list[StatechartNode]) -> Self:
        """
        Factory method for creating an EndStatechart node that activates once *any* of the
        given nodes observes True or succeeded.

        :param nodes: The nodes whose goals end the statechart.
        :return: The new EndStatechart node.
        """
        end = cls()
        end.start_condition = sm.logic_or(
            *[cls._observing_true_or_succeeded(node) for node in nodes]
        )
        return end


@dataclass(eq=False, repr=False, kw_only=True)
class CancelStatechart(TerminalNode):
    """
    Ends the statechart by raising :attr:`exception` at the end of the tick it starts
    in, even if it is interrupted again within that tick.

    Its factory methods mirror :class:`EndStatechart`'s: they read whether a node reached its
    goal, which keeps answering once that node has ended, rather than the observation
    behind it, which is gone by then.
    """

    exception: DataclassException
    """
    The exception that ends the statechart once this node starts.
    """

    _pending_exception: Exception | None = field(default=None, init=False, repr=False)
    """
    The exception this node started with in the current tick, raised once that tick
    is complete.
    """

    plot_specifications: NodePlotSpec = plot_specification_field(
        NodePlotSpec.create_cancel_style
    )
    """
    Draws this node in the style of a node cancelling the statechart.
    """

    def create_structure_copy(self) -> CancelStatechart:
        return CancelStatechart(name=self.name, exception=self.exception)

    def build_artifacts(self) -> NodeArtifacts:
        return NodeArtifacts(observation=Scalar.const_true())

    def on_start(self) -> None:
        self._pending_exception = self.create_exception()

    def raise_pending_exception(self) -> None:
        """
        Raises the exception this node started with in the current tick, if any.
        """
        if self._pending_exception is None:
            return
        exception = self._pending_exception
        self._pending_exception = None
        raise exception

    def create_exception(self) -> Exception:
        """
        :return: The exception that cancels the statechart.
        """
        return self.exception

    @classmethod
    def when_true(
        cls, node: StatechartNode, exception: Exception | None = None
    ) -> Self:
        """
        Factory method for creating a CancelStatechart node that activates once the given
        node observes True or succeeded.

        :param node: The node whose goal activates the created node.
        :param exception: The exception raised on activation, defaults to one naming the given node.
        :return: The new CancelStatechart node.
        """
        if exception is None:
            exception = CancelledByNodeReachingItsGoal(node=node)
        end = cls(exception=exception)
        end.start_condition = cls._observing_true_or_succeeded(node)
        return end

    @classmethod
    def when_failed(
        cls, node: StatechartNode, exception: Exception | None = None
    ) -> Self:
        """
        Factory method for creating a CancelStatechart node that activates once the given
        node ended by failing.

        :param node: The node whose failure activates the created node.
        :param exception: The exception raised on activation, defaults to one naming the given node.
        :return: The new CancelStatechart node.
        """
        if exception is None:
            exception = CancelledByNodeFailing(node=node)
        end = cls(exception=exception)
        end.start_condition = node.is_failed
        return end

    @classmethod
    def when_all_true(cls, nodes: list[StatechartNode], exception: Exception) -> Self:
        """
        Factory method for creating a CancelStatechart node that activates once *all* of the
        given nodes observe True or succeeded.

        :param nodes: The nodes whose goals activate the created node.
        :param exception: The exception raised on activation.
        :return: The new CancelStatechart node.
        """
        end = cls(exception=exception)
        end.start_condition = sm.logic_and(
            *[cls._observing_true_or_succeeded(node) for node in nodes]
        )
        return end

    @classmethod
    def when_any_true(cls, nodes: list[StatechartNode], exception: Exception) -> Self:
        """
        Factory method for creating a CancelStatechart node that activates once *any* of the
        given nodes observes True or succeeded.

        :param nodes: The nodes whose goals activate the created node.
        :param exception: The exception raised on activation.
        :return: The new CancelStatechart node.
        """
        end = cls(exception=exception)
        end.start_condition = sm.logic_or(
            *[cls._observing_true_or_succeeded(node) for node in nodes]
        )
        return end
