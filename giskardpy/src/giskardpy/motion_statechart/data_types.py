from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum, Enum, StrEnum
from typing import Union, FrozenSet

from giskardpy.motion_statechart.exceptions import TransitionHasNoVerdictError
from krrood.symbolic_math.symbolic_math import Scalar, if_eq_cases
from semantic_digital_twin.world_description.geometry import Color

goal_parameter = Union[str, float, bool, dict, list, IntEnum, None]


# %% serialization


class NodeJSONKey(StrEnum):
    """
    Keys a serialized node carries on top of its dataclass fields.
    """

    NODE_ID = "node_id"
    """
    Tells apart the nodes of one JSON document, so every place referring to the same node
    deserializes to the same instance.
    """


# %% life cycle states


class LifeCycleValues(IntEnum):
    """
    Where a node is in its own execution, see
    :class:`~giskardpy.motion_statechart.motion_statechart.MotionStatechart`.
    """

    color: Color
    """
    The color a visualization draws a node in this state in.
    """

    badge: str
    """
    The short text a visualization labels a node in this state with.
    """

    NOT_STARTED = 0, Color.from_hex("#9CA3AF"), "—"
    """
    The node has not run yet.

    Its observation is forced back to unknown every tick.
    """

    RUNNING = 1, Color.from_hex("#3B82F6"), "▶"
    """
    The node is active: its constraints carry weight, its observation expression is
    evaluated and
    :meth:`~giskardpy.motion_statechart.graph_node.MotionStatechartNode.on_tick` is
    called.
    """

    PAUSED = 2, Color.from_hex("#EAB308"), "<B>||</B>"
    """
    The node was running and its pause condition became true.

    Its constraints are deactivated and its observation is frozen at the last value.
    """

    SUCCEEDED = 3, Color.from_hex("#28A745"), "✔"
    """
    The node's success condition held.
    """

    FAILED = 4, Color.from_hex("#EF4444"), "✖"
    """
    The node declared that it cannot continue, through its own fail condition.
    """

    INTERRUPTED = 5, Color.from_hex("#F97316"), "■"
    """
    The node's interrupt condition held, or an ancestor ended and took it down with it.

    Neither is a judgement of the node itself.
    """

    def __new__(cls, value: int, color: Color, badge: str) -> LifeCycleValues:
        """
        :param value: The number this state is stored as.
        :param color: The color a visualization draws a node in this state in.
        :param badge: The short text a visualization labels such a node with.
        :return: The member standing for that state.
        """
        member = int.__new__(cls, value)
        member._value_ = value
        member.color = color
        member.badge = badge
        return member

    @classmethod
    def terminal_states(cls) -> FrozenSet[LifeCycleValues]:
        """
        :return: The states a node can only leave by being reset.
        """
        return frozenset({cls.SUCCEEDED, cls.FAILED, cls.INTERRUPTED})

    @classmethod
    def judged_states(cls) -> FrozenSet[LifeCycleValues]:
        """
        :return: The states a node reaches by being judged on its own terms, as opposed
            to :attr:`INTERRUPTED`.
        """
        return frozenset({cls.SUCCEEDED, cls.FAILED})

    @property
    def is_terminal(self) -> bool:
        """
        :return: Whether a node in this state has ended.
        """
        return self in self.terminal_states()


class FloatEnum(float, Enum):
    """
    Enum where members are also (and must be) floats.
    """


class ObservationStateValues(FloatEnum):
    """
    The trinary truth values used throughout the motion statechart.
    """

    color: Color
    """
    The color a visualization draws a node observing this in.
    """

    badge: str
    """
    The short text a visualization labels a node observing this with.
    """

    FALSE = float(Scalar.const_false()), Color.from_hex("#FF5024"), "False"
    UNKNOWN = float(Scalar.const_trinary_unknown()), Color.from_hex("#8F959E"), "?"
    TRUE = float(Scalar.const_true()), Color.from_hex("#B6E5A0"), "True"

    def __new__(cls, value: float, color: Color, badge: str) -> ObservationStateValues:
        """
        :param value: The number this truth value is stored as.
        :param color: The color a visualization draws a node observing this in.
        :param badge: The short text a visualization labels such a node with.
        :return: The member standing for that truth value.
        """
        member = float.__new__(cls, value)
        member._value_ = value
        member.color = color
        member.badge = badge
        return member


# %% life cycle predicates


@dataclass(frozen=True)
class LifeCyclePredicateDefinition:
    """
    The truth table of a test on a node's life cycle state.
    """

    true_states: FrozenSet[LifeCycleValues]
    """
    The states in which the predicate is true.
    """

    unknown_states: FrozenSet[LifeCycleValues] = frozenset()
    """
    The states in which the predicate has no answer yet.

    Every state that is neither here nor in :attr:`true_states` makes the predicate
    false.
    """

    def truth_value(self, life_cycle_value: LifeCycleValues) -> ObservationStateValues:
        """
        :param life_cycle_value: The state to evaluate the predicate in.
        :return: The trinary value the predicate takes in that state.
        """
        if life_cycle_value in self.true_states:
            return ObservationStateValues.TRUE
        if life_cycle_value in self.unknown_states:
            return ObservationStateValues.UNKNOWN
        return ObservationStateValues.FALSE

    def expression(self, life_cycle: Scalar) -> Scalar:
        """
        The same truth table as :meth:`truth_value`, but read off an expression rather
        than a value, so a predicate can be resolved while the life cycle state it reads
        is still being computed.

        :param life_cycle: The life cycle state to evaluate the predicate in.
        :return: The trinary value the predicate takes in that state.
        """
        return if_eq_cases(
            a=life_cycle,
            b_result_cases=[
                (int(state), Scalar(float(self.truth_value(state))))
                for state in sorted(LifeCycleValues)
            ],
            else_result=Scalar.const_trinary_unknown(),
        )


class LifeCyclePredicate(LifeCyclePredicateDefinition, Enum):
    """
    A test on a node's life cycle state that may be used in transition conditions.

    Verdict predicates are trinary, because *how* a node ended has no answer before it
    ends. :attr:`IS_SUCCEEDED` and :attr:`IS_FAILED` stay unknown until the node is
    judged, which leaves an interrupted node as open as a running one;
    :attr:`IS_TERMINATED` and :attr:`IS_INTERRUPTED` are answered by every way of
    ending. Phase predicates are binary, because *where* a node is right now always has
    an answer.
    """

    IS_NOT_STARTED = frozenset({LifeCycleValues.NOT_STARTED})
    IS_RUNNING = frozenset({LifeCycleValues.RUNNING})
    IS_PAUSED = frozenset({LifeCycleValues.PAUSED})
    IS_TERMINATED = LifeCycleValues.terminal_states()
    IS_SUCCEEDED = (
        frozenset({LifeCycleValues.SUCCEEDED}),
        frozenset(LifeCycleValues) - LifeCycleValues.judged_states(),
    )
    IS_FAILED = (
        frozenset({LifeCycleValues.FAILED}),
        frozenset(LifeCycleValues) - LifeCycleValues.judged_states(),
    )
    IS_INTERRUPTED = (
        frozenset({LifeCycleValues.INTERRUPTED}),
        frozenset(LifeCycleValues) - LifeCycleValues.terminal_states(),
    )

    @property
    def attribute_name(self) -> str:
        """
        :return: The name this predicate is reached under on a node, also used to render
            it inside a condition.
        """
        return self.name.lower()


# %% weights and transitions


class DefaultWeights(FloatEnum):
    WEIGHT_MAXIMUM = 10000.0
    WEIGHT_ABOVE_COLLISION_AVOIDANCE = 2500.0
    WEIGHT_COLLISION_AVOIDANCE = 50.0
    WEIGHT_BELOW_COLLISION_AVOIDANCE = 1.0
    WEIGHT_MINIMUM = 0.0


class TransitionKind(Enum):
    START = 1
    """
    Transitions nodes from NOT_STARTED to RUNNING.
    """

    PAUSE = 2
    """
    Transitions nodes from RUNNING to PAUSED if True, or back if False.
    """

    SUCCEED = 3
    """
    Ends a node from RUNNING or PAUSED as SUCCEEDED, and interrupts its descendants.
    """

    RESET = 4
    """
    Transitions nodes from any state to NOT_STARTED.
    """

    FAIL = 5
    """
    Ends a node from RUNNING or PAUSED as FAILED, because it cannot continue, and
    interrupts its descendants.
    """

    INTERRUPT = 6
    """
    Ends a node from RUNNING or PAUSED as INTERRUPTED, and its descendants with it.
    """

    @property
    def source_states(self) -> FrozenSet[LifeCycleValues]:
        """
        :return: The lifecycle states from which this transition kind can trigger.
        """
        match self:
            case TransitionKind.START:
                return frozenset({LifeCycleValues.NOT_STARTED})
            case TransitionKind.PAUSE:
                return frozenset({LifeCycleValues.RUNNING, LifeCycleValues.PAUSED})
            case TransitionKind.RESET:
                return frozenset(LifeCycleValues)
            case (
                TransitionKind.SUCCEED | TransitionKind.FAIL | TransitionKind.INTERRUPT
            ):
                return frozenset({LifeCycleValues.RUNNING, LifeCycleValues.PAUSED})

    @classmethod
    def ending_kinds(cls) -> tuple[TransitionKind, ...]:
        """
        :return: The transitions that end a node, in the order they take precedence when
            several hold on the same control cycle: a node that arrived did what it was
            asked, and a node that cannot continue says more about itself than being
            stopped does.
        """
        return cls.SUCCEED, cls.FAIL, cls.INTERRUPT

    @property
    def verdict(self) -> LifeCycleValues:
        """
        :return: The terminal state this transition ends a node in.
        :raises TransitionHasNoVerdictError: If this transition does not end a node.
        """
        match self:
            case TransitionKind.SUCCEED:
                return LifeCycleValues.SUCCEEDED
            case TransitionKind.FAIL:
                return LifeCycleValues.FAILED
            case TransitionKind.INTERRUPT:
                return LifeCycleValues.INTERRUPTED
        raise TransitionHasNoVerdictError(transition_kind=self)

    def can_trigger_from(self, life_cycle: LifeCycleValues) -> bool:
        """
        :param life_cycle: The lifecycle state to check.
        :return: Whether this transition can trigger from the given lifecycle state.
        """
        return life_cycle in self.source_states
