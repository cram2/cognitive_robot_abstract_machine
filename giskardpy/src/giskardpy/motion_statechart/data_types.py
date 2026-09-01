from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum, Enum
from typing import Union, FrozenSet

from krrood.symbolic_math.symbolic_math import Scalar, if_eq_cases

goal_parameter = Union[str, float, bool, dict, list, IntEnum, None]


# %% life cycle states


class LifeCycleValues(IntEnum):
    """
    Where a node is in its own execution, see
    :class:`~giskardpy.motion_statechart.motion_statechart.MotionStatechart`.
    """

    NOT_STARTED = 0
    """
    The node has not run yet.

    Its observation is forced back to unknown every tick.
    """

    RUNNING = 1
    """
    The node is active: its constraints carry weight, its observation expression is
    evaluated and
    :meth:`~giskardpy.motion_statechart.graph_node.MotionStatechartNode.on_tick` is
    called.
    """

    PAUSED = 2
    """
    The node was running and its pause condition became true.

    Its constraints are deactivated and its observation is frozen at the last value.
    """

    SUCCEEDED = 3
    """
    The node was ended while it was observing its goal as reached.
    """

    FAILED = 4
    """
    The node was ended while it was not.
    """

    INTERRUPTED = 5
    """
    The node ended without being judged, either because an ancestor ended and took it
    down with it, or because it was not observing anything decisive at the time.
    """

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

    @classmethod
    def verdict_for(cls, observation: ObservationStateValues) -> LifeCycleValues:
        """
        The verdict a node receives when it is ended by its own end condition.

        An observation that has no answer yet is no basis for a judgement, so it leaves
        the node unjudged.

        :param observation: What the node observes at the moment it is ended.
        :return: The terminal state the node reaches.
        """
        match observation:
            case ObservationStateValues.TRUE:
                return cls.SUCCEEDED
            case ObservationStateValues.FALSE:
                return cls.FAILED
            case _:
                return cls.INTERRUPTED

    @property
    def color(self) -> str:
        """
        :return: The color a visualization draws a node in this state in.
        """
        return {
            LifeCycleValues.NOT_STARTED: "#9CA3AF",
            LifeCycleValues.RUNNING: "#3B82F6",
            LifeCycleValues.PAUSED: "#EAB308",
            LifeCycleValues.SUCCEEDED: "#28A745",
            LifeCycleValues.FAILED: "#EF4444",
            LifeCycleValues.INTERRUPTED: "#F97316",
        }[self]

    @property
    def symbol(self) -> str:
        """
        :return: The glyph a visualization labels a node in this state with.
        """
        return {
            LifeCycleValues.NOT_STARTED: "—",
            LifeCycleValues.RUNNING: "▶",
            LifeCycleValues.PAUSED: "<B>||</B>",
            LifeCycleValues.SUCCEEDED: "✔",
            LifeCycleValues.FAILED: "✖",
            LifeCycleValues.INTERRUPTED: "■",
        }[self]


class FloatEnum(float, Enum):
    """
    Enum where members are also (and must be) floats.
    """


class ObservationStateValues(FloatEnum):
    """
    The trinary truth values used throughout the motion statechart.
    """

    FALSE = float(Scalar.const_false())
    UNKNOWN = float(Scalar.const_trinary_unknown())
    TRUE = float(Scalar.const_true())

    @property
    def color(self) -> str:
        """
        :return: The color a visualization draws a node observing this in.
        """
        return {
            ObservationStateValues.FALSE: "#FF5024",
            ObservationStateValues.UNKNOWN: "#8F959E",
            ObservationStateValues.TRUE: "#B6E5A0",
        }[self]

    @property
    def symbol(self) -> str:
        """
        :return: The text a visualization labels a node observing this with.
        """
        return {
            ObservationStateValues.FALSE: "False",
            ObservationStateValues.UNKNOWN: "?",
            ObservationStateValues.TRUE: "True",
        }[self]


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


class LifeCyclePredicate(Enum):
    """
    A test on a node's life cycle state that may be used in transition conditions.

    Verdict predicates are trinary, because *how* a node ended has no answer before it
    ends. :attr:`IS_SUCCEEDED` and :attr:`IS_FAILED` stay unknown until the node is
    judged, which leaves an interrupted node as open as a running one;
    :attr:`IS_TERMINATED` and :attr:`IS_INTERRUPTED` are answered by every way of
    ending. Phase predicates are binary, because *where* a node is right now always has
    an answer.
    """

    IS_NOT_STARTED = LifeCyclePredicateDefinition(
        true_states=frozenset({LifeCycleValues.NOT_STARTED})
    )
    IS_RUNNING = LifeCyclePredicateDefinition(
        true_states=frozenset({LifeCycleValues.RUNNING})
    )
    IS_PAUSED = LifeCyclePredicateDefinition(
        true_states=frozenset({LifeCycleValues.PAUSED})
    )
    IS_TERMINATED = LifeCyclePredicateDefinition(
        true_states=LifeCycleValues.terminal_states()
    )
    IS_SUCCEEDED = LifeCyclePredicateDefinition(
        true_states=frozenset({LifeCycleValues.SUCCEEDED}),
        unknown_states=frozenset(LifeCycleValues) - LifeCycleValues.judged_states(),
    )
    IS_FAILED = LifeCyclePredicateDefinition(
        true_states=frozenset({LifeCycleValues.FAILED}),
        unknown_states=frozenset(LifeCycleValues) - LifeCycleValues.judged_states(),
    )
    IS_INTERRUPTED = LifeCyclePredicateDefinition(
        true_states=frozenset({LifeCycleValues.INTERRUPTED}),
        unknown_states=frozenset(LifeCycleValues) - LifeCycleValues.terminal_states(),
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

    END = 3
    """
    Transitions nodes from RUNNING or PAUSED to a terminal state, and their descendants
    to INTERRUPTED.

    Which terminal state the node itself reaches follows from what it observes at that
    moment, see :meth:`LifeCycleValues.verdict_for`.
    """

    RESET = 4
    """
    Transitions nodes from any state to NOT_STARTED.
    """
