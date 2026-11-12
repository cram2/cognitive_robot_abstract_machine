"""
The enumerations and value types a statechart is described with: life cycle and
observation states, transition kinds and predicates.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum, Enum, auto
from functools import cache

from cramph.exceptions import TransitionHasNoOutcomeError
from krrood.patterns.float_enumeration import FloatEnum
from krrood.symbolic_math.symbolic_math import Scalar, logic_or
from semantic_digital_twin.world_description.geometry import Color

# %% life cycle states


class LifeCycleValues(IntEnum):
    """
    Where a node is in its own execution, see
    :class:`~cramph.statechart.Statechart`.
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
    The node is active: its observation expression is evaluated and
    :meth:`~cramph.node.StatechartNode.on_tick` is called.
    """

    PAUSED = 2, Color.from_hex("#EAB308"), "<B>||</B>"
    """
    The node was running and its pause condition became true.

    Whatever it contributes while running is withdrawn and its observation is frozen at
    the last value.
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
    The node's interrupt condition held, an ancestor ended and took it down with it, or
    a :class:`~cramph.node.CancelStatechart` ended the statechart while it was running
    or paused.

    None of these is a judgement of the node itself.
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
    @cache
    def terminal_states(cls) -> frozenset[LifeCycleValues]:
        """
        :return: The states a node can only leave by being reset, which are the outcomes
            of the transitions that end a node.
        """
        return frozenset(kind.outcome for kind in TransitionKind.ending_kinds())

    @property
    def is_terminal(self) -> bool:
        """
        :return: Whether a node in this state has ended.
        """
        return self in self.terminal_states()


# %% observation states


class ObservationStateValues(FloatEnum):
    """
    The trinary truth values used throughout the statechart.
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


# %% transitions


class TransitionKind(Enum):
    """
    The kinds of transitions a node takes through its life cycle, each decided by one
    of its transition conditions.
    """

    START = auto()
    """
    Transitions nodes from NOT_STARTED to RUNNING, or to PAUSED if their pause condition
    already holds.
    """

    PAUSE = auto()
    """
    Transitions nodes from RUNNING to PAUSED if True, or back if False.
    """

    SUCCEED = auto()
    """
    Ends a node from RUNNING or PAUSED as SUCCEEDED, and interrupts its descendants.
    """

    RESET = auto()
    """
    Transitions nodes from any state to NOT_STARTED.
    """

    FAIL = auto()
    """
    Ends a node from RUNNING or PAUSED as FAILED, because it cannot continue, and
    interrupts its descendants.
    """

    INTERRUPT = auto()
    """
    Ends a node from RUNNING or PAUSED as INTERRUPTED, and its descendants with it.
    """

    @property
    def source_states(self) -> frozenset[LifeCycleValues]:
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

    @property
    def default_condition(self) -> Scalar:
        """
        :return: The condition a node is created with for this transition, before
            anything wires it.
        """
        if self is TransitionKind.START:
            return Scalar.const_true()
        return Scalar.const_false()

    @classmethod
    def owner_wired_kinds(cls) -> tuple[TransitionKind, ...]:
        """
        :return: The transitions the owner of a node wires, which are all but failing:
            a node declares its own failure.
        """
        return cls.START, cls.PAUSE, cls.SUCCEED, cls.INTERRUPT, cls.RESET

    @classmethod
    def ending_kinds(cls) -> tuple[TransitionKind, ...]:
        """
        :return: The transitions that end a node, in the order they take precedence when
            several hold on the same tick: a node that arrived did what it was
            asked, and a node that cannot continue says more about itself than being
            stopped does.
        """
        return cls.SUCCEED, cls.FAIL, cls.INTERRUPT

    @property
    def outcome(self) -> LifeCycleValues:
        """
        :return: The terminal state this transition ends a node in.
        :raises TransitionHasNoOutcomeError: If this transition does not end a node.
        """
        match self:
            case TransitionKind.SUCCEED:
                return LifeCycleValues.SUCCEEDED
            case TransitionKind.FAIL:
                return LifeCycleValues.FAILED
            case TransitionKind.INTERRUPT:
                return LifeCycleValues.INTERRUPTED
        raise TransitionHasNoOutcomeError(transition_kind=self)

    def can_trigger_from(self, life_cycle: LifeCycleValues) -> bool:
        """
        :param life_cycle: The lifecycle state to check.
        :return: Whether this transition can trigger from the given lifecycle state.
        """
        return life_cycle in self.source_states


# %% life cycle predicates


@dataclass(frozen=True)
class LifeCyclePredicateDefinition:
    """
    The truth table of a test on a node's life cycle state.
    """

    true_states: frozenset[LifeCycleValues]
    """
    The states in which the predicate is true; every other state makes it false.
    """

    def truth_value(self, life_cycle_value: LifeCycleValues) -> ObservationStateValues:
        """
        :param life_cycle_value: The state to evaluate the predicate in.
        :return: True if the predicate holds in that state, false otherwise.
        """
        if life_cycle_value in self.true_states:
            return ObservationStateValues.TRUE
        return ObservationStateValues.FALSE

    def expression(self, life_cycle: Scalar) -> Scalar:
        """
        The same truth table as :meth:`truth_value`, but read off an expression rather
        than a value.

        :param life_cycle: The life cycle state to evaluate the predicate in.
        :return: True if the predicate holds in that state, false otherwise.
        """
        return logic_or(
            *[Scalar(life_cycle) == float(state) for state in sorted(self.true_states)]
        )


class LifeCyclePredicate(LifeCyclePredicateDefinition, Enum):
    """
    A test on the life cycle state of a node, which may be used in transition conditions
    and observations.

    Every member is binary: a node that has not ended, or ended some other way, did not
    end the way an outcome predicate asks about.
    """

    IS_NOT_STARTED = frozenset({LifeCycleValues.NOT_STARTED})
    IS_RUNNING = frozenset({LifeCycleValues.RUNNING})
    IS_PAUSED = frozenset({LifeCycleValues.PAUSED})
    IS_TERMINATED = LifeCycleValues.terminal_states()
    IS_SUCCEEDED = frozenset({LifeCycleValues.SUCCEEDED})
    IS_FAILED = frozenset({LifeCycleValues.FAILED})
    IS_INTERRUPTED = frozenset({LifeCycleValues.INTERRUPTED})

    @property
    def attribute_name(self) -> str:
        """
        :return: The name this predicate is reached under on a node, also used to render
            it inside a condition.
        """
        return self.name.lower()


# %% observation predicates


class ObservationReading(Enum):
    """
    Which of a node's observations a test reads.
    """

    CURRENT = auto()
    """
    What the node observes now, which is unknown while it is not running.
    """

    LAST = auto()
    """
    The observation the node took most recently, which it keeps once it has ended.
    """


@dataclass(frozen=True)
class ObservationPredicateDefinition:
    """
    A test whether one of a node's observations is a particular value.
    """

    reading: ObservationReading
    """
    The observation the test reads.
    """

    observed_value: ObservationStateValues
    """
    The value the test is true for; every other value, unknown included, makes it false.
    """

    def truth_value(
        self, observation: ObservationStateValues
    ) -> ObservationStateValues:
        """
        :param observation: The observation to evaluate the test on.
        :return: True if `observation` is :attr:`observed_value`, false otherwise.
        """
        if observation == self.observed_value:
            return ObservationStateValues.TRUE
        return ObservationStateValues.FALSE

    def expression(self, observation: Scalar) -> Scalar:
        """
        The same test as :meth:`truth_value`, read off an expression rather than a value.

        :param observation: The observation to evaluate the test on.
        :return: True if `observation` is :attr:`observed_value`, false otherwise.
        """
        return Scalar(observation) == float(self.observed_value)


class ObservationPredicate(ObservationPredicateDefinition, Enum):
    """
    A two-valued test on what a node observes, which may be used in transition conditions
    and observation expressions alike.
    """

    OBSERVES_TRUE = ObservationReading.CURRENT, ObservationStateValues.TRUE
    OBSERVES_FALSE = ObservationReading.CURRENT, ObservationStateValues.FALSE
    LAST_OBSERVED_TRUE = ObservationReading.LAST, ObservationStateValues.TRUE

    @property
    def attribute_name(self) -> str:
        """
        :return: The name this predicate is reached under on a node, also used to render
            it inside a condition.
        """
        return self.name.lower()
