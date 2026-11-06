from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from typing_extensions import TYPE_CHECKING

from krrood.symbolic_math.symbolic_math import VariableParameters
from semantic_digital_twin.exceptions import NonMonotonicTimeError

if TYPE_CHECKING:
    from giskardpy.motion_statechart.graph_node import DebugExpression
    from krrood.symbolic_math.symbolic_math import CompiledFunction, FloatVariable


@dataclass
class RecordedDebugExpression:
    """
    Records the value of a single :class:`DebugExpression` over the course of a motion.

    The underlying symbolic expression is compiled once on construction and re-evaluated
    against the live world state on every :meth:`record` call, which avoids the repeated
    recompilation that :meth:`DebugExpression.evaluated` would incur per control cycle.
    """

    debug_expression: DebugExpression
    """The debug expression whose value is recorded."""

    _free_variables: list[FloatVariable] = field(init=False)
    """
    The free variables of the expression, resolved against the live world state.
    """

    _compiled_function: CompiledFunction = field(init=False)
    """
    The expression compiled once for efficient per-cycle evaluation.
    """

    _values: list[np.ndarray] = field(init=False, default_factory=list)
    """
    The flattened expression value recorded for each control cycle.
    """

    _is_followed: bool = field(init=False, default=True)
    """
    Whether :attr:`debug_expression` is still evaluated; once it is not, every control
    cycle is recorded without a value.
    """

    def __post_init__(self):
        self.follow(self.debug_expression)

    @classmethod
    def without_values_so_far(
        cls, debug_expression: DebugExpression, cycles_so_far: int
    ) -> RecordedDebugExpression:
        """
        :param debug_expression: The debug expression whose value is recorded.
        :param cycles_so_far: How many control cycles were recorded before
            `debug_expression` existed.
        :return: A record of `debug_expression` that has no value for the control
            cycles so far.
        """
        recorded = cls(debug_expression)
        recorded._values = [recorded._missing_value() for _ in range(cycles_so_far)]
        return recorded

    def follow(self, debug_expression: DebugExpression) -> None:
        """
        Record the value of `debug_expression` from now on, keeping the values recorded
        so far.

        :param debug_expression: The debug expression whose value is recorded.
        """
        self.debug_expression = debug_expression
        self._free_variables = debug_expression.expression.free_variables()
        self._compiled_function = debug_expression.expression.compile(
            VariableParameters.from_lists(self._free_variables)
        )
        self._is_followed = True

    def stop_following(self) -> None:
        """
        Record every control cycle without a value from now on, keeping the values
        recorded so far.
        """
        self._is_followed = False

    def _missing_value(self) -> np.ndarray:
        """
        :return: What is recorded for a control cycle without a value: not a number, for
            every component of the expression.
        """
        rows, columns = self.debug_expression.expression.shape
        return np.full(rows * columns, np.nan)

    @property
    def name(self) -> str:
        """
        The name of the recorded debug expression.
        """
        return self.debug_expression.name

    def record(self) -> None:
        """
        Evaluate the expression against the current world state and store the result.
        """
        if not self._is_followed:
            self._values.append(self._missing_value())
            return
        arguments = np.array(
            [variable.resolve() for variable in self._free_variables], dtype=np.float64
        )
        self._values.append(self._compiled_function(arguments).flatten())

    @property
    def values(self) -> np.ndarray:
        """
        The recorded values with shape ``(number_of_cycles, number_of_components)``.
        """
        return np.asarray(self._values)


@dataclass
class DebugExpressionTrajectory:
    """
    Time series of all debug expressions recorded during a motion statechart execution.
    """

    recorded_debug_expressions: list[RecordedDebugExpression]
    """
    One recorder per debug expression that is tracked.
    """

    _times: list[float] = field(init=False, default_factory=list)
    """
    The simulation time in seconds of each recorded control cycle.
    """

    @classmethod
    def from_debug_expressions(
        cls, debug_expressions: list[DebugExpression]
    ) -> DebugExpressionTrajectory:
        """
        Create a trajectory that records the given debug expressions.
        """
        return cls(
            recorded_debug_expressions=[
                RecordedDebugExpression(debug_expression)
                for debug_expression in debug_expressions
            ]
        )

    def follow(self, debug_expressions: list[DebugExpression]) -> None:
        """
        Record `debug_expressions` from now on, keeping everything recorded so far.

        A recorded debug expression goes on with the given one of the same name. A given
        one that was not recorded yet has no value for the times so far, and a recorded
        one that is no longer given has no value from now on.

        :param debug_expressions: The debug expressions that exist now.
        """
        no_longer_given = list(self.recorded_debug_expressions)
        for debug_expression in debug_expressions:
            recorded = next(
                (
                    recorded
                    for recorded in no_longer_given
                    if recorded.name == debug_expression.name
                ),
                None,
            )
            if recorded is None:
                self.recorded_debug_expressions.append(
                    RecordedDebugExpression.without_values_so_far(
                        debug_expression, cycles_so_far=len(self._times)
                    )
                )
                continue
            no_longer_given.remove(recorded)
            recorded.follow(debug_expression)
        for recorded in no_longer_given:
            recorded.stop_following()

    def append(self, time: float) -> None:
        """
        Record the current value of every tracked debug expression for the given time.

        :param time: The simulation time in seconds. Must be strictly greater than the
            time of the previous recording.
        """
        if self._times and time <= self._times[-1]:
            raise NonMonotonicTimeError(
                last_time=float(self._times[-1]), attempted_time=time
            )
        self._times.append(time)
        for recorded_debug_expression in self.recorded_debug_expressions:
            recorded_debug_expression.record()

    @property
    def times(self) -> np.ndarray:
        """
        The simulation time in seconds of each recorded control cycle.
        """
        return np.asarray(self._times)
