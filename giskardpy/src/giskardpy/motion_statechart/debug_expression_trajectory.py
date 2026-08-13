from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from typing_extensions import TYPE_CHECKING, List

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

    _free_variables: List[FloatVariable] = field(init=False)
    """The free variables of the expression, resolved against the live world state."""

    _compiled_function: CompiledFunction = field(init=False)
    """The expression compiled once for efficient per-cycle evaluation."""

    _values: List[np.ndarray] = field(init=False, default_factory=list)
    """The flattened expression value recorded for each control cycle."""

    def __post_init__(self):
        self._free_variables = self.debug_expression.expression.free_variables()
        self._compiled_function = self.debug_expression.expression.compile(
            VariableParameters.from_lists(self._free_variables)
        )

    @property
    def name(self) -> str:
        """The name of the recorded debug expression."""
        return self.debug_expression.name

    def record(self) -> None:
        """Evaluate the expression against the current world state and store the result."""
        arguments = np.array(
            [variable.resolve() for variable in self._free_variables], dtype=np.float64
        )
        self._values.append(self._compiled_function(arguments).flatten())

    @property
    def values(self) -> np.ndarray:
        """The recorded values with shape ``(number_of_cycles, number_of_components)``."""
        return np.asarray(self._values)


@dataclass
class DebugExpressionTrajectory:
    """
    Time series of all debug expressions recorded during a motion statechart execution.
    """

    recorded_debug_expressions: List[RecordedDebugExpression]
    """One recorder per debug expression that is tracked."""

    _times: List[float] = field(init=False, default_factory=list)
    """The simulation time in seconds of each recorded control cycle."""

    @classmethod
    def from_debug_expressions(
        cls, debug_expressions: List[DebugExpression]
    ) -> DebugExpressionTrajectory:
        """Create a trajectory that records the given debug expressions."""
        return cls(
            recorded_debug_expressions=[
                RecordedDebugExpression(debug_expression)
                for debug_expression in debug_expressions
            ]
        )

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
        """The simulation time in seconds of each recorded control cycle."""
        return np.asarray(self._times)
