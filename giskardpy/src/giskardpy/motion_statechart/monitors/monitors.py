from __future__ import annotations

import abc
from abc import ABC
from dataclasses import field
from typing_extensions import List, Optional

from giskardpy.motion_statechart.context import MotionStatechartContext
from giskardpy.motion_statechart.data_types import ObservationStateValues
from giskardpy.motion_statechart.exceptions import EmptyDegreesOfFreedomError
from giskardpy.motion_statechart.graph_node import (
    MotionStatechartNode,
    NodeArtifacts,
    velocity_convergence_expression,
)
from giskardpy.utils.decorators import dataclass
from krrood.symbolic_math.symbolic_math import FloatVariable
from semantic_digital_twin.world_description.degree_of_freedom import DegreeOfFreedom


@dataclass
class ThreadedPayloadMonitor(MotionStatechartNode, ABC):
    """
    A monitor which executes its __call__ function when start_condition becomes True.

    Subclass this and implement __init__.py and __call__. The __call__ method should
    change self.state to True when it's done. Calls __call__ in a separate thread. Use
    for expensive operations
    """

    state: ObservationStateValues = field(
        init=False, default=ObservationStateValues.UNKNOWN
    )

    @abc.abstractmethod
    def __call__(self):
        pass


@dataclass
class LocalMinimumReached(MotionStatechartNode):
    """
    Checks if the robot has reached a local minimum in the trajectory, by checking if
    all velocities are below a degree of freedoms' max velocity
    *`joint_convergence_threshold`.
    """

    joint_convergence_threshold: float = 0.01
    """
    If a degree of freedom velocity is below its maximum velocity * this value, it is
    considered as not moving.
    """

    minimum_threshold: float = 0.01
    """
    Minimum value for degree of freedom velocity * joint_convergence_threshold.
    """

    maximum_threshold: float = 0.06
    """
    Maximum value for degree of freedom velocity * joint_convergence_threshold.
    """

    windows_size: int = 1
    """
    Windows size for joint convergence check.
    """

    degrees_of_freedom: Optional[List[DegreeOfFreedom]] = None
    """
    Degrees of freedom to check for convergence.

    Defaults to ``context.world.active_degrees_of_freedom`` (every active degree of
    freedom) if left ``None``.
    """

    minimum_time: float = 1.0
    """
    Minimum elapsed control time (in seconds) before the observation can become true.
    """

    measure_from_own_start: bool = True
    """
    Whether ``minimum_time`` is measured from when this monitor itself started, instead
    of from the start of the whole motion chart.

    Set this when the monitor is wrapped around one specific, possibly late-starting
    motion (e.g. via :class:`~giskardpy.motion_statechart.goals.templates.Parallel`)
    -- otherwise ``minimum_time`` could already be satisfied by cycles the chart spent
    on earlier, unrelated motions, before this one ever started.
    """

    _start_cycle_variable: Optional[FloatVariable] = field(
        init=False, default=None, repr=False
    )
    """
    Control-cycle count at which this monitor actually started running, set in
    ``on_start`` when :attr:`measure_from_own_start` is True.
    """

    def on_start(self, context: MotionStatechartContext):
        if self.measure_from_own_start:
            context.float_variable_data.set_value(
                self._start_cycle_variable,
                context.control_cycle_variable.evaluate()[0],
            )

    def build_artifacts(self, context: MotionStatechartContext) -> NodeArtifacts:
        if self.degrees_of_freedom is not None and not self.degrees_of_freedom:
            raise EmptyDegreesOfFreedomError(node=self)
        if self.measure_from_own_start:
            self._start_cycle_variable = FloatVariable(f"{self.name}_start_cycle")
            context.float_variable_data.register_expression(self._start_cycle_variable)
        return NodeArtifacts(
            observation=velocity_convergence_expression(
                context=context,
                joint_convergence_threshold=self.joint_convergence_threshold,
                minimum_threshold=self.minimum_threshold,
                maximum_threshold=self.maximum_threshold,
                degrees_of_freedom=self.degrees_of_freedom,
                minimum_time=self.minimum_time,
                reference_cycle_variable=self._start_cycle_variable,
            )
        )
