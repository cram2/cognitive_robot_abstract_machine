from __future__ import annotations

from dataclasses import field

from krrood.ormatic.utils import classproperty
from cramph.node import EndedByOwner
from cramph.context import ContextExtension
from giskardpy.motion_statechart.context import MotionControlContext
from giskardpy.motion_statechart.exceptions import EmptyDegreesOfFreedomError
from giskardpy.motion_statechart.graph_node import (
    MotionStatechartNode,
    velocity_convergence_expression,
)
from cramph.node import NodeArtifacts
from giskardpy.utils.decorators import dataclass
from krrood.symbolic_math.symbolic_math import FloatVariable
from semantic_digital_twin.world_description.degree_of_freedom import DegreeOfFreedom


@dataclass(repr=False, eq=False, kw_only=True)
class LocalMinimumReached(EndedByOwner, MotionStatechartNode):
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

    degrees_of_freedom: list[DegreeOfFreedom] | None = None
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
    motion (e.g. via :class:`~cramph.composites.Parallel`)
    -- otherwise ``minimum_time`` could already be satisfied by cycles the chart spent
    on earlier, unrelated motions, before this one ever started.
    """

    _start_cycle_variable: FloatVariable | None = field(
        init=False, default=None, repr=False
    )
    """
    Control-cycle count at which this monitor actually started running, set in
    ``on_start`` when :attr:`measure_from_own_start` is True.
    """

    @classproperty
    def required_context_extensions(cls) -> tuple[type[ContextExtension], ...]:
        return super().required_context_extensions + (MotionControlContext,)

    def on_start(self) -> None:
        if self.measure_from_own_start:
            self.context.float_variable_data.set_value(
                self._start_cycle_variable,
                self.context.tick_variable.evaluate()[0],
            )

    def set_up(self) -> None:
        """
        Register the variable holding the tick this monitor started on, if it measures
        from its own start.
        """
        super().set_up()
        if self.measure_from_own_start:
            self._start_cycle_variable = FloatVariable(f"{self.name}_start_cycle")
            self.context.float_variable_data.register_expression(
                self._start_cycle_variable
            )

    def build_artifacts(self) -> NodeArtifacts:
        if self.degrees_of_freedom is not None and not self.degrees_of_freedom:
            raise EmptyDegreesOfFreedomError(node=self)
        return NodeArtifacts(
            observation=velocity_convergence_expression(
                context=self.context,
                joint_convergence_threshold=self.joint_convergence_threshold,
                minimum_threshold=self.minimum_threshold,
                maximum_threshold=self.maximum_threshold,
                degrees_of_freedom=self.degrees_of_freedom,
                minimum_time=self.minimum_time,
                reference_cycle_variable=self._start_cycle_variable,
            )
        )
