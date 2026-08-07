from __future__ import annotations

from dataclasses import dataclass, field

from typing_extensions import List, Optional

import krrood.symbolic_math.symbolic_math as sm
from krrood.symbolic_math.symbolic_math import Scalar
from giskardpy.motion_statechart.context import MotionStatechartContext
from giskardpy.motion_statechart.data_types import (
    LifeCycleValues,
    ObservationStateValues,
)
from giskardpy.motion_statechart.exceptions import (
    NoConvergingTaskError,
    NoProgressError,
)
from giskardpy.motion_statechart.error_signals import ErrorSignal
from giskardpy.motion_statechart.graph_node import (
    CancelMotion,
    ConvergingTask,
    Goal,
    MotionStatechartNode,
    NodeArtifacts,
)
from giskardpy.motion_statechart.monitors.payload_monitors import (
    CountSimulationTimeSeconds,
)

# %% watching a single task


@dataclass(eq=False, repr=False)
class NotApproachingGoal(MotionStatechartNode):
    """
    Turns ``True`` while :attr:`monitored_task` is not closing on its goal fast enough.

    A task that is not running has no meaningful convergence rate, so it is reported as
    not approaching. That makes this node safe to combine with others, but it means the
    node only says something useful about a task while that task runs.

    .. note:: The rate passes through zero whenever the error turns around, for instance
        when the robot drives around an obstacle, so this node on its own is not evidence
        that a task is stuck. :class:`ProgressStalled` requires it to hold for a while.
    """

    monitored_task: ConvergingTask = field(kw_only=True)
    """
    The task whose progress towards its goal is watched.
    """

    minimum_convergence_rate: float = field(default=0.05, kw_only=True)
    """
    Rate below which the task counts as not approaching its goal, as a fraction of the
    task's own threshold per second.
    """

    _sampled_error: Optional[ErrorSignal] = field(default=None, init=False, repr=False)
    """
    The error to difference across control cycles, if it cannot be differentiated.
    """

    _previous_error: Optional[float] = field(default=None, init=False, repr=False)
    """
    Error measured on the previous control cycle.
    """

    _control_dt: float = field(default=0.0, init=False, repr=False)
    """
    Seconds between control cycles, used to turn a difference into a rate.
    """

    @property
    def prerequisite_nodes(self) -> List[MotionStatechartNode]:
        return [self.monitored_task]

    def build_artifacts(self, context: MotionStatechartContext) -> NodeArtifacts:
        """
        Compare the rate of change of the monitored task's error against
        :attr:`minimum_convergence_rate`.

        An error that can be differentiated gives an exact rate from the current joint
        velocities. One that cannot is differenced across control cycles in
        :meth:`on_tick` instead.
        """
        self._control_dt = (
            context.qp_controller_config.control_dt
            or context.qp_controller_config.model_predictive_control_time_step
        )
        error_signal = self.monitored_task.error_signal
        rate = error_signal.create_rate_expression()
        if rate is None:
            self._sampled_error = error_signal
            return NodeArtifacts()
        normalized_rate = rate / self.monitored_task.threshold
        return NodeArtifacts(
            observation=sm.trinary_logic_or(
                self._monitored_task_is_not_running(),
                sm.abs(normalized_rate) <= self.minimum_convergence_rate,
            )
        )

    def _monitored_task_is_not_running(self) -> Scalar:
        """
        :return: ``True`` while the monitored task is in any life cycle state other
            than :attr:`~giskardpy.motion_statechart.data_types.LifeCycleValues.RUNNING`.
        """
        return sm.Scalar(
            self.monitored_task.life_cycle_variable != int(LifeCycleValues.RUNNING)
        )

    def on_start(self, context: MotionStatechartContext):
        self._previous_error = None

    def on_tick(
        self, context: MotionStatechartContext
    ) -> Optional[ObservationStateValues]:
        """
        Measure the convergence rate of an error that cannot be differentiated.

        :return: For a differentiable error, ``None``, leaving the observation to the
            expression built in :meth:`build_artifacts`.
        """
        if self._sampled_error is None:
            return None
        if self.monitored_task.life_cycle_state != LifeCycleValues.RUNNING:
            self._previous_error = None
            return ObservationStateValues.TRUE
        error = float(self._sampled_error.expression.evaluate()[0])
        previous_error = self._previous_error
        self._previous_error = error
        if previous_error is None:
            # No rate is measurable from a single sample, so assume the task is moving.
            return ObservationStateValues.FALSE
        normalized_rate = (error - previous_error) / (
            self._control_dt * self.monitored_task.threshold
        )
        if abs(normalized_rate) <= self.minimum_convergence_rate:
            return ObservationStateValues.TRUE
        return ObservationStateValues.FALSE


@dataclass(eq=False, repr=False)
class AnyMonitoredTaskRunning(MotionStatechartNode):
    """
    Turns ``True`` while at least one of :attr:`monitored_tasks` is running.

    Without this, a set of tasks that have all finished, or have not started, would read
    as "nothing is approaching its goal" and be mistaken for a stall.
    """

    monitored_tasks: List[ConvergingTask] = field(kw_only=True)
    """
    The tasks whose life cycle states are watched.
    """

    def build_artifacts(self, context: MotionStatechartContext) -> NodeArtifacts:
        """
        Each life cycle comparison is ``0`` or ``1``, so their maximum is ``1`` exactly
        when at least one task runs.

        That also stays correct for a single task, unlike an n-ary or.
        """
        return NodeArtifacts(
            observation=sm.max(
                sm.Vector(
                    [
                        sm.Scalar(
                            task.life_cycle_variable == int(LifeCycleValues.RUNNING)
                        )
                        for task in self.monitored_tasks
                    ]
                )
            )
        )


# %% watching a whole goal


@dataclass(eq=False, repr=False)
class ProgressStalled(Goal):
    """
    Turns ``True`` once nothing under :attr:`monitored_node` has approached its goal for
    :attr:`timeout` seconds.

    Watching each converging task separately, rather than one combined error, keeps the
    measure meaningful for a :class:`~giskardpy.motion_statechart.goals.templates.Sequence`,
    whose steps run one after another, and names the task that is actually stuck.

    Wire it to a :class:`~giskardpy.motion_statechart.graph_node.CancelMotion` to abort a
    motion that is no longer making progress.
    """

    monitored_node: MotionStatechartNode = field(kw_only=True)
    """
    The task or goal whose progress is watched.
    """

    timeout: float = field(default=5.0, kw_only=True)
    """
    Seconds of simulated time without progress after which this turns ``True``.
    """

    minimum_convergence_rate: float = field(default=0.05, kw_only=True)
    """
    Rate below which a task counts as not approaching its goal, as a fraction of that
    task's own threshold per second.
    """

    _monitored_tasks: List[ConvergingTask] = field(
        default_factory=list, init=False, repr=False
    )
    """
    The converging tasks found under :attr:`monitored_node`.
    """

    _not_approaching_monitors: List[NotApproachingGoal] = field(
        default_factory=list, init=False, repr=False
    )
    """
    One monitor per entry of :attr:`_monitored_tasks`.
    """

    _timer: CountSimulationTimeSeconds = field(init=False, repr=False, default=None)
    """
    Counts how long every monitored task has been failing to approach its goal.
    """

    @property
    def prerequisite_nodes(self) -> List[MotionStatechartNode]:
        return [self.monitored_node]

    @property
    def monitored_tasks(self) -> List[ConvergingTask]:
        """
        :return: The converging tasks watched by this node, in the order they were found.
        """
        return self._monitored_tasks

    @property
    def stalled_tasks(self) -> List[ConvergingTask]:
        """
        :return: The monitored tasks that are running but not currently approaching their
            goal.
        """
        return [
            monitor.monitored_task
            for monitor in self._not_approaching_monitors
            if monitor.observation_state == ObservationStateValues.TRUE
            and monitor.monitored_task.life_cycle_state == LifeCycleValues.RUNNING
        ]

    def cancel_motion(self) -> CancelMotion:
        """
        :return: A node that aborts the motion with a
            :class:`~giskardpy.motion_statechart.exceptions.NoProgressError` once this
            node turns ``True``.
        """
        cancel = _CancelBecauseNoProgress(progress_monitor=self)
        cancel.start_condition = self.observation_variable
        return cancel

    def expand(self, context: MotionStatechartContext) -> None:
        self._monitored_tasks = self._find_converging_tasks(self.monitored_node)
        if not self._monitored_tasks:
            raise NoConvergingTaskError(node=self, monitored_node=self.monitored_node)
        not_approaching_monitors = [
            NotApproachingGoal(
                name=f"{self.name}/{task.name}",
                monitored_task=task,
                minimum_convergence_rate=self.minimum_convergence_rate,
            )
            for task in self._monitored_tasks
        ]
        any_running = AnyMonitoredTaskRunning(
            name=f"{self.name}/any_running", monitored_tasks=self._monitored_tasks
        )
        self._timer = CountSimulationTimeSeconds(
            name=f"{self.name}/timer", seconds=self.timeout
        )
        self._not_approaching_monitors = not_approaching_monitors
        self.add_nodes(not_approaching_monitors + [any_running, self._timer])

        stalled_now = sm.trinary_logic_and(
            any_running.observation_variable,
            *[monitor.observation_variable for monitor in not_approaching_monitors],
        )
        self._timer.start_condition = stalled_now
        self._timer.reset_condition = sm.trinary_logic_not(stalled_now)

    def build_artifacts(self, context: MotionStatechartContext) -> NodeArtifacts:
        return NodeArtifacts(observation=self._timer.observation_variable)

    def _find_converging_tasks(
        self, node: MotionStatechartNode
    ) -> List[ConvergingTask]:
        """
        Collect every converging task at or below ``node``.

        :param node: The node to search.
        :return: The converging tasks found, depth first.
        """
        if isinstance(node, ConvergingTask):
            return [node]
        if not isinstance(node, Goal):
            return []
        tasks = []
        for child_node in node.nodes:
            tasks.extend(self._find_converging_tasks(child_node))
        return tasks


@dataclass(eq=False, repr=False)
class _CancelBecauseNoProgress(CancelMotion):
    """
    Cancels the motion by raising an error naming the tasks that stopped approaching
    their goals.
    """

    progress_monitor: ProgressStalled = field(kw_only=True)
    """
    The monitor that detected the stall.
    """

    exception: Exception = field(init=False, default=Exception)
    """
    Set to ``init=False``, because this class creates its own exception once it knows
    which tasks are stalled.
    """

    def on_tick(self, context: MotionStatechartContext) -> Optional[float]:
        raise NoProgressError(progress_monitor=self.progress_monitor)
