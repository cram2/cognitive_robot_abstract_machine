from __future__ import annotations

from dataclasses import dataclass, field
from datetime import timedelta

import numpy as np
from typing_extensions import List, Optional

import krrood.symbolic_math.symbolic_math as sm
from krrood.symbolic_math.symbolic_math import (
    CompiledFunction,
    FloatVariable,
    Scalar,
    VariableParameters,
    trinary_logic_not,
)
from giskardpy.motion_statechart.context import MotionStatechartContext
from giskardpy.motion_statechart.data_types import (
    LifeCycleValues,
    ObservationStateValues,
)
from giskardpy.motion_statechart.exceptions import NoProgressError
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

    The task counts as approaching its goal while its error lies below the error it had
    when it last made progress, by at least :attr:`minimum_convergence_rate` times its
    threshold for every second since. An error that moves without getting smaller, as
    it does for an arm that collision avoidance keeps pushing back, therefore never
    counts as approaching, however fast it moves.

    A task that is not running has no meaningful progress, so it is reported as not
    approaching. That makes this node safe to combine with others, but it means the
    node only says something useful about a task while that task runs.

    A task that has reached its goal is reported as not approaching too: its error can
    keep moving within the threshold for as long as it holds the goal, as a grip held
    while something else moves does, and that movement is not progress. Counting it
    would keep a motion whose other tasks are stuck from ever being given up on.

    .. note:: An error also stops falling while the robot drives around an obstacle, so
        this node on its own is not evidence that a task is stuck.
        :class:`StillProgressing` requires it to hold for a while.
    """

    monitored_task: ConvergingTask = field(kw_only=True)
    """
    The task whose progress towards its goal is watched.
    """

    minimum_convergence_rate: float = field(default=0.05, kw_only=True)
    """
    Rate below which the task counts as not approaching its goal, as a fraction of the
    task's own threshold per second.

    0.05 means the error must fall by at least 5% of that task's own success threshold
    every second, or the task counts as not approaching its goal.
    """

    _error_at_last_progress: FloatVariable = field(init=False, repr=False)
    """
    The monitored task's error when it last counted as making progress, or infinity
    while there is nothing to compare with yet, so that the first error measured counts
    as progress.

    Small falls in between do not move it, so the error has to fall far enough below it
    for all the time since, not merely below the lowest value seen.
    """

    _seconds_since_last_progress: FloatVariable = field(init=False, repr=False)
    """
    Simulated time from the last progress up to the previous control cycle.
    """

    _compiled_error: CompiledFunction = field(init=False, repr=False)
    """
    The monitored task's error, compiled once and bound to the world state, so that
    :meth:`on_tick` can record it when the task makes progress.
    """

    _control_dt: float = field(default=0.0, init=False, repr=False)
    """
    Seconds between control cycles.
    """

    @property
    def prerequisite_nodes(self) -> List[MotionStatechartNode]:
        return [self.monitored_task]

    def build_artifacts(self, context: MotionStatechartContext) -> NodeArtifacts:
        self._control_dt = (
            context.qp_controller_config.control_time_step.total_seconds()
        )
        self._error_at_last_progress = self._registered_variable(
            "error_at_last_progress", context
        )
        self._seconds_since_last_progress = self._registered_variable(
            "seconds_since_last_progress", context
        )
        self._forget_progress(context)
        self._compile_error(context)
        error = self.monitored_task.error_signal
        required_fall = (
            self.minimum_convergence_rate
            * self.monitored_task.threshold
            * (self._seconds_since_last_progress + self._control_dt)
        )
        return NodeArtifacts(
            observation=sm.trinary_logic_or(
                self._monitored_task_is_not_running(),
                self.monitored_task.goal_reached_at(error),
                self._error_at_last_progress - error < required_fall,
            )
        )

    def _registered_variable(
        self, name: str, context: MotionStatechartContext
    ) -> FloatVariable:
        """
        :param name: What the variable holds.
        :param context: The context whose float variable data holds its value.
        :return: A new variable of this node, registered with that data.
        """
        variable = FloatVariable(f"{self.name}_{name}")
        context.float_variable_data.register_expression(variable)
        return variable

    def _compile_error(self, context: MotionStatechartContext) -> None:
        """
        Compile the monitored task's error against the world's positions and the float
        variable data, reading both in place.
        """
        self._compiled_error = self.monitored_task.error_signal.compile(
            parameters=VariableParameters.from_lists(
                context.world.state.position_float_variables,
                context.float_variable_data.variables,
            ),
            sparse=False,
        )
        self._compiled_error.bind_args_to_memory_view(0, context.world.state.positions)
        context.float_variable_data.bind_argument(self._compiled_error, 1)

    def _monitored_task_is_not_running(self) -> Scalar:
        """
        :return: ``True`` while the monitored task is in any life cycle state other
            than :attr:`~giskardpy.motion_statechart.data_types.LifeCycleValues.RUNNING`.
        """
        return sm.Scalar(
            self.monitored_task.life_cycle_variable != int(LifeCycleValues.RUNNING)
        )

    def on_start(self, context: MotionStatechartContext):
        self._forget_progress(context)

    def on_tick(self, context: MotionStatechartContext) -> None:
        """
        Move the error at the last progress forward once the task has made progress on
        this control cycle, and count the time while it has not.
        """
        if (
            self.monitored_task.life_cycle_state != LifeCycleValues.RUNNING
            or self.monitored_task.observation_state == ObservationStateValues.TRUE
        ):
            self._forget_progress(context)
            return None
        if self.observation_state == ObservationStateValues.FALSE:
            context.float_variable_data.set_value(
                self._error_at_last_progress,
                float(self._compiled_error.evaluate()[0]),
            )
            context.float_variable_data.set_value(
                self._seconds_since_last_progress, 0.0
            )
            return None
        context.float_variable_data.set_value(
            self._seconds_since_last_progress,
            context.float_variable_data.get_value(self._seconds_since_last_progress)
            + self._control_dt,
        )
        return None

    def _forget_progress(self, context: MotionStatechartContext) -> None:
        """
        Start judging progress afresh, from the next measured error.

        :param context: The context whose float variable data holds the progress.
        """
        context.float_variable_data.set_value(self._error_at_last_progress, np.inf)
        context.float_variable_data.set_value(self._seconds_since_last_progress, 0.0)


@dataclass(eq=False, repr=False)
class AnyMonitoredTaskShortOfItsGoal(MotionStatechartNode):
    """
    Turns ``True`` while at least one of :attr:`monitored_tasks` is running and has not
    reached its goal.

    Without this, a set of tasks that have all finished, or have not started, would read
    as "nothing is approaching its goal" and be mistaken for a stall.

    A task that reached its goal counts as finished even while it is still running:
    nothing ends a task for arriving, so it would otherwise keep this true for the rest
    of the motion and make the first wait after the last goal was reached a stall.
    """

    monitored_tasks: List[ConvergingTask] = field(kw_only=True)
    """
    The tasks whose life cycle states and goals are watched.
    """

    def build_artifacts(self, context: MotionStatechartContext) -> NodeArtifacts:
        """
        Each task contributes ``0`` or ``1``, so their maximum is ``1`` exactly when at
        least one of them is still short of its goal.

        That also stays correct for a single task, unlike an n-ary or.
        """
        return NodeArtifacts(
            observation=sm.max(
                sm.Vector(
                    [
                        sm.Scalar(
                            task.life_cycle_variable == int(LifeCycleValues.RUNNING)
                        )
                        * sm.logic_not(task.goal_reached.is_true())
                        for task in self.monitored_tasks
                    ]
                )
            )
        )


# %% watching a whole goal


@dataclass(eq=False, repr=False)
class StillProgressing(Goal):
    """
    Turns ``False`` once nothing under :attr:`monitored_node` has approached its goal
    for :attr:`timeout`.

    Watching each converging task separately, rather than one combined error, keeps the
    measure meaningful for a :class:`~giskardpy.motion_statechart.goals.templates.Sequence`,
    whose steps run one after another, and names the task that is actually stuck.

    Wire :meth:`cancel_motion` to abort a motion that is no longer making progress, or
    the negation of its observation to a node's end condition to give up on that node.
    """

    monitored_node: MotionStatechartNode = field(kw_only=True)
    """
    The task or goal whose progress is watched.
    """

    timeout: timedelta = field(default=timedelta(seconds=5), kw_only=True)
    """
    Simulated time without progress after which this turns ``False``.
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
            goal, leaving out those that already hold it.
        """
        return [
            monitor.monitored_task
            for monitor in self._not_approaching_monitors
            if monitor.observation_state == ObservationStateValues.TRUE
            and monitor.monitored_task.life_cycle_state == LifeCycleValues.RUNNING
            and monitor.monitored_task.observation_state != ObservationStateValues.TRUE
        ]

    def cancel_motion(self) -> CancelMotion:
        """
        :return: A node that aborts the motion with a
            :class:`~giskardpy.motion_statechart.exceptions.NoProgressError` once this
            node turns ``False``.
        """
        cancel = _CancelBecauseNoProgress(progress_monitor=self)
        cancel.start_condition = sm.trinary_logic_not(self.observation_variable)
        return cancel

    def expand(self, context: MotionStatechartContext) -> None:
        self._monitored_tasks = self._find_converging_tasks(self.monitored_node)
        self._timer = CountSimulationTimeSeconds(
            name=f"{self.name}/timer", seconds=self.timeout.total_seconds()
        )
        self._add_child_to_motion_statechart(self._timer)
        stalled_now = self._expand_stall_detection()
        self._timer.start_condition = stalled_now
        self._timer.reset_condition = sm.trinary_logic_not(stalled_now)

    def _expand_stall_detection(self) -> Scalar:
        """
        Adds one monitor per converging task and combines them into a single signal.

        A node with nothing converging beneath it has nothing that could approach a
        goal, so it counts as stalled for as long as it runs and :attr:`timeout` alone
        decides when it is given up on. That makes this node safe to point at anything,
        including a node built entirely from monitors.

        :return: True while nothing beneath the monitored node is approaching its goal.
        """
        if not self._monitored_tasks:
            return Scalar.const_true()
        self._not_approaching_monitors = [
            NotApproachingGoal(
                name=f"{self.name}/{task.name}",
                monitored_task=task,
                minimum_convergence_rate=self.minimum_convergence_rate,
            )
            for task in self._monitored_tasks
        ]
        still_working = AnyMonitoredTaskShortOfItsGoal(
            name=f"{self.name}/short_of_its_goal",
            monitored_tasks=self._monitored_tasks,
        )
        self._add_children_to_motion_statechart(
            self._not_approaching_monitors + [still_working]
        )
        return sm.trinary_logic_and(
            still_working.observation_variable,
            *[
                monitor.observation_variable
                for monitor in self._not_approaching_monitors
            ],
        )

    def build_artifacts(self, context: MotionStatechartContext) -> NodeArtifacts:
        """
        The timer only turns true once progress has stalled for :attr:`timeout`, so
        every other reading of it means this node has not given up yet.

        The timer is unknown until it starts, which a plain negation would carry through
        to a node that is in fact progressing, so the timer is compared against being
        true rather than negated.
        """
        return NodeArtifacts(
            observation=trinary_logic_not(self._timer.observation_variable.is_true())
        )

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

    progress_monitor: StillProgressing = field(kw_only=True)
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
