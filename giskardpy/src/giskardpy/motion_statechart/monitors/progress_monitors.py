from __future__ import annotations

from dataclasses import dataclass, field
from datetime import timedelta

import numpy as np
from numpy.typing import NDArray

import krrood.symbolic_math.symbolic_math as sm
from krrood.symbolic_math.symbolic_math import (
    CompiledFunction,
    FloatVariable,
    Scalar,
    VariableParameters,
    Vector,
)
from giskardpy.motion_statechart.context import MotionControlContext
from krrood.ormatic.utils import classproperty
from cramph.node import EndedByOwner
from cramph.context import ContextExtension
from cramph.data_types import LifeCycleValues, ObservationStateValues
from giskardpy.motion_statechart.exceptions import NoProgressError
from cramph.node import CancelStatechart, CompositeNode, NodeArtifacts
from giskardpy.motion_statechart.graph_node import ConvergingTask, MotionStatechartNode
from cramph.monitors import CountSimulationTimeSeconds

# %% watching tasks


@dataclass(eq=False, repr=False, kw_only=True)
class NoMonitoredTaskApproachingItsGoal(EndedByOwner, MotionStatechartNode):
    """
    Turns ``True`` while none of :attr:`monitored_tasks` is closing on its goal fast
    enough.

    A task counts as approaching its goal while its error lies below the error it had
    when it last made progress, by at least :attr:`minimum_convergence_rate` times its
    threshold for every second since. An error that moves without getting smaller, as
    it does for an arm that collision avoidance keeps pushing back, therefore never
    counts as approaching, however fast it moves.

    A task that is not running has no meaningful progress, so it counts as not
    approaching. That makes this node safe to combine with others, but it means the
    node only says something useful about the tasks that run.

    A task that has reached its goal counts as not approaching too: its error can keep
    moving within the threshold for as long as it holds the goal, as a grip held while
    something else moves does, and that movement is not progress. Counting it would keep
    a motion whose other tasks are stuck from ever being given up on. For the same
    reason, a task that leaves its goal again approaches it only once its error falls
    below the error it last made progress at, so a task cycling through its goal does not
    count as approaching.

    One node watches every task, rather than one node per task, because a plan holds
    dozens of converging tasks, most of which do not run at any one time.

    .. note:: An error also stops falling while the robot drives around an obstacle, so
        this node on its own is not evidence that a task is stuck.
        :class:`StillProgressing` requires it to hold for a while.
    """

    monitored_tasks: list[ConvergingTask]
    """
    The tasks whose progress towards their goals is watched.
    """

    minimum_convergence_rate: float = 0.05
    """
    Rate below which a task counts as not approaching its goal, as a fraction of the
    task's own threshold per second.

    0.05 means the error must fall by at least 5% of that task's own success threshold
    every second, or the task counts as not approaching its goal.
    """

    _errors_at_last_progress: NDArray[np.float64] = field(init=False, repr=False)
    """
    Each monitored task's error when it last counted as making progress, or infinity
    while there is nothing to compare with yet, so that the first error measured counts
    as progress.

    Small falls in between do not move it, so the error has to fall far enough below it
    for all the time since, not merely below the lowest value seen.
    """

    _seconds_since_last_progress: NDArray[np.float64] = field(init=False, repr=False)
    """
    Simulated time from each monitored task's last progress up to the previous control
    cycle.
    """

    _errors_at_last_progress_variables: Vector = field(init=False, repr=False)
    """
    The variables the observation reads :attr:`_errors_at_last_progress` through,
    updated whenever it changes.
    """

    _seconds_since_last_progress_variables: Vector = field(init=False, repr=False)
    """
    The variables the observation reads :attr:`_seconds_since_last_progress` through,
    updated whenever it changes.
    """

    _compiled_errors: CompiledFunction = field(init=False, repr=False)
    """
    The monitored tasks' errors, compiled once and bound to the world state, so that
    :meth:`on_tick` can measure them all at once.
    """

    _control_dt: float = field(default=0.0, init=False, repr=False)
    """
    Seconds between control cycles.
    """

    _previous_errors: NDArray[np.float64] = field(init=False, repr=False)
    """
    Each monitored task's error on the previous control cycle, which the observation of
    that cycle judged, or NaN where there was none.
    """

    @classproperty
    def required_context_extensions(cls) -> tuple[type[ContextExtension], ...]:
        return super().required_context_extensions + (MotionControlContext,)

    @property
    def prerequisite_nodes(self) -> list[MotionStatechartNode]:
        return list(self.monitored_tasks)

    @property
    def stalled_tasks(self) -> list[ConvergingTask]:
        """
        :return: The monitored tasks that are running but did not approach their goal
            on the last control cycle, leaving out those that already hold it.
        """
        stalled = self._judged() & self._falls_too_slowly(self._previous_errors)
        return [
            task
            for task, is_stalled in zip(self.monitored_tasks, stalled)
            if is_stalled
        ]

    def set_up(self) -> None:
        """
        Register the progress this node keeps track of.
        """
        super().set_up()
        self._control_dt = self.context.require_extension(
            MotionControlContext
        ).qp_controller_config.control_time_step.total_seconds()
        self._errors_at_last_progress_variables = self._registered_variables(
            "error_at_last_progress"
        )
        self._seconds_since_last_progress_variables = self._registered_variables(
            "seconds_since_last_progress"
        )
        self._errors_at_last_progress = np.full(len(self.monitored_tasks), np.inf)
        self._seconds_since_last_progress = np.zeros(len(self.monitored_tasks))
        self._previous_errors = np.full(len(self.monitored_tasks), np.nan)
        self._store_progress()

    def build_artifacts(self) -> NodeArtifacts:
        if not self.monitored_tasks:
            return NodeArtifacts(observation=Scalar.const_true())
        self._compile_errors()
        return NodeArtifacts(
            observation=sm.min(
                sm.Vector(
                    [
                        self._not_approaching(task, index)
                        for index, task in enumerate(self.monitored_tasks)
                    ]
                )
            )
        )

    def _not_approaching(self, task: ConvergingTask, index: int) -> Scalar:
        """
        :param task: One of the monitored tasks.
        :param index: Its position in :attr:`monitored_tasks`.
        :return: Whether `task` is not approaching its goal at its current error.
        """
        error = task.error_signal
        return sm.trinary_logic_or(
            sm.Scalar(task.life_cycle_variable != int(LifeCycleValues.RUNNING)),
            task.goal_reached_at(error),
            self._fell_too_little(
                error_at_last_progress=self._errors_at_last_progress_variables[index],
                error=error,
                seconds_since_last_progress=self._seconds_since_last_progress_variables[
                    index
                ],
                threshold=task.threshold,
            ),
        )

    def _fell_too_little(
        self,
        error_at_last_progress: Scalar | NDArray[np.float64],
        error: Scalar | NDArray[np.float64],
        seconds_since_last_progress: Scalar | NDArray[np.float64],
        threshold: float | NDArray[np.float64],
    ) -> Scalar | NDArray[np.bool_]:
        """
        The rule deciding whether an error fell by too little to count as progress, for
        symbolic expressions as well as numpy arrays, so that the observation and
        :meth:`on_tick` cannot disagree on it.

        :param error_at_last_progress: The error when the task last made progress.
        :param error: The error now.
        :param seconds_since_last_progress: Simulated time from the last progress up to
            the previous control cycle.
        :param threshold: The task's threshold.
        :return: Whether the error fell by less than the minimum rate requires.
        """
        required_fall = (
            self.minimum_convergence_rate
            * threshold
            * (seconds_since_last_progress + self._control_dt)
        )
        return error_at_last_progress - error < required_fall

    def _falls_too_slowly(self, errors: NDArray[np.float64]) -> NDArray[np.bool_]:
        """
        :param errors: An error per monitored task, NaN where there is none.
        :return: Per monitored task, whether the error is known and fell by too little
            since the task last made progress.
        """
        with np.errstate(invalid="ignore"):
            fell_too_little = self._fell_too_little(
                error_at_last_progress=self._errors_at_last_progress,
                error=errors,
                seconds_since_last_progress=self._seconds_since_last_progress,
                threshold=np.array([task.threshold for task in self.monitored_tasks]),
            )
        return ~np.isnan(errors) & fell_too_little

    def _registered_variables(self, name: str) -> Vector:
        """
        :param name: What the variables hold.
        :return: One new variable of this node per monitored task, registered with that
            data.
        """
        variables = Vector(
            [
                FloatVariable(f"{self.name}_{index}_{name}")
                for index in range(len(self.monitored_tasks))
            ]
        )
        if self.monitored_tasks:
            self.context.float_variable_data.register_expression(variables)
        return variables

    def _compile_errors(self) -> None:
        """
        Compile the monitored tasks' errors against the world's positions and the float
        variable data, reading both in place.
        """
        self._compiled_errors = Vector(
            [task.error_signal for task in self.monitored_tasks]
        ).compile(
            parameters=VariableParameters.from_lists(
                self.context.world.state.position_float_variables,
                self.context.float_variable_data.variables,
            ),
            sparse=False,
        )
        self._compiled_errors.bind_args_to_memory_view(
            0, self.context.world.state.positions
        )
        self.context.float_variable_data.bind_argument(self._compiled_errors, 1)

    def _running(self) -> NDArray[np.bool_]:
        """
        :return: Per monitored task, whether it runs.
        """
        return np.array(
            [
                task.life_cycle_state == LifeCycleValues.RUNNING
                for task in self.monitored_tasks
            ],
            dtype=bool,
        )

    def _judged(self) -> NDArray[np.bool_]:
        """
        :return: Per monitored task, whether it runs and has not reached its goal, the
            only tasks whose progress is judged.
        """
        return self._running() & np.array(
            [
                task.observation_state != ObservationStateValues.TRUE
                for task in self.monitored_tasks
            ],
            dtype=bool,
        )

    def on_start(self) -> None:
        self._forget_progress(np.ones(len(self.monitored_tasks), dtype=bool))
        self._store_progress()

    def on_tick(self) -> None:
        """
        Move each task's error at the last progress forward once it made progress on the
        previous control cycle, count the time while it did not, and forget both for the
        tasks that no longer run.

        A tick calls this before it observes anything, so what it reads of the monitored
        tasks is how the previous control cycle ended, and the errors that cycle judged
        are the ones remembered from it.
        """
        if not self.monitored_tasks:
            return None
        errors = np.asarray(self._compiled_errors.evaluate(), dtype=float).reshape(-1)
        previous_errors, self._previous_errors = self._previous_errors, errors.copy()
        still_judged = self._judged()
        judged = still_judged & ~np.isnan(previous_errors)
        stalled = judged & self._falls_too_slowly(previous_errors)
        approached = judged & ~stalled
        self._errors_at_last_progress[approached] = previous_errors[approached]
        self._seconds_since_last_progress[approached] = 0.0
        self._seconds_since_last_progress[stalled] += self._control_dt
        self._forget_progress(~self._running())
        self._store_progress()
        return None

    def _forget_progress(self, tasks: NDArray[np.bool_]) -> None:
        """
        Start judging the progress of `tasks` afresh, from their next measured errors.

        :param tasks: Per monitored task, whether to forget its progress.
        """
        self._errors_at_last_progress[tasks] = np.inf
        self._seconds_since_last_progress[tasks] = 0.0
        self._previous_errors[tasks] = np.nan

    def _store_progress(self) -> None:
        """
        Copy :attr:`_errors_at_last_progress` and :attr:`_seconds_since_last_progress`
        into the float variable data the observation reads.
        """
        if self.monitored_tasks:
            self.context.float_variable_data.set_value(
                self._errors_at_last_progress_variables, self._errors_at_last_progress
            )
            self.context.float_variable_data.set_value(
                self._seconds_since_last_progress_variables,
                self._seconds_since_last_progress,
            )


@dataclass(eq=False, repr=False, kw_only=True)
class AnyMonitoredTaskShortOfItsGoal(EndedByOwner, MotionStatechartNode):
    """
    Turns ``True`` while at least one of :attr:`monitored_tasks` is running and has not
    reached its goal.

    Without this, a set of tasks that have all finished, or have not started, would read
    as "nothing is approaching its goal" and be mistaken for a stall.

    A task that reached its goal counts as finished even while it is still running:
    nothing ends a task for arriving, so it would otherwise keep this true for the rest
    of the motion and make the first wait after the last goal was reached a stall.
    """

    monitored_tasks: list[ConvergingTask]
    """
    The tasks whose life cycle states and goals are watched.
    """

    def build_artifacts(self) -> NodeArtifacts:
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
                        * sm.logic_not(task.observes_true)
                        for task in self.monitored_tasks
                    ]
                )
            )
        )


# %% watching a whole goal


@dataclass(eq=False, repr=False, kw_only=True)
class StillProgressing(EndedByOwner, CompositeNode):
    """
    Turns ``False`` once nothing under :attr:`monitored_node` has approached its goal
    for :attr:`timeout`.

    Watching each converging task separately, rather than one combined error, keeps the
    measure meaningful for a :class:`~cramph.composites.Sequence`, whose steps run one
    after another, and names the task that is actually stuck.

    Wire :meth:`cancel_motion` to abort a motion that is no longer making progress, or
    the negation of its observation to a node's fail condition to give up on that node.
    """

    monitored_node: MotionStatechartNode
    """
    The task or goal whose progress is watched.
    """

    timeout: timedelta = timedelta(seconds=5)
    """
    Simulated time without progress after which this turns ``False``.
    """

    minimum_convergence_rate: float = 0.05
    """
    Rate below which a task counts as not approaching its goal, as a fraction of that
    task's own threshold per second.
    """

    @property
    def prerequisite_nodes(self) -> list[MotionStatechartNode]:
        return [self.monitored_node]

    @property
    def monitored_tasks(self) -> list[ConvergingTask]:
        """
        :return: The converging tasks watched by this node, in the order they were found.
        """
        if self._not_approaching is None:
            return []
        return self._not_approaching.monitored_tasks

    @property
    def _not_approaching(self) -> NoMonitoredTaskApproachingItsGoal | None:
        """
        :return: The child watching every converging task found under
            :attr:`monitored_node`, or None if nothing converges beneath it.
        """
        return next(
            (
                node
                for node in self.nodes
                if isinstance(node, NoMonitoredTaskApproachingItsGoal)
            ),
            None,
        )

    @property
    def _timer(self) -> CountSimulationTimeSeconds:
        """
        :return: The child counting how long every monitored task has been failing to
            approach its goal.
        """
        return self.nodes[0]

    @property
    def stalled_tasks(self) -> list[ConvergingTask]:
        """
        :return: The monitored tasks that are running but not currently approaching their
            goal, leaving out those that already hold it.
        """
        if self._not_approaching is None:
            return []
        return self._not_approaching.stalled_tasks

    def cancel_motion(self) -> CancelStatechart:
        """
        :return: A node that aborts the motion with a
            :class:`~giskardpy.motion_statechart.exceptions.NoProgressError` once this
            node turns ``False``.
        """
        cancel = _CancelBecauseNoProgress(progress_monitor=self)
        cancel.start_condition = self.observes_false
        return cancel

    def expand(self) -> None:
        timer = CountSimulationTimeSeconds(
            name=f"{self.name}/timer", seconds=self.timeout.total_seconds()
        )
        self._add_child_to_statechart(timer)
        stall_monitors = self._expand_stall_detection(
            self._find_converging_tasks(self.monitored_node)
        )
        timer.start_condition = sm.logic_and(
            Scalar.const_true(), *[monitor.observes_true for monitor in stall_monitors]
        )
        timer.reset_condition = sm.logic_or(
            Scalar.const_false(),
            *[monitor.observes_false for monitor in stall_monitors],
        )

    def _expand_stall_detection(
        self, monitored_tasks: list[ConvergingTask]
    ) -> list[MotionStatechartNode]:
        """
        Adds one monitor watching every converging task, next to one observing whether
        any of them is still short of its goal.

        A node with nothing converging beneath it has nothing that could approach a
        goal, so it counts as stalled for as long as it runs and :attr:`timeout` alone
        decides when it is given up on. That makes this node safe to point at anything,
        including a node built entirely from monitors.

        :param monitored_tasks: The converging tasks found under :attr:`monitored_node`.
        :return: The monitors that all observe True while nothing beneath the monitored
            node is approaching its goal, none if nothing converges beneath it.
        """
        if not monitored_tasks:
            return []
        not_approaching = NoMonitoredTaskApproachingItsGoal(
            name=f"{self.name}/not_approaching",
            monitored_tasks=monitored_tasks,
            minimum_convergence_rate=self.minimum_convergence_rate,
        )
        still_working = AnyMonitoredTaskShortOfItsGoal(
            name=f"{self.name}/short_of_its_goal", monitored_tasks=monitored_tasks
        )
        self._add_children_to_statechart([not_approaching, still_working])
        return [not_approaching, still_working]

    def build_artifacts(self) -> NodeArtifacts:
        """
        The timer only reaches what it counts once progress has stalled for
        :attr:`timeout`, so every other reading of it means this node has not given up
        yet. Nothing but this node ending ends the timer, so its live observation is
        there for as long as this node observes anything.

        The timer says nothing until it starts, which counts as not having reached what
        it counts.
        """
        return NodeArtifacts(observation=sm.logic_not(self._timer.observes_true))

    def _find_converging_tasks(
        self, node: MotionStatechartNode
    ) -> list[ConvergingTask]:
        """
        Collect every converging task at or below ``node``.

        :param node: The node to search.
        :return: The converging tasks found, depth first.
        """
        if isinstance(node, ConvergingTask):
            return [node]
        if not isinstance(node, CompositeNode):
            return []
        tasks = []
        for child_node in node.nodes:
            tasks.extend(self._find_converging_tasks(child_node))
        return tasks


@dataclass(eq=False, repr=False, kw_only=True)
class Stalled(StillProgressing):
    """
    Turns ``True`` once nothing under :attr:`monitored_node` has approached its goal for
    :attr:`timeout`.

    The same measurement as :class:`StillProgressing`, said the way a failure monitor is
    read: it observes the thing that goes wrong, not the thing that goes right, so a node
    choosing its child can give up on that child once it fires.
    """

    def build_artifacts(self) -> NodeArtifacts:
        """
        The timer reaches what it counts once progress has stalled for :attr:`timeout`,
        which is exactly when this node has something to report.
        """
        return NodeArtifacts(observation=self._timer.observes_true)


@dataclass(eq=False, repr=False, kw_only=True)
class _CancelBecauseNoProgress(CancelStatechart):
    """
    Cancels the motion by raising an error naming the tasks that stopped approaching
    their goals.
    """

    progress_monitor: StillProgressing
    """
    The monitor that detected the stall.
    """

    exception: Exception = field(init=False, default=Exception)
    """
    Set to ``init=False``, because this class creates its own exception once it knows
    which tasks are stalled.
    """

    def create_exception(self) -> NoProgressError:
        return NoProgressError(progress_monitor=self.progress_monitor)
