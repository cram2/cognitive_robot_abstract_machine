from __future__ import division

from dataclasses import dataclass, field
from datetime import timedelta
from typing import List

from typing_extensions import Optional

from giskardpy.motion_statechart.context import MotionStatechartContext
from giskardpy.motion_statechart.data_types import ObservationStateValues
from giskardpy.motion_statechart.graph_node import (
    CancelMotion,
    Goal,
    MotionStatechartNode,
    NodeArtifacts,
    TerminalNode,
)
from giskardpy.motion_statechart.monitors.progress_monitors import StillProgressing
from giskardpy.motion_statechart.monitors.templates import StoppedWhenTrue
from krrood.exceptions import DataclassException
from krrood.symbolic_math.symbolic_math import (
    Scalar,
    if_cases,
    sum,
    trinary_logic_and,
    trinary_logic_not,
    trinary_logic_or,
)


@dataclass(repr=False, eq=False)
class Sequence(Goal):
    """
    Runs a list of nodes one after another.

    Its observation turns True once the last step reached its goal, and False as soon as
    a step ended short of its own, so a step that was given up on fails the sequence
    rather than leaving it waiting forever.

    .. note:: corresponds to the RPL's SEQ. (McDermott, Drew. A reactive plan language, 1991)
    """

    nodes: List[MotionStatechartNode] = field(default_factory=list, init=True)

    def expand(self, context: MotionStatechartContext) -> None:
        """
        A step ends itself once it observes its goal, which succeeds it, and the next
        step reads that verdict rather than the observation behind it, because only the
        verdict outlasts the step that reached it.

        Reaching its goal is a reason to end a step on top of whatever the caller
        already wired, not instead of it: being given up on is the only way a step ever
        ends short of its goal, since observing false is how a step looks on its way
        there.
        """
        self._check_has_children()
        last_node: Optional[MotionStatechartNode] = None
        for node in self.nodes:
            self.add_node(node)
            if last_node is not None:
                node.start_condition = last_node.is_succeeded
            # A node that ends the motion has nothing left to transition to.
            if not isinstance(node, TerminalNode):
                node.end_condition = trinary_logic_or(
                    node.observation_variable, node.end_condition
                )
            last_node = node

    def build_artifacts(self, context: MotionStatechartContext) -> NodeArtifacts:
        """
        Report success, a failed step, or neither.

        A step short of its goal has not failed, it has not arrived yet, so only a step
        that ended without reaching its goal decides anything. The last step is read
        through its verdict, which outlasts the step that earned it.
        """
        return NodeArtifacts(
            observation=if_cases(
                cases=[
                    (
                        trinary_logic_or(
                            *[
                                node.ended_without_reaching_its_goal
                                for node in self.nodes
                            ]
                        ),
                        Scalar.const_false(),
                    ),
                    (self.nodes[-1].goal_reached.is_true(), Scalar.const_true()),
                ],
                else_result=Scalar.const_trinary_unknown(),
            )
        )


@dataclass(repr=False, eq=False)
class Parallel(Goal):
    """
    Takes a list of nodes and executes them in parallel.

    Its observation turns True once at least :attr:`minimum_success` of them reached
    their goals.
    """

    nodes: List[MotionStatechartNode] = field(default_factory=list, init=True)
    minimum_success: Optional[int] = field(default=None, kw_only=True)
    """
    How many nodes must have reached their goals for this goal to be achieved.

    Defaults to None, which means all of them.
    """

    def expand(self, context: MotionStatechartContext) -> None:
        self._check_has_children()
        for node in self.nodes:
            self.add_node(node)

    def build_artifacts(self, context: MotionStatechartContext) -> NodeArtifacts:
        """
        Count the nodes that reached their goals, and compare that against
        :attr:`minimum_success`.

        This goal ends none of its nodes, so a node that keeps running is counted by
        what it observes now and stops counting once it drifts away from its goal again.
        A node something *else* ended keeps counting, because its verdict outlasts it.
        """
        nodes_at_their_goal = [node.goal_reached.is_true() for node in self.nodes]
        minimum_success = (
            self.minimum_success
            if self.minimum_success is not None
            else len(self.nodes)
        )
        return NodeArtifacts(observation=minimum_success <= sum(*nodes_at_their_goal))


# %% giving a motion an outcome


@dataclass(repr=False, eq=False)
class Attempt(Goal):
    """
    Runs a motion that would never end on its own and decides it, one way or the other.

    A constraint holds itself against the world and observes only whether it is at its
    goal right now, so nothing about it ever concludes. This goal concludes instead: it
    observes True once the task is at its goal, False once one of :attr:`failure_monitors`
    fires, and ends itself as soon as it observes either. That is what lets a maintained
    motion be one step of a plan.

    .. note:: The task is never ended from here. It keeps exerting itself after first
        reaching its goal and comes down only with this goal, so a constraint that was
        pushed off its goal again is still being held.
    """

    task: MotionStatechartNode = field(kw_only=True)
    """
    The motion run until this goal is decided.
    """

    failure_monitors: List[MotionStatechartNode] = field(kw_only=True)
    """
    The nodes whose observing True gives up on the task, any one of which is enough.

    An empty list states that this motion cannot fail, leaving reaching its goal as the
    only way it ends. A monitor is read the way it is written, so one that observes
    being *well* has to be negated before it can be passed here.
    """

    @property
    def any_failure_monitor_fired(self) -> Scalar:
        """
        :return: True once a failure monitor fired, and false while none has or there
            are none to fire.
        """
        if not self.failure_monitors:
            return Scalar.const_false()
        return trinary_logic_or(
            *[monitor.goal_reached.is_true() for monitor in self.failure_monitors]
        )

    @property
    def failure_reasons(self) -> List[MotionStatechartNode]:
        """
        Which monitors gave up on the task, which is what turns a failure into a reason.

        They are read through their verdicts, because ending this goal ends them too and
        a node that ended observes nothing any more.

        :return: The failure monitors that fired, in the order they were given, and
            nothing at all unless this goal ended short of its own goal.
        """
        if self.goal_reached_state != ObservationStateValues.FALSE:
            return []
        return [
            monitor
            for monitor in self.failure_monitors
            if monitor.goal_reached_state == ObservationStateValues.TRUE
        ]

    def expand(self, context: MotionStatechartContext) -> None:
        """
        Add the task and the monitors, and end this goal once it has an answer.

        A condition may only read its own node or a sibling, so this goal reaches that
        answer through its own observation rather than through its children.

        A monitor ends on the control cycle it fires, which keeps what it saw as its
        verdict. A goal reads its children a cycle late, so a monitor that fires only
        briefly would otherwise be indistinguishable afterwards from one that never
        fired at all.
        """
        self.add_node(self.task)
        self.add_nodes(self.failure_monitors)
        for failure_monitor in self.failure_monitors:
            failure_monitor.end_condition = trinary_logic_or(
                failure_monitor.end_condition, failure_monitor.observation_variable
            )
        self.end_condition = trinary_logic_or(
            self.end_condition,
            trinary_logic_not(self.observation_variable.is_unknown()),
        )

    def build_artifacts(self, context: MotionStatechartContext) -> NodeArtifacts:
        """
        Report reaching the goal, being given up on, or neither.

        Reaching the goal outranks a monitor firing on the same control cycle: a task
        that arrived did what it was asked, whatever else was true at that moment. The
        task is read through its verdict, so one that something else ended still counts.
        """
        return NodeArtifacts(
            observation=if_cases(
                cases=[
                    (self.task.goal_reached.is_true(), Scalar.const_true()),
                    (self.any_failure_monitor_fired, Scalar.const_false()),
                ],
                else_result=Scalar.const_trinary_unknown(),
            )
        )


# %% repeating a task


@dataclass(repr=False, eq=False)
class RepeatUntil(Goal):
    """
    Runs a task again from the start whenever an attempt at it fails.

    Its observation turns True once the task succeeds and False once
    :attr:`stop_retry_monitor` calls the retrying off, so a caller can tell "eventually
    worked" from "gave up".

    Hand it a :attr:`failure_monitor` to decide what a failed attempt is, or subclass it
    and derive that decision from the task instead, overriding :attr:`failure_monitor`
    with an ``init=False`` field set in :meth:`__post_init__`.
    """

    task: MotionStatechartNode = field(kw_only=True)
    """
    The node to run, and to run again after every failed attempt.

    Resetting a goal resets everything below it, so a composite task starts over as a
    unit.
    """

    stop_retry_monitor: MotionStatechartNode = field(kw_only=True)
    """
    Stops the retrying once it observes True, which makes this goal observe False.
    """

    failure_monitor: MotionStatechartNode = field(kw_only=True)
    """
    Decides that an attempt failed and the task should run again.

    Subclasses that derive one from the task instead exclude this from their constructor
    and set it themselves in :meth:`__post_init__`.
    """

    @property
    def attempt_failed(self) -> Scalar:
        """
        :return: True on the control cycles on which the running attempt counts as
            failed.
        """
        return self.failure_monitor.observation_variable.is_true()

    def expand(self, context: MotionStatechartContext) -> None:
        """
        Wire the retry loop.

        The failure monitor resets the task and itself with the same observation. A node
        that has been reset observes Unknown, so the reset lasts a single control cycle
        instead of holding the task at the start line, and the monitor is armed again
        for the next attempt.
        """
        self.add_nodes([self.task, self.failure_monitor, self.stop_retry_monitor])

        # Each reading is compared against True, so that an undecided Unknown counts as
        # neither, and the results combine as plain booleans.
        attempt_succeeded = self.task.goal_reached.is_true()
        attempt_failed = self.attempt_failed
        still_trying = trinary_logic_not(
            self.stop_retry_monitor.observation_variable.is_true()
        )

        # Starting is gated as well as ending, because a reset task is not started and
        # ending is not considered while it is not.
        self.task.start_condition = still_trying
        # A failure that arrives on the same control cycle as the success must not undo
        # it, and resetting takes precedence over ending.
        self.task.reset_condition = trinary_logic_and(
            attempt_failed, trinary_logic_not(attempt_succeeded), still_trying
        )
        self.task.end_condition = trinary_logic_or(
            self.task.observation_variable,
            self.stop_retry_monitor.observation_variable,
        )
        self.failure_monitor.reset_condition = trinary_logic_and(
            attempt_failed, still_trying
        )
        self.failure_monitor.end_condition = (
            self.stop_retry_monitor.observation_variable
        )

    def build_artifacts(self, context: MotionStatechartContext) -> NodeArtifacts:
        """
        Report success, giving up, or neither.

        The cases are compared against True rather than combined with trinary logic,
        because an undecided attempt is Unknown and would otherwise read as a success.
        The task is read through its verdict, which outlasts the attempt that earned it.
        """
        return NodeArtifacts(
            observation=if_cases(
                cases=[
                    (self.task.goal_reached.is_true(), Scalar.const_true()),
                    (
                        self.stop_retry_monitor.observation_variable.is_true(),
                        Scalar.const_false(),
                    ),
                ],
                else_result=Scalar.const_trinary_unknown(),
            )
        )


@dataclass(repr=False, eq=False)
class RepeatOnStall(RepeatUntil):
    """
    Runs a task again from the start whenever it stops approaching its goal.

    A task with nothing converging beneath it never approaches anything, so
    :attr:`timeout` alone decides when such an attempt is given up on.
    """

    failure_monitor: MotionStatechartNode = field(init=False, kw_only=True)
    """
    Watches the task's own progress; derived from it in :meth:`__post_init__` rather
    than accepted from the caller.
    """

    timeout: timedelta = field(default=timedelta(seconds=5), kw_only=True)
    """
    Simulated time without progress after which an attempt counts as failed.
    """

    minimum_convergence_rate: float = field(default=0.05, kw_only=True)
    """
    Rate below which a task counts as not approaching its goal, as a fraction of that
    task's own threshold per second.
    """

    def __post_init__(self) -> None:
        super().__post_init__()
        self.failure_monitor = StillProgressing(
            name=f"{self.name}/progress",
            monitored_node=self.task,
            timeout=self.timeout,
            minimum_convergence_rate=self.minimum_convergence_rate,
        )

    @property
    def attempt_failed(self) -> Scalar:
        """
        :return: True once the attempt has stopped approaching its goal, which is what
            its progress monitor observes the negation of.
        """
        return trinary_logic_not(self.failure_monitor.observation_variable)


# %% trying alternatives


@dataclass(repr=False, eq=False)
class TryAll(Goal):
    """
    Takes a list of nodes and executes them in parallel.

    Its observation turns True as soon as any node is True and turns False only when all
    nodes are False, i.e. it only fails if every node fails.
    """

    nodes: List[MotionStatechartNode] = field(default_factory=list, init=True)
    """
    The child nodes executed in parallel.
    """

    def expand(self, context: MotionStatechartContext) -> None:
        """
        Add all child nodes to this goal so they run in parallel.
        """
        self._check_has_children()
        for node in self.nodes:
            self.add_node(node)

    def build_artifacts(self, context: MotionStatechartContext) -> NodeArtifacts:
        """
        Build an observation that is True as soon as any child node reached its goal.

        This goal ends none of its children, so a child that keeps running is judged by
        what it observes now rather than by a verdict it never reaches.
        """
        return NodeArtifacts(
            observation=trinary_logic_or(*[node.goal_reached for node in self.nodes]),
        )


@dataclass(repr=False, eq=False)
class TryInOrder(Goal):
    """
    Tries a list of nodes one after another, short-circuiting on the first success.

    The next alternative only starts once the previous one has ended without reaching
    its goal, not merely while it is still short of it. Its observation turns True as
    soon as an alternative succeeds and False only once every one of them is over, so it
    stays unknown while any of them is still being tried.

    .. note:: Abandoning an alternative that stopped making progress extends the
        construct: RPL knows no alternative timing out, only one that gives up
        explicitly. See :attr:`give_up_after`.
    .. note:: corresponds to the RPL's TRY-IN-ORDER. (McDermott, Drew. A reactive plan language, 1991)
    """

    nodes: List[MotionStatechartNode] = field(default_factory=list, init=True)
    """
    The child nodes tried one after another, in order.
    """

    _alternatives: List[MotionStatechartNode] = field(default_factory=list, init=False)
    """
    The nodes that were passed in, captured before :meth:`expand` adds a progress
    monitor per alternative to :attr:`nodes` alongside them.
    """

    give_up_after: timedelta = field(default=timedelta(seconds=5), kw_only=True)
    """
    Simulated time an alternative may make no progress before it is abandoned and the
    next one is tried.
    """

    def expand(self, context: MotionStatechartContext) -> None:
        """
        Add the child nodes and wire them so each one starts only after the previous one
        ended without reaching its goal, short-circuiting on the first success.

        An alternative is ended once it reaches its goal or once it stops making
        progress; which of the two happened is decided by what the alternative observes
        as it ends, not here. An observation that is merely still false means the
        alternative has not arrived yet, and is no reason to abandon it.

        A progress monitor ends with the alternative it watches, so it does not go on
        measuring progress against a node that has already been decided.
        """
        self._check_has_children()
        self._alternatives = list(self.nodes)
        previous_node: Optional[MotionStatechartNode] = None
        for node in self._alternatives:
            self.add_node(node)
            if previous_node is not None:
                node.start_condition = previous_node.is_failed_or_interrupted
            still_progressing = StillProgressing(
                name=f"{self.name}/progress_of_{node.name}",
                monitored_node=node,
                timeout=self.give_up_after,
            )
            self.add_node(still_progressing)
            still_progressing.start_condition = node.is_running
            still_progressing.end_condition = node.is_terminated
            node.end_condition = trinary_logic_or(
                node.observation_variable,
                trinary_logic_not(still_progressing.observation_variable),
            )
            previous_node = node

    @staticmethod
    def _reached_its_goal(node: MotionStatechartNode) -> Scalar:
        """
        Whether an alternative is at its goal, answering false rather than unknown for
        one that was abandoned before it ever got there.

        :param node: The alternative to read.
        :return: What that alternative observes while it runs, and whether it reached
            its goal once it has ended.
        """
        return trinary_logic_and(
            node.goal_reached,
            trinary_logic_not(node.ended_without_reaching_its_goal),
        )

    def build_artifacts(self, context: MotionStatechartContext) -> NodeArtifacts:
        """
        Build an observation that is True as soon as any alternative succeeded, and
        False only once every one of them is over without one having succeeded.
        """
        return NodeArtifacts(
            observation=trinary_logic_or(
                *[self._reached_its_goal(node) for node in self._alternatives]
            ),
        )


# %% monitored subtrees


@dataclass(repr=False, eq=False)
class CancelledWhenTrue(StoppedWhenTrue):
    """
    Ends the monitored node as soon as the monitor observes True, and ends the motion
    with it.

    Nothing in a plan waits for a node that failed, so a monitor that gives up on its
    subtree has to end the motion rather than leave the rest of the plan waiting for a
    subtree that will never succeed.
    """

    exception: DataclassException = field(kw_only=True)
    """
    The failure reported once the monitor ends the motion.
    """

    def expand(self, context: MotionStatechartContext) -> None:
        """
        Add the monitor and the monitored node, and the node that ends the motion once
        the monitor observes True.
        """
        super().expand(context)
        cancelled = CancelMotion(
            name=f"{self.name}/cancelled", exception=self.exception
        )
        self.add_node(cancelled)
        cancelled.start_condition = self.monitor.observation_variable
