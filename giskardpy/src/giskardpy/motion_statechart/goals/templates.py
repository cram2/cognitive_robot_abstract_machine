from __future__ import division

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import timedelta
from typing import List

from typing_extensions import Optional

from giskardpy.motion_statechart.context import MotionStatechartContext
from giskardpy.motion_statechart.data_types import (
    LifeCycleValues,
    ObservationStateValues,
)
from giskardpy.motion_statechart.exceptions import NodeCannotDecideItselfError
from giskardpy.motion_statechart.graph_node import (
    CancelMotion,
    Goal,
    MaintenanceNode,
    MotionStatechartNode,
    NodeArtifacts,
    SelfDecidingNode,
    TerminalNode,
)
from giskardpy.motion_statechart.monitors.progress_monitors import Stalled
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

# %% giving a motion an outcome


@dataclass(repr=False, eq=False)
class Attempt(SelfDecidingNode, Goal):
    """
    Runs a motion that would never end on its own and decides it, one way or the other.

    A constraint holds itself against the world and observes only whether it is at its
    goal right now, so nothing about it ever concludes. This goal concludes instead: it
    observes True once the task is at its goal, which ends it as a success, and False
    once one of :attr:`failure_monitors` fires, which is what it declares its own
    failure on. That is what lets a maintained motion be one step of a plan.

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
            nothing at all unless this goal declared itself failed.
        """
        if self.life_cycle_state != LifeCycleValues.FAILED:
            return []
        return [
            monitor
            for monitor in self.failure_monitors
            if monitor.goal_reached_state == ObservationStateValues.TRUE
        ]

    def expand(self, context: MotionStatechartContext) -> None:
        """
        Add the task and the monitors, and declare this goal failed once it observes
        being given up on.

        A condition may only read its own node or a sibling, so this goal reaches its
        monitors through its own observation rather than through its children.

        A monitor succeeds on the control cycle it fires, which keeps having fired as
        its verdict. A goal reads its children a cycle late, so a monitor that fires
        only briefly would otherwise be indistinguishable afterwards from one that never
        fired at all.
        """
        self.add_node(self.task)
        self.add_nodes(self.failure_monitors)
        for failure_monitor in self.failure_monitors:
            failure_monitor.success_condition = trinary_logic_or(
                failure_monitor.success_condition, failure_monitor.observation_variable
            )
        self.fail_condition = trinary_logic_or(
            self.fail_condition, trinary_logic_not(self.observation_variable)
        )

    def build_artifacts(self, context: MotionStatechartContext) -> NodeArtifacts:
        """
        Report reaching the goal, being given up on, or neither.

        Reaching the goal outranks a monitor firing on the same control cycle: a task
        that arrived did what it was asked, whatever else was true at that moment. The
        task is read through :attr:`goal_reached`, which is what it observes for as long
        as this goal holds it open.
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


# %% goals built from nodes that end on their own


@dataclass(repr=False, eq=False)
class GoalOverSelfDecidingNodes(SelfDecidingNode, Goal, ABC):
    """
    Base for the goals that order or choose between children, which only works if each
    child reaches a terminal state by itself.

    Such a goal reads its children's verdicts and never their observations, and it owns
    their life cycles: what starts and ends a child is this goal's to decide. What comes
    out decides itself in turn, which is what lets one be a step of another.
    """

    def expand(self, context: MotionStatechartContext) -> None:
        """
        Wire the children, then declare this goal failed once it observes that it cannot
        arrive.

        A condition may only read its own node or a sibling, so this goal reaches its
        children through its own observation rather than through them.
        """
        self.expand_children(context)
        self.fail_condition = trinary_logic_or(
            self.fail_condition, trinary_logic_not(self.observation_variable)
        )

    @abstractmethod
    def expand_children(self, context: MotionStatechartContext) -> None:
        """
        Add the children and wire their life cycles.

        :param context: The build context passed to the expansion.
        """

    def _add_self_deciding(self, node: MotionStatechartNode) -> MotionStatechartNode:
        """
        Adds a child that ends on its own, converting the caller's node where it needs
        converting.

        A :class:`~giskardpy.motion_statechart.graph_node.MaintenanceNode` observes
        whether its constraints are satisfied, which is enough to decide that it reached
        its goal, so one is wrapped in an :class:`Attempt` that states no way of
        failing. Any other node has to declare that it decides itself.

        :param node: The child the caller passed.
        :return: The child to run in its place, which may be `node` itself.
        :raises NodeCannotDecideItselfError: If `node` never ends on its own and cannot
            be converted.
        """
        self._check_caller_wired_no_transitions(node)
        self._check_node_doesnt_belong_to_different_parent(node)
        if isinstance(node, SelfDecidingNode):
            self.add_node(node)
            return node
        if not isinstance(node, MaintenanceNode):
            raise NodeCannotDecideItselfError(node=self, child=node)
        attempt = Attempt(name=f"{node.name}/attempt", task=node, failure_monitors=[])
        # The attempt takes the node's place among the children and becomes its parent,
        # so the children stay in the order the caller wrote them in.
        if node in self.nodes:
            self.nodes[self.nodes.index(node)] = attempt
        self.add_node(attempt)
        node.parent_node = attempt
        return attempt


@dataclass(repr=False, eq=False)
class Sequence(GoalOverSelfDecidingNodes):
    """
    Runs a list of nodes one after another.

    Its observation turns True once the last step reached its goal, and False as soon as
    a step ended short of its own, so a step that was given up on fails the sequence
    rather than leaving it waiting forever.

    .. note:: corresponds to the RPL's SEQ. (McDermott, Drew. A reactive plan language, 1991)
    """

    nodes: List[MotionStatechartNode] = field(default_factory=list, init=True)
    """
    The child nodes run one after another, in order.
    """

    _steps: List[MotionStatechartNode] = field(default_factory=list, init=False)
    """
    The nodes actually run, which is what the caller passed with every plain task
    wrapped in an attempt.
    """

    def expand_children(self, context: MotionStatechartContext) -> None:
        """
        Each step is a node that ends on its own, and the next one waits for the verdict
        it earned, because only a verdict outlasts the step that reached it.
        """
        self._check_has_children()
        previous_step: Optional[MotionStatechartNode] = None
        for node in list(self.nodes):
            # A node that ends the motion decides nothing and has nothing to convert.
            if isinstance(node, TerminalNode):
                self.add_node(node)
                step = node
            else:
                step = self._add_self_deciding(node)
            if previous_step is not None:
                step.start_condition = previous_step.is_succeeded
            self._steps.append(step)
            previous_step = step

    def build_artifacts(self, context: MotionStatechartContext) -> NodeArtifacts:
        """
        Report success, a failed step, or neither, all read off the steps' verdicts.

        A step short of its goal has not failed, it has not arrived yet, so only a step
        that ended without reaching its goal decides anything.
        """
        return NodeArtifacts(
            observation=if_cases(
                cases=[
                    (
                        trinary_logic_or(
                            *[
                                step.ended_without_reaching_its_goal
                                for step in self._steps
                            ]
                        ),
                        Scalar.const_false(),
                    ),
                    (self._steps[-1].ended_at_its_goal, Scalar.const_true()),
                ],
                else_result=Scalar.const_trinary_unknown(),
            )
        )


@dataclass(repr=False, eq=False)
class Parallel(MaintenanceNode, Goal):
    """
    Holds a list of nodes at once until enough of them are at their goals together.

    Its observation turns True once at least :attr:`minimum_success` of them are at
    their goals on the same control cycle.

    Unlike the goals that run steps, this one ends none of its nodes and reads what they
    observe now, because releasing a constraint that reached its goal would let a
    sibling drag the robot back out of it. That also makes it a maintenance node itself:
    a plan step built from one is an attempt wrapping it.
    """

    nodes: List[MotionStatechartNode] = field(default_factory=list, init=True)
    """
    The child nodes held at once.
    """

    minimum_success: Optional[int] = field(default=None, kw_only=True)
    """
    How many nodes must have reached their goals for this goal to be achieved.

    Defaults to None, which means all of them.
    """

    @property
    def required_successes(self) -> int:
        """
        :return: How many nodes have to reach their goals, which is all of them unless
            :attr:`minimum_success` says otherwise.
        """
        if self.minimum_success is None:
            return len(self.nodes)
        return self.minimum_success

    def expand(self, context: MotionStatechartContext) -> None:
        self._check_has_children()
        for node in self.nodes:
            self.add_node(node)

    def build_artifacts(self, context: MotionStatechartContext) -> NodeArtifacts:
        """
        Count the nodes that reached their goals against :attr:`required_successes`.

        This goal ends none of its nodes, so a node that keeps running is counted by
        what it observes now and stops counting once it drifts away from its goal again.
        A node something *else* ended keeps counting, because its verdict outlasts it.

        Observing False means the constraints are not satisfied, not that anything went
        wrong: whether that is worth giving up on is decided outside, by the attempt this
        goal is wrapped in.
        """
        nodes_at_their_goal = [node.goal_reached.is_true() for node in self.nodes]
        return NodeArtifacts(
            observation=self.required_successes <= sum(*nodes_at_their_goal)
        )


# %% repeating a task


@dataclass(repr=False, eq=False)
class RepeatUntil(GoalOverSelfDecidingNodes):
    """
    Runs a task again from the start whenever an attempt at it fails.

    Its observation turns True once the task succeeds and False once
    :attr:`stop_retry_monitor` calls the retrying off, so a caller can tell "eventually
    worked" from "gave up".

    What counts as a failed attempt is stated on the task itself: hand it an
    :class:`Attempt` carrying the failure monitors that decide it, or see
    :class:`RepeatOnStall`, which derives that decision from the task's own progress. A
    task that never ends on its own is attempted with no way of failing, so it is never
    retried and ends only by succeeding or once :attr:`stop_retry_monitor` fires.
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

    _attempt: Optional[MotionStatechartNode] = field(default=None, init=False)
    """
    The node actually run, which is :attr:`task` wrapped in an attempt if it needed one.
    """

    def expand_children(self, context: MotionStatechartContext) -> None:
        """
        Wire the retry loop.

        The attempt declares its own failure, and that verdict is what starts the next
        try: a node reading its own life cycle reads the state it entered the control
        cycle with, so the reset lands the cycle after the failure rather than on it.

        The stop monitor is read through its verdict, which outlasts a monitor that ends
        itself on reaching what it counts.
        """
        self._attempt = self._add_self_deciding(self.task)
        self.add_node(self.stop_retry_monitor)

        retrying_stopped = Scalar(self.stop_retry_monitor.goal_reached)
        still_trying = trinary_logic_not(retrying_stopped)
        # Starting is gated as well as ending, because a reset task is not started and
        # ending is not considered while it is not.
        self._attempt.start_condition = still_trying
        self._attempt.reset_condition = trinary_logic_and(
            self._attempt.is_failed, still_trying
        )
        self._attempt.interrupt_condition = retrying_stopped

    def build_artifacts(self, context: MotionStatechartContext) -> NodeArtifacts:
        """
        Report success, giving up, or neither.

        Both children are read through their verdicts, which outlast them; the attempt's
        is cleared again by the reset that starts the next try.
        """
        return NodeArtifacts(
            observation=if_cases(
                cases=[
                    (self._attempt.ended_at_its_goal, Scalar.const_true()),
                    (
                        self.stop_retry_monitor.goal_reached.is_true(),
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
        """
        Turn the task into an attempt that gives up on a stall, which is the decision
        this subclass exists to make for the caller.

        A task that already is an :class:`Attempt` keeps its own failure monitors and
        gives up on a stall as well.
        """
        super().__post_init__()
        if isinstance(self.task, Attempt):
            self.task.failure_monitors.append(
                self._create_stall_monitor(self.task.task)
            )
            return
        self.task = Attempt(
            name=f"{self.name}/attempt",
            task=self.task,
            failure_monitors=[self._create_stall_monitor(self.task)],
        )

    def _create_stall_monitor(self, monitored_node: MotionStatechartNode) -> Stalled:
        """
        :param monitored_node: The node whose progress is measured.
        :return: A monitor that fires once nothing under `monitored_node` has approached
            its goal for :attr:`timeout`.
        """
        return Stalled(
            name=f"{self.name}/progress",
            monitored_node=monitored_node,
            timeout=self.timeout,
            minimum_convergence_rate=self.minimum_convergence_rate,
        )


# %% trying alternatives


@dataclass(repr=False, eq=False)
class TryAll(GoalOverSelfDecidingNodes):
    """
    Runs a list of alternatives at once and takes the first one that works.

    Its observation turns True as soon as an alternative reached its goal, and False
    only once every one of them ended without doing so.
    """

    nodes: List[MotionStatechartNode] = field(default_factory=list, init=True)
    """
    The child nodes tried at once.
    """

    _alternatives: List[MotionStatechartNode] = field(default_factory=list, init=False)
    """
    The nodes actually run, which is what the caller passed with every plain task
    wrapped in an attempt.
    """

    def expand_children(self, context: MotionStatechartContext) -> None:
        """
        Add every alternative, so they run side by side.
        """
        self._check_has_children()
        self._alternatives = [
            self._add_self_deciding(node) for node in list(self.nodes)
        ]

    def build_artifacts(self, context: MotionStatechartContext) -> NodeArtifacts:
        """
        Report the first alternative that worked, or that none of them did.
        """
        return NodeArtifacts(
            observation=if_cases(
                cases=[
                    (
                        trinary_logic_or(
                            *[
                                alternative.ended_at_its_goal
                                for alternative in self._alternatives
                            ]
                        ),
                        Scalar.const_true(),
                    ),
                    (
                        trinary_logic_and(
                            *[
                                alternative.ended_without_reaching_its_goal
                                for alternative in self._alternatives
                            ]
                        ),
                        Scalar.const_false(),
                    ),
                ],
                else_result=Scalar.const_trinary_unknown(),
            )
        )


@dataclass(repr=False, eq=False)
class TryInOrder(GoalOverSelfDecidingNodes):
    """
    Tries a list of alternatives one after another, short-circuiting on the first
    success.

    The next alternative only starts once the previous one has ended without reaching
    its goal. Its observation turns True as soon as an alternative succeeds and False
    only once every one of them is over, so it stays unknown while any is still running.

    Each alternative decides for itself when to give up, which is why this goal reduces
    to ordering: wrap one in an :class:`Attempt` carrying the monitors that decide it.

    .. note:: corresponds to the RPL's TRY-IN-ORDER. (McDermott, Drew. A reactive plan language, 1991)
    """

    nodes: List[MotionStatechartNode] = field(default_factory=list, init=True)
    """
    The child nodes tried one after another, in order.
    """

    _alternatives: List[MotionStatechartNode] = field(default_factory=list, init=False)
    """
    The nodes actually run, which is what the caller passed with every plain task
    wrapped in an attempt.
    """

    def expand_children(self, context: MotionStatechartContext) -> None:
        """
        Wire each alternative to start once the previous one ended short of its goal,
        which short-circuits on the first success.
        """
        self._check_has_children()
        previous_alternative: Optional[MotionStatechartNode] = None
        for node in list(self.nodes):
            alternative = self._add_self_deciding(node)
            if previous_alternative is not None:
                alternative.start_condition = (
                    previous_alternative.is_failed_or_interrupted
                )
            self._alternatives.append(alternative)
            previous_alternative = alternative

    def build_artifacts(self, context: MotionStatechartContext) -> NodeArtifacts:
        """
        Report the alternative that worked, or that none of them did.
        """
        return NodeArtifacts(
            observation=if_cases(
                cases=[
                    (
                        trinary_logic_or(
                            *[
                                alternative.ended_at_its_goal
                                for alternative in self._alternatives
                            ]
                        ),
                        Scalar.const_true(),
                    ),
                    (
                        trinary_logic_and(
                            *[
                                alternative.ended_without_reaching_its_goal
                                for alternative in self._alternatives
                            ]
                        ),
                        Scalar.const_false(),
                    ),
                ],
                else_result=Scalar.const_trinary_unknown(),
            )
        )


# %% monitored subtrees


@dataclass(repr=False, eq=False)
class CancelledWhenTrue(StoppedWhenTrue):
    """
    Interrupts the monitored node as soon as the monitor observes True, and ends the
    motion with it.

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
