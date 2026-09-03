from __future__ import division

from dataclasses import dataclass, field
from datetime import timedelta
from typing import List, Optional

from typing_extensions import Optional

import krrood.symbolic_math.symbolic_math as sm
from giskardpy.motion_statechart.context import MotionStatechartContext
from giskardpy.motion_statechart.graph_node import (
    Goal,
    MotionStatechartNode,
    NodeArtifacts,
    TerminalNode, CancelMotion,
)
from giskardpy.motion_statechart.monitors.progress_monitors import ProgressStalled
from giskardpy.motion_statechart.monitors.templates import StoppedWhenTrue
from krrood.exceptions import DataclassException
from krrood.symbolic_math.symbolic_math import (
    sum,
    trinary_logic_and,
    trinary_logic_not,
    trinary_logic_or,
)


@dataclass(repr=False, eq=False)
class Sequence(Goal):
    """
    Takes a list of nodes and wires their start/end conditions such that they are
    executed in order.

    Its observation is the observation of the last node in the sequence.
    """

    nodes: List[MotionStatechartNode] = field(default_factory=list, init=True)

    def expand(self, context: MotionStatechartContext) -> None:
        last_node: Optional[MotionStatechartNode] = None
        for i, node in enumerate(self.nodes):
            self.add_node(node)
            if last_node is not None:
                node.start_condition = last_node.observation_variable
            # A node that ends the motion has nothing left to transition to.
            if not isinstance(node, TerminalNode):
                node.end_condition = node.observation_variable
            last_node = node

    def build_artifacts(self, context: MotionStatechartContext) -> NodeArtifacts:
        return NodeArtifacts(observation=self.nodes[-1].observation_variable)


@dataclass(repr=False, eq=False)
class Parallel(Goal):
    """
    Takes a list of nodes and executes them in parallel.

    This nodes' observation state turns True when up to `minimum_success` nodes are
    True.
    """

    nodes: List[MotionStatechartNode] = field(default_factory=list, init=True)
    minimum_success: Optional[int] = field(default=None, kw_only=True)
    """
    Defines the minimum number of nodes that must be True for the goal to be achieved.

    Defaults to None, which means that all nodes must be True.
    """

    def expand(self, context: MotionStatechartContext) -> None:
        for node in self.nodes:
            self.add_node(node)

    def build_artifacts(self, context: MotionStatechartContext) -> NodeArtifacts:
        true_observation_variables = [
            x.observation_variable == True for x in self.nodes
        ]
        minimum_success = (
            self.minimum_success
            if self.minimum_success is not None
            else len(self.nodes)
        )
        return NodeArtifacts(
            observation=minimum_success <= sum(*true_observation_variables)
        )


# %% repeating a task


@dataclass(repr=False, eq=False)
class RepeatUntil(Goal):
    """
    Runs a task again from the start whenever an attempt at it fails.

    Its observation turns True once the task succeeds and False once :attr:`monitor`
    calls the retrying off, so a caller can tell "eventually worked" from "gave up".

    Hand it a :attr:`failure_monitor` to decide what a failed attempt is, or subclass it
    and override :meth:`create_failure_monitor` when that decision has to be derived
    from the task itself.
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

    retry_trigger_monitor: MotionStatechartNode = field(kw_only=True)
    """
    Decides that an attempt failed and the task should run again.

    Subclasses that derive one from the task leave this unset and override
    :meth:`create_failure_monitor`; it holds the node in use once this goal was expanded.
    """

    def expand(self, context: MotionStatechartContext) -> None:
        """
        Wire the retry loop.

        The failure monitor resets the task and itself with the same observation. A node
        that has been reset observes Unknown, so the reset lasts a single control cycle
        instead of holding the task at the start line, and the monitor is armed again
        for the next attempt.
        """
        self.add_nodes([self.task, self.retry_trigger_monitor, self.stop_retry_monitor])

        # Each observation is compared against True, so that an undecided Unknown counts
        # as neither, and the results combine as plain booleans.
        attempt_succeeded = self.task.observation_variable.is_true()

        attempt_failed = self.retry_trigger_monitor.observation_variable.is_true()

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
        self.retry_trigger_monitor.reset_condition = trinary_logic_and(
            attempt_failed, still_trying
        )
        self.retry_trigger_monitor.end_condition = (
            self.stop_retry_monitor.observation_variable
        )

    def build_artifacts(self, context: MotionStatechartContext) -> NodeArtifacts:
        """
        Report success, giving up, or neither.

        The cases are compared against True rather than combined with trinary logic,
        because an undecided attempt is Unknown and would otherwise read as a success.
        """
        return NodeArtifacts(
            observation=sm.if_cases(
                cases=[
                    (
                        self.task.observation_variable.is_true(),
                        sm.Scalar.const_true(),
                    ),
                    (
                        self.stop_retry_monitor.observation_variable.is_true(),
                        sm.Scalar.const_false(),
                    ),
                ],
                else_result=sm.Scalar.const_trinary_unknown(),
            )
        )


@dataclass(repr=False, eq=False)
class RepeatOnStall(RepeatUntil):
    """
    Runs a task again from the start whenever it stops approaching its goal.

    .. note:: The task must contain at least one
        :class:`~giskardpy.motion_statechart.graph_node.ConvergingTask`, because progress
        is measured from a task's error. Watching anything else raises
        :class:`~giskardpy.motion_statechart.exceptions.NoConvergingTaskError` while the
        motion statechart is compiled.
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

    retry_trigger_monitor: MotionStatechartNode = field(init=False)
    """
    Monitors that triggers a retry of the observed node.
    """

    def __post_init__(self):
        super().__post_init__()
        self.retry_trigger_monitor = ProgressStalled(
            name=f"{self.name}/stalled",
            monitored_node=self.task,
            timeout=self.timeout.total_seconds(),
            minimum_convergence_rate=self.minimum_convergence_rate,
        )


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
        for node in self.nodes:
            self.add_node(node)

    def build_artifacts(self, context: MotionStatechartContext) -> NodeArtifacts:
        """
        Build an observation that is True as soon as any child node is True.
        """
        observations = [node.observation_variable for node in self.nodes]
        observation = (
            observations[0]
            if len(observations) == 1
            else trinary_logic_or(*observations)
        )
        return NodeArtifacts(observation=observation)


@dataclass(repr=False, eq=False)
class TryInOrder(Goal):
    """
    Takes a list of nodes and tries them one after another, short-circuiting on the
    first success.

    The next node only starts once the previous node failed; as soon as a node succeeds
    the remaining nodes are skipped. Its observation turns True if any node is True and
    turns False only when all nodes are False, i.e. it only fails if every node fails.
    """

    nodes: List[MotionStatechartNode] = field(default_factory=list, init=True)
    """
    The child nodes tried one after another, in order.
    """

    def expand(self, context: MotionStatechartContext) -> None:
        """
        Add the child nodes and wire them so each one starts only after the previous one
        failed, short-circuiting on the first success.
        """
        last_node: Optional[MotionStatechartNode] = None
        for node in self.nodes:
            self.add_node(node)
            if last_node is not None:
                # Start the next node only if the previous one failed (short-circuit on success).
                node.start_condition = trinary_logic_not(last_node.observation_variable)
            # End this node as soon as it is decided (True or False) so the chain can advance/finish.
            node.end_condition = trinary_logic_or(
                node.observation_variable, trinary_logic_not(node.observation_variable)
            )
            last_node = node

    def build_artifacts(self, context: MotionStatechartContext) -> NodeArtifacts:
        """
        Build an observation that is True as soon as any child node is True.
        """
        observations = [node.observation_variable for node in self.nodes]
        observation = (
            observations[0]
            if len(observations) == 1
            else trinary_logic_or(*observations)
        )
        return NodeArtifacts(observation=observation)


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
