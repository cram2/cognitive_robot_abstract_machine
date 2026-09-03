# used for delayed evaluation of typing until python 3.11 becomes mainstream
from __future__ import annotations

import logging
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing_extensions import (
    Any,
    Callable,
    List,
    Optional,
    Type,
)

from giskardpy.motion_statechart.goals.templates import (
    Parallel,
    RepeatOnStall,
    RepeatUntil,
    Sequence,
    TryAll,
    TryInOrder,
    CancelledWhenTrue,
)
from giskardpy.motion_statechart.graph_node import (
    CancelMotion,
    Goal,
    MotionStatechartNode,
)
from giskardpy.motion_statechart.monitors.payload_monitors import CountNodeResets
from giskardpy.motion_statechart.monitors.templates import (
    MonitoredGoal,
    PausedUntilTrue,
    PausedWhileTrue,
)
from coraplex.plans.executables import (
    GiskardExecutable,
    Executable,
)
from coraplex.datastructures.enums import TaskStatus
from coraplex.plans.failures import (
    AllChildrenFailed,
    PlanCancelled,
    PlanFailure,
    RepetitionsExhausted,
)
from coraplex.plans.motion_state_chart_building import BuildsMotionStateChart
from coraplex.plans.plan_node import PlanNode

logger = logging.getLogger(__name__)


@dataclass(eq=False)
class LanguageNode(PlanNode, BuildsMotionStateChart, ABC):
    """
    Base class for language nodes in a plan.

    Language nodes are nodes that are not directly executable, but manage the execution
    of their children in a certain way.
    """

    motion_state_chart_template: Type[Goal] = field(kw_only=True, default=Sequence)
    """
    Giskard template which this language expression translates to.
    """

    def simplify(self):
        for child in self.children:
            if type(child) != type(self):
                continue

            self.merge(child)

    def notify(self):
        for child in self.children:
            child.notify()

    def parse(self) -> Executable:
        return self.parse_children(self.children)

    def create_goal(self) -> Goal:
        """
        :return: An empty goal of this node's template, describing how its children are
            executed inside a motion state chart.
        """
        return self.motion_state_chart_template(name=type(self).__name__)

    def add_to_motion_state_chart(
        self, parent_goal: Goal, executable: GiskardExecutable
    ) -> Goal:
        """
        Add this node as its own goal below `parent_goal` and add every child that
        contributes motions into it, one at a time.
        """
        goal = self.create_goal()
        parent_goal.add_node(goal)
        self.add_children_to_motion_state_chart(goal, self.children, executable)
        return goal


@dataclass
class ExecutesSequentially(LanguageNode):
    """
    Base class for nodes that execute their children sequentially.
    """


@dataclass
class ExecutesInParallel(LanguageNode, ABC):
    """
    Base class for nodes that execute their children in parallel.
    """

    @classmethod
    def _perform_parallel(cls, nodes: List[PlanNode]):
        """
        Open threads for all nodes and wait for them to finish.

        :param nodes: A list of nodes which should be performed in parallel
        """
        threads = []
        for child in nodes:
            thread = threading.Thread(
                target=child.perform,
            )
            thread.start()
            threads.append(thread)

        for thread in threads:
            thread.join()


@dataclass
class SequentialNode(ExecutesSequentially):
    """
    Executes all children sequentially.

    Any failure is immediately raised.
    """

    motion_state_chart_template: Type[Goal] = field(kw_only=True, default=Sequence)


@dataclass
class ParallelNode(ExecutesInParallel):
    """
    Executes all children in parallel by creating a thread per children and executing
    them in the respective thread.

    All exceptions are raised after all children have finished.
    """

    motion_state_chart_template: Type[Goal] = field(kw_only=True, default=Parallel)

    def notify(self):
        self._perform_parallel(self.children)
        for child in self.children:
            if child.status == TaskStatus.FAILED:
                raise child.reason


@dataclass(eq=False)
class RepeatNode(ExecutesSequentially):
    """
    Executes all children, and executes them again whenever an attempt fails.

    Attempting stops as soon as the children succeed. Running out of attempts is a
    failure, raising :class:`~coraplex.plans.failures.RepetitionsExhausted`.

    .. note:: The default template treats an attempt as failed once it stops making
        progress, which needs at least one converging task among the children.
    """

    maximum_repetitions: int = 1
    """
    How many times the children are attempted before the repeating is given up on.
    """

    repeat_template: Callable[..., RepeatUntil] = field(
        kw_only=True, default=RepeatOnStall
    )
    """
    Builds the giskard goal deciding what counts as a failed attempt.

    Use it for a decision derived from the children, such as a stall. It is called with
    the children's goal, the attempt counter and :attr:`failure_monitor`, so a template
    needing more configuration is passed pre-configured, for instance
    ``partial(RepeatOnStall, timeout=timedelta(seconds=1))``.
    """

    failure_monitor: Optional[MotionStatechartNode] = field(default=None, kw_only=True)
    """
    Node whose True observation means an attempt failed.

    Use it for a decision that stands on its own, such as a force spike, together with
    :class:`~giskardpy.motion_statechart.goals.templates.RepeatUntil` as the template.
    Templates that derive their own reject it.
    """

    def parse(self) -> Executable:
        return self.create_giskard_executable([self])

    def add_to_motion_state_chart(
        self, parent_goal: Goal, executable: GiskardExecutable
    ) -> Goal:
        """
        Add a goal below `parent_goal` that runs this node's children over and over.

        The counter, the children's goal and the node that reports running out of
        attempts all become children of that goal, because a transition condition may
        only name a sibling.
        """
        children_goal = self.create_goal()
        counter = CountNodeResets(
            name=f"{type(self).__name__}/attempts",
            node=children_goal,
            target=self.maximum_repetitions,
        )
        loop = self.repeat_template(
            name=type(self).__name__,
            task=children_goal,
            stop_retry_monitor=counter,
        )
        parent_goal.add_node(loop)
        loop.add_node(children_goal)
        self.add_children_to_motion_state_chart(
            children_goal, self.children, executable
        )

        exhausted = CancelMotion(
            name=f"{type(self).__name__}/exhausted",
            exception=RepetitionsExhausted(
                language_node=self, maximum_repetitions=self.maximum_repetitions
            ),
        )
        exhausted.start_condition = counter.observation_variable
        loop.add_node(exhausted)
        return loop


@dataclass(eq=False)
class TryInOrderNode(ExecutesSequentially):
    """
    Tries all children in order sequentially and fails if all children fail.
    """

    motion_state_chart_template: Type[Goal] = field(kw_only=True, default=TryInOrder)

    def notify(self):
        for child in self.children:
            try:
                child.perform()
            except PlanFailure:
                continue
        failed = all([child.status == TaskStatus.FAILED for child in self.children])
        if failed:
            raise AllChildrenFailed(self)


@dataclass(eq=False)
class TryAllNode(ExecutesInParallel):
    """
    Executes all children in parallel.

    Only raise a failure if all children fail.
    """

    motion_state_chart_template: Type[Goal] = field(kw_only=True, default=TryAll)

    def notify(self):
        self._perform_parallel(self.children)
        failed = all([child.status == TaskStatus.FAILED for child in self.children])
        if failed:
            raise AllChildrenFailed(self)


@dataclass(eq=False)
class MonitorNode(LanguageNode, ABC):
    """
    Executes its children under the observation of a motion state chart monitor.

    The monitor is evaluated by the control loop alongside the children, so it can act
    on them while they are running.
    """

    monitor: MotionStatechartNode = field(kw_only=True)
    """
    The node whose observation controls this node's children.
    """

    def parse(self) -> Executable:
        return self.create_giskard_executable([self])

    def add_to_motion_state_chart(
        self, parent_goal: Goal, executable: GiskardExecutable
    ) -> Goal:
        """
        Add a goal below `parent_goal` that runs this node's children next to its
        monitor.

        The monitor and the children's goal become siblings, which is what lets the
        monitor's observation drive the children's life cycle.
        """
        monitored_goal = self.create_monitored_goal()
        parent_goal.add_node(monitored_goal)
        monitored_goal.add_node(self.monitor)
        children_goal = self.create_goal()
        monitored_goal.add_node(children_goal)
        monitored_goal.monitored_node = children_goal
        self.add_children_to_motion_state_chart(
            children_goal, self.children, executable
        )
        return monitored_goal

    @abstractmethod
    def create_monitored_goal(self) -> MonitoredGoal:
        """
        :return: An empty goal that runs this node's children under its monitor.
        """


@dataclass(eq=False)
class CancelMonitor(MonitorNode):
    """
    Cancels the plan once the monitor observes True.

    Its children are stopped and the whole motion ends, raising
    :class:`~coraplex.plans.failures.PlanCancelled`, because the state the rest of the
    plan assumed no longer holds.
    """

    def create_monitored_goal(self) -> MonitoredGoal:
        return CancelledWhenTrue(
            monitor=self.monitor,
            name=type(self).__name__,
            exception=PlanCancelled(language_node=self),
        )


@dataclass(eq=False)
class PauseMonitor(MonitorNode):
    """
    Holds its children for as long as the monitor observes True.

    .. warning:: A monitor that never turns False again holds the children forever, so the
        motion runs out of control cycles and fails with
        :class:`~coraplex.exceptions.MotionDidNotFinish`. Use :class:`CancelMonitor` to
        give up on the plan instead.
    """

    def create_monitored_goal(self) -> MonitoredGoal:
        return PausedWhileTrue(monitor=self.monitor, name=type(self).__name__)


@dataclass(eq=False)
class PauseUntilMonitor(MonitorNode):
    """
    Holds its children until the monitor observes True.

    .. warning:: A monitor that never turns True holds the children forever, so the
        motion runs out of control cycles and fails with
        :class:`~coraplex.exceptions.MotionDidNotFinish`. Use :class:`CancelMonitor` to
        give up on the plan instead.
    """

    def create_monitored_goal(self) -> MonitoredGoal:
        return PausedUntilTrue(monitor=self.monitor, name=type(self).__name__)


@dataclass
class CodeNode(LanguageNode):
    """
    Executable function in a plan.

    This class' primary purpose is for debugging and testing.
    """

    code: Callable = field(default_factory=lambda: lambda: None, kw_only=True)

    def notify(self) -> Any:
        return self.code()
