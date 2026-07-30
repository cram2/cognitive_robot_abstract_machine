# used for delayed evaluation of typing until python 3.11 becomes mainstream
from __future__ import annotations

import atexit
import logging
import threading
import time
from abc import ABC
from dataclasses import dataclass, field
from typing_extensions import (
    Optional,
    Callable,
    Any,
    List,
    Union,
    Type,
)

from giskardpy.motion_statechart.goals.templates import Sequence, Parallel
from giskardpy.motion_statechart.graph_node import Goal
from coraplex.language_giskard_templates import TryAll, TryInOrder
from coraplex.plans.executables import (
    GiskardExecutable,
    Executable,
    ModelChangeExecutable,
)
from coraplex.datastructures.enums import TaskStatus, MonitorBehavior
from coraplex.plans.attachment_nodes import ModelChangeNode
from coraplex.plans.failures import PlanFailure, AllChildrenFailed
from coraplex.fluent import Fluent
from coraplex.plans.plan_node import (
    PlanNode,
    UnderspecifiedNode,
)
from coraplex.utils import split_list_by_type

logger = logging.getLogger(__name__)


@dataclass(eq=False)
class LanguageNode(PlanNode, ABC):
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
        # Nodes that do not parse into a single motion chart (model changes, and
        # underspecified nodes that are only grounded during execution) split the
        # plan into sequential execution groups instead of one merged chart.
        if any(
            isinstance(child, (ModelChangeNode, UnderspecifiedNode))
            for child in self.descendants
        ):
            return self.parse_with_non_giskard_executable()
        child_execs = [child.parse() for child in self.children]

        return GiskardExecutable(
            motion_mappings=self.merge_motion_mappings(child_execs),
            context=self.plan.context,
        )

    def parse_with_non_giskard_executable(self) -> Executable:
        """
        Build an executable whose execution list keeps non-giskard executables (model
        changes, deferred underspecified nodes) as sequential boundaries, merging only
        the consecutive giskard executables between them.
        """
        child_executables = [node.parse() for node in self.children]

        giskard_exec_groups = split_list_by_type(child_executables, GiskardExecutable)

        exec_list = []

        for group in giskard_exec_groups:
            if isinstance(group[0], GiskardExecutable):
                exec_list.append(
                    GiskardExecutable(
                        motion_mappings=self.merge_motion_mappings(group),
                        context=self.plan.context,
                    )
                )
            else:
                exec_list.extend(group)

        return Executable(execution_list=exec_list, context=self.plan.context)


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

    motion_state_chart_template = Sequence


@dataclass
class ParallelNode(ExecutesInParallel):
    """
    Executes all children in parallel by creating a thread per children and executing
    them in the respective thread.

    All exceptions are raised after all children have finished.
    """

    motion_state_chart_template = Parallel

    def notify(self):
        self._perform_parallel(self.children)
        for child in self.children:
            if child.status == TaskStatus.FAILED:
                raise child.reason


@dataclass(eq=False)
class RepeatNode(ExecutesSequentially):
    """
    Executes all children a given number of times in sequential order.
    """

    repetitions: int = 1
    """
    The number of repetitions of the children.
    """

    def notify(self):
        for _ in range(self.repetitions):
            super().notify()


@dataclass(eq=False)
class MonitorNode(ExecutesSequentially):
    """
    Monitors a Language Expression and interrupts it when the given condition is
    evaluated to True.

    Behaviour:
        Monitors start a new Thread which checks the condition while performing the nodes below it. Monitors can have
        different behaviors, they can Interrupt, Pause or Resume the execution of the children.
        If the behavior is set to Resume the plan will be paused until the condition is met.
    """

    condition: Union[Callable, Fluent] = field(kw_only=True)
    """
    The condition to monitor.
    """

    behavior: MonitorBehavior = field(kw_only=True, default=MonitorBehavior.INTERRUPT)
    """
    What to do on the condition.
    """

    _monitor_thread: Optional[threading.Thread] = field(init=False, default=None)
    """
    Thread for the subplan that is monitored.
    """

    kill_event: threading.Event = field(init=False, default_factory=threading.Event)
    """
    Event used to stop the monitoring thread once the children have finished.
    """

    def __post_init__(self):
        if self.behavior == MonitorBehavior.RESUME:
            self.pause()
        if callable(self.condition):
            self.condition = Fluent(self.condition)

        self._monitor_thread = threading.Thread(
            target=self.monitor, name=f"MonitorThread-{id(self)}"
        )
        self._monitor_thread.start()

    def notify(self):
        super().notify()
        self.kill_event.set()
        self._monitor_thread.join()

    def monitor(self):
        atexit.register(self.kill_event.set)
        while not self.kill_event.is_set():
            if self.condition.get_value():
                if self.behavior == MonitorBehavior.INTERRUPT:
                    self.interrupt()
                    self.kill_event.set()
                elif self.behavior == MonitorBehavior.PAUSE:
                    self.pause()
                    self.kill_event.set()
                elif self.behavior == MonitorBehavior.RESUME:
                    self.resume()
                    self.kill_event.set()
            time.sleep(0.1)


@dataclass(eq=False)
class TryInOrderNode(ExecutesSequentially):
    """
    Tries all children in order sequentially and fails if all children fail.
    """

    motion_state_chart_template = TryInOrder

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

    motion_state_chart_template = TryAll

    def notify(self):
        self._perform_parallel(self.children)
        failed = all([child.status == TaskStatus.FAILED for child in self.children])
        if failed:
            raise AllChildrenFailed(self)


@dataclass
class CodeNode(LanguageNode):
    """
    Executable function in a plan.

    This class' primary purpose is for debugging and testing.
    """

    code: Callable = field(default_factory=lambda: lambda: None, kw_only=True)

    def notify(self) -> Any:
        return self.code()
