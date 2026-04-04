# used for delayed evaluation of typing until python 3.11 becomes mainstream
from __future__ import annotations

import atexit
import logging
import threading
import time
from abc import ABC
from dataclasses import dataclass, field
from queue import Queue

from typing_extensions import (
    Optional,
    Callable,
    Any,
    List,
    Union,
)

from pycram.datastructures.enums import TaskStatus, MonitorBehavior
from pycram.plans.failures import PlanFailure, AllChildrenFailed
from pycram.fluent import Fluent
from pycram.plans.plan_node import (
    PlanNode,
)

logger = logging.getLogger(__name__)


@dataclass(eq=False)
class LanguageNode(PlanNode, ABC):
    """
    Base class for language nodes in a plan.
    Language nodes are nodes that are not directly executable, but manage the execution of their children in a certain
    way.
    """

    def simplify(self):
        for child in self.children:
            if type(child) != type(self):
                continue

            for grand_child in child.children:
                self.plan.add_edge(
                    self, grand_child, child.layer_index + grand_child.layer_index
                )
            self.plan.plan_graph.remove_edge(self.index, child.index)
            self.plan.remove_node(child)


@dataclass
class ExecutesSequentially(LanguageNode):
    """
    Base class for nodes that execute their children sequentially.
    """

    def _perform(self):
        result = [child.perform() for child in self.children]
        return result


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
            t = threading.Thread(
                target=child.perform,
            )
            t.start()
            threads.append(t)

        for thread in threads:
            thread.join()


@dataclass
class SequentialNode(ExecutesSequentially):
    """
    Executes all children sequentially. Any failure is immediately raised.
    """


@dataclass
class ParallelNode(ExecutesInParallel):
    """
    Executes all children in parallel by creating a thread per children and executing them in the respective thread.
    All exceptions are raised after all children have finished.
    """

    def _perform(self):
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

    def _perform(self):
        for _ in range(self.repetitions):
            super()._perform()


@dataclass(eq=False)
class MonitorNode(ExecutesSequentially):
    """
    Monitors a Language Expression and interrupts it when the given condition is evaluated to True.

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

    def __post_init__(self):
        self.kill_event = threading.Event()
        self.exception_queue = Queue()
        if self.behavior == MonitorBehavior.RESUME:
            self.pause()
        if callable(self.condition):
            self.condition = Fluent(self.condition)

        self._monitor_thread = threading.Thread(
            target=self.monitor, name=f"MonitorThread-{id(self)}"
        )
        self._monitor_thread.start()

    def _perform(self):
        super()._perform()
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

    def _perform(self):
        for child in self.children:
            try:
                child.perform()
            except PlanFailure as e:
                pass
        failed = all([child.status == TaskStatus.FAILED for child in self.children])
        if failed:
            raise AllChildrenFailed(self)


@dataclass(eq=False)
class TryAllNode(ExecutesInParallel):
    """
    Executes all children in parallel.
    Only raise a failure if all children fail.
    """

    def _perform(self):
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

    def _perform(self) -> Any:
        return self.code()
