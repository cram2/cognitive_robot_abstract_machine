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
from pycram.failures import PlanFailure
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
                self.plan.add_edge(self, grand_child)
            self.plan.plan_graph.remove_edge(self.index, child.index)
            self.plan.remove_node(child)


@dataclass
class ExecutesSequentially(LanguageNode):
    """
    Base class for nodes that execute their children sequentially.
    """


@dataclass
class SequentialNode(ExecutesSequentially):
    """
    Executes all children sequentially. Any failure is immediately raised.
    """

    def _perform(self):
        result = [child.perform() for child in self.children]
        return result


@dataclass
class ParallelNode(LanguageNode):
    """
    Executes all children in parallel by creating a thread per children and executing them in the respective thread. All
    exceptions during execution will be caught, saved to a list, and returned in the end.

    Behaviour:
        Returns a tuple containing the final state of execution (SUCCEEDED, FAILED) and a list of results from
        each child's perform() method. The state is :py:attr:`~TaskStatus.SUCCEEDED` *iff* all children could be executed without
        an exception. In any other case the State :py:attr:`~TaskStatus.FAILED` will be returned.

    """

    def _perform(self):
        self.perform_parallel(self.children)
        for child in self.children:
            if child.status == TaskStatus.FAILED:
                raise child.reason

    def perform_parallel(self, nodes: List[PlanNode]):
        """
        Behaviour of the parallel node performs the given nodes in parallel in different threads.

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


@dataclass(eq=False)
class RepeatNode(SequentialNode):
    """
    Executes all children a given number of times in sequential order.
    """

    repetitions: int = 1
    """
    The number of repetitions of the children.
    """

    def perform(self):
        """
        Behaviour of repeat, executes all children in a loop as often as stated on initialization.

        :return:
        """
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
    Executes all children sequentially, an exception while executing a child does not terminate the whole process.
    Instead, the exception is saved to a list of all exceptions thrown during execution and returned.

    Behaviour:
        Returns a tuple containing the final state of execution (SUCCEEDED, FAILED) and a list of results from each
        child's perform() method. The state is :py:attr:`~TaskStatus.SUCCEEDED` if one or more children are executed without
        exception. In the case that all children could not be executed the State :py:attr:`~TaskStatus.FAILED` will be returned.
    """

    def _perform(self):
        results = []
        for child in self.children:
            try:
                results.append(child.perform())
            except PlanFailure as e:
                results.append(e)
        return results


@dataclass
class TryAllNode(ParallelNode):
    """
    Executes all children in parallel by creating a thread per children and executing them in the respective thread. All
    exceptions during execution will be caught, saved to a list and returned upon end.

    Behaviour:
        Returns a tuple containing the final state of execution (SUCCEEDED, FAILED) and a list of results from each
        child's perform() method. The state is :py:attr:`~TaskStatus.SUCCEEDED` if one or more children could be executed
        without raising an exception. If all children fail the State :py:attr:`~TaskStatus.FAILED` will be returned.
    """

    def perform(self):
        """
        Behaviour of TryAll, creates a new thread for each child and executes all children in their respective threads.

        :return: The state and list of results according to the behaviour described in :func:`TryAll`
        """
        self.perform_parallel(self.children)
        child_statuses = [child.status for child in self.children]
        self.status = (
            TaskStatus.SUCCEEDED
            if TaskStatus.SUCCEEDED in child_statuses
            else TaskStatus.FAILED
        )

    def __hash__(self):
        return id(self)


@dataclass
class CodeNode(LanguageNode):
    """
    Executable code block in a plan.
    """

    code: Callable = field(default_factory=lambda: lambda: None, kw_only=True)

    def execute(self) -> Any:
        """
        Execute the code with its arguments

        :returns: Anything that the function associated with this object will return.
        """
        return self.code()
