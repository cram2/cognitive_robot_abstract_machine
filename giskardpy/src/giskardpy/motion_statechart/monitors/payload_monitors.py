import logging
import threading
import time
from dataclasses import field, dataclass
from typing import Optional, Callable

from typing_extensions import Self

from giskardpy.motion_statechart.context import MotionStatechartContext
from giskardpy.motion_statechart.data_types import ObservationStateValues
from giskardpy.motion_statechart.graph_node import MotionStatechartNode, NodeArtifacts

logger = logging.getLogger(__name__)


@dataclass(eq=False, repr=False)
class CheckControlCycleCount(MotionStatechartNode):
    """
    Sets observation to True if control cycle count is above threshold.
    """

    threshold: int = field(kw_only=True)
    """
    After this many control cycles, the node will turn True.
    """

    def build(self, context: MotionStatechartContext) -> NodeArtifacts:
        artifacts = NodeArtifacts()
        artifacts.observation = context.control_cycle_variable > self.threshold
        return artifacts


@dataclass(eq=False, repr=False)
class Print(MotionStatechartNode):
    """
    Prints a message to the console every tick.
    """

    message: str = ""

    def on_tick(self, context: MotionStatechartContext) -> ObservationStateValues:
        print(self.message)
        return ObservationStateValues.TRUE


@dataclass
class CountSeconds(MotionStatechartNode):
    """
    This node counts X seconds and then turns True.

    Only counts while in state RUNNING.
    """

    seconds: float = field(kw_only=True)
    _now: Callable[[], float] = field(default=time.monotonic, kw_only=True, repr=False)
    _start_time: float = field(init=False)

    def on_tick(
        self, context: MotionStatechartContext
    ) -> Optional[ObservationStateValues]:
        difference = self._now() - self._start_time
        if difference >= self.seconds - 1e-5:
            return ObservationStateValues.TRUE
        return None

    def on_start(self, context: MotionStatechartContext):
        self._start_time = self._now()


@dataclass(repr=False, eq=False)
class CountControlCycles(MotionStatechartNode):
    """
    This node counts 'threshold'-many control cycles and then turns True.

    Only counts while in state RUNNING.
    """

    _counter: int = field(init=False)
    """
    Keeps track of how many ticks have passed since first True.
    """

    control_cycles: int = field(kw_only=True)
    """
    Turns True after this many control cycles.
    """

    def on_tick(
        self, context: MotionStatechartContext
    ) -> Optional[ObservationStateValues]:
        self._counter += 1
        if self._counter >= self.control_cycles:
            return ObservationStateValues.TRUE
        return ObservationStateValues.FALSE

    def on_start(self, context: MotionStatechartContext):
        self._counter = 0


@dataclass(eq=False, repr=False)
class ThreadedPredicateMonitor(MotionStatechartNode):
    """
    Evaluates an arbitrary boolean predicate in a background thread and exposes the
    result as the node's observation state.

    While the node is RUNNING:

    - On entering RUNNING (``on_start``), the predicate is launched in a daemon
      thread so a slow/blocking evaluation does not stall the control loop.
    - Until the thread finishes, the observation is ``UNKNOWN``.
    - Afterwards the observation is ``TRUE`` / ``FALSE`` based on the predicate's
      return value. If the predicate raises, the error is logged and the
      observation becomes ``FALSE``.

    The predicate is a plain ``Callable[[], bool]`` so this class has no
    dependency on whatever produces it (e.g. a Coraplex/EQL condition is wrapped in
    a lambda by the caller).

    .. warning:: The predicate is not serializable, so this monitor only works in
        a locally ticked statechart, not when the statechart is shipped to a
        remote giskard instance.
    """

    predicate: Optional[Callable[[], bool]] = field(kw_only=True)
    """
    The predicate to evaluate, passed as a constructor argument.
    """

    _thread: Optional[threading.Thread] = field(default=None, init=False, repr=False)
    _result: Optional[bool] = field(default=None, init=False, repr=False)
    _done: bool = field(default=False, init=False, repr=False)
    _error: Optional[BaseException] = field(default=None, init=False, repr=False)

    def _worker(self, predicate: Callable[[], bool]) -> None:
        """
        Wrapper that is executed in the external thread to catch Exceptions and manage
        Observation variables.

        :param predicate: The predicate to evaluate
        """
        result: Optional[bool] = None
        error: Optional[BaseException] = None
        try:
            result = bool(predicate())
        except BaseException as e:  # noqa: BLE001 - reported via observation/logging
            error = e
        self._result = result
        self._error = error
        self._done = True

    def on_start(self, context: MotionStatechartContext) -> None:
        """
        On start of this note construct the external thread with self._worker and start
        it as daemon.
        """
        if self.predicate is None:
            logger.error(
                "%s has no predicate; pass one via the `predicate` argument.",
                self.unique_name,
            )
            return
        self._result = None
        self._error = None
        self._done = False
        self._thread = threading.Thread(
            target=self._worker,
            args=(self.predicate,),
            name=f"{self.__class__.__name__}-{self.name}",
            daemon=True,
        )
        self._thread.start()

    def on_tick(
        self, context: MotionStatechartContext
    ) -> Optional[ObservationStateValues]:
        """
        On tick of the Motion State Chart check if the thread is finished and set the
        ObservationStateValues accordingly to ObservationStateValues.UNKNOWN if the
        thread is still working ObservationStateValues.TRUE if the Thread finished with
        true and ObservationStateValues.FALSE if the Thread finished with false or
        crashed with an exception.
        """
        if not self._done:
            return ObservationStateValues.UNKNOWN
        if self._error is not None:
            logger.warning(
                "%s predicate raised %s; reporting FALSE.",
                self.unique_name,
                self._error,
            )
            raise self._error
        return (
            ObservationStateValues.TRUE
            if self._result
            else ObservationStateValues.FALSE
        )

    def on_reset(self, context: MotionStatechartContext) -> None:
        self._join_thread()
        self._result = None
        self._error = None
        self._done = False

    def cleanup(self, context: MotionStatechartContext) -> None:
        self._join_thread()

    def _join_thread(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            # Don't block the control loop indefinitely on a hung predicate;
            # the thread is a daemon and will be reaped on interpreter exit.
            self._thread.join(timeout=0.1)
        self._thread = None


@dataclass
class Pulse(MotionStatechartNode):
    """
    Will stay True for a single tick, then turn False.
    """

    _counter: int = field(default=0, init=False)
    """
    Keeps track of how many ticks have passed since first True.
    """

    length: int = field(default=1, kw_only=True)
    """
    Number of ticks to stay True.
    """

    def on_start(self, context: MotionStatechartContext):
        self._counter = 0

    def on_tick(
        self, context: MotionStatechartContext
    ) -> Optional[ObservationStateValues]:
        if self._counter < self.length:
            self._triggered = True
            self._counter += 1
            return ObservationStateValues.TRUE
        return ObservationStateValues.FALSE
