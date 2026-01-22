import time
from dataclasses import field, dataclass
from typing import Optional, Callable

from ..context import ExecutionContext, BuildContext
from ..data_types import ObservationStateValues
from ..graph_node import MotionStatechartNode, NodeArtifacts


@dataclass(eq=False, repr=False)
class CheckControlCycleCount(MotionStatechartNode):
    """
    Sets observation to True if control cycle count is above threshold.
    """

    threshold: int = field(kw_only=True)
    """After this many control cycles, the node will turn True."""

    def build(self, context: BuildContext) -> NodeArtifacts:
        artifacts = NodeArtifacts()
        artifacts.observation = context.control_cycle_variable > self.threshold
        return artifacts


@dataclass(eq=False, repr=False)
class Print(MotionStatechartNode):
    """
    Prints a message to the console every tick.
    """
    message: str = ""

    def on_tick(self, context: ExecutionContext) -> ObservationStateValues:
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

    def on_tick(self, context: ExecutionContext) -> Optional[ObservationStateValues]:
        difference = self._now() - self._start_time
        if difference >= self.seconds - 1e-5:
            return ObservationStateValues.TRUE
        return None

    def on_start(self, context: ExecutionContext):
        self._start_time = self._now()


@dataclass(repr=False, eq=False)
class CountControlCycles(MotionStatechartNode):
    """
    This node counts 'threshold'-many control cycles and then turns True.
    Only counts while in state RUNNING.
    """

    _counter: int = field(init=False)
    """Keeps track of how many ticks have passed since first True"""
    control_cycles: int = field(kw_only=True)
    """Turns True after this many control cycles."""

    def on_tick(self, context: ExecutionContext) -> Optional[ObservationStateValues]:
        self._counter += 1
        if self._counter >= self.control_cycles:
            return ObservationStateValues.TRUE
        return ObservationStateValues.FALSE

    def on_start(self, context: ExecutionContext):
        self._counter = 0


@dataclass
class Pulse(MotionStatechartNode):
    """
    Will stay True for a single tick, then turn False.
    """

    _counter: int = field(default=0, init=False)
    """Keeps track of how many ticks have passed since first True"""
    length: int = field(default=1, kw_only=True)
    """Number of ticks to stay True"""

    def on_start(self, context: ExecutionContext):
        self._counter = 0

    def on_tick(self, context: ExecutionContext) -> Optional[ObservationStateValues]:
        if self._counter < self.length:
            self._triggered = True
            self._counter += 1
            return ObservationStateValues.TRUE
        return ObservationStateValues.FALSE
