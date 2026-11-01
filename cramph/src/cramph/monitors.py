import time
from abc import ABC, abstractmethod
from dataclasses import field, dataclass
from typing_extensions import Callable

from cramph.node import EndedByOwner
from cramph.data_types import LifeCycleValues, ObservationStateValues
from cramph.node import StatechartNode, NodeArtifacts

# %% tick count check


@dataclass(eq=False, repr=False, kw_only=True)
class CheckTickCount(EndedByOwner, StatechartNode):
    """
    Sets observation to True if the tick count is above threshold.
    """

    threshold: int
    """
    After this many ticks, the node will turn True.
    """

    def build_artifacts(self) -> NodeArtifacts:
        artifacts = NodeArtifacts()
        artifacts.observation = self.context.tick_variable > self.threshold
        return artifacts


# %% printing


@dataclass(eq=False, repr=False, kw_only=True)
class Print(EndedByOwner, StatechartNode):
    """
    Prints a message to the console every tick.
    """

    message: str = ""
    """
    The text printed on every tick.
    """

    def on_tick(self) -> ObservationStateValues:
        print(self.message)
        return ObservationStateValues.TRUE


# %% counting time and ticks


@dataclass(eq=False, repr=False, kw_only=True)
class CountSeconds(EndedByOwner, StatechartNode):
    """
    This node counts X seconds and then turns True.

    Only counts while in state RUNNING, and it is up to its owner to stop it once it has
    counted far enough.
    """

    seconds: float
    """
    How many seconds to count before turning True.
    """

    tolerance: float = 1e-5
    """
    How many seconds short of :attr:`seconds` still count as having counted them.
    """

    _now: Callable[[], float] = field(default=time.monotonic, repr=False)
    """
    The clock the seconds are read from.
    """

    _start_time: float = field(init=False)
    """
    What :attr:`_now` read when this node started.
    """

    def on_tick(self) -> ObservationStateValues | None:
        difference = self._now() - self._start_time
        if difference >= self.seconds - self.tolerance:
            return ObservationStateValues.TRUE
        return None

    def on_start(self) -> None:
        self._start_time = self._now()


@dataclass(eq=False, repr=False, kw_only=True)
class TickCounter(EndedByOwner, StatechartNode, ABC):
    """
    Base for nodes that count ticks while RUNNING and turn True once a target is
    reached.

    Only counts while in state RUNNING, and it is up to its owner to stop it once it
    reaches its target.
    """

    _counter: int = field(init=False, default=0)
    """
    Number of ticks counted since the last start/reset.
    """

    def on_start(self) -> None:
        self._counter = 0

    def on_tick(self) -> ObservationStateValues | None:
        self._counter += 1
        if self._reached_target():
            return ObservationStateValues.TRUE
        return ObservationStateValues.FALSE

    @abstractmethod
    def _reached_target(self) -> bool:
        """
        Whether the counted target has been reached on the current tick.
        """


@dataclass(eq=False, repr=False, kw_only=True)
class CountSimulationTimeSeconds(TickCounter):
    """
    This node counts X seconds of simulation time (ticks * tick duration) and then turns
    True.

    Only counts while in state RUNNING.
    """

    seconds: float
    """
    How many seconds of simulation time to count.
    """

    def _reached_target(self) -> bool:
        return self.context.require_tick_duration() * self._counter >= self.seconds


@dataclass(eq=False, repr=False, kw_only=True)
class CountTicks(TickCounter):
    """
    This node counts :attr:`ticks`-many ticks and then turns True.

    Only counts while in state RUNNING.
    """

    ticks: int
    """
    Turns True after this many ticks.
    """

    def _reached_target(self) -> bool:
        return self._counter >= self.ticks


# %% pulses


@dataclass(eq=False, repr=False, kw_only=True)
class Pulse(EndedByOwner, StatechartNode):
    """
    Will stay True for a single tick, then turn False.
    """

    _counter: int = field(default=0, init=False)
    """
    Keeps track of how many ticks have passed since first True.
    """

    length: int = 1
    """
    Number of ticks to stay True.
    """

    def on_start(self) -> None:
        self._counter = 0

    def on_tick(self) -> ObservationStateValues | None:
        if self._counter < self.length:
            self._triggered = True
            self._counter += 1
            return ObservationStateValues.TRUE
        return ObservationStateValues.FALSE


# %% counting resets


@dataclass(eq=False, repr=False, kw_only=True)
class CountNodeResets(EndedByOwner, StatechartNode):
    """
    Turns True once :attr:`node` has been reset :attr:`target` times.

    Counts attempts rather than ticks, by watching the node re-enter NOT_STARTED. Its
    count is never cleared, unlike the counters that reset themselves when they start,
    so it survives the resets it is counting.
    """

    node: StatechartNode
    """
    The node whose resets are counted.
    """

    target: int
    """
    Number of resets after which this turns True.
    """

    resets: int = field(default=0, init=False)
    """
    Resets of :attr:`node` seen so far.
    """

    _previous_life_cycle: LifeCycleValues | None = field(
        default=None, init=False, repr=False
    )
    """
    Life cycle state of :attr:`node` on the previous tick.
    """

    def on_tick(self) -> ObservationStateValues | None:
        current_life_cycle = self.node.life_cycle_state
        if (
            self._previous_life_cycle is not None
            and current_life_cycle == LifeCycleValues.NOT_STARTED
            and self._previous_life_cycle != LifeCycleValues.NOT_STARTED
        ):
            self.resets += 1
        self._previous_life_cycle = current_life_cycle
        if self.resets >= self.target:
            return ObservationStateValues.TRUE
        return ObservationStateValues.FALSE
