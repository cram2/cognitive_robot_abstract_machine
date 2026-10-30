from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import timedelta

from typing_extensions import TYPE_CHECKING, TypeVar

from cramph.context import ContextExtension, StatechartContext
from cramph.exceptions import (
    MissingExecutorExtensionError,
    StatechartOfDifferentContextError,
    NonPositiveRealTimeFactorError,
    TickLimitReachedError,
)
from cramph.compiler import RecompileCallback
from cramph.statechart import Statechart
from krrood.symbolic_math.symbolic_math import FloatVariable

if TYPE_CHECKING:
    from semantic_digital_twin.adapters.multi_sim import MujocoSim

logger = logging.getLogger(__name__)


@dataclass
class Pacer(ABC):
    """
    Decides how long a loop waits between two cycles.
    """

    target_frequency: float | None = field(default=None, init=False)
    """
    Frequency of the loop in hertz, set by whoever runs the loop; None until then.
    """

    def pace_ticks_of(self, context: StatechartContext) -> None:
        """
        Sets :attr:`target_frequency` to one cycle per tick of `context`.

        :param context: The context whose ticks this pacer paces.
        :raises TickDurationUnknownError: If `context` does not know how long a tick
            lasts.
        """
        self.target_frequency = 1 / context.require_tick_duration()

    @abstractmethod
    def sleep(self) -> None:
        """
        Wait until the loop may start its next cycle.
        """

    def wait_for_standstill(self) -> None:
        """
        Wait until the world has caught up with what the loop commanded.

        Nothing to wait for unless the loop drives a physics in lockstep.
        """
        return None


@dataclass
class NoPacing(Pacer):
    """
    Lets a loop run as fast as the hardware allows.
    """

    def pace_ticks_of(self, context: StatechartContext) -> None:
        """
        Does nothing, since this pacer never waits.
        """
        return None

    def sleep(self) -> None:
        return None


@dataclass
class ScheduledPacer(Pacer, ABC):
    """
    Holds a loop at a fixed cycle duration by sleeping until the next slot.

    A cycle that overruns its slot is not compensated by a shorter following one; the
    schedule simply skips to the next slot after the current time.
    """

    _next_target_time: float | None = field(default=None, init=False)
    """
    Point in time the next cycle may start at, None until the first sleep.
    """

    @property
    @abstractmethod
    def cycle_duration(self) -> float:
        """
        How many seconds one cycle should take.
        """

    def sleep(self) -> None:
        cycle_duration = self.cycle_duration
        now = time.monotonic()
        if self._next_target_time is None:
            self._next_target_time = now + cycle_duration
        sleep_time = self._next_target_time - now
        if sleep_time > 0:
            time.sleep(sleep_time)
            now = self._next_target_time
        while self._next_target_time <= now:
            self._next_target_time += cycle_duration


@dataclass
class RealTimePacer(ScheduledPacer):
    """
    Holds a loop at its target frequency in wall clock time.
    """

    @property
    def cycle_duration(self) -> float:
        return 1 / self.target_frequency


@dataclass
class SimulationPacer(ScheduledPacer):
    """
    Runs a loop at a multiple of its target frequency to speed up or slow down a
    simulation.
    """

    real_time_factor: float = 1.0
    """
    How much faster than real time the loop runs; ``2.0`` is twice as fast.
    """

    def __post_init__(self) -> None:
        if self.real_time_factor <= 0:
            raise NonPositiveRealTimeFactorError(self.real_time_factor)

    @property
    def cycle_duration(self) -> float:
        return 1 / (self.target_frequency * self.real_time_factor)


@dataclass
class SteppedSimulationPacer(Pacer):
    """
    Holds a loop by stepping a physically simulated world one cycle forward between two
    ticks, so a controller ticking against the world runs in lockstep with its physics.

    Every tick's command lands in the world state, the simulation's servos take it as
    their set point, and the physics advances one cycle before the next tick reads the
    world back.
    """

    simulation: MujocoSim
    """
    The simulation to step; it has to be started with
    :meth:`~semantic_digital_twin.adapters.multi_sim.MujocoSim.start_stepped_simulation`
    already.
    """

    standstill_velocity: float = 0.01
    """
    The joint velocity, in radians or metres per second, below which a joint counts as
    no longer moving.
    """

    standstill_time_limit: timedelta = field(default=timedelta(seconds=10))
    """
    The simulated time the robot is given to come to a standstill.
    """

    def sleep(self) -> None:
        self.simulation.step_simulation(timedelta(seconds=1 / self.target_frequency))

    def wait_for_standstill(self) -> None:
        """
        Step the physics until no servo-driven joint is moving any more, or
        :attr:`standstill_time_limit` passes.

        A controller stops ticking once its command satisfies it, which is before the
        joints have got there. A joint held up by something it has run into counts as
        stopped too.
        """
        servo_names = [
            actuator.name.name for actuator in self.simulation.world.actuators
        ]
        if not servo_names:
            return
        simulator = self.simulation.simulator
        standstill_travel = self.standstill_velocity / self.target_frequency
        previous = simulator.get_actuators_lengths(servo_names).result
        for _ in range(
            int(self.standstill_time_limit.total_seconds() * self.target_frequency)
        ):
            self.sleep()
            current = simulator.get_actuators_lengths(servo_names).result
            travelled = max(
                abs(current[servo_name] - previous[servo_name])
                for servo_name in servo_names
            )
            previous = current
            if travelled < standstill_travel:
                return
        logger.warning(
            "The robot was still moving after %s of simulated time.",
            self.standstill_time_limit,
        )


@dataclass
class ExecutorExtension:
    """
    Adds behaviour to an :class:`Executor` around creating it, compiling a statechart
    and ending a run.

    Every stage does nothing unless an extension overrides it.
    """

    def extend_context(self, context: StatechartContext) -> None:
        """
        Called once when the executor is created, before a :class:`StatechartExecutor`
        lets its pacer read the tick duration of `context`.

        :param context: The context handed to every node of the executed statecharts.
        """
        return None

    def after_compile(self, executor: Executor) -> None:
        """
        Called once the statechart is compiled, before its first tick.

        :param executor: The executor this extension belongs to.
        """
        return None

    def after_run(self, executor: Executor) -> None:
        """
        Called once a run stops, see :meth:`Executor.finish_run`.

        :param executor: The executor this extension belongs to.
        """
        return None


@dataclass
class TickingExecutorExtension(ExecutorExtension):
    """
    Adds behaviour to a :class:`StatechartExecutor` around every tick and every
    recompile, besides the stages of :class:`ExecutorExtension`.
    """

    def before_recompile(self, executor: StatechartExecutor) -> bool:
        """
        Called before a statechart that already compiled compiles again, or before a
        node of it chooses its child, either of which blocks the tick. The statechart
        keeps ticking, calling this every tick, until every extension is at rest.

        :param executor: The executor this extension belongs to.
        :return: Whether what this extension drives is at rest; it is unless overridden.
        """
        return True

    def after_recompile(self, executor: StatechartExecutor) -> None:
        """
        Called every time a statechart that already compiled compiled again, which keeps
        its state, its history and the tick count. Does what :meth:`after_compile` does
        unless overridden.

        :param executor: The executor this extension belongs to.
        """
        self.after_compile(executor)

    def before_tick(self, executor: StatechartExecutor) -> None:
        """
        Called at the start of every tick, before the tick count is advanced.

        :param executor: The executor this extension belongs to.
        """
        return None

    def after_tick(self, executor: StatechartExecutor) -> None:
        """
        Called at the end of every tick, after the statechart was ticked.

        :param executor: The executor this extension belongs to.
        """
        return None


GenericExecutorExtension = TypeVar("GenericExecutorExtension", bound=ExecutorExtension)


# %% executing a statechart


@dataclass
class Executor(ABC):
    """
    Executes a statechart built in its :attr:`context`: :meth:`compile` takes the
    statechart, and :meth:`execute` runs it until it ended.
    """

    context: StatechartContext
    """
    The context handed to every node of the statechart.
    """

    extensions: list[ExecutorExtension] = field(default_factory=list, kw_only=True)
    """
    Add behaviour around compiling and running, called in the order they are listed.
    """

    # %% init False
    statechart: Statechart | None = field(init=False, default=None)
    """
    The statechart that is executed, set by :meth:`compile`.
    """

    def __post_init__(self) -> None:
        """
        Adds the :meth:`default_context_extensions` to :attr:`context` and the
        :meth:`default_executor_extensions` to :attr:`extensions`, keeping any extension
        of the same type that is already there, and lets every extension extend
        :attr:`context`.
        """
        for default_extension in self.default_context_extensions():
            self.context.ensure_extension(default_extension)
        for default_extension in self.default_executor_extensions():
            if not any(
                type(extension) is type(default_extension)
                for extension in self.extensions
            ):
                self.extensions.append(default_extension)
        for extension in self.extensions:
            extension.extend_context(self.context)

    @abstractmethod
    def default_context_extensions(self) -> list[ContextExtension]:
        """
        :return: The context extensions :attr:`context` holds unless one of the same
            type was supplied.
        """

    @abstractmethod
    def default_executor_extensions(self) -> list[ExecutorExtension]:
        """
        :return: The executor extensions appended to :attr:`extensions` unless one of
            the same type was supplied.
        """

    def require_extension(
        self, extension_type: type[GenericExecutorExtension]
    ) -> GenericExecutorExtension:
        """
        :param extension_type: The exact type of the requested extension.
        :return: The first extension in :attr:`extensions` of `extension_type`.
        :raises MissingExecutorExtensionError: If no extension is of `extension_type`.
        """
        for extension in self.extensions:
            if type(extension) is extension_type:
                return extension
        raise MissingExecutorExtensionError(expected_extension=extension_type)

    def compile(self, statechart: Statechart) -> None:
        """
        Takes `statechart` as the one :meth:`execute` runs.

        :param statechart: The statechart to execute.
        :raises StatechartOfDifferentContextError: If `statechart` was not built in
            :attr:`context`.
        """
        if statechart.context is not self.context:
            raise StatechartOfDifferentContextError()
        self.statechart = statechart

    def _notify_compiled(self) -> None:
        """
        Tell every extension that :attr:`statechart` is compiled.
        """
        for extension in self.extensions:
            extension.after_compile(self)

    @abstractmethod
    def execute(self) -> None:
        """
        Runs the compiled statechart until it ended.
        """

    def finish_run(self) -> None:
        """
        Tell every extension that the run stopped, however it stopped.
        """
        for extension in self.extensions:
            extension.after_run(self)


@dataclass
class StatechartExecutor(Executor, RecompileCallback):
    """
    Compiles a statechart and ticks it, counting the ticks.

    A statechart it runs may take new nodes while it runs, see
    :meth:`~cramph.statechart.Statechart.modify`, and builds its nodes again when the
    kinematic structure of its world changes; the extensions then compile again.
    """

    pacer: Pacer = field(default_factory=NoPacing, kw_only=True)
    """
    Paces the loop that ticks this executor.
    """

    def __post_init__(self) -> None:
        super().__post_init__()
        self.pacer.pace_ticks_of(self.context)
        self._create_tick_variable()

    def default_context_extensions(self) -> list[ContextExtension]:
        return []

    def default_executor_extensions(self) -> list[ExecutorExtension]:
        return []

    @property
    def ticking_extensions(self) -> list[TickingExecutorExtension]:
        """
        :return: The extensions in :attr:`extensions` that act around every tick and
            every recompile, in the order they are listed.
        """
        return [
            extension
            for extension in self.extensions
            if isinstance(extension, TickingExecutorExtension)
        ]

    def _create_tick_variable(self) -> None:
        """
        Registers the variable counting the ticks in the context.
        """
        self.context.tick_variable = FloatVariable("tick_count")
        self.context.float_variable_data.register_expression(self.context.tick_variable)

    @property
    def time(self) -> float:
        """
        :return: How many seconds the ticks run so far stand for.
        """
        return self.tick_count * self.context.require_tick_duration()

    @property
    def tick_count(self) -> int:
        """
        :return: The number of ticks run since :meth:`compile`.
        """
        return self.context.tick_count

    @tick_count.setter
    def tick_count(self, value: int) -> None:
        self.context.float_variable_data.set_value(self.context.tick_variable, value)

    def compile(self, statechart: Statechart) -> None:
        """
        Compiles `statechart` and ticks it once, so that nodes whose start condition is
        constant true start immediately.

        A run that raises while it starts is finished, see :meth:`finish_run`, before
        the exception propagates.

        :param statechart: The statechart to execute.
        :raises StatechartOfDifferentContextError: If `statechart` was not built in
            :attr:`context`.
        """
        super().compile(statechart)
        self.tick_count = 0
        self.statechart.compiler.compile()
        self.statechart.compiler.add_recompile_callback(self)
        try:
            self._notify_compiled()
            self.statechart.tick()
        except BaseException:
            self.finish_run()
            raise

    def before_recompile(self) -> bool:
        """
        Tell every extension that the statechart is about to block its tick.

        :return: Whether every extension is at rest.
        """
        answers = [
            extension.before_recompile(self) for extension in self.ticking_extensions
        ]
        return all(answers)

    def after_recompile(self) -> None:
        """
        Let every extension compile again, so what it builds from the nodes covers every
        node of the statechart.

        The tick count is not restarted.
        """
        for extension in self.ticking_extensions:
            extension.after_recompile(self)

    def tick(self) -> None:
        """
        Advances the statechart by one tick.
        """
        ticking_extensions = self.ticking_extensions
        for extension in ticking_extensions:
            extension.before_tick(self)
        self.tick_count += 1
        self.statechart.tick()
        for extension in ticking_extensions:
            extension.after_tick(self)

    def execute(self) -> None:
        """
        Ticks the compiled statechart until it ended, see :meth:`tick_until_end`.
        """
        self.tick_until_end()

    def tick_until_end(self, timeout: int = 1_000) -> None:
        """
        Calls tick until
        :meth:`~cramph.statechart.Statechart.is_ended`
        returns True.

        :param timeout: Max number of ticks to perform.
        :raises TickLimitReachedError: If the statechart has not ended after `timeout`
            ticks.
        """
        try:
            for _ in range(timeout):
                self.tick()
                self.pacer.sleep()
                if self.statechart.is_ended():
                    return
            raise TickLimitReachedError(tick_limit=timeout)
        finally:
            self.finish_run()

    def finish_run(self) -> None:
        """
        Does what :meth:`Executor.finish_run` does, then cleans up the nodes and the
        context.
        """
        super().finish_run()
        self.statechart.cleanup_nodes()
        self.context.cleanup()
