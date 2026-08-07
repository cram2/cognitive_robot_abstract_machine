from __future__ import annotations

import threading
import time
from contextlib import ExitStack, AbstractContextManager
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Tuple, Type

import numpy as np
from typing_extensions import Self

from giskardpy.executor import Executor
from giskardpy.middleware.ros2.control_loop import ControlLoop
from giskardpy.middleware.ros2.feedback_publisher import ActionFeedbackPublisher
from giskardpy.middleware.ros2.input_synchronization import WorldStateInputs
from giskardpy.motion_statechart.motion_statechart import MotionStatechart
from giskardpy.qp.qp_controller import QPController
from krrood.adapters.json_serializer import SubclassJSONSerializer
from krrood.exceptions import DataclassException
from semantic_digital_twin.collision_checking.collision_manager import CollisionManager
from semantic_digital_twin.spatial_computations.forward_kinematics import (
    ForwardKinematicsManager,
)
from semantic_digital_twin.world import World

PhasePath = Tuple[str, ...]
"""
The chain of phase names from the control cycle down to one measured call.
"""

CONTROL_CYCLE_PHASE = "control_cycle"
"""
Name of the outermost phase; a measurement is only taken inside of it.
"""

# %% what is measured


@dataclass(frozen=True)
class PhaseDefinition:
    """
    One method whose runtime is attributed to a phase of the control cycle.
    """

    owner: Type[Any]
    """
    Class the method is defined on.
    """

    method_name: str
    """
    Name of the method that is replaced by a timing wrapper.
    """

    phase_name: str
    """
    Name the measurements are reported under.
    """


CONTROL_CYCLE_PHASES: Tuple[PhaseDefinition, ...] = (
    PhaseDefinition(ControlLoop, "run_cycle", CONTROL_CYCLE_PHASE),
    PhaseDefinition(ControlLoop, "apply_world_updates", "apply_world_updates"),
    PhaseDefinition(WorldStateInputs, "synchronize", "synchronize_inputs"),
    PhaseDefinition(ControlLoop, "raise_if_canceled", "check_cancel"),
    PhaseDefinition(Executor, "tick", "executor_tick"),
    PhaseDefinition(CollisionManager, "compute_collisions", "compute_collisions"),
    PhaseDefinition(MotionStatechart, "tick", "statechart_tick"),
    PhaseDefinition(QPController, "compute_command", "qp_solve"),
    PhaseDefinition(World, "apply_control_commands", "apply_control_commands"),
    PhaseDefinition(World, "notify_state_change", "notify_state_change"),
    PhaseDefinition(ForwardKinematicsManager, "recompute", "forward_kinematics"),
    PhaseDefinition(ControlLoop, "publish_commands", "publish_commands"),
    PhaseDefinition(ActionFeedbackPublisher, "publish_if_changed", "publish_feedback"),
)
"""
The phases of one control cycle, in the order they are entered.
"""

# %% samples


@dataclass
class _OpenPhase:
    """
    A phase that has been entered and not yet left.
    """

    name: str
    """
    Name of the phase.
    """

    started_at: float
    """
    Value of the performance counter when the phase was entered.
    """

    time_spent_in_children: float = 0.0
    """
    Seconds the phases called by this one took together.
    """


@dataclass
class _ThreadLocalPhaseStack(threading.local):
    """
    Gives every thread its own stack of entered phases.
    """

    stack: List[_OpenPhase] = field(default_factory=list)
    """
    The phases the thread has entered and not yet left.
    """


@dataclass
class PhaseSamples(SubclassJSONSerializer):
    """
    Every measurement taken for one phase at one position in the call tree.
    """

    path: PhasePath
    """
    Chain of phase names leading to this phase.
    """

    inclusive_durations: List[float] = field(default_factory=list)
    """
    Seconds each call took, including the phases it called.
    """

    exclusive_durations: List[float] = field(default_factory=list)
    """
    Seconds each call spent outside of the phases it called.
    """

    @property
    def call_count(self) -> int:
        """
        How often the phase was measured.

        :return: The number of measured calls.
        """
        return len(self.inclusive_durations)

    @property
    def inclusive_total(self) -> float:
        """
        Seconds all calls took together, including their children.

        :return: The summed inclusive duration in seconds.
        """
        return float(np.sum(self.inclusive_durations))

    @property
    def exclusive_total(self) -> float:
        """
        Seconds all calls spent in their own code.

        :return: The summed exclusive duration in seconds.
        """
        return float(np.sum(self.exclusive_durations))

    @property
    def inclusive_mean(self) -> float:
        """
        Average seconds per call, including children.

        :return: The mean inclusive duration in seconds.
        """
        return float(np.mean(self.inclusive_durations))

    @property
    def inclusive_maximum(self) -> float:
        """
        Slowest call, including children.

        :return: The largest inclusive duration in seconds.
        """
        return float(np.max(self.inclusive_durations))

    def inclusive_percentile(self, percentile: float) -> float:
        """
        Seconds below which the given share of calls stayed.

        :param percentile: The share of calls to cover, from 0 to 100.
        :return: The inclusive duration at that percentile in seconds.
        """
        return float(np.percentile(self.inclusive_durations, percentile))

    def to_json(self) -> Dict[str, Any]:
        return {
            **super().to_json(),
            "path": list(self.path),
            "inclusive_durations": self.inclusive_durations,
            "exclusive_durations": self.exclusive_durations,
        }

    @classmethod
    def _from_json(cls, data: Dict[str, Any], **kwargs) -> Self:
        """
        Rebuild the samples, turning the stored path back into the tuple that keys it.

        :param data: The json representation of the samples.
        :param kwargs: Additional arguments passed down by the deserializer.
        :return: The restored samples.
        """
        return cls(
            path=tuple(data["path"]),
            inclusive_durations=data["inclusive_durations"],
            exclusive_durations=data["exclusive_durations"],
        )


@dataclass
class CallTreeProfile(SubclassJSONSerializer):
    """
    What one profiled motion spent its time on.
    """

    scenario_name: str
    """
    Name of the motion that was measured.
    """

    control_dt: float
    """
    Seconds one control cycle may take before the robot is commanded too late.
    """

    wall_time: float
    """
    Seconds between the start and the end of the profiled motion.
    """

    compile_duration: float
    """
    Seconds spent turning the motion statechart into a controller before the first
    cycle.
    """

    phases: Dict[PhasePath, PhaseSamples] = field(default_factory=dict)
    """
    Measurements per phase, in the order the phases were first entered.
    """

    @property
    def control_cycle(self) -> PhaseSamples:
        """
        Measurements of the whole control cycle.

        :return: The samples of the outermost phase.
        :raises NoControlCycleMeasuredError: If not a single cycle was measured.
        """
        path = (CONTROL_CYCLE_PHASE,)
        if path not in self.phases:
            raise NoControlCycleMeasuredError(self.scenario_name)
        return self.phases[path]

    @property
    def control_cycles(self) -> int:
        """
        How many control cycles the motion took.

        :return: The number of measured control cycles.
        """
        return self.control_cycle.call_count

    @property
    def cycles_per_second(self) -> float:
        """
        How many control cycles the loop manages per second of cycle time.

        :return: The achievable control frequency in hertz.
        """
        return 1 / self.control_cycle.inclusive_mean

    @property
    def budget_utilization(self) -> float:
        """
        Share of the control budget an average cycle uses; ``1.0`` means the loop is
        exactly fast enough.

        :return: The used share of the control budget.
        """
        return self.control_cycle.inclusive_mean / self.control_dt

    def children_of(self, path: PhasePath) -> List[PhaseSamples]:
        """
        The phases that were called directly by the phase at the given path.

        :param path: Chain of phase names leading to the parent phase.
        :return: The samples of every phase one level below that path.
        """
        return [
            samples
            for samples in self.phases.values()
            if samples.path[:-1] == path and len(samples.path) == len(path) + 1
        ]

    def to_json(self) -> Dict[str, Any]:
        return {
            **super().to_json(),
            "scenario_name": self.scenario_name,
            "control_dt": self.control_dt,
            "wall_time": self.wall_time,
            "compile_duration": self.compile_duration,
            "phases": [samples.to_json() for samples in self.phases.values()],
        }

    @classmethod
    def _from_json(cls, data: Dict[str, Any], **kwargs) -> Self:
        """
        Rebuild a profile, keying the phases by the path each one carries.

        The phases are stored as a list because their keys are paths, which json cannot
        represent.

        :param data: The json representation of the profile.
        :param kwargs: Additional arguments passed down by the deserializer.
        :return: The restored profile.
        """
        phases = [PhaseSamples.from_json(entry) for entry in data["phases"]]
        return cls(
            scenario_name=data["scenario_name"],
            control_dt=data["control_dt"],
            wall_time=data["wall_time"],
            compile_duration=data["compile_duration"],
            phases={samples.path: samples for samples in phases},
        )


@dataclass
class NoControlCycleMeasuredError(DataclassException):
    """
    Raised when a profile is read that never saw a control cycle.
    """

    scenario_name: str

    def error_message(self) -> str:
        return f'No control cycle was measured for "{self.scenario_name}". '

    def suggest_correction(self) -> str:
        return "The profiler has to be entered before the motion is started."


# %% profiler


@dataclass
class ControlLoopProfiler(AbstractContextManager):
    """
    Measures where the control loop spends its time, without changing it.

    Entering the profiler replaces the methods of the control cycle by timing wrappers
    and leaving it puts the originals back. Only calls that happen inside a control
    cycle of the thread that runs the loop are measured, so the idle loop of the motion
    server does not pollute the result.
    """

    scenario_name: str
    """
    Name the resulting profile is reported under.
    """

    control_dt: float
    """
    Seconds one control cycle may take, used to judge the measurements.
    """

    phase_definitions: Tuple[PhaseDefinition, ...] = CONTROL_CYCLE_PHASES
    """
    The methods that are measured.
    """

    # %% init False
    phases: Dict[PhasePath, PhaseSamples] = field(init=False, default_factory=dict)
    """
    Measurements collected so far.
    """

    compile_duration: float = field(init=False, default=0.0)
    """
    Seconds the last :meth:`giskardpy.executor.Executor.compile` took.
    """

    _open_phases: _ThreadLocalPhaseStack = field(
        init=False, default_factory=lambda: _ThreadLocalPhaseStack()
    )
    """
    The phases each thread has entered and not yet left.
    """

    _started_at: float = field(init=False, default=0.0)
    """
    Value of the performance counter when the profiler was entered.
    """

    _wall_time: float = field(init=False, default=0.0)
    """
    Seconds the profiler was active.
    """

    _exit_stack: ExitStack = field(init=False, default_factory=ExitStack)
    """
    Undoes every replaced method when the profiler is left.
    """

    def __enter__(self) -> ControlLoopProfiler:
        for definition in self.phase_definitions:
            self._install_phase(definition)
        self._install_compile_measurement()
        self._started_at = time.perf_counter()
        return self

    def __exit__(self, exception_type, exception, traceback) -> None:
        self._wall_time = time.perf_counter() - self._started_at
        self._exit_stack.close()

    @property
    def profile(self) -> CallTreeProfile:
        """
        Everything that was measured while the profiler was active.

        :return: The profile of the measured motion.
        """
        return CallTreeProfile(
            scenario_name=self.scenario_name,
            control_dt=self.control_dt,
            wall_time=self._wall_time,
            compile_duration=self.compile_duration,
            phases=self.phases,
        )

    def _install_phase(self, definition: PhaseDefinition) -> None:
        """
        Replace the method of the definition by a wrapper that times it.

        :param definition: The method to measure and the phase it is reported under.
        """
        original = getattr(definition.owner, definition.method_name)
        is_control_cycle = definition.phase_name == CONTROL_CYCLE_PHASE

        def timed(instance: Any, *args: Any, **kwargs: Any) -> Any:
            open_phases = self._open_phases_of_current_thread()
            if not open_phases and not is_control_cycle:
                return original(instance, *args, **kwargs)
            phase = _OpenPhase(
                name=definition.phase_name, started_at=time.perf_counter()
            )
            path = tuple(entered.name for entered in open_phases) + (phase.name,)
            open_phases.append(phase)
            try:
                return original(instance, *args, **kwargs)
            finally:
                open_phases.pop()
                duration = time.perf_counter() - phase.started_at
                self._record(path, duration, duration - phase.time_spent_in_children)
                if open_phases:
                    open_phases[-1].time_spent_in_children += duration

        self._replace_method(definition.owner, definition.method_name, original, timed)

    def _install_compile_measurement(self) -> None:
        """
        Measure how long the controller of a goal takes to build.
        """
        original = Executor.compile

        def timed(instance: Executor, *args: Any, **kwargs: Any) -> Any:
            started_at = time.perf_counter()
            try:
                return original(instance, *args, **kwargs)
            finally:
                self.compile_duration = time.perf_counter() - started_at

        self._replace_method(Executor, "compile", original, timed)

    def _replace_method(
        self,
        owner: Type[Any],
        method_name: str,
        original: Callable[..., Any],
        replacement: Callable[..., Any],
    ) -> None:
        """
        Put the replacement on the class and restore the original when the profiler is
        left.

        A method the owner inherited is removed again instead of assigned back, so
        profiling never turns an inherited method into an own one.

        :param owner: Class the method is looked up on.
        :param method_name: Name of the method that is replaced.
        :param original: The method as it was before profiling.
        :param replacement: The method that is installed instead.
        """
        if method_name in owner.__dict__:
            self._exit_stack.callback(setattr, owner, method_name, original)
        else:
            self._exit_stack.callback(delattr, owner, method_name)
        setattr(owner, method_name, replacement)

    def _open_phases_of_current_thread(self) -> List[_OpenPhase]:
        """
        The phases the calling thread has entered and not yet left.

        :return: The phase stack of the calling thread.
        """
        return self._open_phases.stack

    def _record(self, path: PhasePath, inclusive: float, exclusive: float) -> None:
        """
        Remember one measurement of the phase at the given path.

        :param path: Chain of phase names leading to the measured phase.
        :param inclusive: Seconds the call took, including the phases it called.
        :param exclusive: Seconds the call spent outside of the phases it called.
        """
        if path not in self.phases:
            self.phases[path] = PhaseSamples(path=path)
        samples = self.phases[path]
        samples.inclusive_durations.append(inclusive)
        samples.exclusive_durations.append(exclusive)
