import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from typing_extensions import Optional

from krrood.symbolic_math.symbolic_math import FloatVariable
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.world_state import WorldStateTrajectory
from semantic_digital_twin.world_description.world_state_trajectory_plotter import (
    WorldStateTrajectoryPlotter,
)
from .data_types.exceptions import NoQPControllerConfigException
from .motion_statechart.context import BuildContext, ExecutionContext
from .motion_statechart.motion_statechart import MotionStatechart
from .motion_statechart.variable_managers.auxiliary_variable_manager import (
    AuxiliaryVariableManager,
)
from .qp.exceptions import EmptyProblemException
from .qp.qp_controller import QPController
from .qp.qp_controller_config import QPControllerConfig


@dataclass
class Pacer(ABC):
    """
    Tries to achieve a specific frequency by adjusting the sleep time between calls.
    """

    target_frequency: float
    """
    Frequency of the loop in hertz.
    """

    @abstractmethod
    def sleep(self):
        """
        Sleeps according to the pacer's logic to make a loop run at hz frequency.
        """


@dataclass
class SimulationPacer(Pacer):
    target_frequency: float = field(init=False)
    """
    How long a cycle should take in seconds with real_time_factor=1.0.
    """

    real_time_factor: Optional[float] = None
    """
    Allows you to adjust the simulation speed.
    If None, the pacer will not sleep at all.
    If 1.0, the pacer will try to achieve the control_dt frequency, as long as the other code in the loop allows it.
    """

    _next_target_time: Optional[float] = field(default=None, init=False)

    def sleep(self):
        """
        Sleep to maintain a control loop pace defined by `control_dt` and `real_time_factor`.
        - If `real_time_factor` is None, return immediately (no pacing).
        - Otherwise, target interval is `control_dt / real_time_factor`.
        """
        if self.real_time_factor is None:
            return
        if self.real_time_factor <= 0:
            return
        dt = 1 / (self.target_frequency * self.real_time_factor)
        now = time.monotonic()
        if self._next_target_time is None:
            self._next_target_time = now + dt
        sleep_time = self._next_target_time - now
        if sleep_time > 0:
            time.sleep(sleep_time)
            now = self._next_target_time
        else:
            # if we are behind schedule, catch up without sleeping and reschedule to the next slot after now
            pass
        # advance next target time to the next slot strictly after current time
        while self._next_target_time is not None and self._next_target_time <= now:
            self._next_target_time += dt


@dataclass
class Executor:
    """
    Represents the main execution entity that manages motion statecharts, collision
    scenes, and control cycles for the robot's operations.
    """

    context: BuildContext

    tmp_folder: str = field(default="/tmp/")
    """Path to safe temporary files."""
    record_trajectory: bool = False
    """Whether to record the trajectory of the robot."""
    world_state_trajectory: WorldStateTrajectory = field(init=False)
    """The trajectory of the robot's world state."""
    trajectory_plotter: WorldStateTrajectoryPlotter = field(
        default_factory=WorldStateTrajectoryPlotter
    )
    """The trajectory plotter used to plot the robot's trajectory."""

    pacer: Pacer = field(default_factory=SimulationPacer)

    # %% init False
    motion_statechart: MotionStatechart = field(init=False)
    """The motion statechart describing the robot's motion logic."""
    # auxiliary_variable_manager: AuxiliaryVariableManager = field(
    #     default_factory=AuxiliaryVariableManager, init=False
    # )
    # """Manages auxiliary symbolic variables for execution contexts."""
    qp_controller: Optional[QPController] = field(default=None, init=False)
    """Optional quadratic programming controller used for motion control."""

    # control_cycles: int = field(init=False)
    # """Tracks the number of control cycles elapsed during execution."""
    # _control_cycles_variable: FloatVariable = field(init=False)
    # """Auxiliary variable linked to the control_cycles attribute."""
    _control_cycle_index: int = field(init=False)
    """Tracks the index of the current control cycle."""

    _time_variable: FloatVariable = field(init=False)
    """Auxiliary variable representing the current time in seconds since the start of the simulation."""

    @classmethod
    def create_from_parts(cls, world: World, pacer: Pacer | None = None):
        if pacer is None:
            pacer = SimulationPacer()
        return cls(BuildContext(world=world), pacer=pacer)

    @property
    def time(self) -> float:
        return self.control_cycles * self.context.qp_controller_config.control_dt

    def __post_init__(self):
        self.pacer.target_frequency = self.context.qp_controller_config.target_frequency
        self._create_control_cycles_variable()

    def _create_control_cycles_variable(self):
        self.context.control_cycle_variable = FloatVariable("control_cycles")
        self._control_cycle_index = self.context.float_variable_data.add_variable(
            self.context.control_cycle_variable
        )

    @property
    def control_cycles(self):
        return self.context.float_variable_data.data[self._control_cycle_index]

    @control_cycles.setter
    def control_cycles(self, value):
        self.context.float_variable_data.set_value(self._control_cycle_index, value)

    def compile(self, motion_statechart: MotionStatechart):
        self.motion_statechart = motion_statechart
        self.control_cycles = 0
        self.motion_statechart.compile(self.context)
        self._compile_qp_controller(self.context.qp_controller_config)
        self.world_state_trajectory = WorldStateTrajectory.from_world_state(
            self.context.world.state, time=self.time
        )

    @property
    def execution_context(self) -> ExecutionContext:
        return self.context

    def tick(self):
        self.control_cycles += 1
        self.context.collision_manager.compute_collisions()
        self.motion_statechart.tick(self.context)
        if self.qp_controller is None:
            return
        next_cmd = self.qp_controller.get_cmd(
            world_state=self.context.world.state.data,
            life_cycle_state=self.motion_statechart.life_cycle_state.data,
            float_variables=self.context.float_variable_data.data,
        )
        self.context.world.apply_control_commands(
            next_cmd,
            self.qp_controller.config.control_dt,
            self.qp_controller.config.max_derivative,
        )
        self.world_state_trajectory.append(self.context.world.state, self.time)

    def tick_until_end(self, timeout: int = 1_000):
        """
        Calls tick until is_end_motion() returns True.
        :param timeout: Max number of ticks to perform.
        """
        try:
            for i in range(timeout):
                self.tick()
                self.pacer.sleep()
                if self.motion_statechart.is_end_motion():
                    return
            raise TimeoutError("Timeout reached while waiting for end of motion.")
        finally:
            self._set_velocity_acceleration_jerk_to_zero()

    def _set_velocity_acceleration_jerk_to_zero(self):
        self.context.world.state.velocities[:] = 0
        self.context.world.state.accelerations[:] = 0
        self.context.world.state.jerks[:] = 0

    def _compile_qp_controller(self, controller_config: QPControllerConfig):
        ordered_dofs = sorted(
            self.context.world.active_degrees_of_freedom,
            key=lambda dof: self.context.world.state._index[dof.id],
        )
        constraint_collection = (
            self.motion_statechart.combine_constraint_collections_of_nodes()
        )
        if len(constraint_collection._constraints) == 0:
            self.qp_controller = None
            # to not build controller, if there are no constraints
            return
        elif controller_config is None:
            raise NoQPControllerConfigException(
                "constraints but no controller config given."
            )
        self.qp_controller = QPController(
            config=controller_config,
            degrees_of_freedom=ordered_dofs,
            constraint_collection=constraint_collection,
            world_state_symbols=self.context.world.state.get_variables(),
            life_cycle_variables=self.motion_statechart.life_cycle_state.life_cycle_symbols(),
            float_variables=self.context.float_variable_data.variables,
        )
        if self.qp_controller.has_not_free_variables():
            raise EmptyProblemException()

    def plot_trajectory(self, file_name: str = "./trajectory.pdf"):
        self.trajectory_plotter.plot_trajectory(
            self.context.world_state_trajectory, file_name
        )
