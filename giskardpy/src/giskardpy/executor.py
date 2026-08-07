import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from giskardpy.data_types.exceptions import NonPositiveRealTimeFactorError
from giskardpy.motion_statechart.context import MotionStatechartContext
from giskardpy.motion_statechart.exceptions import PlotterNotConfiguredError
from giskardpy.motion_statechart.motion_statechart import MotionStatechart
from giskardpy.motion_statechart.plotters.debug_expression_trajectory_plotter import (
    DebugExpressionTrajectoryPlotter,
)
from giskardpy.qp.exceptions import EmptyProblemException
from giskardpy.qp.qp_controller import QPController
from giskardpy.qp.qp_controller_config import QPControllerConfig
from krrood.symbolic_math.symbolic_math import FloatVariable
from semantic_digital_twin.world_description.world_state_trajectory_plotter import (
    WorldStateTrajectoryPlotter,
)


@dataclass
class Pacer(ABC):
    """
    Decides how long a loop waits between two cycles.
    """

    target_frequency: float = field(init=False)
    """
    Frequency of the loop in hertz, set by whoever runs the loop.
    """

    @abstractmethod
    def sleep(self) -> None:
        """
        Wait until the loop may start its next cycle.
        """


@dataclass
class NoPacing(Pacer):
    """
    Lets a loop run as fast as the hardware allows.
    """

    def sleep(self) -> None:
        pass


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

    def __post_init__(self):
        if self.real_time_factor <= 0:
            raise NonPositiveRealTimeFactorError(self.real_time_factor)

    @property
    def cycle_duration(self) -> float:
        return 1 / (self.target_frequency * self.real_time_factor)


@dataclass
class Executor:
    """
    Represents the main execution entity that manages motion statecharts, collision
    scenes, and control cycles for the robot's operations.
    """

    context: MotionStatechartContext

    trajectory_plotter: WorldStateTrajectoryPlotter | None = field(default=None)
    """
    The trajectory plotter used to plot the robot's trajectory.
    """

    debug_expression_plotter: DebugExpressionTrajectoryPlotter | None = field(
        default=None
    )
    """
    Records and plots how the debug expressions evolved during the motion.
    """

    pacer: Pacer = field(default_factory=NoPacing)
    """
    Paces the loop that ticks this executor.
    """

    # %% init False
    motion_statechart: MotionStatechart | None = field(init=False, default=None)
    """
    The motion statechart describing the robot's motion logic, set by :meth:`compile`.
    """

    qp_controller: QPController | None = field(default=None, init=False)
    """
    Optional quadratic programming controller used for motion control.
    """

    @property
    def time(self) -> float:
        return self.control_cycles * self.context.qp_controller_config.control_dt

    def __post_init__(self):
        self.pacer.target_frequency = self.context.qp_controller_config.target_frequency
        self._create_control_cycles_variable()

    def _create_control_cycles_variable(self):
        self.context.control_cycle_variable = FloatVariable("control_cycles")
        self.context.float_variable_data.register_expression(
            self.context.control_cycle_variable
        )

    @property
    def control_cycles(self) -> float:
        return float(
            self.context.float_variable_data.get_value(
                self.context.control_cycle_variable
            )
        )

    @control_cycles.setter
    def control_cycles(self, value):
        self.context.float_variable_data.set_value(
            self.context.control_cycle_variable, value
        )

    def compile(self, motion_statechart: MotionStatechart):
        self.motion_statechart = motion_statechart
        self.control_cycles = 0
        self.motion_statechart.compile(self.context)
        self._compile_qp_controller(self.context.qp_controller_config)
        if self.trajectory_plotter is not None:
            self.trajectory_plotter.reset(self.context.world.state, self.time)
        if self.debug_expression_plotter is not None:
            self.debug_expression_plotter.reset(
                self.motion_statechart.collect_debug_expressions()
            )
            self.debug_expression_plotter.debug_expression_trajectory.append(self.time)
        self.context.collision_manager.update_collision_matrix()
        # do one tick to immediately active nodes whose start condition is constant true.
        self.motion_statechart.tick(self.context)

    def tick(self):
        self.control_cycles += 1
        if self.context.requires_collision_checking:
            self.context.collision_manager.compute_collisions()
        self.motion_statechart.tick(self.context)
        if self.debug_expression_plotter is not None:
            self.debug_expression_plotter.debug_expression_trajectory.append(self.time)
        if self.qp_controller is None:
            return
        next_cmd = self.qp_controller.compute_command(
            world_state=self.context.world.state._data,
            life_cycle_state=self.motion_statechart.life_cycle_state.data,
            float_variables=self.context.float_variable_data.data,
        )
        self.context.world.apply_control_commands(
            next_cmd,
            self.qp_controller.config.control_dt,
            self.qp_controller.config.max_derivative,
        )
        if self.trajectory_plotter is not None:
            self.trajectory_plotter.world_state_trajectory.append(
                self.context.world.state, self.time
            )

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
            self.set_velocity_acceleration_jerk_to_zero()
            self.motion_statechart.cleanup_nodes(context=self.context)
            self.context.cleanup()

    def set_velocity_acceleration_jerk_to_zero(self):
        """
        Clear all commanded derivatives of the world state.
        """
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

    def plot_debug_expressions(self, file_name: str = "./debug_expressions.pdf"):
        """
        Plot the recorded debug expressions to the given PDF file.
        """
        if self.debug_expression_plotter is None:
            raise PlotterNotConfiguredError("debug expression plotter")
        self.debug_expression_plotter.plot(file_name)
