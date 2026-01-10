import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, InitVar

from typing_extensions import Optional

from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.robots.abstract_robot import (
    AbstractRobot,
)
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.world_state import WorldStateTrajectory
from semantic_digital_twin.world_description.world_state_trajectory_plotter import (
    WorldStateTrajectoryPlotter,
)
from .data_types.exceptions import NoQPControllerConfigException
from .model.better_pybullet_syncer import BulletCollisionDetector
from .model.collision_world_syncer import (
    CollisionWorldSynchronizer,
    CollisionCheckerLib,
)
from .model.collisions import NullCollisionDetector
from .motion_statechart.auxilary_variable_manager import (
    AuxiliaryVariableManager,
    AuxiliaryVariable,
)
from .motion_statechart.context import BuildContext, ExecutionContext
from .motion_statechart.motion_statechart import MotionStatechart
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

    world: World
    """The world object containing the state and entities of the robot's environment."""
    controller_config: QPControllerConfig = field(
        default_factory=QPControllerConfig.create_with_simulation_defaults
    )
    """Optional configuration for the QP Controller. Is only needed when constraints are present in the motion statechart."""
    collision_checker: InitVar[CollisionCheckerLib] = field(
        default=CollisionCheckerLib.none
    )
    """Library used for collision checking. Can be set to Bullet or None."""
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
    collision_scene: Optional[CollisionWorldSynchronizer] = field(
        default=None, init=False
    )
    """The collision scene synchronizer for managing robot collision states."""
    auxiliary_variable_manager: AuxiliaryVariableManager = field(
        default_factory=AuxiliaryVariableManager, init=False
    )
    """Manages auxiliary symbolic variables for execution contexts."""
    qp_controller: Optional[QPController] = field(default=None, init=False)
    """Optional quadratic programming controller used for motion control."""

    control_cycles: int = field(init=False)
    """Tracks the number of control cycles elapsed during execution."""
    _control_cycles_variable: AuxiliaryVariable = field(init=False)
    """Auxiliary variable linked to the control_cycles attribute."""

    _time_variable: AuxiliaryVariable = field(init=False)
    """Auxiliary variable representing the current time in seconds since the start of the simulation."""

    @property
    def time(self) -> float:
        return self.control_cycles * self.controller_config.control_dt

    def __post_init__(self, collision_checker: CollisionCheckerLib):
        if collision_checker == CollisionCheckerLib.bpb:
            collision_detector = BulletCollisionDetector(
                _world=self.world, tmp_folder=self.tmp_folder
            )
        else:
            collision_detector = NullCollisionDetector(_world=self.world)

        self.collision_scene = CollisionWorldSynchronizer(
            world=self.world,
            robots=self.world.get_semantic_annotations_by_type(AbstractRobot),
            collision_detector=collision_detector,
        )
        self.pacer.target_frequency = self.controller_config.target_frequency

    def _create_control_cycles_variable(self):
        self._control_cycles_variable = (
            self.auxiliary_variable_manager.create_float_variable(
                PrefixedName("control_cycles"), lambda: self.control_cycles
            )
        )

    def compile(self, motion_statechart: MotionStatechart):
        self.motion_statechart = motion_statechart
        self.control_cycles = 0
        self._create_control_cycles_variable()
        self.motion_statechart.compile(self.build_context)
        self._compile_qp_controller(self.controller_config)
        self.world_state_trajectory = WorldStateTrajectory.from_world_state(
            self.world.state, time=self.time
        )

    @property
    def build_context(self) -> BuildContext:
        return BuildContext(
            world=self.world,
            auxiliary_variable_manager=self.auxiliary_variable_manager,
            collision_scene=self.collision_scene,
            qp_controller_config=self.controller_config,
            control_cycle_variable=self._control_cycles_variable,
        )

    @property
    def execution_context(self) -> ExecutionContext:
        return ExecutionContext(
            world=self.world,
            external_collision_data_data=self.collision_scene.get_external_collision_data(),
            self_collision_data_data=self.collision_scene.get_self_collision_data(),
            auxiliar_variables_data=self.auxiliary_variable_manager.resolve_auxiliary_variables(),
            control_cycle_counter=self.control_cycles,
        )

    def tick(self):
        self.control_cycles += 1
        self.collision_scene.sync()
        self.collision_scene.check_collisions()
        execution_context = self.execution_context
        self.motion_statechart.tick(execution_context)
        if self.qp_controller is None:
            return
        next_cmd = self.qp_controller.get_cmd(
            world_state=self.world.state.data,
            life_cycle_state=self.motion_statechart.life_cycle_state.data,
            external_collisions=execution_context.external_collision_data_data,
            self_collisions=execution_context.self_collision_data_data,
            auxiliary_variables=execution_context.auxiliar_variables_data,
        )
        self.world.apply_control_commands(
            next_cmd,
            self.qp_controller.config.control_dt,
            self.qp_controller.config.max_derivative,
        )
        self.world_state_trajectory.append(self.world.state, self.time)

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
        self.world.state.velocities[:] = 0
        self.world.state.accelerations[:] = 0
        self.world.state.jerks[:] = 0

    def _compile_qp_controller(self, controller_config: QPControllerConfig):
        ordered_dofs = sorted(
            self.world.active_degrees_of_freedom,
            key=lambda dof: self.world.state._index[dof.id],
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
            world_state_symbols=self.world.state.get_variables(),
            life_cycle_variables=self.motion_statechart.life_cycle_state.life_cycle_symbols(),
            external_collision_avoidance_variables=self.collision_scene.get_external_collision_symbol(),
            self_collision_avoidance_variables=self.collision_scene.get_self_collision_symbol(),
            auxiliary_variables=self.auxiliary_variable_manager.variables,
        )
        if self.qp_controller.has_not_free_variables():
            raise EmptyProblemException()

    def plot_trajectory(self, file_name: str = "./trajectory.pdf"):
        self.trajectory_plotter.plot_trajectory(self.world_state_trajectory, file_name)
