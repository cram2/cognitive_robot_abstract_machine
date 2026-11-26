"""
The executor extensions that turn a motion statechart into commands and record what it
did.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from cramph.context import StatechartContext
from cramph.data_types import LifeCycleValue
from cramph.executor import (
    StatechartExecutor,
    SteppedSimulationPacer,
    TickingExecutorExtension,
)
from giskardpy.motion_statechart.context import MotionControlContext
from giskardpy.motion_statechart.exceptions import WorldStateArrayReplacedError
from giskardpy.motion_statechart.graph_node import MotionStatechartNode
from giskardpy.qp.constraint_collection import ConstraintCollection
from giskardpy.qp.qp_controller import QPController
from giskardpy.qp.qp_controller_config import QPControllerConfig
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.degree_of_freedom import DegreeOfFreedom
from semantic_digital_twin.world_description.world_state_trajectory_plotter import (
    WorldStateTrajectoryPlotter,
)

# %% motion control


@dataclass(kw_only=True)
class MotionControl(TickingExecutorExtension):
    """
    Turns the constraints of the motion statechart nodes into commands and applies them
    to the world after every tick.
    """

    qp_controller_config: QPControllerConfig = field(
        default_factory=QPControllerConfig.create_with_simulation_defaults
    )
    """
    The configuration of the QP controller, whose control time step is the duration of
    one tick.
    """

    speed_at_rest: float = 1e-3
    """
    The fastest a degree of freedom may move for the robot to count as at rest.
    """

    # %% init False
    qp_controller: QPController | None = field(default=None, init=False)
    """
    The controller computing the commands, None while no node adds constraints.
    """

    _holding_still: bool = field(default=False, init=False)
    """
    Whether the statechart asked in this tick to hold still, so the commands follow no
    node of it and the controller brings the robot to rest.
    """

    _measured_degrees_of_freedom: list[DegreeOfFreedom] = field(
        default_factory=list, init=False
    )
    """
    The degrees of freedom whose position the commands leave where it is: while a
    physics steps the world, every one outside the robots, such as a door hinge, which
    moves only as far as a robot pushes it, not to where the controller predicts it.
    """

    _compiled_world_state_data: np.ndarray | None = field(default=None, init=False)
    """
    The world state array the motion statechart was compiled against.

    The compiled updaters read it through a memory view, so it must stay the very same
    array for as long as they are in use.
    """

    def extend_context(self, *, context: StatechartContext) -> None:
        context.add_extension(
            extension=MotionControlContext(
                qp_controller_config=self.qp_controller_config
            )
        )
        context.set_tick_duration(
            tick_duration=self.qp_controller_config.control_time_step.total_seconds()
        )

    def before_recompile(self, *, executor: StatechartExecutor) -> bool:
        """
        Stop following the statechart for this tick, so the controller slows the robot
        down within its limits, since no command is computed while the statechart blocks
        its tick.

        :return: Whether every degree of freedom the controller commands is at rest.
        """
        self._holding_still = True
        if self.qp_controller is None:
            return True
        state = executor.context.world.state
        return all(
            abs(state[degree_of_freedom.id].velocity) <= self.speed_at_rest
            for degree_of_freedom in self.qp_controller.active_degrees_of_freedom
        )

    def after_compile(self, *, executor: StatechartExecutor) -> None:
        """
        Compile :attr:`qp_controller` and decide which degrees of freedom the commands
        leave to the physics.
        """
        self._holding_still = False
        world = executor.context.world
        self._compiled_world_state_data = world.state._data
        self._measured_degrees_of_freedom = []
        if isinstance(executor.pacer, SteppedSimulationPacer):
            robot_bodies = world.robot_body_to_robot_mapping
            self._measured_degrees_of_freedom = [
                degree_of_freedom
                for connection in world.connections
                if connection.child not in robot_bodies
                for degree_of_freedom in connection.active_dofs
            ]
        self._compile_qp_controller(executor=executor)
        world.collision_manager.update_collision_matrix()

    def before_tick(self, *, executor: StatechartExecutor) -> None:
        self._holding_still = False
        self._raise_if_world_state_array_was_replaced(world=executor.context.world)
        if executor.context.require_extension(
            extension_type=MotionControlContext
        ).requires_collision_checking:
            executor.context.world.collision_manager.compute_collisions()

    def after_tick(self, *, executor: StatechartExecutor) -> None:
        """
        Apply the command :attr:`qp_controller` computes for the state the tick left.
        """
        if self.qp_controller is None:
            return
        world = executor.context.world
        next_command = self.qp_controller.compute_command(
            world_state=world.state._data,
            life_cycle_state=self._life_cycle_state_followed(executor=executor),
            float_variables=executor.context.float_variable_data.data,
        )
        world.apply_control_commands(
            commands=next_command,
            dt=self.qp_controller.config.control_time_step.total_seconds(),
            derivative=self.qp_controller.config.max_derivative,
            measured_degrees_of_freedom=self._measured_degrees_of_freedom,
        )

    def _life_cycle_state_followed(self, *, executor: StatechartExecutor) -> np.ndarray:
        """
        :return: The life cycle state the commands follow: the statechart's own, or,
            while holding still, one in which no node runs, so no constraint of the
            statechart applies.
        """
        life_cycle_state = executor.statechart.life_cycle_state.data
        if not self._holding_still:
            return life_cycle_state
        return np.full_like(life_cycle_state, LifeCycleValue.NOT_STARTED)

    def after_run(self, *, executor: StatechartExecutor) -> None:
        self.set_velocity_acceleration_jerk_to_zero(world=executor.context.world)

    @staticmethod
    def set_velocity_acceleration_jerk_to_zero(*, world: World) -> None:
        """
        Clear all commanded derivatives of the world state.

        :param world: The world whose commanded derivatives are cleared.
        """
        world.state.velocities[:] = 0
        world.state.accelerations[:] = 0
        world.state.jerks[:] = 0

    def _raise_if_world_state_array_was_replaced(self, *, world: World) -> None:
        """
        Ensures the world still holds the state array the motion statechart compiled
        against.

        :param world: The world the motion statechart was compiled against.
        :raises WorldStateArrayReplacedError: If the world replaced its state array,
            which leaves the compiled updaters reading a detached copy of the state.
        """
        if self._compiled_world_state_data is None:
            return
        if world.state._data is self._compiled_world_state_data:
            return
        raise WorldStateArrayReplacedError(
            compiled_degrees_of_freedom=self._compiled_world_state_data.shape[1],
            current_degrees_of_freedom=world.state._data.shape[1],
        )

    def _compile_qp_controller(self, *, executor: StatechartExecutor) -> None:
        """
        Builds :attr:`qp_controller` from the constraints of the compiled statechart.

        :param executor: The executor whose statechart was compiled.
        :raises EmptyProblemException: If the constraints do not depend on any degree of
            freedom.
        """
        world = executor.context.world
        ordered_degrees_of_freedom = sorted(
            world.active_degrees_of_freedom,
            key=lambda degree_of_freedom: world.state._index[degree_of_freedom.id],
        )
        constraint_collection = self._combine_constraint_collections_of_nodes(
            executor=executor
        )
        if len(constraint_collection._constraints) == 0:
            self.qp_controller = None
            # to not build controller, if there are no constraints
            return
        self.qp_controller = QPController(
            config=self.qp_controller_config,
            degrees_of_freedom=ordered_degrees_of_freedom,
            constraint_collection=constraint_collection,
            world_state_symbols=world.state.get_variables(),
            life_cycle_variables=executor.statechart.life_cycle_state.life_cycle_symbols(),
            float_variables=executor.context.float_variable_data.variables,
        )

    @staticmethod
    def _combine_constraint_collections_of_nodes(
        *,
        executor: StatechartExecutor,
    ) -> ConstraintCollection:
        """
        :param executor: The executor whose statechart was compiled.
        :return: The constraint collections of all motion nodes, merged into one, with
            each node's constraints prefixed by its
            :attr:`~cramph.node.StatechartNode.unique_name`.
        """
        combined_constraint_collection = ConstraintCollection()
        for node in executor.statechart.get_nodes_by_type(
            node_type=MotionStatechartNode
        ):
            combined_constraint_collection.merge(
                name_prefix=node.unique_name, other=node.constraint_collection
            )
        return combined_constraint_collection


# %% recordings


@dataclass(kw_only=True)
class WorldStateTrajectoryRecording(TickingExecutorExtension):
    """
    Records the world state every tick left behind.

    .. note:: List it after :class:`MotionControl`, so it records the state after the
        commands of the tick were applied.
    """

    plotter: WorldStateTrajectoryPlotter = field(
        default_factory=WorldStateTrajectoryPlotter
    )
    """
    Holds the recorded trajectory and plots it.
    """

    appends_every_tick: bool = True
    """
    Whether every tick appends its world state; while off, the trajectory keeps only the
    state it started with.
    """

    def start_trajectory(self, *, executor: StatechartExecutor) -> None:
        """
        Drops the recorded trajectory and starts a new one with the current world state,
        whether or not :attr:`appends_every_tick` is on.
        """
        self.plotter.reset(executor.context.world.state, executor.time)

    def after_compile(self, *, executor: StatechartExecutor) -> None:
        """
        Starts a new trajectory, see :meth:`start_trajectory`.
        """
        self.start_trajectory(executor=executor)

    def after_recompile(self, *, executor: StatechartExecutor) -> None:
        """
        Keeps the recorded trajectory, unlike :meth:`after_compile`.
        """

    def after_tick(self, *, executor: StatechartExecutor) -> None:
        """
        Appends the world state to the recorded trajectory.

        .. note:: A trajectory holds one set of degrees of freedom, so a world whose
            degrees of freedom changed starts the trajectory anew.
        """
        if not self.appends_every_tick:
            return
        state = executor.context.world.state
        trajectory = self.plotter.world_state_trajectory
        if not trajectory.records_degrees_of_freedom_of(state):
            self.plotter.reset(state, executor.time)
            return
        trajectory.adopt_world_model_of(state)
        trajectory.append(state, executor.time)
