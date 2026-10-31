from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from cramph.context import StatechartContext
from cramph.data_types import LifeCycleValues
from cramph.executor import StatechartExecutor, TickingExecutorExtension
from giskardpy.motion_statechart.context import MotionControlContext
from giskardpy.motion_statechart.exceptions import WorldStateArrayReplacedError
from giskardpy.motion_statechart.graph_node import (
    DebugExpression,
    MotionStatechartNode,
)
from giskardpy.motion_statechart.plotters.debug_expression_trajectory_plotter import (
    DebugExpressionTrajectoryPlotter,
)
from giskardpy.qp.constraint_collection import ConstraintCollection
from giskardpy.qp.exceptions import EmptyProblemException
from giskardpy.qp.qp_controller import QPController
from giskardpy.qp.qp_controller_config import QPControllerConfig
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.world_state_trajectory_plotter import (
    WorldStateTrajectoryPlotter,
)

# %% motion control


@dataclass
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

    _compiled_world_state_data: np.ndarray | None = field(default=None, init=False)
    """
    The world state array the motion statechart was compiled against.

    The compiled updaters read it through a memory view, so it must stay the very same
    array for as long as they are in use.
    """

    def extend_context(self, context: StatechartContext) -> None:
        context.add_extension(
            MotionControlContext(qp_controller_config=self.qp_controller_config)
        )
        context.set_tick_duration(
            self.qp_controller_config.control_time_step.total_seconds()
        )

    def before_recompile(self, executor: StatechartExecutor) -> bool:
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

    def after_compile(self, executor: StatechartExecutor) -> None:
        self._holding_still = False
        world = executor.context.world
        self._compiled_world_state_data = world.state._data
        self._compile_qp_controller(executor)
        world.collision_manager.update_collision_matrix()

    def before_tick(self, executor: StatechartExecutor) -> None:
        self._holding_still = False
        self._raise_if_world_state_array_was_replaced(executor.context.world)
        if executor.context.require_extension(
            MotionControlContext
        ).requires_collision_checking:
            executor.context.world.collision_manager.compute_collisions()

    def after_tick(self, executor: StatechartExecutor) -> None:
        if self.qp_controller is None:
            return
        world = executor.context.world
        next_command = self.qp_controller.compute_command(
            world_state=world.state._data,
            life_cycle_state=self._life_cycle_state_followed(executor),
            float_variables=executor.context.float_variable_data.data,
        )
        world.apply_control_commands(
            next_command,
            self.qp_controller.config.control_time_step.total_seconds(),
            self.qp_controller.config.max_derivative,
        )

    def _life_cycle_state_followed(self, executor: StatechartExecutor) -> np.ndarray:
        """
        :return: The life cycle state the commands follow: the statechart's own, or,
            while holding still, one in which no node runs, so no constraint of the
            statechart applies.
        """
        life_cycle_state = executor.statechart.life_cycle_state.data
        if not self._holding_still:
            return life_cycle_state
        return np.full_like(life_cycle_state, LifeCycleValues.NOT_STARTED)

    def after_run(self, executor: StatechartExecutor) -> None:
        self.set_velocity_acceleration_jerk_to_zero(executor.context.world)

    @staticmethod
    def set_velocity_acceleration_jerk_to_zero(world: World) -> None:
        """
        Clear all commanded derivatives of the world state.

        :param world: The world whose commanded derivatives are cleared.
        """
        world.state.velocities[:] = 0
        world.state.accelerations[:] = 0
        world.state.jerks[:] = 0

    def _raise_if_world_state_array_was_replaced(self, world: World) -> None:
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

    def _compile_qp_controller(self, executor: StatechartExecutor) -> None:
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
        constraint_collection = self._combine_constraint_collections_of_nodes(executor)
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
        if self.qp_controller.has_not_free_variables():
            raise EmptyProblemException()

    @staticmethod
    def _combine_constraint_collections_of_nodes(
        executor: StatechartExecutor,
    ) -> ConstraintCollection:
        """
        :param executor: The executor whose statechart was compiled.
        :return: The constraint collections of all motion nodes, merged into one, with
            each node's constraints prefixed by its
            :attr:`~cramph.node.StatechartNode.unique_name`.
        """
        combined_constraint_collection = ConstraintCollection()
        for node in executor.statechart.get_nodes_by_type(MotionStatechartNode):
            combined_constraint_collection.merge(
                name_prefix=node.unique_name, other=node.constraint_collection
            )
        return combined_constraint_collection


# %% recordings


@dataclass
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

    def after_compile(self, executor: StatechartExecutor) -> None:
        self.plotter.reset(executor.context.world.state, executor.time)

    def after_recompile(self, executor: StatechartExecutor) -> None:
        """
        Keeps the recorded trajectory, unlike :meth:`after_compile`.
        """

    def after_tick(self, executor: StatechartExecutor) -> None:
        """
        Appends the world state to the recorded trajectory.

        .. note:: A trajectory holds one set of degrees of freedom, so a world whose
            degrees of freedom changed starts the trajectory anew.
        """
        state = executor.context.world.state
        trajectory = self.plotter.world_state_trajectory
        if not trajectory.records_degrees_of_freedom_of(state):
            self.plotter.reset(state, executor.time)
            return
        trajectory.adopt_world_model_of(state)
        trajectory.append(state, executor.time)


@dataclass
class DebugExpressionRecording(TickingExecutorExtension):
    """
    Records the value of every debug expression of the statechart after every tick.

    .. note:: List it after :class:`MotionControl`, so its values belong to the same
        world state a :class:`WorldStateTrajectoryRecording` records for that time.
    """

    plotter: DebugExpressionTrajectoryPlotter = field(
        default_factory=DebugExpressionTrajectoryPlotter
    )
    """
    Holds the recorded values and plots them.
    """

    def after_compile(self, executor: StatechartExecutor) -> None:
        self.plotter.reset(DebugExpression.collect_from(executor.statechart))
        self.plotter.debug_expression_trajectory.append(executor.time)

    def after_recompile(self, executor: StatechartExecutor) -> None:
        """
        Keeps the recorded values, unlike :meth:`after_compile`, and records the debug
        expressions the statechart has now, see :meth:`~giskardpy.motion_statechart.debu
        g_expression_trajectory.DebugExpressionTrajectory.follow`.
        """
        self.plotter.debug_expression_trajectory.follow(
            DebugExpression.collect_from(executor.statechart)
        )

    def after_tick(self, executor: StatechartExecutor) -> None:
        self.plotter.debug_expression_trajectory.append(executor.time)
