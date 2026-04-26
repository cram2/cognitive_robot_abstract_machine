"""
Physics models for the pouring domain (D_pour).

Implements Φ_pour using MotionStatechart with PouringTask / CoupledPouringTask
that integrates the fill-level ODE and controls the tilt joint via QP.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from giskardpy.executor import Executor, SimulationPacer
from giskardpy.motion_statechart.context import MotionStatechartContext
from giskardpy.motion_statechart.graph_node import EndMotion
from giskardpy.motion_statechart.monitors.monitors import LocalMinimumReached
from giskardpy.motion_statechart.motion_statechart import MotionStatechart
from giskardpy.motion_statechart.tasks.pouring import CoupledPouringTask, PouringTask
from giskardpy.qp import qp_controller
from giskardpy.qp.qp_controller_config import QPControllerConfig

from pycram.body_motion_problem.types import (
    Effect,
    PhysicsModel,
)
from semantic_digital_twin.physics.pouring_equations import (
    HasFillLevel,
    PouringEquation,
)
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import Connection
from semantic_digital_twin.world_description.world_entity import Body


@dataclass
class PouringMSCModel(PhysicsModel):
    """
    Concrete physics model Φ_pour: execute a PouringTask MotionStatechart against a World.

    The MotionStatechart contains a single :class:`PouringTask` that integrates the
    fill-level ODE and drives the tilt joint via QP. This model compiles the MSC,
    ticks it until EndMotion, records both the tilt and fill trajectories, and
    resets the World state before returning.

    ODE parameters and both connections are carried by :attr:`fill_equation`.
    """

    fill_equation: PouringEquation
    """Pouring ODE — owns the tilt connection, fill connection, and k."""

    target_outflow: float = field(default=0.05)
    """Desired rate of decrease for the normalized fill level."""

    initial_tilt: float = field(default=0.1)
    """Initial tilt angle in radians to help the gradient-driven task."""

    timeout: int = field(default=900)
    """Maximum number of ticks before aborting the rollout."""

    last_fill_trajectory: List[float] = field(default_factory=list, init=False)
    """Fill-level trajectory from the most recent :meth:`run` call."""

    def run(self, effect: Effect, world: World) -> Tuple[Optional[List[float]], bool]:
        """
        Simulate pouring via the PouringTask MotionStatechart and return the tilt trajectory.

        :param effect: Desired fill-level effect; provides goal_value and tolerance.
        :param world: World whose state is simulated and reset on return.
        :return: ``(tilt_trajectory, achieved)`` — tilt samples at each tick and success flag.
        """
        msc = self._build_msc(effect)
        context = MotionStatechartContext(
            world=world,
            qp_controller_config=QPControllerConfig(
                target_frequency=50,
                prediction_horizon=30,
            ),
        )
        executor = Executor(
            context=context,
            pacer=SimulationPacer(real_time_factor=1.0),
        )
        executor.compile(motion_statechart=msc)

        tilt_connection = self.fill_equation.tilt_connection
        fill_connection = self.fill_equation.fill_connection

        tilt_trajectory: List[float] = []
        fill_trajectory: List[float] = []
        with world.reset_state_context():
            # Start with a small tilt to jump-start the gradient-driven task
            world.set_positions_1DOF_connection({tilt_connection: self.initial_tilt})

            for _ in range(self.timeout):
                executor.tick()
                tilt_trajectory.append(float(tilt_connection.position))
                fill_trajectory.append(float(fill_connection.position))
                if msc.is_end_motion():
                    break

            achieved = effect.is_achieved()

        context.cleanup()
        self.last_fill_trajectory = fill_trajectory
        return tilt_trajectory, achieved

    def build_secondary_trajectories(
        self, effect: Effect
    ) -> List[Tuple[Connection, List[float]]]:
        """
        Return the fill-level trajectory recorded by the most recent run().

        :param effect: Effect whose target_object provides the fill connection.
        :return: Single-entry list of (fill_connection, fill_level_positions).
        """
        return [(effect.target_object.fill_connection, self.last_fill_trajectory)]

    def _build_msc(self, effect: Effect) -> MotionStatechart:
        """
        Build the MotionStatechart for a single pouring manoeuvre.

        :param effect: Effect providing goal_value and tolerance for PouringTask.
        :return: Compiled-ready MotionStatechart with PouringTask and EndMotion.
        """
        msc = MotionStatechart()
        task = PouringTask(
            fill_equation=self.fill_equation,
            goal_value=effect.goal_value,
            tolerance=effect.tolerance,
            target_outflow=self.target_outflow,
        )
        msc.add_node(task)
        msc.add_node(local_min := LocalMinimumReached())
        msc.add_node(EndMotion.when_true(local_min))
        return msc


@dataclass
class CoupledPouringMSCModel(PhysicsModel):
    """
    Physics model for the coupled pouring chain: source → receiver.

    Runs a :class:`~giskardpy.motion_statechart.tasks.pouring.CoupledPouringTask`
    MotionStatechart that drives tilt, source fill, receiver fill, and cup body
    position simultaneously via QP constraints.

    :param source: Semantic annotation carrying ``fill_equation``, ``fill_connection``,
        ``container_geometry``, and ``root`` (cup body = tip link).
    :param receiver: Semantic annotation carrying ``fill_connection``,
        ``container_geometry``, and ``root`` (used to read world position).
    :param root_link: Root of the kinematic chain that positions the source cup.
    :param target_outflow: Desired rate of decrease for the normalized fill level.
    :param initial_tilt: Initial tilt angle in radians to help the gradient-driven task.
    :param timeout: Maximum simulation ticks before aborting.
    :param height_above_receiver: Desired rim clearance above the receiver opening (m).
    """

    source: HasFillLevel
    receiver: HasFillLevel
    root_link: Body
    target_outflow: float = field(default=0.05)
    initial_tilt: float = field(default=0.1)
    timeout: int = field(default=2000)
    height_above_receiver: float = field(default=0.1)

    last_source_fill_trajectory: List[float] = field(default_factory=list, init=False)
    last_receiver_fill_trajectory: List[float] = field(default_factory=list, init=False)
    _last_chain_trajectories: Dict[Connection, List[float]] = field(
        default_factory=dict, init=False
    )

    def run(self, effect: Effect, world: World) -> Tuple[Optional[List[float]], bool]:
        """
        Simulate the full coupled pouring chain and return the tilt trajectory.

        :param effect: ReceiverFillEffect targeting the receiver container.
        :param world: World whose state is simulated and reset on return.
        :return: ``(tilt_trajectory, achieved)`` — tilt samples and whether the
            receiver fill reached the effect's goal value.
        """
        self.receiver.inflow_equation.source = self.source
        msc = self._build_msc(effect)
        context = MotionStatechartContext(world=world)
        executor = Executor(
            context=context, pacer=SimulationPacer(real_time_factor=1.0)
        )
        executor.compile(motion_statechart=msc)

        tilt_connection = self.source.fill_equation.tilt_connection
        source_fill_connection = self.source.fill_connection
        receiver_fill_connection = self.receiver.fill_connection
        chain_connections = self._get_kinematic_chain_connections()

        tilt_traj: List[float] = []
        source_fill_traj: List[float] = []
        receiver_fill_traj: List[float] = []
        chain_trajs: Dict[Connection, List[float]] = {c: [] for c in chain_connections}

        with world.reset_state_context():
            # Start with a small tilt to jump-start the gradient-driven task
            world.set_positions_1DOF_connection({tilt_connection: self.initial_tilt})

            for _ in range(self.timeout):
                executor.tick()
                tilt_traj.append(float(tilt_connection.position))
                source_fill_traj.append(float(source_fill_connection.position))
                receiver_fill_traj.append(float(receiver_fill_connection.position))
                for conn in chain_connections:
                    chain_trajs[conn].append(float(conn.position))
                if msc.is_end_motion():
                    break
            achieved = effect.is_achieved()

        context.cleanup()
        self.receiver.inflow_equation.source = None
        self.last_source_fill_trajectory = source_fill_traj
        self.last_receiver_fill_trajectory = receiver_fill_traj
        self._last_chain_trajectories = chain_trajs
        return tilt_traj, achieved

    def build_secondary_trajectories(
        self, effect: Effect
    ) -> List[Tuple[Connection, List[float]]]:
        """
        Return secondary trajectories recorded by the most recent run().

        :param effect: Effect (unused; receiver fill connection is taken from ``self.receiver``).
        :return: Source fill, receiver fill, and all kinematic chain DOF trajectories.
        """
        result: List[Tuple[Connection, List[float]]] = [
            (self.source.fill_connection, self.last_source_fill_trajectory),
            (self.receiver.fill_connection, self.last_receiver_fill_trajectory),
        ]
        result.extend(self._last_chain_trajectories.items())
        return result

    def _get_kinematic_chain_connections(self) -> List[Connection]:
        """Collect all connections from just above the tilt joint up to root_link."""
        connections: List[Connection] = []
        current = self.source.fill_equation.tilt_connection.parent
        while current is not None and current != self.root_link:
            conn = current.parent_connection
            connections.append(conn)
            current = conn.parent
        return connections

    def _build_msc(self, effect: Effect) -> MotionStatechart:
        """
        Build the MotionStatechart for the full coupled pouring manoeuvre.

        :param effect: ReceiverFillEffect providing goal_value and tolerance.
        :return: Compiled-ready MotionStatechart with CoupledPouringTask and EndMotion.
        """
        receiver_center = self.receiver.root.global_transform.to_position().to_np()[:3]
        receiver_target = receiver_center.copy()
        receiver_target[2] += (
            self.receiver.container_geometry.height / 2 + self.height_above_receiver
        )

        msc = MotionStatechart()
        task = CoupledPouringTask(
            fill_equation=self.source.fill_equation,
            goal_value=0.0,
            tolerance=effect.tolerance,
            target_outflow=self.target_outflow,
            receiver_fill_connection=self.receiver.fill_connection,
            root_link=self.root_link,
            tip_link=self.source.root,
            source_geometry=self.source.container_geometry,
            receiver_target=receiver_target,
            goal_receiver_fill=effect.goal_value,
        )
        msc.add_node(task)
        msc.add_node(EndMotion.when_true(task))
        return msc
