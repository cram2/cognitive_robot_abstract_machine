"""
Physics model for the pouring domain (D_pour).

Implements Φ_pour using a MotionStatechart with a PouringTask that integrates
the fill-level ODE and controls the tilt joint via QP.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
from giskardpy.executor import Executor, SimulationPacer
from giskardpy.motion_statechart.context import MotionStatechartContext
from giskardpy.motion_statechart.graph_node import EndMotion
from giskardpy.motion_statechart.motion_statechart import MotionStatechart
from giskardpy.motion_statechart.tasks.pouring import CoupledPouringTask, PouringTask

from semantic_digital_twin.reasoning.body_motion_problem.pouring.articulated import (
    PouringEquation,
)
from semantic_digital_twin.reasoning.body_motion_problem.types import (
    Effect,
    PhysicsModel,
)
from semantic_digital_twin.semantic_annotations.mixins import ContainerGeometry
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import Connection


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

    theta_max: float = field(default=1.0471975511965976)  # math.pi / 3
    """Maximum tilt angle in radians."""

    ramp_margin: float = field(default=0.15)
    """
    Fill-level units above goal_value at which tilt ramp-down begins.
    Passed directly to :class:`PouringTask`.
    """

    timeout: int = field(default=500)
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
        context = MotionStatechartContext(world=world)
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
            for _ in range(self.timeout):
                executor.tick()
                tilt_trajectory.append(float(tilt_connection.position))
                fill_trajectory.append(float(fill_connection.position))
                if msc.is_end_motion():
                    break

            achieved = effect.is_achieved()

        self.last_fill_trajectory = fill_trajectory
        return tilt_trajectory, achieved

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
            theta_max=self.theta_max,
            ramp_margin=self.ramp_margin,
        )
        msc.add_node(task)
        msc.add_node(EndMotion.when_true(task))
        return msc


@dataclass
class CoupledPouringMSCModel(PhysicsModel):
    """
    Physics model for the coupled pouring chain: source → receiver.

    Runs a :class:`~giskardpy.motion_statechart.tasks.pouring.CoupledPouringTask`
    MotionStatechart that drives tilt, source fill, receiver fill, and cup body
    position simultaneously via QP constraints. The kinematic rim-positioning
    constraints keep the spilling rim above the receiver at every tick.

    :param source_model: Configured source-side physics model; used as a config
        holder for ``fill_equation``, ``theta_max``, ``ramp_margin``, and ``timeout``.
    :param source_geometry: Interior dimensions of the source cup.
    :param receiver_geometry: Interior dimensions of the receiver cup.
    :param x_translation_connection: Prismatic x-DOF of the source cup body.
    :param z_translation_connection: Prismatic z-DOF of the source cup body.
    :param receiver_opening_world: World-frame xyz of the receiver opening top-centre.
    :param height_above_receiver: Desired rim clearance above the receiver opening.
    """

    source_model: PouringMSCModel
    source_geometry: ContainerGeometry
    receiver_geometry: ContainerGeometry
    x_translation_connection: Connection
    z_translation_connection: Connection
    receiver_opening_world: np.ndarray
    height_above_receiver: float = field(default=0.1)

    last_receiver_fill_trajectory: List[float] = field(default_factory=list, init=False)
    last_x_trajectory: List[float] = field(default_factory=list, init=False)
    last_z_trajectory: List[float] = field(default_factory=list, init=False)

    def run(self, effect: Effect, world: World) -> Tuple[Optional[List[float]], bool]:
        """
        Simulate the full coupled pouring chain and return the tilt trajectory.

        :param effect: ReceiverFillEffect targeting the receiver container.
        :param world: World whose state is simulated and reset on return.
        :return: ``(tilt_trajectory, achieved)`` — tilt samples and whether the
            receiver fill reached the effect's goal value.
        """
        msc = self._build_msc(effect)
        context = MotionStatechartContext(world=world)
        executor = Executor(
            context=context, pacer=SimulationPacer(real_time_factor=1.0)
        )
        executor.compile(motion_statechart=msc)

        tilt_connection = self.source_model.fill_equation.tilt_connection
        source_fill_connection = self.source_model.fill_equation.fill_connection
        receiver_fill_connection = effect.target_object.fill_connection

        tilt_traj: List[float] = []
        source_fill_traj: List[float] = []
        receiver_fill_traj: List[float] = []
        x_traj: List[float] = []
        z_traj: List[float] = []

        with world.reset_state_context():
            for _ in range(self.source_model.timeout):
                executor.tick()
                tilt_traj.append(float(tilt_connection.position))
                source_fill_traj.append(float(source_fill_connection.position))
                receiver_fill_traj.append(float(receiver_fill_connection.position))
                x_traj.append(float(self.x_translation_connection.position))
                z_traj.append(float(self.z_translation_connection.position))
                if msc.is_end_motion():
                    break
            achieved = effect.is_achieved()

        self.source_model.last_fill_trajectory = source_fill_traj
        self.last_receiver_fill_trajectory = receiver_fill_traj
        self.last_x_trajectory = x_traj
        self.last_z_trajectory = z_traj
        return tilt_traj, achieved

    def _build_msc(self, effect: Effect) -> MotionStatechart:
        """
        Build the MotionStatechart for the full coupled pouring manoeuvre.

        :param effect: ReceiverFillEffect providing goal_value and tolerance.
        :return: Compiled-ready MotionStatechart with CoupledPouringTask and EndMotion.
        """
        msc = MotionStatechart()
        task = CoupledPouringTask(
            fill_equation=self.source_model.fill_equation,
            goal_value=0.0,
            tolerance=effect.tolerance,
            theta_max=self.source_model.theta_max,
            ramp_margin=self.source_model.ramp_margin,
            receiver_fill_connection=effect.target_object.fill_connection,
            x_translation_connection=self.x_translation_connection,
            z_translation_connection=self.z_translation_connection,
            source_geometry=self.source_geometry,
            receiver_geometry=self.receiver_geometry,
            receiver_x=float(self.receiver_opening_world[0]),
            receiver_z_rim=float(self.receiver_opening_world[2])
            + self.height_above_receiver,
            goal_receiver_fill=effect.goal_value,
        )
        msc.add_node(task)
        msc.add_node(EndMotion.when_true(task))
        return msc
