"""
Giskardpy Tasks for the pouring domain (D_pour) of the BMP framework.

Both the fill-level ODE (Torricelli) and the tilt goal are expressed as
symbolic QP constraints in ``build()``. ``on_tick()`` is a pure observation
check; it performs no integration and no QP manipulation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import krrood.symbolic_math.symbolic_math as sm
from giskardpy.motion_statechart.context import MotionStatechartContext
from giskardpy.motion_statechart.data_types import (
    DefaultWeights,
    ObservationStateValues,
)
from giskardpy.motion_statechart.graph_node import NodeArtifacts, Task
from semantic_digital_twin.reasoning.body_motion_problem.pouring.articulated import (
    PouringEquation,
)
from semantic_digital_twin.semantic_annotations.mixins import ContainerGeometry
from semantic_digital_twin.spatial_types import Point3
from semantic_digital_twin.world_description.connections import Connection
from semantic_digital_twin.world_description.world_entity import Body


@dataclass(eq=False, repr=False)
class PouringTask(Task):
    """
    MSC Task that controls tilt and fill level via two symbolic QP constraints.

    ``build()`` registers:

    1. A velocity equality constraint on the fill joint encoding the fill-level ODE.
    2. A position equality constraint on the tilt joint whose target is a
       continuous function of the fill level: tilt ramps linearly from ``theta_max``
       (high fill) to ``0`` (fill ≤ goal).  ``ramp_margin`` controls how many
       fill-level units above the goal the ramp-down begins.

    ``on_tick()`` only reads state and returns the terminal observation; it does not
    integrate the ODE or manipulate QP float variables.
    """

    fill_equation: PouringEquation
    """Pouring ODE coupling the tilt joint to the fill-level DOF."""

    goal_value: float
    """Target fill level to achieve."""

    tolerance: float
    """Acceptance band around goal_value."""

    theta_max: float = field(default=math.pi / 3, kw_only=True)
    """Maximum tilt angle in radians."""

    ramp_margin: float = field(default=0.15, kw_only=True)
    """
    Fill-level units above goal_value at which tilt ramp-down begins.
    Larger values start the ramp earlier, reducing overshoot at the cost of slower pour.
    """

    weight: float = field(default=DefaultWeights.WEIGHT_BELOW_CA, kw_only=True)
    """QP constraint weight for the tilt joint."""

    max_velocity: float = field(default=1.0, kw_only=True)
    """Maximum velocity for the tilt joint in rad/s."""

    def build(self, context: MotionStatechartContext) -> NodeArtifacts:
        """
        Register the fill ODE velocity constraint and the symbolic tilt goal constraint.

        :param context: MSC build context providing world state and the QP config.
        :return: NodeArtifacts with one velocity constraint (fill) and one equality
                 constraint (tilt).
        """
        artifacts = NodeArtifacts()
        tilt_connection = self.fill_equation.tilt_connection
        fill_connection = self.fill_equation.fill_connection

        current_tilt = tilt_connection.dof.variables.position
        fill_sym = fill_connection.dof.variables.position

        # Constraint 1: fill velocity follows the Torricelli ODE.
        fill_vel_goal = self.fill_equation.symbolic_velocity(fill_sym, current_tilt)
        artifacts.constraints.add_velocity_eq_constraint(
            velocity_goal=fill_vel_goal,
            quadratic_weight=DefaultWeights.WEIGHT_ABOVE_CA,
            task_expression=fill_sym,
            velocity_limit=self.fill_equation.k,
            name=str(fill_connection.name),
        )

        # Constraint 2: tilt goal is a continuous function of fill level.
        tilt_goal = self._tilt_goal_expression(fill_sym)

        velocity_upper = tilt_connection.dof.limits.upper.velocity
        velocity_lower = tilt_connection.dof.limits.lower.velocity
        if velocity_lower is not None and velocity_upper is not None:
            velocity = sm.limit(self.max_velocity, velocity_lower, velocity_upper)
        else:
            velocity = self.max_velocity

        if not tilt_connection.dof.has_position_limits():
            tilt_error = sm.shortest_angular_distance(current_tilt, tilt_goal)
        else:
            tilt_error = tilt_goal - current_tilt

        artifacts.constraints.add_equality_constraint(
            name=str(tilt_connection.name),
            reference_velocity=velocity,
            equality_bound=tilt_error,
            quadratic_weight=self.weight,
            task_expression=current_tilt,
        )
        return artifacts

    def _tilt_goal_expression(self, fill_sym) -> sm.Expression:
        """
        Symbolic tilt target as a function of fill level, capped at theta_max.

        :param fill_sym: Symbolic fill-level variable.
        :return: Symbolic expression for the desired tilt angle.
        """
        tilt_fraction = sm.limit(
            (fill_sym - self.goal_value) / self.ramp_margin,
            sm.Scalar(0.0),
            sm.Scalar(1.0),
        )
        theta_floor = sm.limit(
            self.fill_equation.symbolic_tilt_floor(fill_sym),
            sm.Scalar(0.0),
            sm.Scalar(self.theta_max),
        )
        return (
            tilt_fraction * self.theta_max
            + (sm.Scalar(1.0) - tilt_fraction) * theta_floor
        )

    def on_tick(
        self, context: MotionStatechartContext
    ) -> Optional[ObservationStateValues]:
        """
        Return TRUE when fill reaches the goal tolerance band, else None.

        The QP fill-velocity constraint ensures outflow stops once the tilt returns
        below ``theta_threshold``, so checking only the fill level is sufficient.

        :param context: MSC runtime context (unused; state is read directly from connections).
        :return: ``TRUE`` when fill ≤ goal + tolerance, else ``None``.
        """
        fill = float(self.fill_equation.fill_connection.position)
        if fill <= self.goal_value + self.tolerance:
            return ObservationStateValues.TRUE
        return None


@dataclass(eq=False, repr=False)
class CoupledPouringTask(PouringTask):
    """
    Extends :class:`PouringTask` with receiver fill and kinematic rim-positioning constraints.

    Adds two groups of QP constraints on top of the source fill ODE and tilt ramp:

    1. Receiver fill velocity: ``ḣ_r = max(0, −ḣ_s) · (h_s·r_s) / (h_r·r_r)``
    2. Rim position: full FK from ``root_link`` to ``tip_link`` keeps the spilling rim
       at ``receiver_target`` by jointly optimising all DOFs in the kinematic chain.

    :param receiver_fill_connection: Fill-level DOF of the receiving container.
    :param root_link: Root of the kinematic chain that positions the source cup.
    :param tip_link: Tip body (cup body) of the kinematic chain.
    :param source_geometry: Physical dimensions of the source container.
    :param receiver_geometry: Physical dimensions of the receiver container.
    :param receiver_target: World-frame [x, y, z] target for the spilling rim
        (= receiver opening xyz + height_above_receiver in z).
    :param goal_receiver_fill: Target normalised fill level for the receiver.
    """

    receiver_fill_connection: Connection = field(kw_only=True)
    root_link: Body = field(kw_only=True)
    tip_link: Body = field(kw_only=True)
    source_geometry: ContainerGeometry = field(kw_only=True)
    receiver_geometry: ContainerGeometry = field(kw_only=True)
    receiver_target: np.ndarray = field(kw_only=True)
    goal_receiver_fill: float = field(kw_only=True)

    def _tilt_goal_expression(self, fill_sym) -> sm.Expression:
        """
        Tilt target driven by receiver fill proximity to goal, capped at theta_max.

        When receiver fill is far below goal, tilt stays at theta_max.
        As receiver fill approaches goal within ramp_margin, tilt ramps back toward
        theta_floor, so outflow decelerates before the goal is reached.

        :param fill_sym: Symbolic source fill-level variable (used for theta_floor).
        :return: Symbolic expression for the desired tilt angle.
        """
        receiver_fill_sym = self.receiver_fill_connection.dof.variables.position
        tilt_fraction = sm.limit(
            (self.goal_receiver_fill - self.tolerance - receiver_fill_sym)
            / self.ramp_margin,
            sm.Scalar(0.0),
            sm.Scalar(1.0),
        )
        theta_floor = sm.limit(
            self.fill_equation.symbolic_tilt_floor(fill_sym),
            sm.Scalar(0.0),
            sm.Scalar(self.theta_max),
        )
        return (
            tilt_fraction * self.theta_max
            + (sm.Scalar(1.0) - tilt_fraction) * theta_floor
        )

    def build(self, context: MotionStatechartContext) -> NodeArtifacts:
        """
        Build source constraints via super(), then add receiver fill and rim positioning constraints.

        :param context: MSC build context.
        :return: NodeArtifacts with source fill ODE, tilt ramp, receiver fill ODE, and rim constraints.
        """
        artifacts = super().build(context)

        source_fill_sym = self.fill_equation.fill_connection.dof.variables.position
        tilt_sym = self.fill_equation.tilt_connection.dof.variables.position
        receiver_fill_sym = self.receiver_fill_connection.dof.variables.position

        r = self.source_geometry.half_width
        A = self.source_geometry.height

        source_vel = self.fill_equation.symbolic_velocity(source_fill_sym, tilt_sym)
        outflow = sm.max(sm.Scalar(0.0), -source_vel)
        volume_ratio = (r * A) / (
            self.receiver_geometry.half_width * self.receiver_geometry.height
        )

        artifacts.constraints.add_velocity_eq_constraint(
            velocity_goal=outflow * volume_ratio,
            quadratic_weight=DefaultWeights.WEIGHT_ABOVE_CA,
            task_expression=receiver_fill_sym,
            velocity_limit=self.fill_equation.k * volume_ratio,
            name=str(self.receiver_fill_connection.name),
        )

        root_T_tip = context.world.compose_forward_kinematics_expression(
            self.root_link, self.tip_link
        )
        tip_P_rim = Point3(r, 0.0, A, reference_frame=self.tip_link)
        root_P_rim = root_T_tip @ tip_P_rim

        for axis, rim_sym, target in [
            ("x", root_P_rim.x, float(self.receiver_target[0])),
            ("y", root_P_rim.y, float(self.receiver_target[1])),
            ("z", root_P_rim.z, float(self.receiver_target[2])),
        ]:
            artifacts.constraints.add_equality_constraint(
                name=f"{self.tip_link.name}_rim_{axis}",
                reference_velocity=1.0,
                equality_bound=sm.Scalar(target) - rim_sym,
                quadratic_weight=DefaultWeights.WEIGHT_BELOW_CA,
                task_expression=rim_sym,
            )
        return artifacts

    def on_tick(
        self, context: MotionStatechartContext
    ) -> Optional[ObservationStateValues]:
        """
        Return TRUE when the receiver fill reaches the goal, else None.

        :param context: MSC runtime context (unused).
        :return: ``TRUE`` when receiver fill ≥ goal_receiver_fill − tolerance, else ``None``.
        """
        fill = float(self.receiver_fill_connection.position)
        if fill >= self.goal_receiver_fill - self.tolerance:
            return ObservationStateValues.TRUE
        return None
