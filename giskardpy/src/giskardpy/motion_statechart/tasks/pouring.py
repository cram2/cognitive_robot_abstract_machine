"""
Giskardpy Tasks for the pouring domain (D_pour) of the BMP framework.

Both the fill-level ODE (Torricelli) and the tilt goal are expressed as
symbolic QP constraints in ``build()``. ``on_tick()`` is a pure observation
check; it performs no integration and no QP manipulation.
"""

from __future__ import annotations

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
from semantic_digital_twin.physics.pouring_equations import (
    PouringEquation,
)
from semantic_digital_twin.world_description.geometry import ContainerGeometry
from semantic_digital_twin.spatial_types import Point3
from semantic_digital_twin.world_description.connections import Connection
from semantic_digital_twin.world_description.world_entity import Body


@dataclass(eq=False, repr=False)
class PouringTask(Task):
    """
    MSC Task that controls tilt and fill level via two symbolic QP constraints.

    ``build()`` registers:

    1. A velocity equality constraint on the fill joint encoding the fill-level ODE.
    2. A gradient-driven tilt goal: it targets a desired outflow velocity
       on the ODE symbolic expression, forcing the QP to find the necessary
       tilt joint velocities to achieve that outflow. Outflow stops once the
       fill level reaches ``goal_value``.
    """

    fill_equation: PouringEquation
    """Pouring ODE coupling the tilt joint to the fill-level DOF."""

    goal_value: float
    """Target fill level to achieve."""

    tolerance: float
    """Acceptance band around goal_value."""

    target_outflow: float = field(default=0.05, kw_only=True)
    """Desired rate of decrease for the normalized fill level."""

    weight: float = field(default=DefaultWeights.WEIGHT_BELOW_CA, kw_only=True)
    """QP constraint weight for the tilt-driving gradient."""

    max_velocity: float = field(default=1.0, kw_only=True)
    """Maximum velocity for the tilt joint in rad/s."""

    def build(self, context: MotionStatechartContext) -> NodeArtifacts:
        """
        Register the fill ODE velocity constraint and the gradient-driven tilt constraint.

        :param context: MSC build context.
        :return: NodeArtifacts with one velocity constraint (fill) and one gradient
                 constraint (tilt-driving).
        """
        artifacts = NodeArtifacts()
        fill_connection = self.fill_equation.fill_connection
        fill_sym = fill_connection.dof.variables.position
        tilt_connection = self.fill_equation.tilt_connection

        self.fill_vel_ode = self.fill_equation.symbolic_velocity()

        artifacts.constraints.add_equality_constraint(
            name=f"{tilt_connection.name} 3",
            equality_bound=sm.Scalar(self.goal_value) - fill_sym,
            quadratic_weight=DefaultWeights.WEIGHT_ABOVE_CA,
            task_expression=fill_sym
            + self.fill_vel_ode,  # This is linear approximation of fill sym as a function of fill and tilt. (one-step-ahead prediction, can be scaled by multiplication)
            reference_velocity=self.target_outflow,
        )
        return artifacts

    def on_tick(
        self, context: MotionStatechartContext
    ) -> Optional[ObservationStateValues]:
        """
        Return TRUE when fill reaches the goal tolerance band, else None.

        The QP fill-velocity constraint ensures outflow stops once the tilt returns
        upright, so checking only the fill level is sufficient.

        :param context: MSC runtime context (unused; state is read directly from connections).
        :return: ``TRUE`` when fill ≤ goal + tolerance, else ``None``.
        """
        fill = float(self.fill_equation.fill_connection.position)
        outflow = float(self.fill_vel_ode.evaluate()[0])
        if fill <= self.goal_value + self.tolerance and outflow <= 0.0:
            return ObservationStateValues.TRUE
        return None


@dataclass(eq=False, repr=False)
class CoupledPouringTask(PouringTask):
    """
    Extends :class:`PouringTask` with kinematic rim-positioning constraints.

    Receiver fill is driven by :class:`~semantic_digital_twin.physics.pouring_equations.InflowEquation`
    wired on the receiver's :class:`~semantic_digital_twin.world_description.connections.LiquidConnection`
    before the simulation starts, so no QP constraint for receiver fill is needed here.

    :param receiver_fill_connection: Fill-level DOF of the receiving container.
    :param root_link: Root of the kinematic chain that positions the source cup.
    :param tip_link: Tip body (cup body) of the kinematic chain.
    :param source_geometry: Physical dimensions of the source container.
    :param receiver_target: World-frame [x, y, z] target for the spilling rim
        (= receiver opening xyz + height_above_receiver in z).
    :param goal_receiver_fill: Target normalised fill level for the receiver.
    """

    receiver_fill_connection: Connection = field(kw_only=True)
    root_link: Body = field(kw_only=True)
    tip_link: Body = field(kw_only=True)
    source_geometry: ContainerGeometry = field(kw_only=True)
    receiver_target: np.ndarray = field(kw_only=True)
    goal_receiver_fill: float = field(kw_only=True)

    def build(self, context: MotionStatechartContext) -> NodeArtifacts:
        """
        Build source constraints via super(), then add rim positioning constraints.

        :param context: MSC build context.
        :return: NodeArtifacts with source tilt gradient and rim constraints.
        """
        artifacts = super().build(context)

        r = self.source_geometry.half_width
        A = self.source_geometry.height

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
                quadratic_weight=DefaultWeights.WEIGHT_BELOW_CA * 0.1,
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
