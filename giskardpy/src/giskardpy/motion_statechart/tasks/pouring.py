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
    tilt_expression_from_fk,
)
from semantic_digital_twin.world_description.geometry import ContainerGeometry
from semantic_digital_twin.spatial_types import Point3
from semantic_digital_twin.world_description.connections import Connection
from semantic_digital_twin.world_description.world_entity import Body


@dataclass(eq=False, repr=False)
class PouringTask(Task):
    """
    Motion Statechart task for controlling the tilt and fill level of a container.
    """

    fill_equation: PouringEquation
    """Pouring ODE coupling tilt to the fill-level DOF."""

    root_link: Body = field(kw_only=True)
    """Root of the kinematic chain used to derive the cup tilt expression."""

    tip_link: Body = field(kw_only=True)
    """Tip of the kinematic chain (the cup body)."""

    goal_value: float
    """Target fill level to achieve."""

    tolerance: float
    """Acceptance band around goal_value."""

    reference_velocity: float = field(default=0.05, kw_only=True)
    """Desired rate of decrease for the normalized fill level."""

    weight: float = field(default=DefaultWeights.WEIGHT_ABOVE_CA, kw_only=True)
    """QP constraint weight for the tilt-driving gradient."""

    def build(self, context: MotionStatechartContext) -> NodeArtifacts:
        """
        Creates the constraints for the fill level and the tilt angle.

        :param context: The build context.
        :return: The generated task artifacts.
        """
        artifacts = NodeArtifacts()
        fill_connection = self.fill_equation.fill_connection
        fill_sym = fill_connection.dof.variables.position

        root_T_tip = context.world.compose_forward_kinematics_expression(
            self.root_link, self.tip_link
        )
        tilt_expr = tilt_expression_from_fk(root_T_tip)
        self.fill_vel_ode = self.fill_equation.symbolic_velocity(tilt_expr)

        artifacts.constraints.add_equality_constraint(
            name=f"{fill_connection.name}",
            equality_bound=sm.Scalar(self.goal_value) - fill_sym,
            quadratic_weight=self.weight,
            task_expression=fill_sym
            + self.fill_vel_ode,  # This is a linear approximation of fill sym as a function of fill and tilt.
            reference_velocity=self.reference_velocity,
        )
        return artifacts

    def on_tick(
        self, context: MotionStatechartContext
    ) -> Optional[ObservationStateValues]:
        """
        Checks if the goal fill level has been reached and that the outflow is zero.

        :param context: The runtime context.
        :return: The observation state.
        """
        fill = float(self.fill_equation.fill_connection.position)
        outflow = float(self.fill_vel_ode.evaluate()[0])
        if fill <= self.goal_value + self.tolerance and outflow <= 0.0:
            return ObservationStateValues.TRUE
        return None


@dataclass(eq=False, repr=False)
class CoupledPouringTask(PouringTask):
    """
    Motion Statechart task for pouring from one container into another.
    """

    receiver_fill_connection: Connection = field(kw_only=True)
    source_geometry: ContainerGeometry = field(kw_only=True)
    receiver_target: np.ndarray = field(kw_only=True)
    goal_receiver_fill: float = field(kw_only=True)

    def build(self, context: MotionStatechartContext) -> NodeArtifacts:
        """
        Creates the pouring and rim positioning constraints.

        :param context: The build context.
        :return: The generated task artifacts.
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
        Checks if the receiver goal fill level has been reached.

        :param context: The runtime context.
        :return: The observation state.
        """
        fill = float(self.receiver_fill_connection.position)
        if fill >= self.goal_receiver_fill - self.tolerance:
            return ObservationStateValues.TRUE
        return None
