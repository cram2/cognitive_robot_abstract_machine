from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import ClassVar, Optional

from krrood.symbolic_math.symbolic_math import Scalar

from giskardpy.motion_statechart.context import MotionStatechartContext
from giskardpy.motion_statechart.data_types import (
    DefaultWeights,
    ObservationStateValues,
)
from giskardpy.motion_statechart.exceptions import NodeInitializationError
from giskardpy.motion_statechart.graph_node import (
    DebugExpression,
    NodeArtifacts,
    Task,
)
from semantic_digital_twin.physics.equations.pouring_equations import (
    PouringEquation,
    SymbolicFillContext,
    tilt_expression_from_fk,
)
from semantic_digital_twin.semantic_annotations.mixins import HasFillLevel, LiquidSource
from semantic_digital_twin.spatial_types.spatial_types import Point3
from semantic_digital_twin.world_description.connections import LiquidConnection
from semantic_digital_twin.world_description.geometry import Color
from semantic_digital_twin.world_description.world_entity import Body


@dataclass(eq=False, repr=False)
class TerminalFillConstraintTask(Task, ABC):
    """
    Base for tasks that drive a container's predicted terminal fill level to a goal.

    Subclasses resolve the fill connection and build the symbolic fill-velocity ODE; this base
    linearizes that ODE into the terminal-state prediction constraint and reports convergence once
    the fill reaches the goal and its rate has settled to zero.
    """

    goal_value: float
    """Target fill level to achieve in terms of percentage."""

    fill_level_tolerance: float
    """Tolerance threshold around :attr:`goal_value`."""

    outflow_tolerance: float = field(default=0.001, kw_only=True)
    """Tolerance threshold around zero for the residual fill rate."""

    reference_velocity: float = field(default=0.05, kw_only=True)
    """Desired rate of change of the normalized fill level."""

    weight: float = field(default=DefaultWeights.WEIGHT_ABOVE_CA, kw_only=True)
    """QP constraint weight for the fill-driving gradient."""

    @abstractmethod
    def _resolve_fill_connection(
        self, context: MotionStatechartContext
    ) -> LiquidConnection:
        """
        Resolves and validates the live fill connection whose DOF position the constraint drives.

        :param context: The build context.
        :return: The world-resident fill connection.
        """

    @abstractmethod
    def _fill_velocity(self, context: MotionStatechartContext) -> Scalar:
        """
        Builds the symbolic fill-velocity ODE to linearize; :attr:`fill_connection` is resolved.

        :param context: The build context.
        :return: Symbolic normalized fill velocity at the current operating point.
        """

    @abstractmethod
    def _fill_goal_reached(self, fill_level: float) -> bool:
        """
        Whether the fill level has reached the goal in the task's fill direction.

        :param fill_level: The current normalized fill level.
        """

    def build(self, context: MotionStatechartContext) -> NodeArtifacts:
        """
        Linearizes the fill ODE into a single terminal-state prediction row over the horizon.

        The fill ODE is linearized at the current operating point and its discrete-time recursion
        unrolled analytically, so the resulting QP row drives the MPC-predicted terminal fill toward
        :attr:`goal_value`.  Because the row couples earlier velocity decisions to a larger share of
        the predicted change, the optimizer eases off before overshooting rather than reacting late.

        :param context: The build context.
        :return: The generated task artifacts.
        """
        artifacts = NodeArtifacts()
        self.fill_connection = self._resolve_fill_connection(context)
        self.fill_vel_ode = self._fill_velocity(context)
        artifacts.constraints.add_terminal_state_prediction_constraint(
            name=f"{self.fill_connection.name}",
            state_velocity=self.fill_vel_ode,
            state_variable=self.fill_connection.dof.variables.position,
            goal_value=self.goal_value,
            quadratic_weight=self.weight,
            reference_velocity=self.reference_velocity,
        )
        return artifacts

    def on_tick(
        self, context: MotionStatechartContext
    ) -> Optional[ObservationStateValues]:
        """
        Reports success once the fill reaches the goal and its rate has settled to zero.

        :param context: The runtime context.
        :return: The observation state.
        """
        fill_level = float(self.fill_connection.position)
        fill_rate = float(self.fill_vel_ode.evaluate()[0])
        rate_settled = -self.outflow_tolerance < fill_rate < self.outflow_tolerance
        if rate_settled and self._fill_goal_reached(fill_level):
            return ObservationStateValues.TRUE
        return None


@dataclass(eq=False, repr=False)
class PouringTask(TerminalFillConstraintTask):
    """
    Motion Statechart task for controlling the tilt and fill level of a held container.

    Tilts a container the robot holds so its own fill level drains toward :attr:`goal_value`; the
    pouring ODE couples the controlled tilt to the passive fill DOF.
    """

    fill_equation: PouringEquation
    """Pouring ODE coupling tilt to the fill-level DOF."""

    fill_connection: LiquidConnection
    """Virtual DOF whose position encodes fill level in [0, 1]."""

    root_link: Body = field(kw_only=True)
    """Root of the kinematic chain used to derive the cup tilt expression; must be the world root."""

    tip_link: Body = field(kw_only=True)
    """Tip of the kinematic chain (the cup body)."""

    def _resolve_fill_connection(
        self, context: MotionStatechartContext
    ) -> LiquidConnection:
        """
        :raises NodeInitializationError: if ``root_link`` is not the world root, since the tilt
            expression is only valid relative to the vertical world-root frame.
        """
        if self.root_link is not context.world.root:
            raise NodeInitializationError(
                self,
                "root_link must be the world root; the cup tilt is derived against the vertical "
                "world-root frame and is otherwise mispredicted",
            )
        return context.world.get_connection(
            self.fill_connection.parent, self.fill_connection.child
        )

    def _fill_velocity(self, context: MotionStatechartContext) -> Scalar:
        root_T_tip = context.world.compose_forward_kinematics_expression(
            self.root_link, self.tip_link
        )
        tilt_expression = tilt_expression_from_fk(root_T_tip)
        return self.fill_equation.symbolic_velocity(
            SymbolicFillContext(
                tilt_expression=tilt_expression,
                fill_position=self.fill_connection.dof.variables.position,
            )
        )

    def _fill_goal_reached(self, fill_level: float) -> bool:
        return fill_level <= self.goal_value + self.fill_level_tolerance


@dataclass(eq=False, repr=False)
class FillByTransferTask(TerminalFillConstraintTask):
    """
    Motion Statechart task that fills a receiver by tilting a separate source container.

    Unlike :class:`PouringTask`, the controlled degrees of freedom (the arm holding the source)
    do not belong to the container whose fill level is the goal.  The receiver's inflow ODE
    depends symbolically on the source arm configuration through the gated source outflow, so
    driving the receiver's predicted terminal fill toward the goal makes the optimizer tilt and
    position the source.
    """

    receiver: HasFillLevel
    """The container whose fill level is driven up to :attr:`goal_value`."""

    def _resolve_fill_connection(
        self, context: MotionStatechartContext
    ) -> LiquidConnection:
        """
        :raises NodeInitializationError: if the receiver has no inflow equation, meaning
            ``receive_outflow_from`` was not called to couple it to a source.
        """
        self.receiver.ensure_inflow_coupling(context.world)
        fill_connection = context.world.get_connection(
            self.receiver.fill_connection.parent, self.receiver.fill_connection.child
        )
        if fill_connection.inflow_equation is None:
            raise NodeInitializationError(
                self, "receiver has no inflow equation; call receive_outflow_from first"
            )
        return fill_connection

    def _fill_velocity(self, context: MotionStatechartContext) -> Scalar:
        self.inflow_equation = self.fill_connection.inflow_equation
        return self.inflow_equation.symbolic_velocity(self.fill_connection)

    def _fill_goal_reached(self, fill_level: float) -> bool:
        return fill_level >= self.goal_value - self.fill_level_tolerance


@dataclass(eq=False, repr=False)
class KeepProjectileInReceiver(Task):
    """
    Positions the source so the poured liquid's projectile lands in the receiver opening.

    Drives the predicted projectile landing point of the source's pour toward the receiver's
    opening centre, so as the source tilts the optimizer moves the gripper to keep the liquid
    landing inside the receiver — the no-spill counterpart to :class:`FillByTransferTask`.
    """

    receiver: HasFillLevel
    """The container the liquid must land in; must already be coupled via ``receive_outflow_from``."""

    source: LiquidSource
    """The liquid source being poured from."""

    reference_velocity: float = field(default=0.2, kw_only=True)
    """Reference velocity for normalization in m/s."""

    threshold: float = field(default=0.02, kw_only=True)
    """Distance threshold for the landing point to count as inside the opening, in metres."""

    weight: float = field(default=DefaultWeights.WEIGHT_ABOVE_CA, kw_only=True)
    """QP constraint weight for the landing-point goal."""

    EXIT_POINT_COLOR: ClassVar[Color] = Color(R=0.0, G=0.6, B=1.0, A=1.0)
    """Color of the exit-point marker (blue)."""

    LANDING_POINT_COLOR: ClassVar[Color] = Color(R=1.0, G=0.0, B=0.0, A=1.0)
    """Color of the landing-point marker (red)."""

    def build(self, context: MotionStatechartContext) -> NodeArtifacts:
        """
        Creates the constraint driving the pour's projectile landing point to the receiver opening.

        :param context: The build context.
        :return: The generated task artifacts.
        """
        artifacts = NodeArtifacts()
        self.receiver.ensure_inflow_coupling(context.world)
        inflow_equation = self.receiver.fill_connection.inflow_equation
        if inflow_equation is None:
            raise NodeInitializationError(
                self, "receiver has no inflow equation; call receive_outflow_from first"
            )
        exit_speed = self.source.current_outflow_velocity(context.world)
        if exit_speed is None:
            exit_speed = inflow_equation.exit_speed
        landing_point = self.receiver.projectile_landing_point(
            self.source, context.world, exit_speed
        )
        receiver_opening = context.world.compose_forward_kinematics_expression(
            context.world.root, self.receiver.root
        ).to_position()
        artifacts.geometry.add_point_goal_constraints(
            name=f"{self.receiver.root.name}_projectile",
            frame_P_goal=receiver_opening,
            frame_P_current=landing_point,
            reference_velocity=self.reference_velocity,
            quadratic_weight=self.weight,
        )
        artifacts.debug_expressions.extend(
            self._build_visualization_debug_expressions(context, landing_point)
        )
        artifacts.observation = (
            receiver_opening.euclidean_distance(landing_point) < self.threshold
        )
        return artifacts

    def _build_visualization_debug_expressions(
        self, context: MotionStatechartContext, landing_point: Point3
    ) -> list[DebugExpression]:
        """
        Build the debug expressions that visualize where the pour leaves and where it lands.

        :param context: The build context.
        :param landing_point: The projectile landing point on the receiver's opening plane.
        :return: Debug expressions for the exit point and the landing point.
        """
        exit_point = self.source.liquid_exit_point(context.world)
        return [
            DebugExpression(
                name="exit", expression=exit_point, color=self.EXIT_POINT_COLOR
            ),
            DebugExpression(
                name="landing", expression=landing_point, color=self.LANDING_POINT_COLOR
            ),
        ]
