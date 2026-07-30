from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import ClassVar, Optional

import krrood.symbolic_math.symbolic_math as sm
from krrood.symbolic_math.symbolic_math import (
    CompiledFunction,
    Scalar,
    VariableParameters,
)

from giskardpy.motion_statechart.context import MotionStatechartContext
from giskardpy.motion_statechart.data_types import (
    DefaultWeights,
    ObservationStateValues,
)
from giskardpy.motion_statechart.exceptions import (
    MissingExitSpeedError,
    MissingInflowEquationError,
    NonPositiveClearanceError,
    RootLinkNotWorldRootError,
)
from giskardpy.motion_statechart.graph_node import (
    DebugExpression,
    NodeArtifacts,
    Task,
)
from semantic_digital_twin.physics.equations.pouring_equations import (
    GatedInflowEquation,
    PouringEquation,
    SymbolicFillContext,
    tilt_expression_from_fk,
)
from semantic_digital_twin.semantic_annotations.mixins import HasFillLevel, LiquidSource
from semantic_digital_twin.spatial_types.spatial_types import Point3, Vector3
from semantic_digital_twin.world_description.connections import LiquidConnection
from semantic_digital_twin.world_description.geometry import Color
from semantic_digital_twin.world_description.world_entity import Body


@dataclass(eq=False, repr=False)
class TerminalFillConstraintTask(Task, ABC):
    """
    Base for tasks that drive a container's predicted terminal fill level to a goal.

    Subclasses resolve the fill connection and build the symbolic fill-velocity ODE;
    this base linearizes that ODE into the terminal-state prediction constraint and
    reports convergence once the fill reaches the goal and its rate has settled to zero.
    """

    goal_value: float
    """
    Target fill level to achieve in terms of percentage.
    """

    fill_level_tolerance: float
    """
    Tolerance threshold around :attr:`goal_value`.
    """

    outflow_tolerance: float = field(default=0.001, kw_only=True)
    """
    Tolerance threshold around zero for the residual fill rate.
    """

    reference_velocity: float = field(default=0.05, kw_only=True)
    """
    Desired rate of change of the normalized fill level.
    """

    weight: float = field(
        default=DefaultWeights.WEIGHT_ABOVE_COLLISION_AVOIDANCE, kw_only=True
    )
    """
    QP constraint weight for the fill-driving gradient.
    """

    fill_connection: LiquidConnection = field(init=False)
    """
    World-resident fill connection resolved by :meth:`build`.
    """

    fill_velocity_expression: Scalar = field(init=False)
    """
    Symbolic fill-velocity ODE built by :meth:`build`.
    """

    _compiled_fill_velocity: CompiledFunction = field(init=False, repr=False)
    """
    Compiled :attr:`fill_velocity_expression`, evaluated once per tick.
    """

    @abstractmethod
    def _resolve_fill_connection(
        self, context: MotionStatechartContext
    ) -> LiquidConnection:
        """
        Resolves and validates the live fill connection whose DOF position the
        constraint drives.

        :param context: The build context.
        :return: The world-resident fill connection.
        """

    @abstractmethod
    def _fill_velocity(self, context: MotionStatechartContext) -> Scalar:
        """
        Builds the symbolic fill-velocity ODE to linearize; :attr:`fill_connection` is
        resolved.

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
        Linearizes the fill ODE into a single terminal-state prediction row over the
        horizon.

        The fill ODE is linearized at the current operating point and its discrete-time
        recursion unrolled analytically, so the resulting QP row drives the MPC-
        predicted terminal fill toward :attr:`goal_value`.  Because the row couples
        earlier velocity decisions to a larger share of the predicted change, the
        optimizer eases off before overshooting rather than reacting late.

        :param context: The build context.
        :return: The generated task artifacts.
        """
        artifacts = NodeArtifacts()
        self.fill_connection = self._resolve_fill_connection(context)
        self.fill_velocity_expression = self._fill_velocity(context)
        self._compiled_fill_velocity = self.fill_velocity_expression.compile(
            parameters=VariableParameters.from_lists(
                context.world.state.position_float_variables,
                context.float_variable_data.variables,
            ),
            sparse=False,
        )
        self._compiled_fill_velocity.bind_args_to_memory_view(
            0, context.world.state.positions
        )
        self._compiled_fill_velocity.bind_args_to_memory_view(
            1, context.float_variable_data.data
        )
        artifacts.constraints.add_terminal_state_prediction_constraint(
            name=f"{self.fill_connection.name}",
            state_velocity=self.fill_velocity_expression,
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
        fill_rate = float(self._compiled_fill_velocity.evaluate()[0])
        rate_settled = -self.outflow_tolerance < fill_rate < self.outflow_tolerance
        if rate_settled and self._fill_goal_reached(fill_level):
            return ObservationStateValues.TRUE
        return None


@dataclass(eq=False, repr=False)
class PouringTask(TerminalFillConstraintTask):
    """
    Motion Statechart task for controlling the tilt and fill level of a held container.

    Tilts a container the robot holds so its own fill level drains toward
    :attr:`goal_value`; the pouring ODE couples the controlled tilt to the passive fill
    DOF.
    """

    fill_equation: PouringEquation
    """
    Pouring ODE coupling tilt to the fill-level DOF.
    """

    fill_connection: LiquidConnection
    """
    Virtual DOF whose position encodes fill level in [0, 1].
    """

    root_link: Body = field(kw_only=True)
    """
    Root of the kinematic chain used to derive the cup tilt expression; must be the
    world root.
    """

    tip_link: Body = field(kw_only=True)
    """
    Tip of the kinematic chain (the cup body).
    """

    def _resolve_fill_connection(
        self, context: MotionStatechartContext
    ) -> LiquidConnection:
        """
        :raises RootLinkNotWorldRootError: if ``root_link`` is not the world root, since the tilt
            expression is only valid relative to the vertical world-root frame.
        """
        if self.root_link is not context.world.root:
            raise RootLinkNotWorldRootError(
                node=self, root_link=self.root_link, world_root=context.world.root
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

    Unlike :class:`PouringTask`, the controlled degrees of freedom (the arm holding the
    source) do not belong to the container whose fill level is the goal.  The receiver's
    inflow ODE depends symbolically on the source arm configuration through the gated
    source outflow, so driving the receiver's predicted terminal fill toward the goal
    makes the optimizer tilt and position the source.
    """

    receiver: HasFillLevel
    """
    The container whose fill level is driven up to :attr:`goal_value`.
    """

    def _resolve_fill_connection(
        self, context: MotionStatechartContext
    ) -> LiquidConnection:
        """
        :raises MissingInflowEquationError: if the receiver has no inflow equation, meaning
            ``receive_outflow_from`` was not called to couple it to a source.
        """
        self.receiver.ensure_inflow_coupling(context.world)
        fill_connection = context.world.get_connection(
            self.receiver.fill_connection.parent, self.receiver.fill_connection.child
        )
        if fill_connection.inflow_equation is None:
            raise MissingInflowEquationError(node=self)
        return fill_connection

    def _fill_velocity(self, context: MotionStatechartContext) -> Scalar:
        inflow_equation = self.fill_connection.inflow_equation
        return inflow_equation.symbolic_velocity(self.fill_connection)

    def _fill_goal_reached(self, fill_level: float) -> bool:
        return fill_level >= self.goal_value - self.fill_level_tolerance


@dataclass(eq=False, repr=False)
class KeepProjectileInReceiver(Task):
    """
    Positions the source so the poured liquid's projectile lands in the receiver
    opening.

    Drives the predicted projectile landing point of the source's pour toward the
    receiver's opening centre, so as the source tilts the optimizer moves the gripper to
    keep the liquid landing inside the receiver — the no-spill counterpart to
    :class:`FillByTransferTask`.
    """

    receiver: HasFillLevel
    """
    The container the liquid must land in; must already be coupled via
    ``receive_outflow_from``.
    """

    source: LiquidSource
    """
    The liquid source being poured from.
    """

    reference_velocity: float = field(default=0.2, kw_only=True)
    """
    Reference velocity for normalization in m/s.
    """

    threshold: float = field(default=0.02, kw_only=True)
    """
    Distance threshold for the landing point to count as inside the opening, in metres.
    """

    weight: float = field(
        default=DefaultWeights.WEIGHT_ABOVE_COLLISION_AVOIDANCE, kw_only=True
    )
    """
    QP constraint weight for the landing-point goal.
    """

    EXIT_POINT_COLOR: ClassVar[Color] = Color(R=0.0, G=0.6, B=1.0, A=1.0)
    """
    Color of the exit-point marker (blue).
    """

    LANDING_POINT_COLOR: ClassVar[Color] = Color(R=1.0, G=0.0, B=0.0, A=1.0)
    """
    Color of the landing-point marker (red).
    """

    def build(self, context: MotionStatechartContext) -> NodeArtifacts:
        """
        Creates the constraint driving the pour's projectile landing point to the
        receiver opening.

        :param context: The build context.
        :return: The generated task artifacts.
        """
        artifacts = NodeArtifacts()
        self.receiver.ensure_inflow_coupling(context.world)
        inflow_equation = self.receiver.fill_connection.inflow_equation
        if inflow_equation is None:
            raise MissingInflowEquationError(node=self)
        exit_speed = self.source.current_outflow_velocity(context.world)
        if exit_speed is None:
            if not isinstance(inflow_equation, GatedInflowEquation):
                raise MissingExitSpeedError(node=self)
            exit_speed = inflow_equation.exit_speed
        landing_point = self.receiver.projectile_landing_point(
            self.source, context.world, exit_speed
        )
        receiver_opening = self.receiver.opening_point(context.world)
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
        Build the debug expressions that visualize where the pour leaves and where it
        lands.

        :param context: The build context.
        :param landing_point: The projectile landing point on the receiver's opening
            plane.
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


@dataclass(eq=False, repr=False)
class KeepSourceRimAboveReceiverRim(Task):
    """
    Keeps the pouring source's rim above the receiver's rim so the rims never collide.

    Constrains the height of the source's pouring lip (its lowest rim point while
    tilting) above the receiver's rim to stay within a clearance band.  Because the lip
    is derived from the live forward kinematics, the constraint accounts for the lip
    descending as the source tilts, so the clearance is a true rim-to-rim gap rather
    than a hand-tuned offset on the cup origins.

    The task stores only the source and receiver, building the symbolic lip and rim on
    the target world, so it survives serialization to a standalone Giskard process
    (unlike a task that would carry a pre-built symbolic point).
    """

    receiver: HasFillLevel
    """
    The container whose rim the source's rim must stay above.
    """

    source: LiquidSource
    """
    The pouring source whose rim must stay above the receiver's rim.
    """

    minimum_clearance: float = field(default=0.05, kw_only=True)
    """
    Lower bound on the source-lip-above-receiver-rim clearance, in metres.

    Must be positive: a band reaching down to zero would ask the optimizer to hold the rims in
    contact, and since the bound is enforced softly the lip would settle below the receiver rim.
    Choose it above the clearance the optimizer actually tracks, not at the collision limit.
    """

    clearance_band: float = field(default=0.05, kw_only=True)
    """
    Width of the clearance band above :attr:`minimum_clearance`, in metres.

    A band, rather than a one-sided lower bound, keeps the optimization well-
    conditioned: it is the only constraint pinning the source's vertical position,
    because the landing point that :class:`KeepProjectileInReceiver` aims lies in the
    receiver's opening plane by construction and so carries no vertical error.
    """

    weight: float = field(
        default=DefaultWeights.WEIGHT_ABOVE_COLLISION_AVOIDANCE, kw_only=True
    )
    """
    QP constraint weight for the clearance.
    """

    maximum_velocity: float = field(default=0.2, kw_only=True)
    """
    Maximum allowed vertical speed for the clearance motion, in metres per second.
    """

    @property
    def maximum_clearance(self) -> float:
        """
        Upper end of the clearance band, in metres.
        """
        return self.minimum_clearance + self.clearance_band

    def build(self, context: MotionStatechartContext) -> NodeArtifacts:
        """
        Creates the constraint keeping the source's pouring lip above the receiver's
        rim.

        :param context: The build context.
        :return: The generated task artifacts.
        :raises NonPositiveClearanceError: if the clearance band is not entirely above
            the rim, which would describe a physically impossible pour.
        """
        if self.minimum_clearance <= 0.0:
            raise NonPositiveClearanceError(
                node=self, minimum_clearance=self.minimum_clearance
            )
        artifacts = NodeArtifacts()
        source_lip = self.source.liquid_exit_point(context.world)
        receiver_rim = self.receiver.opening_point(context.world)
        clearance = (source_lip - receiver_rim) @ Vector3.Z()
        artifacts.constraints.add_inequality_constraint(
            reference_velocity=self.maximum_velocity,
            lower_error=self.minimum_clearance - clearance,
            upper_error=self.maximum_clearance - clearance,
            quadratic_weight=self.weight,
            task_expression=clearance,
            name=f"{self.name}_clearance",
        )
        artifacts.observation = sm.logic_and(
            sm.if_less_eq(clearance, self.maximum_clearance, 1, 0),
            sm.if_greater_eq(clearance, self.minimum_clearance, 1, 0),
        )
        return artifacts
