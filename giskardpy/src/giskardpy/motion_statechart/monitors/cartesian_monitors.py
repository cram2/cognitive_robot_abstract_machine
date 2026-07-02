from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from typing_extensions import Optional

import krrood.symbolic_math.symbolic_math as sm
from giskardpy.motion_statechart.binding_policy import (
    ForwardKinematicsBinding,
    GoalBindingPolicy,
)
from giskardpy.motion_statechart.context import MotionStatechartContext
from giskardpy.motion_statechart.graph_node import MotionStatechartNode, NodeArtifacts
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.spatial_types import (
    Point3,
    Vector3,
    RotationMatrix,
    HomogeneousTransformationMatrix,
)
from semantic_digital_twin.spatial_types.spatial_types import SpatialType
from semantic_digital_twin.world_description.world_entity import Body


@dataclass(eq=False, repr=False)
class RootRelativeGoalMonitor(MotionStatechartNode, ABC):
    """
    Base for monitors whose goal is captured relative to the kinematic chain via a forward
    kinematics binding. The :class:`GoalBindingPolicy` decides whether the goal is frozen once at
    build time or recaptured every time the monitor starts.
    """

    root_link: Body = field(kw_only=True)
    """Root link of the kinematic chain the goal is expressed in."""
    binding_policy: GoalBindingPolicy = field(
        default=GoalBindingPolicy.Bind_on_start, kw_only=True
    )
    """When the goal reference frame is captured. See :class:`GoalBindingPolicy`."""

    _forward_kinematics_binding: Optional[ForwardKinematicsBinding] = field(
        default=None, init=False, repr=False
    )
    """Binding used to freeze the forward kinematics of the goal reference frame."""

    @property
    @abstractmethod
    def goal(self) -> SpatialType:
        """
        :return: The goal spatial object, expressed in its own reference frame.
        """

    def resolve_root_goal(self, context: MotionStatechartContext) -> SpatialType:
        """
        Express the goal in the root link frame, captured relative to the goal reference frame via
        a forward kinematics binding.
        """
        self._forward_kinematics_binding = ForwardKinematicsBinding(
            name=PrefixedName("root_T_goal_ref", str(self.name)),
            root=self.root_link,
            tip=self.goal.reference_frame,
            float_variable_data=context.float_variable_data,
        )
        self._forward_kinematics_binding.bind(context.world)
        return self._forward_kinematics_binding.root_T_tip @ self.goal

    def on_start(self, context: MotionStatechartContext) -> None:
        if self.binding_policy == GoalBindingPolicy.Bind_on_start:
            self._forward_kinematics_binding.bind(context.world)


@dataclass(eq=False, repr=False)
class PoseReached(RootRelativeGoalMonitor):
    """
    Observes ``True`` once the tip link has reached the goal pose within the position and
    orientation thresholds.
    """

    tip_link: Body = field(kw_only=True)
    """Link that should reach the goal pose."""
    goal_pose: HomogeneousTransformationMatrix = field(kw_only=True)
    """Target pose to reach."""
    position_threshold: float = field(default=0.01, kw_only=True)
    """Distance threshold for the position part of the goal in meters."""
    orientation_threshold: float = field(default=0.01, kw_only=True)
    """Rotation error threshold for the orientation part of the goal in radians."""

    @property
    def goal(self) -> HomogeneousTransformationMatrix:
        return self.goal_pose

    def build(self, context: MotionStatechartContext) -> NodeArtifacts:
        root_T_goal = self.resolve_root_goal(context)

        root_P_goal = root_T_goal.to_position()
        root_P_current = context.world.compose_forward_kinematics_expression(
            self.root_link, self.tip_link
        ).to_position()
        distance_to_goal = root_P_goal.euclidean_distance(root_P_current)
        position_reached = distance_to_goal < self.position_threshold

        root_R_goal = root_T_goal.to_rotation_matrix()
        root_R_current = context.world.compose_forward_kinematics_expression(
            self.root_link, self.tip_link
        ).to_rotation_matrix()
        rotation_error = root_R_current.rotational_error(root_R_goal)
        orientation_reached = sm.abs(rotation_error) < self.orientation_threshold

        return NodeArtifacts(
            observation=sm.logic_and(position_reached, orientation_reached)
        )


@dataclass(eq=False, repr=False)
class PositionReached(RootRelativeGoalMonitor):
    """
    Observes ``True`` once the tip link is within ``threshold`` of the goal point.
    """

    tip_link: Body = field(kw_only=True)
    """Link that should reach the goal point."""
    goal_point: Point3 = field(kw_only=True)
    """Target 3D point to reach."""
    threshold: float = field(default=0.01, kw_only=True)
    """Distance threshold for goal achievement in meters."""

    @property
    def goal(self) -> Point3:
        return self.goal_point

    def build(self, context: MotionStatechartContext) -> NodeArtifacts:
        root_P_goal = self.resolve_root_goal(context)

        root_P_current = context.world.compose_forward_kinematics_expression(
            self.root_link, self.tip_link
        ).to_position()
        distance_to_goal = root_P_goal.euclidean_distance(root_P_current)
        return NodeArtifacts(observation=distance_to_goal < self.threshold)


@dataclass(eq=False, repr=False)
class OrientationReached(RootRelativeGoalMonitor):
    """
    Observes ``True`` once the tip link orientation is within ``threshold`` of the goal orientation.
    """

    tip_link: Body = field(kw_only=True)
    """Link that should reach the goal orientation."""
    goal_orientation: RotationMatrix = field(kw_only=True)
    """Target orientation to reach."""
    threshold: float = field(default=0.01, kw_only=True)
    """Rotation error threshold for goal achievement in radians."""

    @property
    def goal(self) -> RotationMatrix:
        return self.goal_orientation

    def build(self, context: MotionStatechartContext) -> NodeArtifacts:
        root_R_goal = self.resolve_root_goal(context)

        root_R_current = context.world.compose_forward_kinematics_expression(
            self.root_link, self.tip_link
        ).to_rotation_matrix()
        rotation_error = root_R_current.rotational_error(root_R_goal)
        return NodeArtifacts(observation=sm.abs(rotation_error) < self.threshold)


@dataclass(eq=False, repr=False)
class PointingAt(MotionStatechartNode):
    """
    Observes ``True`` once the pointing axis of the tip link is aimed at the goal point within
    ``threshold``.
    """

    tip_link: Body = field(kw_only=True)
    """Link whose pointing axis is checked."""
    goal_point: Point3 = field(kw_only=True)
    """Point the tip link should point at."""
    root_link: Body = field(kw_only=True)
    """Reference link the goal point is expressed in."""
    pointing_axis: Vector3 = field(kw_only=True)
    """Axis of the tip link that should point at the goal point."""
    threshold: float = field(default=0.01, kw_only=True)
    """Distance threshold between the goal point and the pointing line in meters."""

    def build(self, context: MotionStatechartContext) -> NodeArtifacts:
        root_P_goal_point = context.world.transform(
            target_frame=self.root_link, spatial_object=self.goal_point
        )

        tip_V_pointing_axis = context.world.transform(
            target_frame=self.tip_link, spatial_object=self.pointing_axis
        )
        tip_V_pointing_axis.scale(1)
        root_T_tip = context.world.compose_forward_kinematics_expression(
            self.root_link, self.tip_link
        )
        root_P_tip = root_T_tip.to_position()
        root_V_pointing_axis = root_T_tip.dot(tip_V_pointing_axis)

        _, distance = root_P_goal_point.project_to_line(
            line_point=root_P_tip, line_direction=root_V_pointing_axis
        )
        return NodeArtifacts(observation=sm.abs(distance) < self.threshold)


@dataclass(eq=False, repr=False)
class VectorsAligned(MotionStatechartNode):
    """
    Observes ``True`` once the tip normal is aligned with the goal normal within ``threshold``.
    """

    root_link: Body = field(kw_only=True)
    """Reference link the goal normal is expressed in."""
    tip_link: Body = field(kw_only=True)
    """Link the tip normal is expressed in."""
    goal_normal: Vector3 = field(kw_only=True)
    """Reference normal the tip normal should align with."""
    tip_normal: Vector3 = field(kw_only=True)
    """Normal of the tip link that should align with the goal normal."""
    threshold: float = field(default=0.01, kw_only=True)
    """Angle threshold between the two normals in radians."""

    def build(self, context: MotionStatechartContext) -> NodeArtifacts:
        tip_V_tip_normal = context.world.transform(
            target_frame=self.tip_link, spatial_object=self.tip_normal
        )
        tip_V_tip_normal.scale(1)

        root_V_root_normal = context.world.transform(
            target_frame=self.root_link, spatial_object=self.goal_normal
        )
        root_V_root_normal.scale(1)

        root_R_tip = context.world.compose_forward_kinematics_expression(
            self.root_link, self.tip_link
        ).to_rotation_matrix()
        root_V_tip_normal = root_R_tip.dot(tip_V_tip_normal)
        error = root_V_tip_normal.angle_between(root_V_root_normal)
        return NodeArtifacts(observation=error < self.threshold)


@dataclass(eq=False, repr=False)
class DistanceToLine(MotionStatechartNode):
    """
    Observes ``True`` once the tip link is within ``threshold`` of the line segment centered at
    ``center_point`` along ``line_axis``.
    """

    root_link: Body = field(kw_only=True)
    """Reference link the line is expressed in."""
    tip_link: Body = field(kw_only=True)
    """Link whose distance to the line is checked."""
    center_point: Point3 = field(kw_only=True)
    """Center of the line segment."""
    line_axis: Vector3 = field(kw_only=True)
    """Direction of the line segment."""
    line_length: float = field(kw_only=True)
    """Length of the line segment."""
    threshold: float = field(default=0.01, kw_only=True)
    """Distance threshold to the line segment in meters."""

    def build(self, context: MotionStatechartContext) -> NodeArtifacts:
        root_P_current = context.world.compose_forward_kinematics_expression(
            self.root_link, self.tip_link
        ).to_position()
        root_V_line_axis = context.world.transform(
            target_frame=self.root_link, spatial_object=self.line_axis
        )
        root_V_line_axis.scale(1)
        root_P_center = context.world.transform(
            target_frame=self.root_link, spatial_object=self.center_point
        )
        root_P_line_start = root_P_center + root_V_line_axis * (self.line_length / 2)
        root_P_line_end = root_P_center - root_V_line_axis * (self.line_length / 2)

        distance, _ = root_P_current.distance_to_line_segment(
            root_P_line_start, root_P_line_end
        )
        return NodeArtifacts(observation=distance < self.threshold)
