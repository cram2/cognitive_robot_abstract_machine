"""
Monitors observing whether the tip of a kinematic chain has reached a goal.
"""

from __future__ import annotations

from abc import ABC
from dataclasses import dataclass

import krrood.symbolic_math.symbolic_math as sm
from cramph.node import EndedByOwner, NodeArtifacts
from giskardpy.motion_statechart.graph_node import (
    BindsGoalReferenceFrame,
    MotionStatechartNode,
)
from giskardpy.motion_statechart.mixins import (
    HasGoalOrientationParameter,
    HasGoalPointParameter,
    HasGoalPoseParameter,
    HasKinematicChainParameters,
    HasPoseThresholdParameters,
    HasThresholdParameter,
)
from krrood_spatial_math.spatial_types import Vector3, Point3
from semantic_digital_twin.world_description.world_entity import (
    KinematicStructureEntity,
)

# %% goals given in a frame that may move


@dataclass(eq=False, repr=False, kw_only=True)
class RootRelativeGoalMonitor(
    EndedByOwner,
    MotionStatechartNode,
    BindsGoalReferenceFrame,
    HasKinematicChainParameters,
    ABC,
):
    """
    A monitor observing :attr:`tip_link` against a goal whose frame is read relative to
    :attr:`root_link`.
    """

    @property
    def goal_binding_root(self) -> KinematicStructureEntity:
        return self.root_link


@dataclass(eq=False, repr=False, kw_only=True)
class PoseReached(
    RootRelativeGoalMonitor, HasGoalPoseParameter, HasPoseThresholdParameters
):
    """
    Observes ``True`` once the tip link has reached :attr:`goal_pose` within the
    position and orientation thresholds.
    """

    @property
    def goal_reference_frame(self) -> KinematicStructureEntity:
        return self.goal_pose.reference_frame

    def build_artifacts(self) -> NodeArtifacts:
        root_T_goal = self.root_T_goal_reference_frame @ self.goal_pose
        root_T_current = self.context.world.compose_forward_kinematics_expression(
            self.root_link, self.tip_link
        )
        distance_to_goal = root_T_goal.position.euclidean_distance(
            root_T_current.position
        )
        position_reached = distance_to_goal < self.position_threshold

        rotation_error = root_T_current.rotation_matrix.rotational_distance(
            root_T_goal.rotation_matrix
        )
        orientation_reached = sm.abs(rotation_error) < self.orientation_threshold

        return NodeArtifacts(
            observation=sm.logic_and(position_reached, orientation_reached)
        )


@dataclass(eq=False, repr=False, kw_only=True)
class PositionReached(
    RootRelativeGoalMonitor, HasGoalPointParameter, HasThresholdParameter
):
    """
    Observes ``True`` once the tip link is within :attr:`threshold` meters of
    :attr:`goal_point`.
    """

    @property
    def goal_reference_frame(self) -> KinematicStructureEntity:
        return self.goal_point.reference_frame

    def build_artifacts(self) -> NodeArtifacts:
        root_P_goal = self.root_T_goal_reference_frame @ self.goal_point
        root_P_current = self.context.world.compose_forward_kinematics_expression(
            self.root_link, self.tip_link
        ).position
        distance_to_goal = root_P_goal.euclidean_distance(root_P_current)
        return NodeArtifacts(observation=distance_to_goal < self.threshold)


@dataclass(eq=False, repr=False, kw_only=True)
class OrientationReached(
    RootRelativeGoalMonitor, HasGoalOrientationParameter, HasThresholdParameter
):
    """
    Observes ``True`` once the orientation of the tip link is within :attr:`threshold`
    radians of :attr:`goal_orientation`.
    """

    @property
    def goal_reference_frame(self) -> KinematicStructureEntity:
        return self.goal_orientation.reference_frame

    def build_artifacts(self) -> NodeArtifacts:
        root_R_goal = self.root_T_goal_reference_frame @ self.goal_orientation
        root_R_current = self.context.world.compose_forward_kinematics_expression(
            self.root_link, self.tip_link
        ).rotation_matrix
        rotation_error = root_R_current.rotational_distance(root_R_goal)
        return NodeArtifacts(observation=sm.abs(rotation_error) < self.threshold)


# %% goals given in a fixed frame


@dataclass(eq=False, repr=False, kw_only=True)
class PointingAt(
    EndedByOwner,
    MotionStatechartNode,
    HasKinematicChainParameters,
    HasGoalPointParameter,
    HasThresholdParameter,
):
    """
    Observes ``True`` once the pointing axis of the tip link passes within
    :attr:`threshold` meters of :attr:`goal_point`.
    """

    pointing_axis: Vector3
    """
    Axis of the tip link that should point at the goal point.
    """

    def build_artifacts(self) -> NodeArtifacts:
        root_P_goal_point = self.context.world.transform(
            target_frame=self.root_link, spatial_object=self.goal_point
        )

        tip_V_pointing_axis = self.context.world.transform(
            target_frame=self.tip_link, spatial_object=self.pointing_axis
        )
        tip_V_pointing_axis.scale(1)
        root_T_tip = self.context.world.compose_forward_kinematics_expression(
            self.root_link, self.tip_link
        )
        root_P_tip = root_T_tip.position
        root_V_pointing_axis = root_T_tip.dot(tip_V_pointing_axis)

        _, distance = root_P_goal_point.project_to_line(
            line_point=root_P_tip, line_direction=root_V_pointing_axis
        )
        return NodeArtifacts(observation=sm.abs(distance) < self.threshold)


@dataclass(eq=False, repr=False, kw_only=True)
class VectorsAligned(
    EndedByOwner,
    MotionStatechartNode,
    HasKinematicChainParameters,
    HasThresholdParameter,
):
    """
    Observes ``True`` once :attr:`tip_normal` is aligned with :attr:`reference_normal`
    within :attr:`threshold` radians.
    """

    reference_normal: Vector3
    """
    Reference normal the tip normal should align with.
    """

    tip_normal: Vector3
    """
    Normal of the tip link that should align with :attr:`reference_normal`.
    """

    def build_artifacts(self) -> NodeArtifacts:
        tip_V_tip_normal = self.context.world.transform(
            target_frame=self.tip_link, spatial_object=self.tip_normal
        )
        tip_V_tip_normal.scale(1)

        root_V_root_normal = self.context.world.transform(
            target_frame=self.root_link, spatial_object=self.reference_normal
        )
        root_V_root_normal.scale(1)

        root_R_tip = self.context.world.compose_forward_kinematics_expression(
            self.root_link, self.tip_link
        ).rotation_matrix
        root_V_tip_normal = root_R_tip.dot(tip_V_tip_normal)
        error = root_V_tip_normal.angle_between(root_V_root_normal)
        return NodeArtifacts(observation=error < self.threshold)


@dataclass(eq=False, repr=False, kw_only=True)
class DistanceToLine(
    EndedByOwner,
    MotionStatechartNode,
    HasKinematicChainParameters,
    HasThresholdParameter,
):
    """
    Observes ``True`` once the tip link is within :attr:`threshold` meters of the line
    segment centered at ``center_point`` along ``line_axis``.
    """

    center_point: Point3
    """
    Center of the line segment.
    """

    line_axis: Vector3
    """
    Direction of the line segment.
    """

    line_length: float
    """
    Length of the line segment.
    """

    def build_artifacts(self) -> NodeArtifacts:
        root_P_current = self.context.world.compose_forward_kinematics_expression(
            self.root_link, self.tip_link
        ).position
        root_V_line_axis = self.context.world.transform(
            target_frame=self.root_link, spatial_object=self.line_axis
        )
        root_V_line_axis.scale(1)
        root_P_center = self.context.world.transform(
            target_frame=self.root_link, spatial_object=self.center_point
        )
        root_P_line_start = root_P_center + root_V_line_axis * (self.line_length / 2)
        root_P_line_end = root_P_center - root_V_line_axis * (self.line_length / 2)

        distance, _ = root_P_current.distance_to_line_segment(
            root_P_line_start, root_P_line_end
        )
        return NodeArtifacts(observation=distance < self.threshold)
