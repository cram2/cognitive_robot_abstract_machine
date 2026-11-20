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
    HasKinematicChainParameter,
    HasPositionThresholdParameter,
    HasOrientationThresholdParameter,
    HasPointingAxisParameter,
    HasReferenceNormalParameter,
    HasThresholdParameter,
    HasTipNormalParameter,
)
from krrood_spatial_math.spatial_types import Vector3Dimensional, Point3Dimensional
from semantic_digital_twin.world_description.world_entity import (
    KinematicStructureEntity,
)

# %% goals given in a frame that may move


@dataclass(eq=False, repr=False, kw_only=True)
class RootRelativeGoalMonitor(
    EndedByOwner,
    MotionStatechartNode,
    BindsGoalReferenceFrame,
    HasKinematicChainParameter,
    ABC,
):
    """
    A monitor observing the tip of :attr:`kinematic_chain` against a goal whose frame is
    read relative to its root.
    """

    @property
    def goal_binding_root(self) -> KinematicStructureEntity:
        return self.kinematic_chain.root


@dataclass(eq=False, repr=False, kw_only=True)
class PoseReached(
    RootRelativeGoalMonitor,
    HasGoalPoseParameter,
    HasPositionThresholdParameter,
    HasOrientationThresholdParameter,
):
    """
    Observes ``True`` once the tip link has reached :attr:`goal_pose` within the
    position and orientation thresholds.
    """

    @property
    def goal_reference_frame(self) -> KinematicStructureEntity:
        return self.goal_pose.reference_frame

    def build_artifacts(self) -> NodeArtifacts:
        """
        :return: Artifacts observing whether the tip is within
            :attr:`position_threshold` of the goal position and within
            :attr:`orientation_threshold` of the goal orientation.
        """
        root_T_goal = self.root_T_goal_reference_frame @ self.goal_pose
        root_T_current = self.context.world.compose_forward_kinematics_expression(
            self.kinematic_chain.root, self.kinematic_chain.tip
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
        """
        :return: Artifacts observing whether the tip is within :attr:`threshold` of
            :attr:`goal_point`.
        """
        root_P_goal = self.root_T_goal_reference_frame @ self.goal_point
        root_P_current = self.context.world.compose_forward_kinematics_expression(
            self.kinematic_chain.root, self.kinematic_chain.tip
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
        """
        :return: Artifacts observing whether the tip is turned less than
            :attr:`threshold` away from :attr:`goal_orientation`.
        """
        root_R_goal = (
            self.root_T_goal_reference_frame @ self.goal_orientation
        ).rotation_matrix
        root_R_current = self.context.world.compose_forward_kinematics_expression(
            self.kinematic_chain.root, self.kinematic_chain.tip
        ).rotation_matrix
        rotation_error = root_R_current.rotational_distance(root_R_goal)
        return NodeArtifacts(observation=sm.abs(rotation_error) < self.threshold)


# %% goals given in a fixed frame


@dataclass(eq=False, repr=False, kw_only=True)
class PointingAt(
    EndedByOwner,
    MotionStatechartNode,
    HasKinematicChainParameter,
    HasGoalPointParameter,
    HasPointingAxisParameter,
    HasThresholdParameter,
):
    """
    Observes ``True`` once the pointing axis of the tip link passes within
    :attr:`threshold` meters of :attr:`goal_point`.
    """

    def build_artifacts(self) -> NodeArtifacts:
        """
        :return: Artifacts observing whether the line along :attr:`pointing_axis` from
            the tip passes within :attr:`threshold` of :attr:`goal_point`.
        """
        root_P_goal_point = self.context.world.transform(
            target_frame=self.kinematic_chain.root, spatial_object=self.goal_point
        )

        tip_V_pointing_axis = self.context.world.transform(
            target_frame=self.kinematic_chain.tip, spatial_object=self.pointing_axis
        )
        tip_V_pointing_axis.scale(1)
        root_T_tip = self.context.world.compose_forward_kinematics_expression(
            self.kinematic_chain.root, self.kinematic_chain.tip
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
    HasKinematicChainParameter,
    HasTipNormalParameter,
    HasReferenceNormalParameter,
    HasThresholdParameter,
):
    """
    Observes ``True`` once :attr:`tip_normal` is aligned with :attr:`reference_normal`
    within :attr:`threshold` radians.
    """

    def build_artifacts(self) -> NodeArtifacts:
        """
        :return: Artifacts observing whether the angle between :attr:`tip_normal` and
            :attr:`reference_normal` is below :attr:`threshold`.
        """
        tip_V_tip_normal = self.context.world.transform(
            target_frame=self.kinematic_chain.tip, spatial_object=self.tip_normal
        )
        tip_V_tip_normal.scale(1)

        root_V_root_normal = self.context.world.transform(
            target_frame=self.kinematic_chain.root, spatial_object=self.reference_normal
        )
        root_V_root_normal.scale(1)

        root_R_tip = self.context.world.compose_forward_kinematics_expression(
            self.kinematic_chain.root, self.kinematic_chain.tip
        ).rotation_matrix
        root_V_tip_normal = root_R_tip.dot(tip_V_tip_normal)
        error = root_V_tip_normal.angle_between(root_V_root_normal)
        return NodeArtifacts(observation=error < self.threshold)


@dataclass(eq=False, repr=False, kw_only=True)
class DistanceToLine(
    EndedByOwner,
    MotionStatechartNode,
    HasKinematicChainParameter,
    HasThresholdParameter,
):
    """
    Observes ``True`` once the tip link is within :attr:`threshold` meters of the line
    segment centered at ``center_point`` along ``line_axis``.
    """

    center_point: Point3Dimensional
    """
    Center of the line segment.
    """

    line_axis: Vector3Dimensional
    """
    Direction of the line segment.
    """

    line_length: float
    """
    Length of the line segment.
    """

    def build_artifacts(self) -> NodeArtifacts:
        """
        :return: Artifacts observing whether the tip is within :attr:`threshold` of the
            line segment of :attr:`line_length` centred on :attr:`center_point`.
        """
        root_P_current = self.context.world.compose_forward_kinematics_expression(
            self.kinematic_chain.root, self.kinematic_chain.tip
        ).position
        root_V_line_axis = self.context.world.transform(
            target_frame=self.kinematic_chain.root, spatial_object=self.line_axis
        )
        root_V_line_axis.scale(1)
        root_P_center = self.context.world.transform(
            target_frame=self.kinematic_chain.root, spatial_object=self.center_point
        )
        root_P_line_start = root_P_center + root_V_line_axis * (self.line_length / 2)
        root_P_line_end = root_P_center - root_V_line_axis * (self.line_length / 2)

        distance, _ = root_P_current.distance_to_line_segment(
            root_P_line_start, root_P_line_end
        )
        return NodeArtifacts(observation=distance < self.threshold)
