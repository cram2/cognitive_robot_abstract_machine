from __future__ import annotations

from dataclasses import dataclass

from giskardpy.motion_statechart.graph_node import MotionNodeArtifacts
from giskardpy.motion_statechart.tasks.cartesian_tasks import CartesianTask
from semantic_digital_twin.spatial_types import Point3, Vector3
from semantic_digital_twin.world_description.world_entity import (
    KinematicStructureEntity,
)


@dataclass(eq=False, repr=False, kw_only=True)
class Pointing(CartesianTask):
    """
    Will orient pointing_axis at goal_point.
    """

    goal_point: Point3
    """
    Where to point pointing_axis at.
    """

    pointing_axis: Vector3
    """
    The axis of tip_link that will be used for pointing.
    """

    max_velocity: float = 0.3
    """
    Reference angular velocity for normalization in rad/s.
    """

    threshold: float = 0.01
    """
    Observation is true if the pointing angle is below this threshold.
    """

    @property
    def goal_reference_frame(self) -> KinematicStructureEntity:
        return self.goal_point.reference_frame

    def build_artifacts(self) -> MotionNodeArtifacts:
        """
        Build motion constraints that swing the pointing axis onto :attr:`goal_point`.

        :return: The artifacts of this task, whose error is the angle between the
            pointing axis and the goal direction.
        """
        artifacts = MotionNodeArtifacts()

        tip_V_pointing_axis = self.context.world.transform(
            target_frame=self.tip_link, spatial_object=self.pointing_axis
        )
        tip_V_pointing_axis.scale(1)

        root_T_tip = self.context.world.compose_forward_kinematics_expression(
            self.root_link, self.tip_link
        )

        root_P_goal_point = self.root_T_goal_reference_frame @ self.goal_point

        root_V_goal_axis = root_P_goal_point - root_T_tip.position
        root_V_goal_axis.scale(1)
        root_V_pointing_axis = root_T_tip @ tip_V_pointing_axis
        root_V_pointing_axis.visualisation_frame = self.tip_link
        root_V_goal_axis.visualisation_frame = self.tip_link

        tip_V_goal_axis = root_T_tip.inverse() @ root_V_goal_axis
        tip_V_goal_axis.scale(1)
        artifacts.geometry.add_vector_goal_constraints(
            frame_V_current=tip_V_goal_axis,
            frame_V_goal=tip_V_pointing_axis,
            reference_velocity=self.max_velocity,
            quadratic_weight=self.weight,
        )
        self.add_goal_and_current_debug_expressions(
            artifacts, goal=root_V_goal_axis, current=root_V_pointing_axis
        )

        artifacts.error = root_V_pointing_axis.angle_between(root_V_goal_axis)
        return artifacts


@dataclass(eq=False, repr=False, kw_only=True)
class PointingCone(CartesianTask):
    """
    Will orient pointing_axis at goal_point with a cone-shaped tolerance region.
    """

    goal_point: Point3
    """
    Where to point pointing_axis at.
    """

    pointing_axis: Vector3
    """
    The axis of tip_link that will be used for pointing.
    """

    cone_theta: float = 0.0
    """
    Slack cone region in radians.
    """

    max_velocity: float = 0.3
    """
    Reference angular velocity for normalization in rad/s.
    """

    threshold: float = 0.01
    """
    Observation is true if the pointing angle is below this threshold.
    """

    @property
    def goal_reference_frame(self) -> KinematicStructureEntity:
        return self.goal_point.reference_frame

    def build_artifacts(self) -> MotionNodeArtifacts:
        """
        Build motion constraints that swing the pointing axis into the goal cone.

        :return: The artifacts of this task, whose error is the angle between the
            pointing axis and the nearest direction inside the cone.
        """
        artifacts = MotionNodeArtifacts()
        tip_V_pointing_axis = self.context.world.transform(
            target_frame=self.tip_link, spatial_object=self.pointing_axis
        )
        tip_V_pointing_axis.scale(1)

        root_T_tip = self.context.world.compose_forward_kinematics_expression(
            self.root_link, self.tip_link
        )

        root_P_goal_point = self.root_T_goal_reference_frame @ self.goal_point

        root_V_goal_axis = root_P_goal_point - root_T_tip.position
        root_V_goal_axis.scale(1)
        root_V_pointing_axis = root_T_tip.dot(tip_V_pointing_axis)
        root_V_pointing_axis.visualisation_frame = self.tip_link

        root_V_goal_axis.visualisation_frame = self.tip_link

        root_V_goal_axis_proj = root_V_pointing_axis.project_to_cone(
            root_V_goal_axis, self.cone_theta
        )
        root_V_goal_axis_proj.visualisation_frame = self.tip_link

        artifacts.geometry.add_vector_goal_constraints(
            frame_V_current=root_V_pointing_axis,
            frame_V_goal=root_V_goal_axis_proj,
            reference_velocity=self.max_velocity,
            quadratic_weight=self.weight,
        )
        self.add_goal_and_current_debug_expressions(
            artifacts, goal=root_V_goal_axis_proj, current=root_V_pointing_axis
        )

        artifacts.error = root_V_pointing_axis.angle_between(root_V_goal_axis_proj)
        return artifacts
