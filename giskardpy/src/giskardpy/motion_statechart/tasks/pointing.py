from __future__ import annotations

from dataclasses import dataclass

from giskardpy.motion_statechart.graph_node import MotionNodeArtifacts
from giskardpy.motion_statechart.tasks.cartesian_tasks import CartesianTask
from krrood_spatial_math.spatial_types import Point3, Vector3


@dataclass(eq=False, repr=False, kw_only=True)
class Pointing(CartesianTask[Point3]):
    """
    Will orient :attr:`pointing_axis` at :attr:`goal`, with an error in radians.
    """

    pointing_axis: Vector3
    """
    The axis of tip_link that will be used for pointing.
    """

    reference_velocity: float = 0.3
    """
    The velocity the constraints of this task are normalized with, in rad/s.
    """

    def build_artifacts(self) -> MotionNodeArtifacts:
        """
        Build motion constraints that swing the pointing axis onto :attr:`goal`.

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

        root_P_goal_point = self.root_T_goal_reference_frame @ self.goal

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
            reference_velocity=self.reference_velocity,
            quadratic_weight=self.weight,
        )
        self.add_goal_and_current_debug_expressions(
            artifacts, goal=root_V_goal_axis, current=root_V_pointing_axis
        )

        artifacts.error = root_V_pointing_axis.angle_between(root_V_goal_axis)
        return artifacts


@dataclass(eq=False, repr=False, kw_only=True)
class PointingCone(CartesianTask[Point3]):
    """
    Will orient :attr:`pointing_axis` at :attr:`goal` with a cone-shaped tolerance
    region, with an error in radians.
    """

    pointing_axis: Vector3
    """
    The axis of tip_link that will be used for pointing.
    """

    cone_theta: float = 0.0
    """
    Slack cone region in radians.
    """

    reference_velocity: float = 0.3
    """
    The velocity the constraints of this task are normalized with, in rad/s.
    """

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

        root_P_goal_point = self.root_T_goal_reference_frame @ self.goal

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
            reference_velocity=self.reference_velocity,
            quadratic_weight=self.weight,
        )
        self.add_goal_and_current_debug_expressions(
            artifacts, goal=root_V_goal_axis_proj, current=root_V_pointing_axis
        )

        artifacts.error = root_V_pointing_axis.angle_between(root_V_goal_axis_proj)
        return artifacts
