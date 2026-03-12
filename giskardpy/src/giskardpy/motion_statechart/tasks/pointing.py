from __future__ import division

from dataclasses import dataclass, field
from functools import cached_property

from giskardpy.motion_statechart.tasks.cartesian_tasks import CartesianTask
from krrood.symbolic_math.float_variable_data import FloatVariableData
from semantic_digital_twin.spatial_types import Point3, Vector3
from semantic_digital_twin.world_description.geometry import Color
from semantic_digital_twin.world_description.world_entity import (
    KinematicStructureEntity,
)
from giskardpy.motion_statechart.context import MotionStatechartContext
from giskardpy.motion_statechart.data_types import DefaultWeights
from giskardpy.motion_statechart.graph_node import NodeArtifacts, DebugExpression
from giskardpy.motion_statechart.graph_node import Task


@dataclass(eq=False, repr=False)
class Pointing(CartesianTask):
    """
    Will orient pointing_axis at goal_point.
    """

    goal_point: Point3 = field(kw_only=True)
    """where to point pointing_axis at."""
    pointing_axis: Vector3 = field(kw_only=True)
    """the axis of tip_link that will be used for pointing"""

    max_velocity: float = field(default=0.3, kw_only=True)
    threshold: float = field(default=0.01, kw_only=True)
    """
    Observation is true if the pointing angle is below this threshold.
    """

    @property
    def goal_reference_frame(self) -> KinematicStructureEntity:
        return self.goal_point.reference_frame

    def build(self, context: MotionStatechartContext) -> NodeArtifacts:
        artifacts = super().build(context)

        tip_V_pointing_axis = context.world.transform(
            target_frame=self.tip_link, spatial_object=self.pointing_axis
        )
        tip_V_pointing_axis.scale(1)

        root_T_tip = context.world.compose_forward_kinematics_expression(
            self.root_link, self.tip_link
        )

        root_P_goal_point = self.root_T_goal_reference_frame @ self.goal_point

        root_V_goal_axis = root_P_goal_point - root_T_tip.to_position()
        root_V_goal_axis.scale(1)
        root_V_pointing_axis = root_T_tip @ tip_V_pointing_axis
        root_V_pointing_axis.vis_frame = self.tip_link
        root_V_goal_axis.vis_frame = self.tip_link

        artifacts.constraints.add_vector_goal_constraints(
            frame_V_current=root_V_pointing_axis,
            frame_V_goal=root_V_goal_axis,
            reference_velocity=self.max_velocity,
            weight=self.weight,
        )
        artifacts.observation = (
            root_V_pointing_axis.angle_between(root_V_goal_axis) <= self.threshold
        )
        return artifacts


@dataclass(eq=False, repr=False)
class PointingCone(CartesianTask):
    """
    Will orient pointing_axis at goal_point with a cone-shaped tolerance region.
    """

    goal_point: Point3 = field(kw_only=True)
    """where to point pointing_axis at."""
    pointing_axis: Vector3 = field(kw_only=True)
    """the axis of tip_link that will be used for pointing"""

    cone_theta: float = field(default=0.0, kw_only=True)
    """Slack cone region in radians"""
    max_velocity: float = field(default=0.3, kw_only=True)
    threshold: float = field(default=0.01, kw_only=True)
    """
    Observation is true if the pointing angle is below this threshold.
    """

    @property
    def goal_reference_frame(self) -> KinematicStructureEntity:
        return self.goal_point.reference_frame

    def build(self, context: MotionStatechartContext) -> NodeArtifacts:
        artifacts = super().build(context)

        tip_V_pointing_axis = context.world.transform(
            target_frame=self.tip_link, spatial_object=self.pointing_axis
        )
        tip_V_pointing_axis.scale(1)

        root_T_tip = context.world.compose_forward_kinematics_expression(
            self.root_link, self.tip_link
        )

        root_P_goal_point = self.root_T_goal_reference_frame @ self.goal_point

        root_V_goal_axis = root_P_goal_point - root_T_tip.to_position()
        root_V_goal_axis.scale(1)
        root_V_pointing_axis = root_T_tip.dot(tip_V_pointing_axis)
        root_V_pointing_axis.vis_frame = self.tip_link

        root_V_goal_axis.vis_frame = self.tip_link

        root_V_goal_axis_proj = root_V_pointing_axis.project_to_cone(
            root_V_goal_axis, self.cone_theta
        )
        root_V_goal_axis_proj.vis_frame = self.tip_link

        artifacts.constraints.add_vector_goal_constraints(
            frame_V_current=root_V_pointing_axis,
            frame_V_goal=root_V_goal_axis_proj,
            reference_velocity=self.max_velocity,
            weight=self.weight,
        )
        artifacts.observation = (
            root_V_pointing_axis.angle_between(root_V_goal_axis_proj) <= self.threshold
        )
        return artifacts
