from __future__ import division

from dataclasses import dataclass, field
from functools import cached_property

from krrood.symbolic_math.float_variable_data import FloatVariableData
from semantic_digital_twin.spatial_types import Point3, Vector3
from semantic_digital_twin.world_description.geometry import Color
from semantic_digital_twin.world_description.world_entity import (
    KinematicStructureEntity,
)
from ..context import MotionStatechartContext
from ..data_types import DefaultWeights
from ..graph_node import NodeArtifacts, DebugExpression
from ..graph_node import Task


@dataclass(eq=False, repr=False)
class Pointing(Task):
    """
    Will orient pointing_axis at goal_point.
    """

    tip_link: KinematicStructureEntity = field(kw_only=True)
    """tip link of the kinematic chain."""
    root_link: KinematicStructureEntity = field(kw_only=True)
    """root link of the kinematic chain."""

    goal_point: Point3 = field(kw_only=True)
    """where to point pointing_axis at."""
    pointing_axis: Vector3 = field(kw_only=True)
    """the axis of tip_link that will be used for pointing"""

    max_velocity: float = field(default=0.3, kw_only=True)
    threshold: float = field(default=0.01, kw_only=True)
    """
    Observation is true if the pointing angle is below this threshold.
    """
    weight: float = field(default=DefaultWeights.WEIGHT_BELOW_CA, kw_only=True)
    """
    Priority weight for the pointing constraint in the optimization problem.
    """
    _root_P_goal_point_index: int = field(init=False)
    """
    Index of the FloatVariables for root_P_goal_point in the FloatVariableData. 
    """
    _float_variable_data: FloatVariableData = field(init=False)
    """
    Reference to the data structure used to store auxiliary variables.
    """

    def build(self, context: MotionStatechartContext) -> NodeArtifacts:
        artifacts = NodeArtifacts()

        self._float_variable_data = context.float_variable_data

        tip_V_pointing_axis = context.world.transform(
            target_frame=self.tip_link, spatial_object=self.pointing_axis
        )
        tip_V_pointing_axis.scale(1)

        root_T_tip = context.world.compose_forward_kinematics_expression(
            self.root_link, self.tip_link
        )
        self.root_P_goal_point.reference_frame = self.root_link

        root_V_goal_axis = self.root_P_goal_point - root_T_tip.to_position()
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
        self.set_root_P_goal_point_data(
            context.world.transform(
                target_frame=self.root_link, spatial_object=self.goal_point
            )
        )
        return artifacts

    @cached_property
    def root_P_goal_point(self) -> Point3:
        root_P_goal_point = Point3.create_with_variables(
            f"{self.name}.root_P_goal_point",
        )
        self._root_P_goal_point_index = (
            self._float_variable_data.add_variables_of_expression(root_P_goal_point)
        )
        return root_P_goal_point

    def set_root_P_goal_point_data(self, value: Point3):
        self._float_variable_data.set_values(
            self._root_P_goal_point_index, value.to_np()[:3]
        )


@dataclass(eq=False, repr=False)
class PointingCone(Task):
    """
    Will orient pointing_axis at goal_point with a cone-shaped tolerance region.
    """

    tip_link: KinematicStructureEntity = field(kw_only=True)
    """tip link of the kinematic chain."""
    root_link: KinematicStructureEntity = field(kw_only=True)
    """root link of the kinematic chain."""

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
    weight: float = field(default=DefaultWeights.WEIGHT_BELOW_CA, kw_only=True)
    """
    Priority weight for the pointing constraint in the optimization problem.
    """
    _root_P_goal_point_index: int = field(init=False)
    """
    Index of the FloatVariables for root_P_goal_point in the FloatVariableData. 
    """
    _float_variable_data: FloatVariableData = field(init=False)
    """
    Reference to the data structure used to store auxiliary variables.
    """

    def build(self, context: MotionStatechartContext) -> NodeArtifacts:
        artifacts = NodeArtifacts()

        self._float_variable_data = context.float_variable_data

        tip_V_pointing_axis = context.world.transform(
            target_frame=self.tip_link, spatial_object=self.pointing_axis
        )
        tip_V_pointing_axis.scale(1)

        root_T_tip = context.world.compose_forward_kinematics_expression(
            self.root_link, self.tip_link
        )
        self.root_P_goal_point.reference_frame = self.root_link

        root_V_goal_axis = self.root_P_goal_point - root_T_tip.to_position()
        root_V_goal_axis.scale(1)
        root_V_pointing_axis = root_T_tip.dot(tip_V_pointing_axis)
        root_V_pointing_axis.vis_frame = self.tip_link

        root_V_goal_axis.vis_frame = self.tip_link
        current_dbg = DebugExpression(
            name=f"{self.name}/root_V_pointing_axis",
            expression=root_V_pointing_axis,
            color=Color(1, 0, 0, 1),
        )
        goal_dbg = DebugExpression(
            name=f"{self.name}/goal_point",
            expression=self.root_P_goal_point,
            color=Color(0, 0, 1, 1),
        )
        artifacts.debug_expressions.append(current_dbg)
        artifacts.debug_expressions.append(goal_dbg)

        root_V_goal_axis_proj = root_V_pointing_axis.project_to_cone(
            root_V_goal_axis, self.cone_theta
        )
        root_V_goal_axis_proj.vis_frame = self.tip_link
        cone_axis_dbg = DebugExpression(
            name=f"{self.name}/cone_axis",
            expression=root_V_goal_axis,
            color=Color(1, 1, 0, 1),
        )
        projected_axis_dbg = DebugExpression(
            name=f"{self.name}/projected_axis",
            expression=root_V_goal_axis_proj,
            color=Color(1, 1, 0, 1),
        )
        artifacts.debug_expressions.append(cone_axis_dbg)
        artifacts.debug_expressions.append(projected_axis_dbg)

        artifacts.constraints.add_vector_goal_constraints(
            frame_V_current=root_V_pointing_axis,
            frame_V_goal=root_V_goal_axis_proj,
            reference_velocity=self.max_velocity,
            weight=self.weight,
        )
        artifacts.observation = (
            root_V_pointing_axis.angle_between(root_V_goal_axis_proj) <= self.threshold
        )
        self.set_root_P_goal_point_data(
            context.world.transform(
                target_frame=self.root_link, spatial_object=self.goal_point
            )
        )
        return artifacts

    @cached_property
    def root_P_goal_point(self) -> Point3:
        root_P_goal_point = Point3.create_with_variables(
            f"{self.name}.root_P_goal_point",
        )
        self._root_P_goal_point_index = (
            self._float_variable_data.add_variables_of_expression(root_P_goal_point)
        )
        return root_P_goal_point

    def set_root_P_goal_point_data(self, value: Point3):
        self._float_variable_data.set_values(
            self._root_P_goal_point_index, value.to_np()[:3]
        )
