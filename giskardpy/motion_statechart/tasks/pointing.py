from __future__ import division

from dataclasses import dataclass, field

import semantic_digital_twin.spatial_types.spatial_types as cas
from giskardpy.motion_statechart.context import BuildContext
from giskardpy.motion_statechart.data_types import DefaultWeights
from giskardpy.motion_statechart.graph_node import NodeArtifacts, DebugExpression
from giskardpy.motion_statechart.graph_node import Task
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.world_description.geometry import Color
from semantic_digital_twin.world_description.world_entity import (
    Body,
    KinematicStructureEntity,
)


@dataclass
class Pointing(Task):
    """
    Will orient pointing_axis at goal_point.
    """

    tip_link: KinematicStructureEntity = field(kw_only=True)
    """tip link of the kinematic chain."""
    root_link: KinematicStructureEntity = field(kw_only=True)
    """root link of the kinematic chain."""

    goal_point: cas.Point3 = field(kw_only=True)
    """where to point pointing_axis at."""
    pointing_axis: cas.Vector3 = field(kw_only=True)
    """the axis of tip_link that will be used for pointing"""

    max_velocity: float = field(default=0.3, kw_only=True)
    threshold: float = field(default=0.01, kw_only=True)
    weight: float = field(default=DefaultWeights.WEIGHT_BELOW_CA, kw_only=True)

    def build(self, context: BuildContext) -> NodeArtifacts:
        artifacts = NodeArtifacts()

        self._root_P_goal_point = (
            context.world.transform(
                target_frame=self.root_link, spatial_object=self.goal_point
            )
            .to_np()
            .tolist()
        )

        tip_V_pointing_axis = context.world.transform(
            target_frame=self.tip_link, spatial_object=self.pointing_axis
        )
        tip_V_pointing_axis.scale(1)

        root_T_tip = context.world.compose_forward_kinematics_expression(
            self.root_link, self.tip_link
        )
        root_P_goal_point = context.auxiliary_variable_manager.create_point3(
            name=PrefixedName("goal", str(self.name)),
            provider=lambda: self._root_P_goal_point,
        )
        root_P_goal_point.reference_frame = self.root_link

        root_V_goal_axis = root_P_goal_point - root_T_tip.to_position()
        root_V_goal_axis.scale(1)
        root_V_pointing_axis = root_T_tip.dot(tip_V_pointing_axis)
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


@dataclass
class PointingCone(Task):
    """
    Will orient pointing_axis at goal_point.
    """

    tip_link: Body = field(kw_only=True)
    """tip link of the kinematic chain."""
    goal_point: cas.Point3 = field(kw_only=True)
    """where to point pointing_axis at."""

    root_link: Body = field(kw_only=True)
    """root link of the kinematic chain."""
    pointing_axis: cas.Vector3 = field(kw_only=True)
    """the axis of tip_link that will be used for pointing"""

    cone_theta: float = 0.0
    max_velocity: float = 0.3
    threshold: float = 0.01
    weight: float = DefaultWeights.WEIGHT_BELOW_CA

    def build(self, context: BuildContext) -> NodeArtifacts:
        artifacts = NodeArtifacts()

        self._root_P_goal_point = (
            context.world.transform(
                target_frame=self.root_link, spatial_object=self.goal_point
            )
            .to_np()
            .tolist()
        )

        tip_V_pointing_axis = context.world.transform(
            target_frame=self.tip_link, spatial_object=self.pointing_axis
        )
        tip_V_pointing_axis.scale(1)

        root_T_tip = context.world.compose_forward_kinematics_expression(
            self.root_link, self.tip_link
        )
        root_P_goal_point = context.auxiliary_variable_manager.create_point3(
            name=PrefixedName("goal", str(self.name)),
            provider=lambda: self._root_P_goal_point,
        )
        root_P_goal_point.reference_frame = self.root_link

        root_V_goal_axis = root_P_goal_point - root_T_tip.to_position()
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
            expression=root_P_goal_point,
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

        return artifacts