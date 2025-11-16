from __future__ import division

from dataclasses import dataclass, field

import semantic_digital_twin.spatial_types.spatial_types as cas
from giskardpy.god_map import god_map
from giskardpy.motion_statechart.context import BuildContext
from giskardpy.motion_statechart.data_types import DefaultWeights
from giskardpy.motion_statechart.graph_node import NodeArtifacts
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
    tip_link: Body = field(kw_only=True)
    goal_point: cas.Point3 = field(kw_only=True)
    root_link: Body = field(kw_only=True)
    pointing_axis: cas.Vector3 = field(kw_only=True)
    cone_theta: float = 0.0
    max_velocity: float = 0.3
    threshold: float = 0.01
    weight: float = DefaultWeights.WEIGHT_BELOW_CA

    def __post_init__(self):
        """
        Will orient pointing_axis at goal_point.
        :param tip_link: tip link of the kinematic chain.
        :param goal_point: where to point pointing_axis at.
        :param root_link: root link of the kinematic chain.
        :param pointing_axis: the axis of tip_link that will be used for pointing
        :param max_velocity: rad/s
        :param weight:
        """
        self.root_P_goal_point = context.world.transform(
            target_frame=self.root_link, spatial_object=self.goal_point
        ).to_np()

        tip_V_pointing_axis = context.world.transform(
            target_frame=self.tip_link, spatial_object=self.pointing_axis
        )
        tip_V_pointing_axis.scale(1)

        root_T_tip = context.world._forward_kinematic_manager.compose_expression(
            self.root_link, self.tip_link
        )
        root_P_goal_point = symbol_manager.register_point3(
            name=f"{self.name}.root_P_goal_point",
            provider=lambda: self.root_P_goal_point,
        )
        root_P_goal_point.reference_frame = self.root_link
        tip_V_pointing_axis = cas.Vector3.from_iterable(tip_V_pointing_axis)

        root_V_goal_axis = root_P_goal_point - root_T_tip.to_position()
        root_V_goal_axis.scale(1)
        root_V_pointing_axis = root_T_tip.dot(tip_V_pointing_axis)
        root_V_pointing_axis.vis_frame = self.tip_link
        root_V_goal_axis.vis_frame = self.tip_link
        god_map.debug_expression_manager.add_debug_expression(
            "root_V_pointing_axis", root_V_pointing_axis, color=Color(1, 0, 0, 1)
        )
        god_map.debug_expression_manager.add_debug_expression(
            "goal_point", root_P_goal_point, color=Color(0, 0, 1, 1)
        )

        root_V_goal_axis_proj = root_V_pointing_axis.project_to_cone(
            root_V_goal_axis, self.cone_theta
        )
        root_V_goal_axis_proj.vis_frame = self.tip_link
        god_map.debug_expression_manager.add_debug_expression(
            "cone_axis", root_V_goal_axis, color=Color(1, 1, 0, 1)
        )
        god_map.debug_expression_manager.add_debug_expression(
            "projected_axis", root_V_goal_axis_proj, color=Color(1, 1, 0, 1)
        )

        self.add_vector_goal_constraints(
            frame_V_current=root_V_pointing_axis,
            frame_V_goal=root_V_goal_axis_proj,
            reference_velocity=self.max_velocity,
            weight=self.weight,
        )
        self.observation_expression = (
            root_V_pointing_axis.angle_between(root_V_goal_axis_proj) <= self.threshold
        )
