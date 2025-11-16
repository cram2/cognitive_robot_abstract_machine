from dataclasses import dataclass

import semantic_digital_twin.spatial_types.spatial_types as cas
from giskardpy.god_map import god_map
from giskardpy.motion_statechart.data_types import DefaultWeights
from giskardpy.motion_statechart.graph_node import Task
from semantic_digital_twin.world_description.geometry import Color
from semantic_digital_twin.world_description.world_entity import Body


@dataclass
class AlignPlanes(Task):
    root_link: Body
    tip_link: Body
    goal_normal: cas.Vector3
    tip_normal: cas.Vector3
    threshold: float = 0.01
    reference_velocity: float = 0.5
    weight: float = DefaultWeights.WEIGHT_ABOVE_CA

    def __post_init__(self):
        """
        This goal will use the kinematic chain between tip and root to align tip_normal with goal_normal.
        :param root_link: root link of the kinematic chain
        :param tip_link: tip link of the kinematic chain
        :param goal_normal:
        :param tip_normal:
        :param reference_velocity: rad/s
        :param weight:
        """
        self.tip_V_tip_normal = context.world.transform(
            target_frame=self.tip_link, spatial_object=self.tip_normal
        )
        self.tip_V_tip_normal.scale(1)

        self.root_V_root_normal = context.world.transform(
            target_frame=self.root_link, spatial_object=self.goal_normal
        )
        self.root_V_root_normal.scale(1)

        root_R_tip = context.world._forward_kinematic_manager.compose_expression(
            self.root_link, self.tip_link
        ).to_rotation_matrix()
        root_V_tip_normal = root_R_tip @ self.tip_V_tip_normal
        self.add_vector_goal_constraints(
            frame_V_current=root_V_tip_normal,
            frame_V_goal=self.root_V_root_normal,
            reference_velocity=self.reference_velocity,
            weight=self.weight,
        )
        root_V_tip_normal.vis_frame = self.tip_link
        god_map.debug_expression_manager.add_debug_expression(
            f"{self.name}/current_normal", root_V_tip_normal, color=Color(1, 0, 0, 1)
        )
        self.root_V_root_normal.vis_frame = self.tip_link
        god_map.debug_expression_manager.add_debug_expression(
            f"{self.name}/goal_normal", self.root_V_root_normal, color=Color(0, 0, 1, 1)
        )

        self.observation_expression = (
            root_V_tip_normal.angle_between(self.root_V_root_normal) <= self.threshold
        )
