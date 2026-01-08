from __future__ import division

from dataclasses import dataclass

from giskardpy.motion_statechart.data_types import DefaultWeights
from giskardpy.motion_statechart.graph_node import Task
from semantic_digital_twin.spatial_types import Point3, Vector3
from semantic_digital_twin.world_description.world_entity import Body


@dataclass
class GraspBar(Task):
    root_link: Body
    tip_link: Body
    tip_grasp_axis: Vector3
    bar_center: Point3
    bar_axis: Vector3
    bar_length: float
    threshold: float = 0.01
    reference_linear_velocity: float = 0.1
    reference_angular_velocity: float = 0.5
    weight: float = DefaultWeights.WEIGHT_ABOVE_CA

    def __post_init__(self):
        """
        Like a CartesianPose but with more freedom.
        tip_link is allowed to be at any point along bar_axis, that is without bar_center +/- bar_length.
        It will align tip_grasp_axis with bar_axis, but allows rotation around it.
        :param root_link: root link of the kinematic chain
        :param tip_link: tip link of the kinematic chain
        :param tip_grasp_axis: axis of tip_link that will be aligned with bar_axis
        :param bar_center: center of the bar to be grasped
        :param bar_axis: alignment of the bar to be grasped
        :param bar_length: length of the bar to be grasped
        :param reference_linear_velocity: m/s
        :param reference_angular_velocity: rad/s
        :param weight:
        """
        bar_center = context.world.transform(
            target_frame=self.root_link, spatial_object=self.bar_center
        )

        tip_grasp_axis = context.world.transform(
            target_frame=self.tip_link, spatial_object=self.tip_grasp_axis
        )
        tip_grasp_axis.scale(1)

        bar_axis = context.world.transform(
            target_frame=self.root_link, spatial_object=self.bar_axis
        )
        bar_axis.scale(1)

        self.bar_axis = bar_axis
        self.tip_grasp_axis = tip_grasp_axis
        self.bar_center = bar_center

        root_V_bar_axis = self.bar_axis
        tip_V_tip_grasp_axis = self.tip_grasp_axis
        root_P_bar_center = self.bar_center

        root_T_tip = context.world._forward_kinematic_manager.compose_expression(
            self.root_link, self.tip_link
        )
        root_V_tip_normal = root_T_tip @ tip_V_tip_grasp_axis

        self.add_vector_goal_constraints(
            frame_V_current=root_V_tip_normal,
            frame_V_goal=root_V_bar_axis,
            reference_velocity=self.reference_angular_velocity,
            weight=self.weight,
        )

        root_P_tip = context.world._forward_kinematic_manager.compose_expression(
            self.root_link, self.tip_link
        ).to_position()

        root_P_line_start = root_P_bar_center + root_V_bar_axis * self.bar_length / 2
        root_P_line_end = root_P_bar_center - root_V_bar_axis * self.bar_length / 2

        dist, nearest = root_P_tip.distance_to_line_segment(
            root_P_line_start, root_P_line_end
        )

        self.add_point_goal_constraints(
            frame_P_current=root_T_tip.to_position(),
            frame_P_goal=nearest,
            reference_velocity=self.reference_linear_velocity,
            weight=self.weight,
        )

        self.observation_expression = dist <= self.threshold
