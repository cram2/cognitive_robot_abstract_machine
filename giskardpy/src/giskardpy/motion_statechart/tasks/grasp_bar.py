from __future__ import annotations

from dataclasses import dataclass, field

from giskardpy.motion_statechart.context import MotionStatechartContext
from giskardpy.motion_statechart.data_types import DefaultWeights
from giskardpy.motion_statechart.graph_node import NodeArtifacts, Task
from semantic_digital_twin.spatial_types import Point3, Vector3
from semantic_digital_twin.world_description.world_entity import Body


@dataclass(eq=False, repr=False)
class GraspBar(Task):
    """
    Like a CartesianPose but with more freedom: the tip link is allowed to be at any
    point along the bar axis within ``bar_center +/- bar_length / 2``.

    It aligns ``tip_grasp_axis`` with ``bar_axis`` while allowing rotation around it.
    """

    root_link: Body = field(kw_only=True)
    """
    Root link of the kinematic chain.
    """

    tip_link: Body = field(kw_only=True)
    """
    Tip link of the kinematic chain.
    """

    tip_grasp_axis: Vector3 = field(kw_only=True)
    """
    Axis of the tip link that will be aligned with the bar axis.
    """

    bar_center: Point3 = field(kw_only=True)
    """
    Center of the bar to be grasped.
    """

    bar_axis: Vector3 = field(kw_only=True)
    """
    Alignment of the bar to be grasped.
    """

    bar_length: float = field(kw_only=True)
    """
    Length of the bar to be grasped.
    """

    threshold: float = field(default=0.01, kw_only=True)
    """
    Distance threshold to the bar for goal achievement in meters.
    """

    reference_linear_velocity: float = field(default=0.1, kw_only=True)
    """
    Reference linear velocity for normalization in m/s.
    """

    reference_angular_velocity: float = field(default=0.5, kw_only=True)
    """
    Reference angular velocity for normalization in rad/s.
    """

    weight: float = field(
        default=DefaultWeights.WEIGHT_ABOVE_COLLISION_AVOIDANCE, kw_only=True
    )
    """
    Priority weight relative to other tasks.
    """

    def build(self, context: MotionStatechartContext) -> NodeArtifacts:
        artifacts = NodeArtifacts()

        root_P_bar_center = context.world.transform(
            target_frame=self.root_link, spatial_object=self.bar_center
        )

        tip_V_tip_grasp_axis = context.world.transform(
            target_frame=self.tip_link, spatial_object=self.tip_grasp_axis
        )
        tip_V_tip_grasp_axis.scale(1)

        root_V_bar_axis = context.world.transform(
            target_frame=self.root_link, spatial_object=self.bar_axis
        )
        root_V_bar_axis.scale(1)

        root_T_tip = context.world.compose_forward_kinematics_expression(
            self.root_link, self.tip_link
        )
        root_V_tip_normal = root_T_tip @ tip_V_tip_grasp_axis

        artifacts.geometry.add_vector_goal_constraints(
            frame_V_current=root_V_tip_normal,
            frame_V_goal=root_V_bar_axis,
            reference_velocity=self.reference_angular_velocity,
            quadratic_weight=self.weight,
        )

        root_P_tip = root_T_tip.to_position()

        root_P_line_start = root_P_bar_center + root_V_bar_axis * self.bar_length / 2
        root_P_line_end = root_P_bar_center - root_V_bar_axis * self.bar_length / 2

        distance, nearest = root_P_tip.distance_to_line_segment(
            root_P_line_start, root_P_line_end
        )

        artifacts.geometry.add_point_goal_constraints(
            frame_P_current=root_P_tip,
            frame_P_goal=nearest,
            reference_velocity=self.reference_linear_velocity,
            quadratic_weight=self.weight,
        )

        artifacts.observation = distance <= self.threshold
        return artifacts
