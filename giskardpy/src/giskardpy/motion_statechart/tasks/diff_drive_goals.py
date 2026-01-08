from __future__ import division

from dataclasses import dataclass
from typing import Optional

import krrood.symbolic_math.symbolic_math as sm
from giskardpy.motion_statechart.data_types import DefaultWeights
from giskardpy.motion_statechart.graph_node import Task
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.spatial_types import (
    Point3,
    Vector3,
    RotationMatrix,
    Quaternion,
)
from semantic_digital_twin.world_description.world_entity import Body


@dataclass
class DiffDriveTangentialToPoint(Task):
    goal_point: Point3
    forward: Optional[Vector3] = None
    group_name: Optional[Vector3] = None
    weight: bool = DefaultWeights.WEIGHT_ABOVE_CA
    drive: bool = False

    def __post_init__(self):
        self.tip = context.world.get_kinematic_structure_entity_by_name(
            PrefixedName("base_footprint", prefix=self.group_name)
        )
        self.root = context.world.root()
        self.goal_point = context.world.transform(
            target_frame=context.world.root_link_name, spatial_object=self.goal_point
        )
        self.goal_point.z = 0
        if self.forward is not None:
            self.tip_V_pointing_axis = context.world.transform(
                target_frame=self.tip, spatial_object=self.forward
            )
            self.tip_V_pointing_axis.scale(1)
        else:
            self.tip_V_pointing_axis = Vector3(1, 0, 0)
            self.tip_V_pointing_axis.reference_frame = self.tip

        map_P_center = self.goal_point
        map_T_base = context.world._forward_kinematic_manager.compose_expression(
            self.root, self.tip
        )
        map_P_base = map_T_base.to_position()
        map_V_base_to_center = map_P_center - map_P_base
        map_V_base_to_center = map_V_base_to_center.scale(1)
        map_V_up = Quaternion(0, 0, 1, 0)
        map_V_tangent = map_V_base_to_center.cross(map_V_up)
        tip_V_pointing_axis = Vector3(self.tip_V_pointing_axis)
        map_V_forward = map_T_base @ tip_V_pointing_axis

        if self.drive:
            angle = sm.abs(map_V_forward.angle_between(map_V_tangent))
            self.add_equality_constraint(
                reference_velocity=0.5,
                equality_bound=-angle,
                weight=self.weight,
                task_expression=angle,
                name="/rot",
            )
        else:
            map_R_goal = RotationMatrix.from_vectors(
                x=map_V_tangent, y=None, z=Vector3(0, 0, 1)
            )
            goal_angle = map_R_goal.to_angle(lambda axis: axis[2])
            map_R_base = map_T_base.to_rotation_matrix()
            axis, map_current_angle = map_R_base.to_axis_angle()
            map_current_angle = sm.if_greater_zero(
                axis[2], map_current_angle, -map_current_angle
            )
            angle_error = sm.shortest_angular_distance(map_current_angle, goal_angle)
            self.add_equality_constraint(
                reference_velocity=0.5,
                equality_bound=angle_error,
                weight=self.weight,
                task_expression=map_current_angle,
                name="/rot",
            )


@dataclass
class KeepHandInWorkspace(Task):
    tip_link: Body
    base_footprint: Optional[Body] = None
    map_frame: Optional[Body] = None
    pointing_axis: Optional[Vector3] = None
    max_velocity: float = 0.3
    weight: float = DefaultWeights.WEIGHT_ABOVE_CA

    def __post_init__(self):
        if self.base_footprint is None:
            self.base_footprint = context.world.search_for_link_name("base_footprint")
        if self.map_frame is None:
            self.map_frame = context.world.root_link_name

        if self.pointing_axis is not None:
            self.map_V_pointing_axis = context.world.transform(
                target_frame=self.base_footprint, spatial_object=self.pointing_axis
            )
            self.map_V_pointing_axis.scale(1)
        else:
            self.map_V_pointing_axis = Vector3(1, 0, 0)
            self.map_V_pointing_axis.reference_frame = self.map_frame

        weight = DefaultWeights.WEIGHT_ABOVE_CA
        base_footprint_V_pointing_axis = Vector3(self.map_V_pointing_axis)
        map_T_base_footprint = (
            context.world._forward_kinematic_manager.compose_expression(
                self.map_frame, self.base_footprint
            )
        )
        map_V_pointing_axis = map_T_base_footprint @ base_footprint_V_pointing_axis
        map_T_tip = context.world._forward_kinematic_manager.compose_expression(
            self.map_frame, self.tip_link
        )
        map_V_tip = Vector3(map_T_tip.to_position())
        map_V_tip.y = 0
        map_V_tip.z = 0
        map_P_tip = map_T_tip.to_position()
        map_P_tip.z = 0
        map_P_base_footprint = map_T_base_footprint.to_position()
        map_P_base_footprint.z = 0
        base_footprint_V_tip = map_P_tip - map_P_base_footprint

        map_V_tip.scale(1)
        angle_error = base_footprint_V_tip.angle_between(map_V_pointing_axis)
        self.add_inequality_constraint(
            reference_velocity=0.5,
            lower_error=-angle_error - 0.2,
            upper_error=-angle_error + 0.2,
            weight=weight,
            task_expression=angle_error,
            name="/rot",
        )
