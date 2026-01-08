from dataclasses import dataclass
from typing import List

import krrood.symbolic_math.symbolic_math as sm
from giskardpy.motion_statechart.graph_node import MotionStatechartNode
from semantic_digital_twin.spatial_types import (
    Point3,
    Vector3,
    RotationMatrix,
    HomogeneousTransformationMatrix,
)
from semantic_digital_twin.world_description.connections import OmniDrive
from semantic_digital_twin.world_description.world_entity import Body


@dataclass
class InWorldSpace(MotionStatechartNode):
    tip_link: Body
    xyz: List[float]

    def __post_init__(self):
        self.joint: OmniDrive = context.world.get_connections_by_type(OmniDrive)[0]
        self.drive_link = self.joint.child
        self.tip_link = self.tip_link
        self.map = self.joint.parent

        map_T_tip = context.world._forward_kinematic_manager.compose_expression(
            self.map, self.tip_link
        )
        map_T_drive = context.world._forward_kinematic_manager.compose_expression(
            self.map, self.drive_link
        )

        # project to floor
        map_T_tip.z = 0

        error = map_T_tip.to_position() - map_T_drive.to_position()
        error.vis_frame = self.drive_link
        context.context.add_debug_expression(f"{self.name}/error", error)
        self.observation_expression = sm.logic_and(
            sm.abs(error.x) <= self.xyz[0],
            sm.abs(error.y) <= self.xyz[1],
        )


@dataclass
class PoseReached(MotionStatechartNode):
    root_link: Body
    tip_link: Body
    goal_pose: HomogeneousTransformationMatrix
    position_threshold: float = 0.01
    orientation_threshold: float = 0.01
    absolute: bool = False

    def __post_init__(self):
        if self.absolute:
            root_T_goal = context.world.transform(
                target_frame=self.root_link, spatial_object=self.goal_pose
            )
        else:
            root_T_x = context.world._forward_kinematic_manager.compose_expression(
                self.root_link, self.goal_pose.reference_frame
            )
            root_T_goal = root_T_x @ self.goal_pose
            root_T_goal = self.update_expression_on_starting(root_T_goal)

        # %% position error
        r_P_g = root_T_goal.to_position()
        r_P_c = context.world._forward_kinematic_manager.compose_expression(
            self.root_link, self.tip_link
        ).to_position()
        distance_to_goal = r_P_g.euclidean_distance(r_P_c)
        position_reached = distance_to_goal < self.position_threshold

        # %% orientation error
        r_R_g = root_T_goal.to_rotation_matrix()
        r_R_c = context.world._forward_kinematic_manager.compose_expression(
            self.root_link, self.tip_link
        ).to_rotation_matrix()
        rotation_error = r_R_c.rotational_error(r_R_g)
        orientation_reached = sm.abs(rotation_error) < self.orientation_threshold

        self.observation_expression = sm.logic_and(
            position_reached, orientation_reached
        )


@dataclass
class PositionReached(MotionStatechartNode):
    root_link: Body
    tip_link: Body
    goal_point: Point3
    threshold: float = 0.01
    absolute: bool = False

    def __post_init__(self):
        if self.absolute:
            root_P_goal = context.world.transform(
                target_frame=self.root_link, spatial_object=self.goal_point
            )
        else:
            root_P_x = context.world._forward_kinematic_manager.compose_expression(
                self.root_link, self.goal_point.reference_frame
            )
            root_P_goal = root_P_x.dot(self.goal_point)
            root_P_goal = self.update_expression_on_starting(root_P_goal)

        r_P_c = context.world._forward_kinematic_manager.compose_expression(
            self.root_link, self.tip_link
        ).to_position()
        distance_to_goal = root_P_goal.euclidean_distance(r_P_c)
        self.observation_expression = distance_to_goal < self.threshold


@dataclass
class OrientationReached(MotionStatechartNode):
    root_link: Body
    tip_link: Body
    goal_orientation: RotationMatrix
    threshold: float = 0.01
    absolute: bool = False

    def __post_init__(self):
        if self.absolute:
            r_R_g = context.world.transform(
                target_frame=self.root_link, spatial_object=self.goal_orientation
            )
        else:
            root_T_x = context.world._forward_kinematic_manager.compose_expression(
                self.root_link, self.goal_orientation.reference_frame
            )
            root_R_goal = root_T_x @ self.goal_orientation
            r_R_g = self.update_expression_on_starting(root_R_goal)

        r_R_c = context.world._forward_kinematic_manager.compose_expression(
            self.root_link, self.tip_link
        ).to_rotation_matrix()
        rotation_error = r_R_c.rotational_error(r_R_g)
        self.observation_expression = sm.abs(rotation_error) < self.threshold


@dataclass
class PointingAt(MotionStatechartNode):
    tip_link: Body
    goal_point: Point3
    root_link: Body
    pointing_axis: Vector3
    threshold: float = 0.01

    def __post_init__(self):
        self.root_P_goal_point = context.world.transform(
            target_frame=self.root_link, spatial_object=self.goal_point
        )

        tip_V_pointing_axis = context.world.transform(
            target_frame=self.tip_link, spatial_object=self.pointing_axis
        )
        tip_V_pointing_axis.scale(1)
        root_T_tip = context.world._forward_kinematic_manager.compose_expression(
            self.root_link, self.tip_link
        )
        root_P_tip = root_T_tip.to_position()

        root_V_pointing_axis = root_T_tip.dot(tip_V_pointing_axis)
        root_V_pointing_axis.vis_frame = self.tip_link
        distance = self.root_P_goal_point.distance_point_to_line(
            frame_P_line_point=root_P_tip,
            frame_V_line_direction=root_V_pointing_axis,
        )
        expr = sm.abs(distance) < self.threshold
        self.observation_expression = expr


@dataclass
class VectorsAligned(MotionStatechartNode):
    root_link: Body
    tip_link: Body
    goal_normal: Vector3
    tip_normal: Vector3
    threshold: float = 0.01

    def __post_init__(self):
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
        root_V_tip_normal = root_R_tip.dot(self.tip_V_tip_normal)
        error = root_V_tip_normal.angle_between_vector(self.root_V_root_normal)
        expr = error < self.threshold
        self.observation_expression = expr


@dataclass
class DistanceToLine(MotionStatechartNode):
    root_link: Body
    tip_link: Body
    center_point: Point3
    line_axis: Vector3
    line_length: float
    threshold: float = 0.01

    def __post_init__(self):
        root_P_current = context.world._forward_kinematic_manager.compose_expression(
            self.root_link, self.tip_link
        ).to_position()
        root_V_line_axis = context.world.transform(
            target_frame=self.root_link, spatial_object=self.line_axis
        )
        root_V_line_axis.scale(1)
        root_P_center = context.world.transform(
            target_frame=self.root_link, spatial_object=self.center_point
        )
        root_P_line_start = root_P_center + root_V_line_axis * (self.line_length / 2)
        root_P_line_end = root_P_center - root_V_line_axis * (self.line_length / 2)

        distance, closest_point = sm.distance_point_to_line_segment(
            frame_P_current=root_P_current,
            frame_P_line_start=root_P_line_start,
            frame_P_line_end=root_P_line_end,
        )
        expr = distance < self.threshold
        self.observation_expression = expr
