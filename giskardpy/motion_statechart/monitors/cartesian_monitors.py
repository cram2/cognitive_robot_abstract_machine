from typing import List

import semantic_digital_twin.spatial_types.spatial_types as cas
from giskardpy.god_map import god_map
from giskardpy.motion_statechart.monitors.monitors import Monitor
from giskardpy.utils.decorators import validated_dataclass
from semantic_digital_twin.world_description.connections import OmniDrive
from semantic_digital_twin.world_description.world_entity import Body


@validated_dataclass
class InWorldSpace(Monitor):
    tip_link: Body
    xyz: List[float]

    def __post_init__(self):
        self.joint: OmniDrive = god_map.world.get_connections_by_type(OmniDrive)[0]
        self.drive_link = self.joint.child
        self.tip_link = self.tip_link
        self.map = self.joint.parent

        map_T_tip = god_map.world.compose_forward_kinematics_expression(
            self.map, self.tip_link
        )
        map_T_drive = god_map.world.compose_forward_kinematics_expression(
            self.map, self.drive_link
        )

        # project to floor
        map_T_tip.z = 0

        error = map_T_tip.to_position() - map_T_drive.to_position()
        error.vis_frame = self.drive_link
        god_map.debug_expression_manager.add_debug_expression(
            f"{self.name}/error", error
        )
        self.observation_expression = cas.logic_and(
            cas.abs(error.x) <= self.xyz[0],
            cas.abs(error.y) <= self.xyz[1],
        )


@validated_dataclass
class PoseReached(Monitor):
    root_link: Body
    tip_link: Body
    goal_pose: cas.TransformationMatrix
    position_threshold: float = 0.01
    orientation_threshold: float = 0.01
    absolute: bool = False

    def __post_init__(self):
        if self.absolute:
            root_T_goal = god_map.world.transform(
                target_frame=self.root_link, spatial_object=self.goal_pose
            )
        else:
            root_T_x = god_map.world.compose_forward_kinematics_expression(
                self.root_link, self.goal_pose.reference_frame
            )
            root_T_goal = root_T_x @ self.goal_pose
            root_T_goal = self.update_expression_on_starting(root_T_goal)

        # %% position error
        r_P_g = root_T_goal.to_position()
        r_P_c = god_map.world.compose_forward_kinematics_expression(
            self.root_link, self.tip_link
        ).to_position()
        distance_to_goal = r_P_g.euclidean_distance(r_P_c)
        position_reached = distance_to_goal < self.position_threshold

        # %% orientation error
        r_R_g = root_T_goal.to_rotation_matrix()
        r_R_c = god_map.world.compose_forward_kinematics_expression(
            self.root_link, self.tip_link
        ).to_rotation_matrix()
        rotation_error = r_R_c.rotational_error(r_R_g)
        orientation_reached = cas.abs(rotation_error) < self.orientation_threshold

        self.observation_expression = cas.logic_and(
            position_reached, orientation_reached
        )


@validated_dataclass
class PositionReached(Monitor):
    root_link: Body
    tip_link: Body
    goal_point: cas.Point3
    threshold: float = 0.01
    absolute: bool = False

    def __post_init__(self):
        if self.absolute:
            root_P_goal = god_map.world.transform(
                target_frame=self.root_link, spatial_object=self.goal_point
            )
        else:
            root_P_x = god_map.world.compose_forward_kinematics_expression(
                self.root_link, self.goal_point.reference_frame
            )
            root_P_goal = root_P_x.dot(self.goal_point)
            root_P_goal = self.update_expression_on_starting(root_P_goal)

        r_P_c = god_map.world.compose_forward_kinematics_expression(
            self.root_link, self.tip_link
        ).to_position()
        distance_to_goal = root_P_goal.euclidean_distance(r_P_c)
        self.observation_expression = distance_to_goal < self.threshold


@validated_dataclass
class OrientationReached(Monitor):
    root_link: Body
    tip_link: Body
    goal_orientation: cas.RotationMatrix
    threshold: float = 0.01
    absolute: bool = False

    def __post_init__(self):
        if self.absolute:
            r_R_g = god_map.world.transform(
                target_frame=self.root_link, spatial_object=self.goal_orientation
            )
        else:
            root_T_x = god_map.world.compose_forward_kinematics_expression(
                self.root_link, self.goal_orientation.reference_frame
            )
            root_R_goal = root_T_x @ self.goal_orientation
            r_R_g = self.update_expression_on_starting(root_R_goal)

        r_R_c = god_map.world.compose_forward_kinematics_expression(
            self.root_link, self.tip_link
        ).to_rotation_matrix()
        rotation_error = r_R_c.rotational_error(r_R_g)
        self.observation_expression = cas.abs(rotation_error) < self.threshold


@validated_dataclass
class PointingAt(Monitor):
    tip_link: Body
    goal_point: cas.Point3
    root_link: Body
    pointing_axis: cas.Vector3
    threshold: float = 0.01

    def __post_init__(self):
        self.root_P_goal_point = god_map.world.transform(
            target_frame=self.root_link, spatial_object=self.goal_point
        )

        tip_V_pointing_axis = god_map.world.transform(
            target_frame=self.tip_link, spatial_object=self.pointing_axis
        )
        tip_V_pointing_axis.scale(1)
        root_T_tip = god_map.world.compose_forward_kinematics_expression(
            self.root_link, self.tip_link
        )
        root_P_tip = root_T_tip.to_position()

        root_V_pointing_axis = root_T_tip.dot(tip_V_pointing_axis)
        root_V_pointing_axis.vis_frame = self.tip_link
        distance = self.root_P_goal_point.distance_point_to_line(
            frame_P_line_point=root_P_tip,
            frame_V_line_direction=root_V_pointing_axis,
        )
        expr = cas.abs(distance) < self.threshold
        self.observation_expression = expr


@validated_dataclass
class VectorsAligned(Monitor):
    root_link: Body
    tip_link: Body
    goal_normal: cas.Vector3
    tip_normal: cas.Vector3
    threshold: float = 0.01

    def __post_init__(self):
        self.tip_V_tip_normal = god_map.world.transform(
            target_frame=self.tip_link, spatial_object=self.tip_normal
        )
        self.tip_V_tip_normal.scale(1)

        self.root_V_root_normal = god_map.world.transform(
            target_frame=self.root_link, spatial_object=self.goal_normal
        )
        self.root_V_root_normal.scale(1)

        root_R_tip = god_map.world.compose_forward_kinematics_expression(
            self.root_link, self.tip_link
        ).to_rotation_matrix()
        root_V_tip_normal = root_R_tip.dot(self.tip_V_tip_normal)
        error = root_V_tip_normal.angle_between_vector(self.root_V_root_normal)
        expr = error < self.threshold
        self.observation_expression = expr


@validated_dataclass
class DistanceToLine(Monitor):
    root_link: Body
    tip_link: Body
    center_point: cas.Point3
    line_axis: cas.Vector3
    line_length: float
    threshold: float = 0.01

    def __post_init__(self):
        root_P_current = god_map.world.compose_forward_kinematics_expression(
            self.root_link, self.tip_link
        ).to_position()
        root_V_line_axis = god_map.world.transform(
            target_frame=self.root_link, spatial_object=self.line_axis
        )
        root_V_line_axis.scale(1)
        root_P_center = god_map.world.transform(
            target_frame=self.root_link, spatial_object=self.center_point
        )
        root_P_line_start = root_P_center + root_V_line_axis * (self.line_length / 2)
        root_P_line_end = root_P_center - root_V_line_axis * (self.line_length / 2)

        distance, closest_point = cas.distance_point_to_line_segment(
            frame_P_current=root_P_current,
            frame_P_line_start=root_P_line_start,
            frame_P_line_end=root_P_line_end,
        )
        expr = cas.less(distance, self.threshold)
        self.observation_expression = expr
