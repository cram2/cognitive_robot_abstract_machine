from dataclasses import dataclass

import numpy as np

import krrood.symbolic_math.symbolic_math as sm
from giskardpy.motion_statechart.data_types import DefaultWeights
from giskardpy.motion_statechart.graph_node import Goal
from giskardpy.motion_statechart.graph_node import Task
from semantic_digital_twin.spatial_types import (
    Point3,
    HomogeneousTransformationMatrix,
    Vector3,
)
from semantic_digital_twin.world_description.connections import ActiveConnection1DOF
from semantic_digital_twin.world_description.world_entity import Body


@dataclass
class PrePushDoor(Goal):
    root_link: Body
    tip_link: Body
    door_object: Body
    door_handle: Body
    reference_linear_velocity: float = 0.1
    reference_angular_velocity: float = 0.5
    weight: float = DefaultWeights.WEIGHT_BELOW_CA

    def __post_init__(self):
        """
        The objective is to push the object until desired rotation is reached
        """
        object_joint_name = self.door_object.get_first_parent_connection_of_type(
            ActiveConnection1DOF
        )
        object_V_object_rotation_axis = Vector3(
            context.world.get_joint(object_joint_name).axis
        )

        root_T_tip = context.world._forward_kinematic_manager.compose_expression(
            self.root_link, self.tip_link
        )
        root_T_door = context.world._forward_kinematic_manager.compose_expression(
            self.root_link, self.door_object
        )
        door_P_handle = context.world.compute_forward_kinematics(
            self.door_object, self.door_handle
        )
        temp_point = np.asarray(
            [door_P_handle.x.to_np(), door_P_handle.y.to_np(), door_P_handle.z.to_np()]
        )

        door_V_v1 = np.zeros(3)
        # axis pointing in the direction of handle frame from door joint frame
        direction_axis = np.argmax(abs(temp_point))
        door_V_v1[direction_axis] = 1
        door_V_v2 = object_V_object_rotation_axis  # B
        door_V_v1 = Vector3(door_V_v1)  # A

        door_Pose_tip = context.world._forward_kinematic_manager.compose_expression(
            self.door_object, self.tip_link
        )
        door_P_tip = door_Pose_tip.to_position()
        dist, door_P_nearest = sm.distance_point_to_plane(
            door_P_tip, door_V_v2, door_V_v1
        )

        root_P_nearest_in_rotated_door = HomogeneousTransformationMatrix(
            root_T_door
        ) @ Point3.from_iterable(door_P_nearest)

        context.context.add_debug_expression(
            "goal_point_on_plane",
            Point3.from_iterable(root_P_nearest_in_rotated_door),
        )

        push_door_task = Task(name="pre push door")
        self.add_task(push_door_task)
        push_door_task.add_point_goal_constraints(
            frame_P_current=root_T_tip.to_position(),
            frame_P_goal=Point3.from_iterable(root_P_nearest_in_rotated_door),
            reference_velocity=self.reference_linear_velocity,
            weight=self.weight,
        )
