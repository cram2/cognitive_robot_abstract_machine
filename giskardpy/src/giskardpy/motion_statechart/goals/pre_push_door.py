from dataclasses import dataclass

import numpy as np

from krrood.symbolic_math.symbolic_math import Scalar
from giskardpy.motion_statechart.context import BuildContext
from giskardpy.motion_statechart.data_types import DefaultWeights
from giskardpy.motion_statechart.graph_node import (
    Goal,
    Task,
    NodeArtifacts,
    DebugExpression,
)
from semantic_digital_twin.spatial_types import (
    Point3,
    Vector3,
)
from semantic_digital_twin.world_description.connections import ActiveConnection1DOF
from semantic_digital_twin.world_description.world_entity import Body
from semantic_digital_twin.world_description.geometry import Color


@dataclass
class PrePushDoor(Goal):
    root_link: Body
    tip_link: Body
    door_object: Body
    door_handle: Body
    threshold: float = 0.01
    reference_linear_velocity: float = 0.1
    reference_angular_velocity: float = 0.5
    weight: float = DefaultWeights.WEIGHT_BELOW_CA

    def expand(self, context: BuildContext) -> None:
        """
        The objective is to push the object until desired rotation is reached.
        """
        # Get the door's rotation joint and its axis
        object_connection: ActiveConnection1DOF = (
            self.door_object.get_first_parent_connection_of_type(ActiveConnection1DOF)
        )
        object_V_object_rotation_axis = Vector3.from_iterable(
            object_connection.axis.to_np()
        )

        # Compute forward kinematics
        root_T_tip = context.world._forward_kinematic_manager.compose_expression(
            self.root_link, self.tip_link
        )
        root_T_door = context.world._forward_kinematic_manager.compose_expression(
            self.root_link, self.door_object
        )
        door_P_handle = context.world.compute_forward_kinematics(
            self.door_object, self.door_handle
        )

        # Find the dominant axis from door joint to handle frame
        temp_point = np.asarray(
            [door_P_handle.x.to_np(), door_P_handle.y.to_np(), door_P_handle.z.to_np()]
        )
        door_V_v1 = np.zeros(3)
        direction_axis = np.argmax(abs(temp_point))
        door_V_v1[direction_axis] = 1
        door_V_v2 = object_V_object_rotation_axis  # rotation axis (B)
        door_V_v1 = Vector3.from_iterable(door_V_v1)  # handle direction axis (A)

        # Project the tip onto the plane defined by the door axes
        door_Pose_tip = context.world._forward_kinematic_manager.compose_expression(
            self.door_object, self.tip_link
        )
        door_P_tip = door_Pose_tip.to_position()
        door_P_nearest, _ = door_P_tip.project_to_plane(door_V_v1, door_V_v2)

        # Transform the nearest point back into the root frame
        root_P_nearest_in_rotated_door = root_T_door @ door_P_nearest

        # Compute the completion condition
        dist = root_T_tip.to_position().euclidean_distance(
            root_P_nearest_in_rotated_door
        )
        self.observation_expression = dist <= Scalar(self.threshold)

        # Create and register the task
        push_door_task = _PrePushDoorTask(
            name="pre push door",
            frame_P_current=root_T_tip.to_position(),
            frame_P_goal=root_P_nearest_in_rotated_door,
            goal_point_on_plane=root_P_nearest_in_rotated_door,
            reference_velocity=self.reference_linear_velocity,
            weight=self.weight,
        )
        self.add_node(push_door_task)


@dataclass
class _PrePushDoorTask(Task):
    frame_P_current: Point3
    frame_P_goal: Point3
    goal_point_on_plane: Point3
    reference_velocity: float
    weight: float

    def build(self, context: BuildContext) -> NodeArtifacts:
        artifacts = NodeArtifacts()
        artifacts.constraints.add_point_goal_constraints(
            frame_P_current=self.frame_P_current,
            frame_P_goal=self.frame_P_goal,
            reference_velocity=self.reference_velocity,
            weight=self.weight,
        )
        artifacts.debug_expressions.append(
            DebugExpression(
                name="goal_point_on_plane",
                expression=self.goal_point_on_plane,
                color=Color(0, 0.5, 0.5, 1),
            )
        )
        return artifacts
