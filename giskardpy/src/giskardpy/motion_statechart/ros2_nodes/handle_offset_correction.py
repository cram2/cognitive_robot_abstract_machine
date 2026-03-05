from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from geometry_msgs.msg import Vector3Stamped as ROSVector3Stamped

from giskardpy.motion_statechart.context import MotionStatechartContext
from giskardpy.motion_statechart.data_types import ObservationStateValues
from giskardpy.motion_statechart.ros2_nodes.topic_monitor import TopicSubscriberNode
from semantic_digital_twin.spatial_types import Vector3, HomogeneousTransformationMatrix
from semantic_digital_twin.world_description.connections import Connection6DoF
from semantic_digital_twin.world_description.world_entity import (
    KinematicStructureEntity,
)


@dataclass(eq=False, repr=False)
class HandleOffsetCorrection(TopicSubscriberNode[ROSVector3Stamped]):
    """
    Monitors the /robokudo/handle_offset topic and adjusts the door's floating
    Connection6DoF transform at runtime to correct for handle position drift.

    Turns True once the received offset vector's magnitude falls below `threshold`.

    :param root_link: Root of the kinematic chain (e.g. map frame).
    :param tip_link: Tip of the kinematic chain (e.g. camera_link).
    :param door_move_connection: The Connection6DoF of the door used to adjust position.
    :param goal_vector: Initial goal vector in root_link frame; overwritten at runtime
                        by incoming topic messages.
    :param threshold: Magnitude threshold below which the node turns True.
    :param error_adjustment: Divisor applied to the error vector to reduce the per-step
                             correction distance and make motion more responsive.
    :param topic_name: ROS2 topic to subscribe to for offset corrections.
    """

    root_link: KinematicStructureEntity = field(kw_only=True)
    tip_link: KinematicStructureEntity = field(kw_only=True)
    door_move_connection: Connection6DoF = field(kw_only=True)
    goal_vector: Vector3 = field(kw_only=True)
    threshold: float = field(default=50.0, kw_only=True)
    error_adjustment: float = field(default=20.0, kw_only=True)
    topic_name: str = field(default="/robokudo/handle_offset", kw_only=True)
    msg_type: type = field(init=False, default=ROSVector3Stamped)

    # Cached root-frame goal vector, updated by the subscriber callback
    _root_V_goal: Vector3 = field(init=False, repr=False)

    def build(self, context: MotionStatechartContext):
        # Transform the initial goal vector into the root frame once at build time
        self._root_V_goal = context.world.transform(
            target_frame=self.root_link,
            spatial_object=self.goal_vector,
        )
        return super().build(context)

    def callback(self, msg: ROSVector3Stamped) -> None:
        """
        ROS2 subscriber callback. Converts the incoming ROS message into a
        semantic_digital_twin Vector3 and stores it in root_link frame.

        The frame_id in the message header is expected to resolve to a body in the world.
        """
        # Store raw values; frame transform happens on_tick when world state is current
        self._root_V_goal = Vector3(
            x=msg.vector.x,
            y=msg.vector.y,
            z=msg.vector.z,
            reference_frame=self.root_link,
        )

    def on_tick(
        self, context: MotionStatechartContext
    ) -> Optional[ObservationStateValues]:
        super().on_tick(context)

        if not self.has_msg():
            return ObservationStateValues.UNKNOWN

        # Current tip position in root frame
        root_T_tip = context.world.compute_forward_kinematics(
            self.root_link, self.tip_link
        )
        root_P_tip = root_T_tip.to_np()[:3, 3]

        # Goal point = tip position + scaled goal vector
        goal_vec_np = np.array(
            [
                self._root_V_goal.x,
                self._root_V_goal.y,
                self._root_V_goal.z,
            ]
        )
        root_P_goal = root_P_tip + goal_vec_np

        # Error vector reduced by adjustment factor
        root_V_error = (root_P_goal - root_P_tip) / self.error_adjustment

        # Compute current parent_T_child for the door connection and apply correction
        parent_T_child = context.world.compute_forward_kinematics(
            self.door_move_connection.parent, self.door_move_connection.child
        )
        parent_T_child_np = parent_T_child.to_np()

        # Transform error into parent frame of the door connection
        parent_T_root = context.world.compute_forward_kinematics(
            self.door_move_connection.parent, self.root_link
        )
        parent_V_error = parent_T_root.to_np()[:3, :3] @ root_V_error

        # Apply correction to the translation component
        parent_T_child_np[:3, 3] += parent_V_error

        # Write corrected transform back into the Connection6DoF
        self.door_move_connection.origin = HomogeneousTransformationMatrix(
            data=parent_T_child_np,
            reference_frame=self.door_move_connection.parent,
            child_frame=self.door_move_connection.child,
        )

        # Turn True when the offset magnitude falls below threshold
        norm = float(np.linalg.norm(goal_vec_np))
        if norm <= self.threshold:
            return ObservationStateValues.TRUE
        return ObservationStateValues.FALSE

    def on_reset(self, context: MotionStatechartContext):
        super().on_reset(context)
        # Re-transform the initial goal vector on reset in case the world state changed
        self._root_V_goal = context.world.transform(
            target_frame=self.root_link,
            spatial_object=self.goal_vector,
        )
