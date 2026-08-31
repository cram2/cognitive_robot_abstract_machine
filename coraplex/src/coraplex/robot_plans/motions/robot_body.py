from dataclasses import dataclass
from typing import Optional

from typing_extensions import List

from giskardpy.motion_statechart.goals.templates import Parallel
from giskardpy.motion_statechart.tasks.joint_tasks import (
    JointPositionList,
    JointState,
    JointVelocityLimit,
)
from giskardpy.motion_statechart.tasks.pointing import Pointing
from coraplex.robot_plans.mixins import HasMaxJointVelocity
from coraplex.robot_plans.motions.base import BaseMotion
from semantic_digital_twin.robots.robot_parts import Camera
from semantic_digital_twin.spatial_types import Vector3
from semantic_digital_twin.spatial_types.spatial_types import Pose


@dataclass
class MoveJointsMotion(BaseMotion, HasMaxJointVelocity):
    """
    Moves any joint on the robot.
    """

    names: List[str]
    """
    List of joint names that should be moved 
    """
    positions: List[float]
    """
    Target positions of joints, should correspond to the list of names.
    """

    align: Optional[bool] = False
    """
    If True, aligns the end-effector with a specified axis (optional).
    """

    tip_link: Optional[str] = None
    """
    Name of the tip link to align with, e.g the object (optional).
    """

    tip_normal: Optional[Vector3] = None
    """
    Normalized vector representing the current orientation axis of the end-effector
    (optional).
    """

    root_link: Optional[str] = None
    """
    Base link of the robot; typically set to the torso (optional).
    """

    root_normal: Optional[Vector3] = None
    """
    Normalized vector representing the desired orientation axis to align with
    (optional).
    """

    def perform(self):
        return

    @property
    def _motion_chart(self):
        dofs = [self.world.get_connection_by_name(name) for name in self.names]
        joint_task = JointPositionList(
            goal_state=JointState.from_mapping(dict(zip(dofs, self.positions))),
        )
        if self.max_joint_velocity is None:
            return joint_task
        return Parallel(
            [
                joint_task,
                JointVelocityLimit(
                    connections=dofs, max_velocity=self.max_joint_velocity
                ),
            ]
        )


@dataclass
class LookingMotion(BaseMotion):
    """
    Lets the robot look at a point.
    """

    target: Pose
    """
    Target pose to look at.
    """

    camera: Camera
    """
    Camera annotation that should look at the target.
    """

    def perform(self):
        return

    @property
    def _motion_chart(self):
        return Pointing(
            root_link=self.robot.get_torso().root,
            tip_link=self.camera.root,
            goal_point=self.target.to_position(),
            pointing_axis=self.camera.forward_facing_axis,
        )
