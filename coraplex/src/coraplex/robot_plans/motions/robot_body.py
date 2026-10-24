from __future__ import annotations

# %% Imports
from dataclasses import dataclass


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


# %% Joint motions
@dataclass
class MoveJointsMotion(BaseMotion, HasMaxJointVelocity):
    """
    Moves any joint on the robot.
    """

    names: list[str]
    """
    List of joint names that should be moved 
    """
    positions: list[float]
    """
    Target positions of joints, should correspond to the list of names.
    """

    align: bool | None = False
    """
    If True, aligns the end-effector with a specified axis (optional).
    """

    tip_link: str | None = None
    """
    Name of the tip link to align with, e.g the object (optional).
    """

    tip_normal: Vector3 | None = None
    """
    Normalized vector representing the current orientation axis of the end-effector
    (optional).
    """

    root_link: str | None = None
    """
    Base link of the robot; typically set to the torso (optional).
    """

    root_normal: Vector3 | None = None
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


# %% Camera pointing
@dataclass
class LookingMotion(BaseMotion):
    """
    Align the camera's optical forward direction with the target.
    """

    target: Pose
    """
    Target pose to look at.
    """

    camera: Camera
    """
    Camera annotation that should look at the target.
    """

    def perform(self) -> None:
        return

    @property
    def _motion_chart(self) -> Pointing | Parallel:
        task = Pointing(
            root_link=self.robot.get_torso().root,
            tip_link=self.camera.root,
            goal_point=self.target.position,
            pointing_axis=self.camera.viewing_direction,
        )
        hold_base = self.keep_base_still()
        if not hold_base:
            return task
        return Parallel([task, *hold_base])
