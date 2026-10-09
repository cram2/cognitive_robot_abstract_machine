from dataclasses import dataclass

from typing_extensions import List

from giskardpy.motion_statechart.goals.templates import Parallel
from giskardpy.motion_statechart.tasks.joint_tasks import (
    JointPositionList,
    JointState,
    JointVelocityLimit,
)
from giskardpy.motion_statechart.tasks.pointing import Pointing
from coraplex.robot_plans.mixins import (
    CameraTargetParameters,
    MaxJointVelocityParameter,
)
from coraplex.robot_plans.motions.base import BaseMotion


@dataclass
class MoveJointsMotion(BaseMotion, MaxJointVelocityParameter):
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
class LookingMotion(BaseMotion, CameraTargetParameters):
    """
    Lets the robot look at a point.
    """

    def perform(self):
        return

    @property
    def _motion_chart(self):
        return Pointing(
            root_link=self.robot.get_torso().root,
            tip_link=self.camera.root,
            goal_point=self.target.position,
            pointing_axis=self.camera.forward_facing_axis,
        )
