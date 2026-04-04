from __future__ import annotations

import datetime
from dataclasses import dataclass
from datetime import timedelta
from typing import Tuple, List

from typing_extensions import Optional, Dict, Any

from pycram.datastructures.enums import AxisIdentifier, Arms

from pycram.datastructures.trajectory import PoseTrajectory
from pycram.plans.factories import execute_single, sequential
from pycram.robot_plans.actions.base import ActionDescription
from pycram.robot_plans.motions.gripper import MoveGripperMotion, MoveTCPWaypointsMotion
from pycram.robot_plans.motions.robot_body import MoveJointsMotion
from pycram.validation.goal_validator import create_multiple_joint_goal_validator
from pycram.view_manager import ViewManager
from semantic_digital_twin.datastructures.definitions import (
    TorsoState,
    GripperState,
    StaticJointState,
)


@dataclass
class MoveTorsoAction(ActionDescription):
    """
    Move the torso of the robot up and down.
    """

    torso_state: TorsoState
    """
    The state of the torso that should be set
    """

    def execute(self) -> None:
        joint_state = self.robot.torso.get_joint_state_by_type(self.torso_state)
        self.add_subplan(
            execute_single(
                MoveJointsMotion(
                    [c.name.name for c in joint_state.connections],
                    joint_state.target_values,
                ),
            )
        ).perform()

    def validate(
        self,
        result: Optional[Any] = None,
        max_wait_time: timedelta = timedelta(seconds=2),
    ):
        """
        Create a goal validator for the joint positions and wait until the goal is achieved or the timeout is reached.
        """

        joint_positions: dict = (
            RobotDescription.current_robot_description.get_static_joint_chain(
                "torso", self.torso_state
            )
        )
        validator = create_multiple_joint_goal_validator(
            World.current_world.robot, joint_positions
        )
        validator.wait_until_goal_is_achieved(
            max_wait_time=max_wait_time, time_per_read=timedelta(milliseconds=20)
        )
        if not validator.goal_achieved:
            raise TorsoGoalNotReached(validator)


@dataclass
class SetGripperAction(ActionDescription):
    """
    Set the gripper state of the robot.
    """

    gripper: Arms
    """
    The gripper that should be set 
    """
    motion: GripperState
    """
    The motion that should be set on the gripper
    """

    def execute(self) -> None:
        arms = [Arms.LEFT, Arms.RIGHT] if self.gripper == Arms.BOTH else [self.gripper]
        self.add_subplan(
            sequential(
                [MoveGripperMotion(gripper=arm, motion=self.motion) for arm in arms]
            )
        ).perform()

    def validate(
        self,
        result: Optional[Any] = None,
        max_wait_time: timedelta = timedelta(seconds=2),
    ):
        """
        Needs gripper state to be read or perceived.
        """
        pass


@dataclass
class ParkArmsAction(ActionDescription):
    """
    Park the arms of the robot.
    """

    arm: Arms
    """
    Entry from the enum for which arm should be parked.
    """

    def execute(self) -> None:
        joint_names, joint_poses = self.get_joint_poses()

        self.add_subplan(
            execute_single(MoveJointsMotion(names=joint_names, positions=joint_poses))
        ).perform()

    def get_joint_poses(self) -> Tuple[List[str], List[float]]:
        """
        :return: The joint positions that should be set for the arm to be in the park position.
        """
        arm_chain = ViewManager().get_all_arm_views(self.arm, self.robot)
        names = []
        values = []
        for arm in arm_chain:
            joint_state = arm.get_joint_state_by_type(StaticJointState.PARK)
            names.extend([c.name.name for c in joint_state.connections])
            values.extend(joint_state.target_values)
        return names, values

    def validate(
        self,
        result: Optional[Any] = None,
        max_wait_time: timedelta = timedelta(seconds=2),
    ):
        """
        Create a goal validator for the joint positions and wait until the goal is achieved or the timeout is reached.
        """
        joint_poses = self.get_joint_poses()
        validator = create_multiple_joint_goal_validator(
            World.current_world.robot, joint_poses
        )
        validator.wait_until_goal_is_achieved(
            max_wait_time=max_wait_time, time_per_read=timedelta(milliseconds=20)
        )
        if not validator.goal_achieved:
            raise ConfigurationNotReached(
                validator, configuration_type=StaticJointState.Park
            )


@dataclass
class CarryAction(ActionDescription):
    """
    Parks the robot's arms. And align the arm with the given Axis of a frame.
    """

    arm: Arms
    """
    Entry from the enum for which arm should be parked.
    """

    align: Optional[bool] = False
    """
    If True, aligns the end-effector with a specified axis.
    """

    tip_link: Optional[str] = None
    """
    Name of the tip link to align with, e.g the object.
    """

    tip_axis: Optional[AxisIdentifier] = None
    """
    Tip axis of the tip link, that should be aligned.
    """

    root_link: Optional[str] = None
    """
    Base link of the robot; typically set to the torso.
    """

    root_axis: Optional[AxisIdentifier] = None
    """
    Goal axis of the root link, that should be used to align with.
    """

    def execute(self) -> None:
        joint_poses = self.get_joint_poses()
        tip_normal = self.axis_to_vector3_stamped(self.tip_axis, link=self.tip_link)
        root_normal = self.axis_to_vector3_stamped(self.root_axis, link=self.root_link)

        self.add_subplan(
            execute_single(
                MoveJointsMotion(
                    names=list(joint_poses.keys()),
                    positions=list(joint_poses.values()),
                    align=self.align,
                    tip_link=self.tip_link,
                    tip_normal=tip_normal,
                    root_link=self.root_link,
                    root_normal=root_normal,
                )
            )
        ).perform()

    def get_joint_poses(self) -> Dict[str, float]:
        """
        :return: The joint positions that should be set for the arm to be in the park position.
        """
        joint_poses = {}
        arm_chains = RobotDescription.current_robot_description.get_arm_chain(self.arm)
        if type(arm_chains) is not list:
            joint_poses = arm_chains.get_static_joint_states(StaticJointState.Park)
        else:
            for arm_chain in RobotDescription.current_robot_description.get_arm_chain(
                self.arm
            ):
                joint_poses.update(
                    arm_chain.get_static_joint_states(StaticJointState.Park)
                )
        return joint_poses

    def axis_to_vector3_stamped(
        self, axis: AxisIdentifier, link: str = "base_link"
    ) -> Vector3:
        v = {
            AxisIdentifier.X: Vector3(x=1.0, y=0.0, z=0.0),
            AxisIdentifier.Y: Vector3(x=0.0, y=1.0, z=0.0),
            AxisIdentifier.Z: Vector3(x=0.0, y=0.0, z=1.0),
        }[axis]
        v.frame_id = link
        return v

    def validate(
        self,
        result: Optional[Any] = None,
        max_wait_time: timedelta = timedelta(seconds=2),
    ):
        """
        Create a goal validator for the joint positions and wait until the goal is achieved or the timeout is reached.
        """
        joint_poses = self.get_joint_poses()
        validator = create_multiple_joint_goal_validator(
            World.current_world.robot, joint_poses
        )
        validator.wait_until_goal_is_achieved(
            max_wait_time=max_wait_time, time_per_read=timedelta(milliseconds=20)
        )
        if not validator.goal_achieved:
            raise ConfigurationNotReached(
                validator, configuration_type=StaticJointState.Park
            )


@dataclass
class FollowToolCenterPointPathAction(ActionDescription):
    """
    Represents an action to move a robotic arm's TCP (Tool Center Point) along a
    path of poses.
    """

    target_locations: PoseTrajectory
    """
    Path poses for the TCP motion.
    """

    arm: Arms
    """
    Entry from the enum for which arm should be parked.
    """

    def execute(self) -> None:
        target_locations = list(self.target_locations.poses)

        motion = MoveTCPWaypointsMotion(
            target_locations,
            self.arm,
            allow_gripper_collision=True,
        )

        self.add_subplan(execute_single(motion)).perform()

    def validate(
        self,
        result: Optional[Any] = None,
        max_wait_time: timedelta = timedelta(seconds=2),
    ):
        pass
