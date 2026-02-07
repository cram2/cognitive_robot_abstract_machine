from __future__ import annotations

import datetime
from dataclasses import dataclass
from datetime import timedelta
from typing import Tuple, List

from typing_extensions import Union, Optional, Type, Dict, Any, Iterable

from semantic_digital_twin.datastructures.definitions import (
    TorsoState,
    GripperState,
    StaticJointState,
)
from ....datastructures.enums import AxisIdentifier, Arms
from ....datastructures.partial_designator import PartialDesignator
from ....datastructures.pose import Vector3Stamped
from ....failures import TorsoGoalNotReached, ConfigurationNotReached
from ....has_parameters import has_parameters
from ....language import SequentialPlan
from ....robot_description import ViewManager
from ....robot_plans.actions.base import ActionDescription
from ....robot_plans.motions.gripper import MoveGripperMotion
from ....robot_plans.motions.robot_body import MoveJointsMotion
from ....validation.goal_validator import create_multiple_joint_goal_validator


@has_parameters
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
        joint_state = self.robot_view.torso.get_joint_state_by_type(self.torso_state)

        SequentialPlan(
            self.context,
            MoveJointsMotion(
                [c.name.name for c in joint_state.connections],
                joint_state.target_values,
            ),
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

    @classmethod
    def description(
        cls, torso_state: Union[Iterable[TorsoState], TorsoState]
    ) -> PartialDesignator[MoveTorsoAction]:
        return PartialDesignator[MoveTorsoAction](
            MoveTorsoAction, torso_state=torso_state
        )


@has_parameters
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
        for arm in arms:
            SequentialPlan(
                self.context, MoveGripperMotion(gripper=arm, motion=self.motion)
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

    @classmethod
    def description(
        cls,
        gripper: Union[Iterable[Arms], Arms],
        motion: Union[Iterable[GripperState], GripperState] = None,
    ) -> PartialDesignator[SetGripperAction]:
        return PartialDesignator[SetGripperAction](
            SetGripperAction, gripper=gripper, motion=motion
        )


@has_parameters
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

        SequentialPlan(
            self.context, MoveJointsMotion(names=joint_names, positions=joint_poses)
        ).perform()

    def get_joint_poses(self) -> Tuple[List[str], List[float]]:
        """
        :return: The joint positions that should be set for the arm to be in the park position.
        """
        arm_chain = ViewManager().get_all_arm_views(self.arm, self.robot_view)
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

    @classmethod
    def description(
        cls, arm: Union[Iterable[Arms], Arms]
    ) -> PartialDesignator[ParkArmsAction]:
        return PartialDesignator[ParkArmsAction](cls, arm=arm)


@has_parameters
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

        SequentialPlan(
            self.context,
            MoveJointsMotion(
                names=list(joint_poses.keys()),
                positions=list(joint_poses.values()),
                align=self.align,
                tip_link=self.tip_link,
                tip_normal=tip_normal,
                root_link=self.root_link,
                root_normal=root_normal,
            ),
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
    ) -> Vector3Stamped:
        v = {
            AxisIdentifier.X: Vector3Stamped(x=1.0, y=0.0, z=0.0),
            AxisIdentifier.Y: Vector3Stamped(x=0.0, y=1.0, z=0.0),
            AxisIdentifier.Z: Vector3Stamped(x=0.0, y=0.0, z=1.0),
        }[axis]
        v.frame_id = link
        v.header.stamp = datetime.datetime.now()
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

    @classmethod
    def description(
        cls,
        arm: Union[Iterable[Arms], Arms],
        align: Optional[bool] = False,
        tip_link: Optional[str] = None,
        tip_axis: Optional[AxisIdentifier] = None,
        root_link: Optional[str] = None,
        root_axis: Optional[AxisIdentifier] = None,
    ) -> PartialDesignator[CarryAction]:
        return PartialDesignator[CarryAction](
            cls,
            arm=arm,
            align=align,
            tip_link=tip_link,
            tip_axis=tip_axis,
            root_link=root_link,
            root_axis=root_axis,
        )


MoveTorsoActionDescription = MoveTorsoAction.description
SetGripperActionDescription = SetGripperAction.description
ParkArmsActionDescription = ParkArmsAction.description
CarryActionDescription = CarryAction.description
