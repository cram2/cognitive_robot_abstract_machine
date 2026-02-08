from dataclasses import dataclass
from typing import Self

from .abstract_robot import (
    AbstractRobot,
    Arm,
    Neck,
    Finger,
    ParallelGripper,
    Camera,
    Torso,
    Base,
)
from .robot_mixins import HasNeck, HasArms
from ..datastructures.definitions import StaticJointState, GripperState, TorsoState
from ..datastructures.joint_state import JointState
from ..datastructures.prefixed_name import PrefixedName
from ..spatial_types import Quaternion
from ..spatial_types.spatial_types import Vector3
from ..world import World
from ..world_description.connections import FixedConnection


@dataclass(eq=False)
class Stretch(AbstractRobot, HasArms, HasNeck):
    """
    Class that describes the Stretch Robot.
    """

    def load_srdf(self):
        """
        Loads the SRDF file for the Stretch robot, if it exists.
        """
        ...

    @classmethod
    def from_world(cls, world: World) -> Self:
        """
        Creates a Stretch robot view from the given world.

        :param world: The world from which to create the robot view.

        :return: A Stretch robot view.
        """

        with world.modify_world():
            stretch = cls(
                name=PrefixedName("stretch", prefix=world.name),
                root=world.get_body_by_name("base_link"),
                _world=world,
            )

            # Create arm
            gripper_thumb = Finger(
                name=PrefixedName("gripper_thumb", prefix=stretch.name.name),
                root=world.get_body_by_name("link_gripper_finger_left"),
                tip=world.get_body_by_name("link_gripper_fingertip_left"),
                _world=world,
            )

            gripper_finger = Finger(
                name=PrefixedName("gripper_finger", prefix=stretch.name.name),
                root=world.get_body_by_name("link_gripper_finger_right"),
                tip=world.get_body_by_name("link_gripper_fingertip_right"),
                _world=world,
            )

            gripper = ParallelGripper(
                name=PrefixedName("gripper", prefix=stretch.name.name),
                root=world.get_body_by_name("link_straight_gripper"),
                tool_frame=world.get_body_by_name("link_grasp_center"),
                front_facing_orientation=Quaternion(0, 0, 0, 1),
                front_facing_axis=Vector3(0, 0, 1),
                thumb=gripper_thumb,
                finger=gripper_finger,
                _world=world,
            )

            arm = Arm(
                name=PrefixedName("arm", prefix=stretch.name.name),
                root=world.get_body_by_name("link_mast"),
                tip=world.get_body_by_name("link_wrist_roll"),
                manipulator=gripper,
                _world=world,
            )

            stretch.add_arm(arm)

            # Create camera and neck
            camera_color = Camera(
                name=PrefixedName(
                    "camera_color_optical_frame", prefix=stretch.name.name
                ),
                root=world.get_body_by_name("camera_color_optical_frame"),
                forward_facing_axis=Vector3(0, 0, 1),
                minimal_height=1.322,
                maximal_height=1.322,
                _world=world,
            )

            camera_depth = Camera(
                name=PrefixedName(
                    "camera_depth_optical_frame", prefix=stretch.name.name
                ),
                root=world.get_body_by_name("camera_depth_optical_frame"),
                forward_facing_axis=Vector3(0, 0, 1),
                minimal_height=1.307,
                maximal_height=1.307,
                _world=world,
            )

            camera_infra1 = Camera(
                name=PrefixedName(
                    "camera_infra1_optical_frame", prefix=stretch.name.name
                ),
                root=world.get_body_by_name("camera_infra1_optical_frame"),
                forward_facing_axis=Vector3(0, 0, 1),
                minimal_height=1.307,
                maximal_height=1.307,
                _world=world,
            )

            camera_infra2 = Camera(
                name=PrefixedName(
                    "camera_infra2_optical_frame", prefix=stretch.name.name
                ),
                root=world.get_body_by_name("camera_infra2_optical_frame"),
                forward_facing_axis=Vector3(0, 0, 1),
                minimal_height=1.257,
                maximal_height=1.257,
                _world=world,
            )

            neck = Neck(
                name=PrefixedName("neck", prefix=stretch.name.name),
                sensors=[camera_color, camera_depth, camera_infra1, camera_infra2],
                root=world.get_body_by_name("link_head"),
                tip=world.get_body_by_name("link_head_tilt"),
                pitch_body=world.get_body_by_name("link_head_tilt"),
                yaw_body=world.get_body_by_name("link_head_pan"),
                _world=world,
            )
            stretch.add_neck(neck)

            # Create torso
            torso = Torso(
                name=PrefixedName("torso", prefix=stretch.name.name),
                root=world.get_body_by_name("link_mast"),
                tip=world.get_body_by_name("link_lift"),
                _world=world,
            )
            stretch.add_torso(torso)

            # Create states
            arm_park = JointState.from_mapping(
                name=PrefixedName("arm_park", prefix=stretch.name.name),
                mapping=dict(
                    zip(
                        [c for c in arm.connections if type(c) != FixedConnection],
                        [0.0] * len(list(arm.connections)),
                    )
                ),
                state_type=StaticJointState.PARK,
            )

            arm.add_joint_state(arm_park)

            gripper_joints = [
                world.get_connection_by_name("joint_gripper_finger_left"),
                world.get_connection_by_name("joint_gripper_finger_right"),
            ]

            gripper_open = JointState.from_mapping(
                name=PrefixedName("gripper_open", prefix=stretch.name.name),
                mapping=dict(zip(gripper_joints, [0.59, 0.59])),
                state_type=GripperState.OPEN,
            )

            gripper_close = JointState.from_mapping(
                name=PrefixedName("gripper_close", prefix=stretch.name.name),
                mapping=dict(zip(gripper_joints, [0.0, 0.0])),
                state_type=GripperState.CLOSE,
            )

            gripper.add_joint_state(gripper_open)
            gripper.add_joint_state(gripper_close)

            torso_joint = [world.get_connection_by_name("joint_lift")]

            torso_low = JointState.from_mapping(
                name=PrefixedName("torso_low", prefix=stretch.name.name),
                mapping=dict(zip(torso_joint, [0.0])),
                state_type=TorsoState.LOW,
            )

            torso_mid = JointState.from_mapping(
                name=PrefixedName("torso_mid", prefix=stretch.name.name),
                mapping=dict(zip(torso_joint, [0.5])),
                state_type=TorsoState.MID,
            )

            torso_high = JointState.from_mapping(
                name=PrefixedName("torso_high", prefix=stretch.name.name),
                mapping=dict(zip(torso_joint, [1.0])),
                state_type=TorsoState.HIGH,
            )

            torso.add_joint_state(torso_low)
            torso.add_joint_state(torso_mid)
            torso.add_joint_state(torso_high)

            # Create the robot base
            base = Base(
                name=PrefixedName("base", prefix=stretch.name.name),
                root=world.get_body_by_name("base_link"),
                tip=world.get_body_by_name("base_link"),
                _world=world,
                main_axis=Vector3(0, -1, 0, world.get_body_by_name("base_link")),
            )

            stretch.add_base(base)
            stretch.full_body_controlled = True

            world.add_semantic_annotation(stretch)

        return stretch
