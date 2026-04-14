from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from semantic_digital_twin.datastructures.definitions import (
    StaticJointState,
    GripperState,
    TorsoState,
)
from semantic_digital_twin.datastructures.joint_state import JointState
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.robots.abstract_robot import (
    HasLetRightArm,
    AbstractRobot,
    HasTorso,
    HasMobileBase,
)
from semantic_digital_twin.robots.robot_parts import (
    Finger,
    Arm,
    Camera,
    FieldOfView,
    Torso,
    HumanoidGripper,
)
from semantic_digital_twin.spatial_types import Quaternion, Vector3
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import (
    ActiveConnection1DOF,
)
from semantic_digital_twin.world_description.world_entity import Body


@dataclass(eq=False)
class ICub3(AbstractRobot, HasLetRightArm, HasTorso, HasMobileBase):
    """
    Class that describes the iCub3 Robot.
    """

    def load_srdf(self):
        """
        Loads the SRDF file for the iCub3 robot, if it exists.
        """
        ...

    @classmethod
    def _get_robot_root_body(cls, world: World) -> Body:
        return world.get_body_by_name("base_footprint")

    def _setup_collision_rules(self):
        pass

    def _setup_other_hardware_interfaces(self):
        pass

    def _setup_arm_semantic_annotations(self):
        world = self._world

        # Create left arm
        left_gripper_thumb = Finger.create_and_add_to_world(
            name=PrefixedName("left_gripper_thumb", prefix=self.name.name),
            root_name="l_hand_thumb_0",
            tip_name="l_hand_thumb_tip",
            world=world,
        )

        left_gripper_index_finger = Finger.create_and_add_to_world(
            name=PrefixedName("left_gripper_index_finger", prefix=self.name.name),
            root_name="l_hand_index_0",
            tip_name="l_hand_index_tip",
            world=world,
        )

        left_gripper_middle_finger = Finger.create_and_add_to_world(
            name=PrefixedName("left_gripper_middle_finger", prefix=self.name.name),
            root_name="l_hand_middle_0",
            tip_name="l_hand_middle_tip",
            world=world,
        )

        left_gripper_ring_finger = Finger.create_and_add_to_world(
            name=PrefixedName("left_gripper_ring_finger", prefix=self.name.name),
            root_name="l_hand_ring_0",
            tip_name="l_hand_ring_tip",
            world=world,
        )

        left_gripper_little_finger = Finger.create_and_add_to_world(
            name=PrefixedName("left_gripper_little_finger", prefix=self.name.name),
            root_name="l_hand_little_0",
            tip_name="l_hand_little_tip",
            world=world,
        )

        left_gripper = HumanoidGripper.create_and_add_to_world(
            name=PrefixedName("left_gripper", prefix=self.name.name),
            root_name="l_hand",
            tool_frame_name="l_gripper_tool_frame",
            front_facing_orientation=Quaternion(0.5, 0.5, 0.5, 0.5),
            thumb=left_gripper_thumb,
            fingers=[
                left_gripper_index_finger,
                left_gripper_middle_finger,
                left_gripper_ring_finger,
                left_gripper_little_finger,
            ],
            world=world,
        )
        left_arm = Arm.create_and_add_to_world(
            name=PrefixedName("left_arm", prefix=self.name.name),
            root_name="root_link",
            tip_name="l_hand",
            manipulator=left_gripper,
            world=world,
        )

        self.add_arm(left_arm)

        # Create right arm
        right_gripper_thumb = Finger.create_and_add_to_world(
            name=PrefixedName("right_gripper_thumb", prefix=self.name.name),
            root_name="r_hand_thumb_0",
            tip_name="r_hand_thumb_tip",
            world=world,
        )
        right_gripper_index_finger = Finger.create_and_add_to_world(
            name=PrefixedName("right_gripper_index_finger", prefix=self.name.name),
            root_name="r_hand_index_0",
            tip_name="r_hand_index_tip",
            world=world,
        )
        right_gripper_middle_finger = Finger.create_and_add_to_world(
            name=PrefixedName("right_gripper_middle_finger", prefix=self.name.name),
            root_name="r_hand_middle_0",
            tip_name="r_hand_middle_tip",
            world=world,
        )
        right_gripper_ring_finger = Finger.create_and_add_to_world(
            name=PrefixedName("right_gripper_ring_finger", prefix=self.name.name),
            root_name="r_hand_ring_0",
            tip_name="r_hand_ring_tip",
            world=world,
        )
        right_gripper_little_finger = Finger.create_and_add_to_world(
            name=PrefixedName("right_gripper_little_finger", prefix=self.name.name),
            root_name="r_hand_little_0",
            tip_name="r_hand_little_tip",
            world=world,
        )

        right_gripper = HumanoidGripper.create_and_add_to_world(
            name=PrefixedName("right_gripper", prefix=self.name.name),
            root_name="r_hand",
            tool_frame_name="r_gripper_tool_frame",
            front_facing_orientation=Quaternion(0, 0, -0.707, 0.707),
            thumb=right_gripper_thumb,
            fingers=[
                right_gripper_index_finger,
                right_gripper_middle_finger,
                right_gripper_ring_finger,
                right_gripper_little_finger,
            ],
            world=world,
        )
        right_arm = Arm.create_and_add_to_world(
            name=PrefixedName("right_arm", prefix=self.name.name),
            root_name="root_link",
            tip_name="r_hand",
            manipulator=right_gripper,
            world=world,
        )

        self.add_arm(right_arm)

    def _setup_arm_hardware_interfaces(self):
        pass

    def _setup_arm_joint_state(self):
        left_arm = self.left_arm
        right_arm = self.right_arm
        left_gripper = left_arm.manipulator
        right_gripper = right_arm.manipulator

        # Create states
        left_arm_park = JointState.from_mapping(
            name=PrefixedName("left_arm_park", prefix=self.name.name),
            mapping=dict(
                zip(
                    [
                        c
                        for c in left_arm.connections
                        if isinstance(c, ActiveConnection1DOF)
                    ],
                    [0.0] * len(list(left_arm.connections)),
                )
            ),
            state_type=StaticJointState.PARK,
        )

        left_arm.add_joint_state(left_arm_park)

        right_arm_park = JointState.from_mapping(
            name=PrefixedName("right_arm_park", prefix=self.name.name),
            mapping=dict(
                zip(
                    [
                        c
                        for c in right_arm.connections
                        if isinstance(c, ActiveConnection1DOF)
                    ],
                    [0.0] * len(list(right_arm.connections)),
                )
            ),
            state_type=StaticJointState.PARK,
        )

        right_arm.add_joint_state(right_arm_park)

        left_gripper_joints = [
            c for c in left_gripper.connections if isinstance(c, ActiveConnection1DOF)
        ]

        left_gripper_open = JointState.from_mapping(
            name=PrefixedName("left_gripper_open", prefix=self.name.name),
            mapping=dict(
                zip(
                    left_gripper_joints,
                    [0.0] * len(list(left_gripper_joints)),
                )
            ),
            state_type=GripperState.OPEN,
        )

        left_gripper_close = JointState.from_mapping(
            name=PrefixedName("left_gripper_close", prefix=self.name.name),
            mapping=dict(
                zip(
                    left_gripper_joints,
                    [
                        np.pi / 2,
                        np.pi / 2,
                        np.pi / 2,
                        np.pi / 2,
                        -0.3490658503988659,
                        np.pi / 2,
                        np.pi / 2,
                        np.pi / 2,
                        0.3490658503988659,
                        np.pi / 2,
                        np.pi / 2,
                        np.pi / 2,
                        0.3490658503988659,
                        np.pi / 2,
                        np.pi / 2,
                        np.pi / 2,
                        0.3490658503988659,
                        np.pi / 2,
                        np.pi / 2,
                        np.pi / 2,
                    ],
                )
            ),
            state_type=GripperState.CLOSE,
        )

        left_gripper.add_joint_state(left_gripper_close)
        left_gripper.add_joint_state(left_gripper_open)

        right_gripper_joints = [
            c for c in right_gripper.connections if isinstance(c, ActiveConnection1DOF)
        ]

        right_gripper_open = JointState.from_mapping(
            name=PrefixedName("right_gripper_open", prefix=self.name.name),
            mapping=dict(
                zip(
                    right_gripper_joints,
                    [0.0] * len(list(right_gripper_joints)),
                )
            ),
            state_type=GripperState.OPEN,
        )

        right_gripper_close = JointState.from_mapping(
            name=PrefixedName("right_gripper_close", prefix=self.name.name),
            mapping=dict(
                zip(
                    right_gripper_joints,
                    [
                        np.pi / 2,
                        np.pi / 2,
                        np.pi / 2,
                        np.pi / 2,
                        -0.3490658503988659,
                        np.pi / 2,
                        np.pi / 2,
                        np.pi / 2,
                        0.3490658503988659,
                        np.pi / 2,
                        np.pi / 2,
                        np.pi / 2,
                        0.3490658503988659,
                        np.pi / 2,
                        np.pi / 2,
                        np.pi / 2,
                        0.3490658503988659,
                        np.pi / 2,
                        np.pi / 2,
                        np.pi / 2,
                    ],
                )
            ),
            state_type=GripperState.CLOSE,
        )

        right_gripper.add_joint_state(right_gripper_close)
        right_gripper.add_joint_state(right_gripper_open)

    def _setup_torso_semantic_annotations(self):
        world = self._world
        # Create camera and neck
        camera = Camera.create_and_add_to_world(
            name=PrefixedName("eye_camera", prefix=self.name.name),
            root_name="head",
            forward_facing_axis=Vector3(1, 0, 0),
            field_of_view=FieldOfView(horizontal_angle=0.99483, vertical_angle=0.75049),
            minimal_height=1.27,
            maximal_height=1.85,
            world=world,
            default_camera=True,
        )

        # Create torso
        torso = Torso.create_and_add_to_world(
            name=PrefixedName("torso", prefix=self.name.name),
            root_name="root_link",
            tip_name="chest",
            world=world,
            sensors=[camera],
        )
        self.add_torso(torso)

    def _setup_torso_hardware_interfaces(self):
        pass

    def _setup_torso_joint_state(self):
        torso = self.torso
        torso_joint = [
            c for c in torso.connections if isinstance(c, ActiveConnection1DOF)
        ]

        torso_low = JointState.from_mapping(
            name=PrefixedName("torso_low", prefix=self.name.name),
            mapping=dict(zip(torso_joint, [0.0])),
            state_type=TorsoState.LOW,
        )

        torso_mid = JointState.from_mapping(
            name=PrefixedName("torso_mid", prefix=self.name.name),
            mapping=dict(zip(torso_joint, [0.0])),
            state_type=TorsoState.MID,
        )

        torso_high = JointState.from_mapping(
            name=PrefixedName("torso_high", prefix=self.name.name),
            mapping=dict(zip(torso_joint, [0.0])),
            state_type=TorsoState.HIGH,
        )

        torso.add_joint_state(torso_low)
        torso.add_joint_state(torso_mid)
        torso.add_joint_state(torso_high)

    def _setup_mobile_base_semantic_annotations(self):
        pass
