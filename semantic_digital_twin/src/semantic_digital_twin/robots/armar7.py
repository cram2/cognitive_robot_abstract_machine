from __future__ import annotations

from dataclasses import dataclass
from typing import Self

from semantic_digital_twin.robots.robot_mixins import HasNeck, SpecifiesLeftRightArm
from semantic_digital_twin.datastructures.definitions import (
    StaticJointState,
    GripperState,
    TorsoState,
)
from semantic_digital_twin.datastructures.joint_state import JointState
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.robots.abstract_robot import (
    Neck,
    Finger,
    Arm,
    Camera,
    FieldOfView,
    Torso,
    AbstractRobot,
    HumanoidGripper,
    Base,
)
from semantic_digital_twin.spatial_types import Quaternion, Vector3

from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import (
    FixedConnection,
    ActiveConnection1DOF,
)


@dataclass(eq=False)
class Armar7(AbstractRobot, SpecifiesLeftRightArm, HasNeck):
    """
    Class that describes the Armar7 Robot.
    """

    def _setup_joint_states(self):
        left_arm_park = JointState.from_mapping(
            name=PrefixedName("left_arm_park", prefix=self.name.name),
            mapping=dict(
                zip(
                    [
                        c
                        for c in self.left_arm.connections
                        if isinstance(c, ActiveConnection1DOF)
                    ],
                    [0.0, 0.0, 0.25, 0.5, 1.0, 1.0, 0.0, 0.0],
                )
            ),
            state_type=StaticJointState.PARK,
        )

        self.left_arm.add_joint_state(left_arm_park)

        right_arm_park = JointState.from_mapping(
            name=PrefixedName("right_arm_park", prefix=self.name.name),
            mapping=dict(
                zip(
                    [
                        c
                        for c in self.right_arm.connections
                        if isinstance(c, ActiveConnection1DOF)
                    ],
                    [0.0, 0.0, 0.25, -0.5, 1.0, -1.0, 0.0, 0.0],
                )
            ),
            state_type=StaticJointState.PARK,
        )

        self.right_arm.add_joint_state(right_arm_park)

        left_gripper_joints = [
            c
            for c in self.left_arm.manipulator.connections
            if type(c) != FixedConnection
        ]

        left_gripper_open = JointState.from_mapping(
            name=PrefixedName("left_gripper_open", prefix=self.name.name),
            mapping=dict(
                zip(
                    left_gripper_joints,
                    [0] * len(left_gripper_joints),
                )
            ),
            state_type=GripperState.OPEN,
        )

        left_gripper_close = JointState.from_mapping(
            name=PrefixedName("left_gripper_close", prefix=self.name.name),
            mapping=dict(
                zip(
                    left_gripper_joints,
                    [1.57] * len(left_gripper_joints),
                )
            ),
            state_type=GripperState.CLOSE,
        )

        self.left_arm.manipulator.add_joint_state(left_gripper_close)
        self.left_arm.manipulator.add_joint_state(left_gripper_open)

        right_gripper_joints = [
            c
            for c in self.right_arm.manipulator.connections
            if type(c) != FixedConnection
        ]

        right_gripper_open = JointState.from_mapping(
            name=PrefixedName("right_gripper_open", prefix=self.name.name),
            mapping=dict(
                zip(
                    right_gripper_joints,
                    [0] * len(right_gripper_joints),
                )
            ),
            state_type=GripperState.OPEN,
        )

        right_gripper_close = JointState.from_mapping(
            name=PrefixedName("right_gripper_close", prefix=self.name.name),
            mapping=dict(
                zip(
                    right_gripper_joints,
                    [1.57] * len(right_gripper_joints),
                )
            ),
            state_type=GripperState.CLOSE,
        )

        self.right_arm.manipulator.add_joint_state(right_gripper_close)
        self.right_arm.manipulator.add_joint_state(right_gripper_open)

        torso_joints = [
            c for c in self.torso.connections if isinstance(c, ActiveConnection1DOF)
        ]

        torso_low = JointState.from_mapping(
            name=PrefixedName("torso_low", prefix=self.name.name),
            mapping=dict(zip(torso_joints, [-0.757037, 1.74533, 2.18166 / 2])),
            state_type=TorsoState.LOW,
        )

        torso_mid = JointState.from_mapping(
            name=PrefixedName("torso_mid", prefix=self.name.name),
            mapping=dict(zip(torso_joints, [-0.757037 / 2, 1.74533 / 2, 2.18166 / 4])),
            state_type=TorsoState.MID,
        )

        torso_high = JointState.from_mapping(
            name=PrefixedName("torso_high", prefix=self.name.name),
            mapping=dict(zip(torso_joints, [0.0] * len(torso_joints))),
            state_type=TorsoState.HIGH,
        )

        self.torso.add_joint_state(torso_low)
        self.torso.add_joint_state(torso_mid)
        self.torso.add_joint_state(torso_high)

    def _setup_velocity_limits(self):
        pass

    def _setup_hardware_interfaces(self):
        pass

    def _setup_collision_rules(self):
        pass

    def _setup_semantic_annotations(self):
        base = Base(
            name=PrefixedName("base", prefix=self.name.name),
            root=self._world.get_body_by_name("Platform_body_link"),
            tip=self._world.get_body_by_name("Platform_body_link"),
            main_axis=Vector3.Y(),
            _world=self._world,
        )

        self.add_base(base)

        # Create left arm
        left_gripper_thumb = Finger(
            name=PrefixedName("left_gripper_thumb", prefix=self.name.name),
            root=self._world.get_body_by_name("Hand L Palm_link"),
            tip=self._world.get_body_by_name("Thumb L Tip_link"),
            _world=self._world,
        )

        left_gripper_ring = Finger(
            name=PrefixedName("left_gripper_ring", prefix=self.name.name),
            root=self._world.get_body_by_name("Hand L Palm_link"),
            tip=self._world.get_body_by_name("Ring L Tip_link"),
            _world=self._world,
        )

        left_gripper_pinky = Finger(
            name=PrefixedName("left_gripper_pinky", prefix=self.name.name),
            root=self._world.get_body_by_name("Hand L Palm_link"),
            tip=self._world.get_body_by_name("Pinky L Tip_link"),
            _world=self._world,
        )

        left_gripper_middle = Finger(
            name=PrefixedName("left_gripper_middle", prefix=self.name.name),
            root=self._world.get_body_by_name("Hand L Palm_link"),
            tip=self._world.get_body_by_name("Middle L Tip_link"),
            _world=self._world,
        )

        left_gripper_index = Finger(
            name=PrefixedName("left_gripper_index", prefix=self.name.name),
            root=self._world.get_body_by_name("Hand L Palm_link"),
            # The actual tip link of the index finger is "Hand L Index TCP_link". But im unsure how its supposed to
            # be used, so ill ignore it for now.
            # tip=self._world.get_body_by_name("Index L Tip_link"),
            tip=self._world.get_body_by_name("Index L Tip_link"),
            _world=self._world,
        )

        left_gripper = HumanoidGripper(
            name=PrefixedName("left_gripper", prefix=self.name.name),
            root=self._world.get_body_by_name("ArmL8_Wrist_Hemisphere_B_link"),
            tool_frame=self._world.get_body_by_name("Hand L TCP_link"),
            front_facing_orientation=Quaternion(
                -0.5,
                0.5,
                -0.5,
                0.5,
            ),
            front_facing_axis=Vector3.Z(),
            thumb=left_gripper_thumb,
            fingers=[
                left_gripper_ring,
                left_gripper_pinky,
                left_gripper_middle,
                left_gripper_index,
            ],
            _world=self._world,
        )
        left_arm = Arm(
            name=PrefixedName("left_arm", prefix=self.name.name),
            root=self._world.get_body_by_name("CenterArms_fixed_link"),
            tip=self._world.get_body_by_name("ArmL8_Wrist_Hemisphere_B_link"),
            manipulator=left_gripper,
            _world=self._world,
        )

        self.add_arm(left_arm)

        # Create right arm
        right_gripper_thumb = Finger(
            name=PrefixedName("right_gripper_thumb", prefix=self.name.name),
            root=self._world.get_body_by_name("Hand R Palm_link"),
            tip=self._world.get_body_by_name("Thumb R Tip_link"),
            _world=self._world,
        )

        right_gripper_ring = Finger(
            name=PrefixedName("right_gripper_ring", prefix=self.name.name),
            root=self._world.get_body_by_name("Hand R Palm_link"),
            tip=self._world.get_body_by_name("Ring R Tip_link"),
            _world=self._world,
        )

        right_gripper_pinky = Finger(
            name=PrefixedName("right_gripper_pinky", prefix=self.name.name),
            root=self._world.get_body_by_name("Hand R Palm_link"),
            tip=self._world.get_body_by_name("Pinky R Tip_link"),
            _world=self._world,
        )

        right_gripper_middle = Finger(
            name=PrefixedName("right_gripper_middle", prefix=self.name.name),
            root=self._world.get_body_by_name("Hand R Palm_link"),
            tip=self._world.get_body_by_name("Middle R Tip_link"),
            _world=self._world,
        )

        right_gripper_index = Finger(
            name=PrefixedName("right_gripper_index", prefix=self.name.name),
            root=self._world.get_body_by_name("Hand R Palm_link"),
            tip=self._world.get_body_by_name("Index R Tip_link"),
            _world=self._world,
        )

        right_gripper = HumanoidGripper(
            name=PrefixedName("right_gripper", prefix=self.name.name),
            root=self._world.get_body_by_name("ArmR8_Wrist_Hemisphere_B_link"),
            tool_frame=self._world.get_body_by_name("Hand R TCP_link"),
            front_facing_orientation=Quaternion(
                -0.5,
                0.5,
                -0.5,
                0.5,
            ),
            front_facing_axis=Vector3.Z(),
            thumb=right_gripper_thumb,
            fingers=[
                right_gripper_ring,
                right_gripper_pinky,
                right_gripper_middle,
                right_gripper_index,
            ],
            _world=self._world,
        )

        right_arm = Arm(
            name=PrefixedName("right_arm", prefix=self.name.name),
            root=self._world.get_body_by_name("CenterArms_fixed_link"),
            tip=self._world.get_body_by_name("ArmR8_Wrist_Hemisphere_B_link"),
            manipulator=right_gripper,
            _world=self._world,
        )

        self.add_arm(right_arm)

        # Create camera and neck
        camera = Camera(
            name=PrefixedName("AzureKinect_RGB_link", prefix=self.name.name),
            root=self._world.get_body_by_name("AzureKinect_RGB_link"),
            forward_facing_axis=Vector3(0, 0, 1),
            field_of_view=FieldOfView(horizontal_angle=0.99483, vertical_angle=0.75049),
            minimal_height=1.371500015258789,
            maximal_height=1.7365000247955322,
            _world=self._world,
        )

        neck = Neck(
            name=PrefixedName("neck", prefix=self.name.name),
            sensors=[camera],
            root=self._world.get_body_by_name("Neck_Root_link"),
            tip=self._world.get_body_by_name("Head_Root_link"),
            pitch_body=self._world.get_body_by_name("Neck_2_Pitch_link"),
            yaw_body=self._world.get_body_by_name("Neck_1_Yaw_link"),
            _world=self._world,
        )
        self.add_neck(neck)

        # Create torso
        torso = Torso(
            name=PrefixedName("torso", prefix=self.name.name),
            root=self._world.get_body_by_name("Platform_link"),
            tip=self._world.get_body_by_name("CenterArms_fixed_link"),
            _world=self._world,
        )
        self.add_torso(torso)

        self._world.add_semantic_annotation(self)

    @classmethod
    def _init_empty_robot(cls, world: World) -> Self:
        return cls(
            name=PrefixedName(name="armar7", prefix=world.name),
            # the actual root here is called "root", but this is such a generic name that i fear it will be
            # duplicated in the world
            root=world.get_body_by_name(
                "Dummy_Platform_link"
            ).parent_kinematic_structure_entity,
            _world=world,
        )
