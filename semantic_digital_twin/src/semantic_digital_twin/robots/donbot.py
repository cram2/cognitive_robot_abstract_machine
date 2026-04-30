from __future__ import annotations

from dataclasses import dataclass

from semantic_digital_twin.datastructures.definitions import (
    StaticJointState,
    GripperState,
)
from semantic_digital_twin.datastructures.joint_state import JointState
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.robots.abstract_robot import (
    AbstractRobot,
    HasMobileBase,
    HasOneArm,
)
from semantic_digital_twin.robots.robot_parts import (
    Finger,
    ParallelGripper,
    Arm,
    Camera,
    FieldOfView,
)
from semantic_digital_twin.spatial_types import Quaternion, Vector3
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import (
    ActiveConnection1DOF,
)
from semantic_digital_twin.world_description.world_entity import Body


@dataclass(eq=False)
class Donbot(AbstractRobot, HasOneArm, HasMobileBase):
    """
    Class that describes the Donbot Robot.
    """

    def load_srdf(self):
        """
        Loads the SRDF file for the Donbot robot, if it exists.
        """
        ...

    @classmethod
    def _get_robot_root_body(cls, world: World) -> Body:
        return world.get_body_by_name("base_footprint")

    def _setup_collision_rules(self):
        pass

    def _setup_arm_semantic_annotations(self):
        world = self._world
        # Create arm
        gripper_thumb = Finger.create_and_add_to_world(
            name=PrefixedName("gripper_thumb", prefix=self.name.name),
            root_name="gripper_gripper_left_link",
            tip_name="gripper_finger_right_link",
            world=world,
        )

        gripper_finger = Finger.create_and_add_to_world(
            name=PrefixedName("gripper_finger", prefix=self.name.name),
            root_name="gripper_gripper_left_link",
            tip_name="gripper_finger_left_link",
            world=world,
        )

        camera = Camera.create_and_add_to_world(
            name=PrefixedName("camera_link", prefix=self.name.name),
            root_name="camera_link",
            forward_facing_axis=Vector3(0, 0, 1),
            field_of_view=FieldOfView(horizontal_angle=0.99483, vertical_angle=0.75049),
            minimal_height=0.5,
            maximal_height=1.2,
            world=world,
            default_camera=True,
        )

        gripper = ParallelGripper.create_and_add_to_world(
            name=PrefixedName("gripper", prefix=self.name.name),
            root_name="gripper_base_link",
            tool_frame_name="gripper_tool_frame",
            front_facing_orientation=Quaternion(0.707, -0.707, 0.707, -0.707),
            thumb=gripper_thumb,
            finger=gripper_finger,
            sensors=[camera],
            world=world,
        )
        arm = Arm.create_and_add_to_world(
            name=PrefixedName("arm", prefix=self.name.name),
            root_name="ur5_base_link",
            tip_name="ur5_wrist_3_link",
            manipulator=gripper,
            world=world,
        )

        self.add_arm(arm)

    def _setup_arm_hardware_interfaces(self):
        self.arm._default_hardware_interface_setup()

    def _setup_arm_joint_state(self):
        arm_park = JointState.from_mapping(
            name=PrefixedName("arm_park", prefix=self.name.name),
            mapping=dict(
                zip(
                    [
                        c
                        for c in self.arm.connections
                        if isinstance(c, ActiveConnection1DOF)
                    ],
                    [3.23, -1.51, -1.57, 0.0, 1.57, -1.65],
                )
            ),
            state_type=StaticJointState.PARK,
        )

        self.arm.add_joint_state(arm_park)

        gripper = self.arm.manipulator
        gripper_joints = [
            c for c in gripper.connections if isinstance(c, ActiveConnection1DOF)
        ]

        gripper_open = JointState.from_mapping(
            name=PrefixedName("gripper_open", prefix=self.name.name),
            mapping=dict(zip(gripper_joints, [0.109, -0.055])),
            state_type=GripperState.OPEN,
        )

        gripper_close = JointState.from_mapping(
            name=PrefixedName("gripper_close", prefix=self.name.name),
            mapping=dict(zip(gripper_joints, [0.0065, -0.0027])),
            state_type=GripperState.CLOSE,
        )

        gripper.add_joint_state(gripper_close)
        gripper.add_joint_state(gripper_open)

    def _setup_mobile_base_semantic_annotations(self):
        pass
