from dataclasses import dataclass
from typing import Self

from semantic_digital_twin.robots.robot_parts import (
    Arm,
    Finger,
    ParallelGripper,
)
from semantic_digital_twin.robots.abstract_robot import (
    HasArms,
    AbstractRobot,
    HasOneArm,
)
from semantic_digital_twin.datastructures.definitions import (
    StaticJointState,
    GripperState,
)
from semantic_digital_twin.datastructures.joint_state import JointState
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.spatial_types import Quaternion
from semantic_digital_twin.spatial_types.spatial_types import Vector3
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import FixedConnection
from semantic_digital_twin.world_description.world_entity import Body


@dataclass(eq=False)
class UR5(AbstractRobot, HasOneArm):
    """
    Class that describes the UR5 Robot.
    """

    def load_srdf(self):
        """
        Loads the SRDF file for the UR5 robot, if it exists.
        """
        ...

    @classmethod
    def _get_robot_root_body(cls, world: World) -> Body:
        return world.get_body_by_name("base_link")

    def _setup_collision_rules(self):
        pass

    def setup_arm_semantic_annotations(self):
        world = self._world
        # Create arm
        gripper_thumb = Finger.create_and_add_to_world(
            name=PrefixedName("gripper_thumb", prefix=self.name.name),
            root_name="robotiq_85_left_inner_knuckle_link",
            tip_name="robotiq_85_left_finger_tip_link",
            world=world,
        )

        gripper_finger = Finger.create_and_add_to_world(
            name=PrefixedName("gripper_finger", prefix=self.name.name),
            root_name="robotiq_85_right_inner_knuckle_link",
            tip_name="robotiq_85_right_finger_tip_link",
            world=world,
        )

        gripper = ParallelGripper.create_and_add_to_world(
            name=PrefixedName("gripper", prefix=self.name.name),
            root_name="robotiq_85_base_link",
            tool_frame_name="robotiq_85_right_finger_link",
            front_facing_orientation=Quaternion(0, 0, 0, 1),
            thumb=gripper_thumb,
            finger=gripper_finger,
            world=world,
        )

        arm = Arm.create_and_add_to_world(
            name=PrefixedName("arm", prefix=self.name.name),
            root_name="base_link",
            tip_name="wrist_3_link",
            manipulator=gripper,
            world=world,
        )

        self.add_arm(arm)

    def _setup_arm_hardware_interfaces(self):
        self.arm._default_hardware_interface_setup()

    def _setup_arm_joint_state(self):
        gripper = self.arm.manipulator
        # Create states
        arm_park = JointState.from_mapping(
            name=PrefixedName("arm_park", prefix=self.name.name),
            mapping=dict(
                zip(
                    [c for c in self.arm.connections if type(c) != FixedConnection],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                )
            ),
            state_type=StaticJointState.PARK,
        )

        self.arm.add_joint_state(arm_park)

        gripper_joints = [c for c in gripper.connections if type(c) != FixedConnection]

        gripper_open = JointState.from_mapping(
            name=PrefixedName("gripper_open", prefix=self.name.name),
            mapping=dict(zip(gripper_joints, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0])),
            state_type=GripperState.OPEN,
        )

        gripper_close = JointState.from_mapping(
            name=PrefixedName("gripper_close", prefix=self.name.name),
            mapping=dict(zip(gripper_joints, [1.0, 1.0, 1.0, 1.0, 1.0, 1.0])),
            state_type=GripperState.CLOSE,
        )

        gripper.add_joint_state(gripper_open)
        gripper.add_joint_state(gripper_close)
