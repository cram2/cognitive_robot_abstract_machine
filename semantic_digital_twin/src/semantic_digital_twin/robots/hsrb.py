from collections import defaultdict
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
    FieldOfView,
    JointState,
)
from .robot_mixins import HasNeck, HasArms
from ..datastructures.prefixed_name import PrefixedName
from ..spatial_types import Quaternion
from ..spatial_types.spatial_types import Vector3
from ..world import World


@dataclass
class HSRB(AbstractRobot, HasArms, HasNeck):
    """
    Class that describes the Human Support Robot variant B (https://upmroboticclub.wordpress.com/robot/).
    """

    def __hash__(self):
        return hash(
            tuple(
                [self.__class__]
                + sorted([kse.name for kse in self.kinematic_structure_entities])
            )
        )

    def load_srdf(self):
        """
        Loads the SRDF file for the PR2 robot, if it exists.
        """
        ...

    @classmethod
    def from_world(cls, world: World) -> Self:
        """
        Creates an HSRB (Human Support Robot B) semantic annotation from a World that was parsed from
        resources/urdf/robots/hsrb.urdf. Assumes all URDF link names exist in the world.
        """
        with world.modify_world():
            hsrb = cls(
                name=PrefixedName("hsrb", prefix=world.name),
                root=world.get_body_by_name("base_footprint"),
                _world=world,
            )

            gripper_thumb = Finger(
                name=PrefixedName("thumb", prefix=hsrb.name.name),
                root=world.get_body_by_name("hand_l_proximal_link"),
                tip=world.get_body_by_name("hand_l_finger_tip_frame"),
                _world=world,
            )

            gripper_finger = Finger(
                name=PrefixedName("finger", prefix=hsrb.name.name),
                root=world.get_body_by_name("hand_r_proximal_link"),
                tip=world.get_body_by_name("hand_r_finger_tip_frame"),
                _world=world,
            )

            gripper = ParallelGripper(
                name=PrefixedName("gripper", prefix=hsrb.name.name),
                root=world.get_body_by_name("hand_palm_link"),
                tool_frame=world.get_body_by_name("hand_gripper_tool_frame"),
                thumb=gripper_thumb,
                finger=gripper_finger,
                front_facing_axis=Vector3(0, 0, 1),
                front_facing_orientation=Quaternion(-1, 0, -1, 0),
                _world=world,
            )

            # the min and max height are incorrect, same with the FoV. needs to be corrected using the real robot
            hand_camera = Camera(
                name=PrefixedName("hand_camera", prefix=hsrb.name.name),
                root=world.get_body_by_name("hand_camera_frame"),
                forward_facing_axis=Vector3(1, 0, 0),
                field_of_view=FieldOfView(
                    horizontal_angle=0.99483, vertical_angle=0.75049
                ),
                minimal_height=0.75049,
                maximal_height=0.99483,
                _world=world,
            )

            arm = Arm(
                name=PrefixedName("arm", prefix=hsrb.name.name),
                root=world.get_body_by_name("torso_lift_link"),
                tip=world.get_body_by_name("hand_palm_link"),
                manipulator=gripper,
                sensors={hand_camera},
                _world=world,
            )
            hsrb.add_arm(arm)

            # Create camera and neck
            head_center_camera = Camera(
                name=PrefixedName("head_center_camera", prefix=hsrb.name.name),
                root=world.get_body_by_name("head_center_camera_frame"),
                forward_facing_axis=Vector3(1, 0, 0),
                field_of_view=FieldOfView(
                    horizontal_angle=0.99483, vertical_angle=0.75049
                ),
                minimal_height=0.75049,
                maximal_height=0.99483,
                _world=world,
            )

            head_r_camera = Camera(
                name=PrefixedName("head_right_camera", prefix=hsrb.name.name),
                root=world.get_body_by_name("head_r_stereo_camera_link"),
                forward_facing_axis=Vector3(1, 0, 0),
                field_of_view=FieldOfView(
                    horizontal_angle=0.99483, vertical_angle=0.75049
                ),
                minimal_height=0.75049,
                maximal_height=0.99483,
                _world=world,
            )

            head_l_camera = Camera(
                name=PrefixedName("head_left_camera", prefix=hsrb.name.name),
                root=world.get_body_by_name("head_l_stereo_camera_link"),
                forward_facing_axis=Vector3(1, 0, 0),
                field_of_view=FieldOfView(
                    horizontal_angle=0.99483, vertical_angle=0.75049
                ),
                minimal_height=0.75049,
                maximal_height=0.99483,
                _world=world,
            )

            head_rgbd_camera = Camera(
                name=PrefixedName("head_rgbd_camera", prefix=hsrb.name.name),
                root=world.get_body_by_name("head_rgbd_sensor_link"),
                forward_facing_axis=Vector3(1, 0, 0),
                field_of_view=FieldOfView(
                    horizontal_angle=0.99483, vertical_angle=0.75049
                ),
                minimal_height=0.75049,
                maximal_height=0.99483,
                _world=world,
            )

            neck = Neck(
                name=PrefixedName("neck", prefix=hsrb.name.name),
                sensors={
                    head_center_camera,
                    head_r_camera,
                    head_l_camera,
                    head_rgbd_camera,
                },
                root=world.get_body_by_name("head_pan_link"),
                tip=world.get_body_by_name("head_tilt_link"),
                _world=world,
            )
            hsrb.add_neck(neck)

            # Create torso
            torso = Torso(
                name=PrefixedName("torso", prefix=hsrb.name.name),
                root=world.get_body_by_name("base_link"),
                tip=world.get_body_by_name("torso_lift_link"),
                _world=world,
            )
            hsrb.add_torso(torso)

            #Create states
            arm_park = JointState(
                name=PrefixedName("arm_park", prefix=hsrb.name.name),
                joint_names=[world.get_body_by_name("arm_flex_joint"), world.get_body_by_name("arm_roll_joint"),
                             world.get_body_by_name("wrist_flex_joint"), world.get_body_by_name("wrist_roll_joint")],
                joint_positions=[0.0, 1.5, -1.85, 0.0],
                state_type="Park",
                kinematic_chains=[arm],
                _world=world,
            )

            gripper_joints = [world.get_body_by_name("hand_l_proximal_joint"),
                             world.get_body_by_name("hand_r_proximal_joint"),
                             world.get_body_by_name("hand_motor_joint")]

            gripper_open = JointState(
                name=PrefixedName("gripper_open", prefix=hsrb.name.name),
                joint_names=gripper_joints,
                joint_positions=[0.3, 0.3, 0.3],
                state_type="Open",
                kinematic_chains=[gripper],
                _world=world,
            )

            gripper_close = JointState(
                name=PrefixedName("gripper_close", prefix=hsrb.name.name),
                joint_names=gripper_joints,
                joint_positions=[0.0, 0.0, 0.0],
                state_type="Close",
                kinematic_chains=[gripper],
                _world=world,
            )

            torso_joint = [world.get_body_by_name("torso_lift_joint")]

            torso_low = JointState(
                name=PrefixedName("torso_low", prefix=hsrb.name.name),
                joint_names=torso_joint,
                joint_positions=[0.0],
                state_type="Low",
                kinematic_chains=[torso],
                _world=world,
            )

            torso_mid = JointState(
                name=PrefixedName("torso_mid", prefix=hsrb.name.name),
                joint_names=torso_joint,
                joint_positions=[0.17],
                state_type="Mid",
                kinematic_chains=[torso],
                _world=world,
            )

            torso_high = JointState(
                name=PrefixedName("torso_high", prefix=hsrb.name.name),
                joint_names=torso_joint,
                joint_positions=[0.34],
                state_type="High",
                kinematic_chains=[torso],
                _world=world,
            )

            hsrb.add_joint_states([arm_park, gripper_open, gripper_close, torso_low, torso_mid, torso_high])

            world.add_semantic_annotation(hsrb)

            vel_limits = defaultdict(lambda: 1)
            hsrb.tighten_dof_velocity_limits_of_1dof_connections(new_limits=vel_limits)

        return hsrb
