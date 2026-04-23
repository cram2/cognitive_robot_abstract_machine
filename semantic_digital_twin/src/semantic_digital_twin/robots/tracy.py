from __future__ import annotations

from abc import ABC
from dataclasses import dataclass

from semantic_digital_twin.robots.robot_part_mixins import (
    HasLeftRightArm,
    HasParallelGripper,
    HasCameras,
)
from semantic_digital_twin.robots.robot_parts import (
    Finger,
    ParallelGripper,
    Arm,
    Camera,
    FieldOfView,
    AbstractRobot,
)
from semantic_digital_twin.spatial_types import Quaternion, Vector3
from semantic_digital_twin.world import World


@dataclass(eq=False)
class TracyFinger(Finger, ABC): ...


@dataclass(eq=False)
class TracyLeftGripperLeftFinger(TracyFinger):

    @classmethod
    def setup_default_configuration_in_world(cls, world: World):
        finger = cls(
            root=world.get_body_by_name("left_robotiq_85_left_knuckle_link"),
            tip=world.get_body_by_name("left_robotiq_85_left_finger_tip_link"),
        )
        world.add_semantic_annotation(finger)
        return finger


@dataclass(eq=False)
class TracyLeftGripperRightFinger(TracyFinger):

    @classmethod
    def setup_default_configuration_in_world(cls, world: World):
        finger = cls(
            root=world.get_body_by_name("left_robotiq_85_right_knuckle_link"),
            tip=world.get_body_by_name("left_robotiq_85_right_finger_tip_link"),
        )
        world.add_semantic_annotation(finger)
        return finger


@dataclass(eq=False)
class TracyRightGripperLeftFinger(TracyFinger):

    @classmethod
    def setup_default_configuration_in_world(cls, world: World):
        finger = cls(
            root=world.get_body_by_name("right_robotiq_85_left_knuckle_link"),
            tip=world.get_body_by_name("right_robotiq_85_left_finger_tip_link"),
        )
        world.add_semantic_annotation(finger)
        return finger


@dataclass(eq=False)
class TracyRightGripperRightFinger(TracyFinger):

    @classmethod
    def setup_default_configuration_in_world(cls, world: World):
        finger = cls(
            root=world.get_body_by_name("right_robotiq_85_right_knuckle_link"),
            tip=world.get_body_by_name("right_robotiq_85_right_finger_tip_link"),
        )
        world.add_semantic_annotation(finger)
        return finger


@dataclass(eq=False)
class TracyLeftGripper(ParallelGripper):

    @classmethod
    def setup_default_configuration_in_world(cls, world: World):
        gripper = cls(
            root=world.get_body_by_name("left_robotiq_85_base_link"),
            tool_frame=world.get_body_by_name("l_gripper_tool_frame"),
            front_facing_orientation=Quaternion(0.5, 0.5, 0.5, 0.5),
        )
        world.add_semantic_annotation(gripper)
        return gripper

    def setup_finger_semantic_annotations(self):
        thumb = TracyLeftGripperLeftFinger.setup_default_configuration_in_world(
            self._world
        )
        self.add_thumb(thumb)
        finger = TracyLeftGripperRightFinger.setup_default_configuration_in_world(
            self._world
        )
        self.add_finger(finger)


@dataclass(eq=False)
class TracyRightGripper(ParallelGripper):

    @classmethod
    def setup_default_configuration_in_world(cls, world: World):
        gripper = cls(
            root=world.get_body_by_name("right_robotiq_85_base_link"),
            tool_frame=world.get_body_by_name("r_gripper_tool_frame"),
            front_facing_orientation=Quaternion(0.5, 0.5, 0.5, 0.5),
        )
        world.add_semantic_annotation(gripper)
        return gripper

    def setup_finger_semantic_annotations(self):
        thumb = TracyRightGripperLeftFinger.setup_default_configuration_in_world(
            self._world
        )
        self.add_thumb(thumb)
        finger = TracyRightGripperRightFinger.setup_default_configuration_in_world(
            self._world
        )
        self.add_finger(finger)


@dataclass(eq=False)
class TracyLeftArm(Arm, HasParallelGripper):

    @classmethod
    def setup_default_configuration_in_world(cls, world: World):
        arm = cls(
            root=world.get_body_by_name("table"),
            tip=world.get_body_by_name("left_wrist_3_link"),
        )
        world.add_semantic_annotation(arm)
        return arm

    def setup_end_effector_semantic_annotation(self):
        gripper = TracyLeftGripper.setup_default_configuration_in_world(self._world)
        self.add_end_effector(gripper)
        gripper.setup_finger_semantic_annotations()


@dataclass(eq=False)
class TracyRightArm(Arm, HasParallelGripper):

    @classmethod
    def setup_default_configuration_in_world(cls, world: World):
        arm = cls(
            root=world.get_body_by_name("table"),
            tip=world.get_body_by_name("right_wrist_3_link"),
        )
        world.add_semantic_annotation(arm)
        return arm

    def setup_end_effector_semantic_annotation(self):
        gripper = TracyRightGripper.setup_default_configuration_in_world(self._world)
        self.add_end_effector(gripper)
        gripper.setup_finger_semantic_annotations()


@dataclass(eq=False)
class TracyCamera(Camera):

    @classmethod
    def setup_default_configuration_in_world(cls, world: World):
        camera = cls(
            root=world.get_body_by_name("camera_link"),
            forward_facing_axis=Vector3(0, 0, 1),
            field_of_view=FieldOfView(horizontal_angle=1.047, vertical_angle=0.785),
            minimal_height=0.8,
            maximal_height=1.7,
            default_camera=True,
        )
        world.add_semantic_annotation(camera)
        return camera


@dataclass(eq=False)
class Tracy(AbstractRobot, HasLeftRightArm, HasCameras):
    """
    Represents two UR10e Arms on a table, with a pole between them holding a small camera.
     Example can be found at: https://vib.ai.uni-bremen.de/page/comingsoon/the-tracebot-laboratory/
    """

    @classmethod
    def _get_root_body_name(cls) -> str:
        return "table"

    def setup_arm_semantic_annotations(self):
        left_arm = TracyLeftArm.setup_default_configuration_in_world(self._world)
        self.add_arm(left_arm)
        left_arm.setup_end_effector_semantic_annotation()

        right_arm = TracyRightArm.setup_default_configuration_in_world(self._world)
        self.add_arm(right_arm)
        right_arm.setup_end_effector_semantic_annotation()

    def setup_sensor_semantic_annotations(self):
        camera = TracyCamera.setup_default_configuration_in_world(self._world)
        self.add_camera(camera)

    def setup_robot_part_semantic_annotations(self):
        self.setup_arm_semantic_annotations()
        self.setup_sensor_semantic_annotations()
