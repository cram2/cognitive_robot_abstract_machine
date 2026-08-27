"""
Tests for reading a world's sem_dt robot-part annotations and publishing them.
"""

from dataclasses import dataclass, field

from typing_extensions import Any, List, Optional

from cramera.robot_parts import ArmSide, RobotPartAnnotation, RobotPartRole

# %% mimics standing in for the sem_dt annotations of a world


@dataclass
class NamedBody:
    """
    A world body carrying a model-prefixed name.
    """

    name: str


@dataclass
class PartWithBodies:
    """
    A robot part exposing the bodies whose link names get published.
    """

    bodies: List[Any] = field(default_factory=list)


@dataclass
class EndEffectorPart(PartWithBodies):
    """
    An end effector attached to an arm.
    """


@dataclass
class ArmPart(PartWithBodies):
    """
    An arm, optionally carrying an end effector.
    """

    end_effector: Optional[EndEffectorPart] = None


@dataclass
class TwoArmedRobot:
    """
    A robot naming which of its arms is the left and which is the right one.
    """

    left: ArmPart
    right: ArmPart

    def get_arms(self) -> List[ArmPart]:
        return [self.left, self.right]

    def get_left_arm_if_specified(self) -> ArmPart:
        return self.left

    def get_right_arm_if_specified(self) -> ArmPart:
        return self.right


@dataclass
class OneArmedRobot:
    """
    A robot that specifies neither a left nor a right arm.
    """

    arm: ArmPart

    def get_arms(self) -> List[ArmPart]:
        return [self.arm]

    def get_left_arm_if_specified(self) -> None:
        return None

    def get_right_arm_if_specified(self) -> None:
        return None


# %% link names


class TestLinkNames:
    def test_the_model_prefix_is_stripped(self):
        part = PartWithBodies(bodies=[NamedBody("pr2/l_wrist_link")])
        assert RobotPartAnnotation.link_names(part) == ["l_wrist_link"]

    def test_an_unprefixed_name_is_kept(self):
        part = PartWithBodies(bodies=[NamedBody("l_wrist_link")])
        assert RobotPartAnnotation.link_names(part) == ["l_wrist_link"]

    def test_a_part_without_bodies_has_no_links(self):
        assert RobotPartAnnotation.link_names(PartWithBodies()) == []


# %% reading the annotations off a robot


class TestDescribeRobotParts:
    def test_each_arm_is_published_with_its_end_effector(self):
        """
        An arm and the end effector it carries are published as two annotations, the end
        effector naming the arm it is attached to.
        """
        gripper = EndEffectorPart(bodies=[NamedBody("pr2/l_gripper_link")])
        arm = ArmPart(
            bodies=[NamedBody("pr2/l_upper_arm_link"), NamedBody("pr2/l_gripper_link")],
            end_effector=gripper,
        )
        robot = OneArmedRobot(arm=arm)

        assert RobotPartAnnotation.of_robot(robot) == [
            RobotPartAnnotation(
                name="ArmPart",
                role=RobotPartRole.ARM,
                side=None,
                links=["l_upper_arm_link"],
            ),
            RobotPartAnnotation(
                name="EndEffectorPart",
                role=RobotPartRole.END_EFFECTOR,
                side=None,
                links=["l_gripper_link"],
                attached_to="ArmPart",
            ),
        ]

    def test_the_side_comes_from_the_robots_own_left_right_annotation(self):
        """
        Which arm is the left one is what the robot annotation says, not what the part
        or link names happen to spell.
        """
        robot = TwoArmedRobot(
            left=ArmPart(bodies=[NamedBody("pr2/first_link")]),
            right=ArmPart(bodies=[NamedBody("pr2/second_link")]),
        )
        sides = [annotation.side for annotation in RobotPartAnnotation.of_robot(robot)]
        assert sides == [ArmSide.LEFT, ArmSide.RIGHT]

    def test_an_arm_of_a_robot_without_a_left_and_a_right_arm_has_no_side(self):
        robot = OneArmedRobot(arm=ArmPart(bodies=[NamedBody("stretch/arm_link")]))
        [annotation] = RobotPartAnnotation.of_robot(robot)
        assert annotation.side is None


# %% the published shape


class TestRobotPartAnnotationPayload:
    def test_a_payload_round_trips(self):
        annotation = RobotPartAnnotation(
            name="PR2LeftGripper",
            role=RobotPartRole.END_EFFECTOR,
            side=ArmSide.LEFT,
            links=["l_gripper_link"],
            attached_to="PR2LeftArm",
        )
        assert RobotPartAnnotation.from_payload(annotation.to_payload()) == annotation

    def test_the_payload_names_the_side_in_lower_case(self):
        annotation = RobotPartAnnotation(
            name="PR2LeftArm", role=RobotPartRole.ARM, side=ArmSide.LEFT
        )
        assert annotation.to_payload() == {
            "name": "PR2LeftArm",
            "role": "arm",
            "side": "left",
            "links": [],
            "attachedTo": None,
        }

    def test_a_sideless_payload_round_trips(self):
        annotation = RobotPartAnnotation(
            name="StretchArm", role=RobotPartRole.ARM, side=None
        )
        assert RobotPartAnnotation.from_payload(annotation.to_payload()) == annotation
