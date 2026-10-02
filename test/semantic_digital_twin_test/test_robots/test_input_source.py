from __future__ import annotations

from dataclasses import dataclass

import pytest

from semantic_digital_twin.input_synchronization import (
    InputSynchronizer,
    WorldStateInputs,
)
from semantic_digital_twin.robots.exceptions import (
    MissingInputSourceError,
    UndeclaredTopicError,
    UnexpectedInputSourceError,
)
from semantic_digital_twin.robots.input_source import (
    BasePoseSource,
    JointPositionSource,
    RobotTopic,
    SimulatedBasePoseSource,
    SimulatedJointPositionSource,
)
from semantic_digital_twin.robots.pr2 import (
    PR2,
    PR2BaseLidar,
    PR2LeftArm,
    PR2MobileBase,
    PR2Topic,
)
from semantic_digital_twin.robots.hsrb import HSRBBaseLidar, HSRBTopic
from semantic_digital_twin.robots.robot_part_mixins import HasInputSource
from semantic_digital_twin.robots.stretch import StretchBaseLidar, StretchTopic
from semantic_digital_twin.robots.tiago import TiagoBaseLidar, TiagoTopic
from semantic_digital_twin.robots.tracy import (
    TracyLeftArm,
    TracyLeftGripperLeftFinger,
    TracyRightArm,
    TracyRightGripperRightFinger,
    TracyTopic,
)
from semantic_digital_twin.robots.robot_parts import KinematicChain
from semantic_digital_twin.world import World

# %% stand-ins for the sources a part can be read from


@dataclass
class TopicReadingSource(JointPositionSource):
    """
    A source that stands in for one reading a robot, remembering only the topic it was
    pointed at.
    """

    topic_name: str
    """
    The topic this source was pointed at.
    """


@dataclass
class RewritingSource(JointPositionSource, InputSynchronizer):
    """
    A source that writes what it last read in every cycle, standing in for the reader a
    loop needs when it moves the world state away from the robot between cycles.
    """

    def apply(self) -> bool:
        return False


@dataclass
class AppliedSource(JointPositionSource, InputSynchronizer):
    """
    A source a loop has to apply, standing in for one that writes what a robot reports
    into the world state.
    """

    def apply(self) -> bool:
        return False

    def rewriting_every_cycle(self) -> RewritingSource:
        return RewritingSource(world=self.world)


# %% stand-ins for the parts that can be told where they are read from


@dataclass(eq=False)
class PartReadFromADeclaredTopic(HasInputSource[JointPositionSource]):
    """
    A part whose description names the topic its robot publishes its state on.
    """

    topic_name = RobotTopic.JOINT_STATES

    @classmethod
    def simulated_source(cls) -> JointPositionSource:
        return SimulatedJointPositionSource()

    def real_source(self, node) -> JointPositionSource:
        return TopicReadingSource(topic_name=self.topic_name)


@dataclass(eq=False)
class PartWithoutADeclaredTopic(PartReadFromADeclaredTopic):
    """
    A part whose description names no topic.
    """

    topic_name = None


# %% the kind of source a part can be read from


def test_a_part_is_read_from_the_kind_of_source_it_binds():
    assert PartReadFromADeclaredTopic.source_family() is JointPositionSource


def test_a_chain_is_read_from_the_positions_of_its_joints():
    assert KinematicChain.source_family() is JointPositionSource


def test_a_mobile_base_is_read_from_its_pose():
    assert PR2MobileBase.source_family() is BasePoseSource


def test_a_part_is_not_read_from_a_source_of_another_kind():
    part = PartReadFromADeclaredTopic()

    with pytest.raises(UnexpectedInputSourceError) as raised:
        part.use_source(SimulatedBasePoseSource())

    assert raised.value.robot_part is part
    assert raised.value.expected_source_family is JointPositionSource
    assert isinstance(raised.value.source, SimulatedBasePoseSource)


def test_a_part_that_was_not_told_where_it_is_read_from_says_so():
    part = PartReadFromADeclaredTopic()

    with pytest.raises(MissingInputSourceError) as raised:
        part.validate()

    assert raised.value.robot_part is part


# %% switching one part


def test_a_switched_part_reads_the_world_it_stands_in_again():
    part = PartReadFromADeclaredTopic()
    part.use_real_source(node=None)

    part.use_simulated_source()

    assert isinstance(part.source, SimulatedJointPositionSource)


def test_a_switched_part_reads_the_topic_it_declares():
    part = PartReadFromADeclaredTopic()

    part.use_real_source(node=None)

    assert part.source.topic_name == RobotTopic.JOINT_STATES


def test_a_part_declaring_no_topic_cannot_be_read_from_its_robot():
    part = PartWithoutADeclaredTopic()

    with pytest.raises(UndeclaredTopicError) as raised:
        part.use_real_source(node=None)

    assert raised.value.robot_part is part


# %% the topics the robots declare


def test_a_chain_reads_the_joint_states_every_robot_publishes():
    assert PR2LeftArm.topic_name == RobotTopic.JOINT_STATES


def test_the_pr2_base_reads_the_odometry_its_interface_names():
    assert PR2MobileBase.topic_name == PR2Topic.ODOMETRY


@pytest.mark.parametrize(
    "lidar, topic",
    [
        (PR2BaseLidar, PR2Topic.LASER_SCAN),
        (HSRBBaseLidar, HSRBTopic.LASER_SCAN),
        (TiagoBaseLidar, TiagoTopic.LASER_SCAN),
        (StretchBaseLidar, StretchTopic.LASER_SCAN),
    ],
)
def test_a_base_lidar_reads_the_scanner_topic_its_robot_declares(lidar, topic):
    assert lidar.topic_name == topic


def test_an_arm_of_a_robot_publishing_per_controller_reads_its_own_topic():
    assert TracyLeftArm.topic_name == TracyTopic.LEFT_ARM_JOINT_STATES
    assert TracyRightArm.topic_name == TracyTopic.RIGHT_ARM_JOINT_STATES


def test_a_finger_reads_the_topic_of_the_gripper_it_belongs_to():
    assert TracyLeftGripperLeftFinger.topic_name == TracyTopic.LEFT_GRIPPER_JOINT_STATES
    assert (
        TracyRightGripperRightFinger.topic_name == TracyTopic.RIGHT_GRIPPER_JOINT_STATES
    )


# %% an annotated robot


@pytest.fixture
def annotated_pr2(pr2_world_copy: World) -> PR2:
    """
    The PR2 annotated in a world of its own, which the tests below switch around.
    """
    return pr2_world_copy.get_semantic_annotations_by_type(PR2)[0]


def test_every_chain_of_an_annotated_robot_reads_the_world_it_stands_in(annotated_pr2):
    chains = [
        part for part in annotated_pr2._robot_parts if isinstance(part, KinematicChain)
    ]

    assert chains
    assert all(
        isinstance(chain.source, SimulatedJointPositionSource) for chain in chains
    )


def test_the_base_of_an_annotated_robot_reads_the_world_it_stands_in(annotated_pr2):
    assert isinstance(annotated_pr2.mobile_base.source, SimulatedBasePoseSource)


def test_an_annotated_robot_has_nothing_to_apply(annotated_pr2):
    assert annotated_pr2.get_input_synchronizers() == []


def test_a_part_read_from_the_robot_becomes_an_input_to_apply(annotated_pr2):
    source = AppliedSource(world=annotated_pr2._world)

    annotated_pr2.left_arm.use_source(source)

    [synchronizer] = annotated_pr2.get_input_synchronizers()
    assert synchronizer is source


def test_a_switched_robot_reads_the_world_it_stands_in_again(annotated_pr2):
    annotated_pr2.left_arm.use_source(AppliedSource(world=annotated_pr2._world))

    annotated_pr2.use_simulated_sources()

    assert isinstance(annotated_pr2.left_arm.source, SimulatedJointPositionSource)
    assert annotated_pr2.get_input_synchronizers() == []


# %% the loops that apply what a robot reports


def test_a_loop_applies_every_source_a_robots_parts_are_read_from(annotated_pr2):
    source = AppliedSource(world=annotated_pr2._world)
    annotated_pr2.left_arm.use_source(source)
    inputs = WorldStateInputs(world=annotated_pr2._world)

    inputs.read_robot(annotated_pr2)

    assert inputs.synchronizers == [source]


def test_a_loop_reapplying_its_inputs_reads_a_rewriting_source(annotated_pr2):
    annotated_pr2.left_arm.use_source(AppliedSource(world=annotated_pr2._world))
    inputs = WorldStateInputs(world=annotated_pr2._world, reapplies_inputs=True)

    inputs.read_robot(annotated_pr2)

    [synchronizer] = inputs.synchronizers
    assert isinstance(synchronizer, RewritingSource)


def test_a_loop_reading_a_simulated_robot_applies_nothing(annotated_pr2):
    inputs = WorldStateInputs(world=annotated_pr2._world)

    inputs.read_robot(annotated_pr2)

    assert inputs.synchronizers == []
