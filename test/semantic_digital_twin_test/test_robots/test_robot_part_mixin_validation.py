from __future__ import annotations

from dataclasses import dataclass

import pytest

from semantic_digital_twin.robots.exceptions import (
    MissingEndEffectorError,
    MissingLidarError,
    MissingMobileBaseError,
    MissingNeckError,
    MissingSensorsError,
    MissingTorsoError,
    TooFewArmsError,
    TooFewFingersError,
    UnexpectedArmCountError,
    UnexpectedFingerCountError,
)
from semantic_digital_twin.robots.robot_part_mixins import (
    HasArms,
    HasEndEffector,
    HasFingers,
    HasLeftRightArm,
    HasLidar,
    HasMobileBase,
    HasNeck,
    HasOneArm,
    HasSensors,
    HasTorso,
    HasTwoFingers,
)

# %% stand-ins for the parts the mixins bind


@dataclass(eq=False)
class MountedPart:
    """
    A part a mixin can hold, carrying nothing the validation looks at.
    """


@dataclass(eq=False)
class Thumb(MountedPart):
    """
    The finger :meth:`HasFingers.thumb` singles out.
    """


@dataclass(eq=False)
class OpposingFinger(MountedPart):
    """
    The finger a thumb closes against.
    """


# %% parts combining mixins


@dataclass(eq=False)
class PartCombiningIndependentMixins(HasTorso[MountedPart], HasLidar[MountedPart]):
    """
    A part whose two mixins are unrelated, so neither one's assumptions replace the
    other's.
    """


@dataclass(eq=False)
class PartNarrowingAMixin(HasTwoFingers[Thumb, OpposingFinger]):
    """
    A part whose mixin narrows another one, replacing its assumption about how many
    fingers there are.
    """


# %% assumptions of independent mixins


def test_every_independent_mixin_is_checked():
    part = PartCombiningIndependentMixins(torso=MountedPart())

    with pytest.raises(MissingLidarError):
        part.validate_assumptions()


def test_a_part_satisfying_every_independent_mixin_passes():
    part = PartCombiningIndependentMixins(torso=MountedPart(), lidar=MountedPart())

    part.validate_assumptions()


# %% assumptions a narrowing mixin replaces


def test_a_narrowed_assumption_replaces_the_one_it_narrows():
    part = PartNarrowingAMixin(fingers=[Thumb(), OpposingFinger()])

    part.validate_assumptions()

    with pytest.raises(TooFewFingersError):
        HasFingers.validate(part)


# %% the exception an unmet assumption raises


@dataclass(eq=False)
class PartWithoutItsSingleChild(
    HasTorso[MountedPart],
    HasNeck[MountedPart],
    HasLidar[MountedPart],
    HasEndEffector[MountedPart],
    HasMobileBase[MountedPart],
    HasSensors[MountedPart],
):
    """
    A part combining every mixin that requires a single child, carrying none of them.
    """


@dataclass(eq=False)
class PartWithManyFingers(HasFingers[Thumb, OpposingFinger]):
    """
    A part whose mixin requires more fingers than a thumb and one opposing finger.
    """


@dataclass(eq=False)
class PartWithManyArms(HasArms[MountedPart, MountedPart, MountedPart]):
    """
    A part whose mixin requires more arms than a left and a right one.
    """


@dataclass(eq=False)
class PartWithOneArm(HasOneArm[MountedPart]):
    """
    A part whose mixin requires exactly one arm.
    """


@dataclass(eq=False)
class PartWithLeftAndRightArm(HasLeftRightArm[MountedPart, MountedPart]):
    """
    A part whose mixin requires exactly two arms.
    """


@pytest.mark.parametrize(
    "mixin, error",
    [
        (HasTorso, MissingTorsoError),
        (HasNeck, MissingNeckError),
        (HasLidar, MissingLidarError),
        (HasEndEffector, MissingEndEffectorError),
        (HasMobileBase, MissingMobileBaseError),
        (HasSensors, MissingSensorsError),
    ],
)
def test_a_mixin_missing_its_child_names_the_child_it_misses(mixin, error):
    part = PartWithoutItsSingleChild()

    with pytest.raises(error):
        mixin.validate(part)


def test_too_few_fingers_carries_the_counts():
    part = PartWithManyFingers(fingers=[Thumb(), OpposingFinger()])

    with pytest.raises(TooFewFingersError) as raised:
        part.validate_assumptions()

    assert raised.value.robot_part is part
    assert raised.value.minimum_count == HasFingers.minimum_finger_count
    assert raised.value.actual_count == len(part.fingers)


def test_a_wrong_number_of_fingers_carries_the_counts():
    part = PartNarrowingAMixin(fingers=[Thumb()])

    with pytest.raises(UnexpectedFingerCountError) as raised:
        part.validate_assumptions()

    assert raised.value.expected_count == HasTwoFingers.finger_count
    assert raised.value.actual_count == len(part.fingers)


def test_too_few_arms_carries_the_counts():
    part = PartWithManyArms(arms=[MountedPart(), MountedPart()])

    with pytest.raises(TooFewArmsError) as raised:
        part.validate_assumptions()

    assert raised.value.minimum_count == HasArms.minimum_arm_count
    assert raised.value.actual_count == len(part.arms)


@pytest.mark.parametrize(
    "part_type, expected_count",
    [
        (PartWithOneArm, HasOneArm.arm_count),
        (PartWithLeftAndRightArm, HasLeftRightArm.arm_count),
    ],
)
def test_a_wrong_number_of_arms_carries_the_counts(part_type, expected_count):
    part = part_type(arms=[])

    with pytest.raises(UnexpectedArmCountError) as raised:
        part.validate_assumptions()

    assert raised.value.expected_count == expected_count
    assert raised.value.actual_count == 0
