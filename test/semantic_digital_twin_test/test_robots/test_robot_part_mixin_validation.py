from __future__ import annotations

from dataclasses import dataclass

import pytest

from semantic_digital_twin.robots.robot_part_mixins import (
    HasFingers,
    HasLaser,
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
class PartCombiningIndependentMixins(
    HasTorso[MountedPart], HasLaser[MountedPart]
):
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

    with pytest.raises(AssertionError):
        part.validate_assumptions()


def test_a_part_satisfying_every_independent_mixin_passes():
    part = PartCombiningIndependentMixins(
        torso=MountedPart(), laser=MountedPart()
    )

    part.validate_assumptions()


# %% assumptions a narrowing mixin replaces


def test_a_narrowed_assumption_replaces_the_one_it_narrows():
    part = PartNarrowingAMixin(fingers=[Thumb(), OpposingFinger()])

    part.validate_assumptions()

    with pytest.raises(AssertionError):
        HasFingers.validate(part)
