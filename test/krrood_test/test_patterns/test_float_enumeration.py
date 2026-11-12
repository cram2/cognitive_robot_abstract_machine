"""
Tests that a member of a :class:`~krrood.patterns.float_enumeration.FloatEnum` behaves
as the number it names.
"""

from __future__ import annotations

from krrood.patterns.float_enumeration import FloatEnum

# %% enumeration under test


class SpeedLimit(FloatEnum):
    """
    Speed limits of a vehicle in metres per second.
    """

    WALKING = 1.5
    """
    The limit at walking speed.
    """

    CRAWLING = 0.25
    """
    The limit at crawling speed.
    """


# %% a member is its number


def test_member_equals_its_number() -> None:
    """
    A member can be compared with a plain float without reading its value first.
    """
    assert SpeedLimit.WALKING == SpeedLimit.WALKING.value


def test_member_is_a_float() -> None:
    """
    A member can be passed wherever a float is expected.
    """
    assert isinstance(SpeedLimit.CRAWLING, float)


def test_member_is_found_by_its_number() -> None:
    """
    A number read back from storage turns into the member it stands for.
    """
    assert SpeedLimit(SpeedLimit.CRAWLING.value) is SpeedLimit.CRAWLING


# %% a member prints as its number, like the members of IntEnum and StrEnum


def test_member_converts_to_the_text_of_its_number() -> None:
    """
    Converting a member to text gives its number, not its name.
    """
    assert str(SpeedLimit.WALKING) == str(SpeedLimit.WALKING.value)


def test_member_formats_as_its_number() -> None:
    """
    A member in a format string is written as its number, not its name.
    """
    assert f"{SpeedLimit.CRAWLING}" == f"{SpeedLimit.CRAWLING.value}"
