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
    assert SpeedLimit.WALKING == SpeedLimit.WALKING.value


def test_member_is_a_float() -> None:
    assert isinstance(SpeedLimit.CRAWLING, float)


def test_member_is_found_by_its_number() -> None:
    assert SpeedLimit(SpeedLimit.CRAWLING.value) is SpeedLimit.CRAWLING


# %% a member prints as its number, like the members of IntEnum and StrEnum


def test_member_converts_to_the_text_of_its_number() -> None:
    assert str(SpeedLimit.WALKING) == str(SpeedLimit.WALKING.value)


def test_member_formats_as_its_number() -> None:
    assert f"{SpeedLimit.CRAWLING}" == f"{SpeedLimit.CRAWLING.value}"
