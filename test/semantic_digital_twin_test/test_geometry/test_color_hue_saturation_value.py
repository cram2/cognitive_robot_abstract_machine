from __future__ import annotations

# %% Imports

import pytest

from krrood.adapters.json_serializer import from_json, to_json
from semantic_digital_twin.world_description.geometry import (
    Color,
    HueSaturationValue,
)
from semantic_digital_twin.world_description.exceptions import InvalidColorChannelCount

# %% Stored hue, saturation, and value


@pytest.mark.parametrize("saturation, value", [(0.0, 0.5), (0.75, 0.0), (0.75, 0.5)])
def test_reading_rgb_preserves_stored_hue_saturation_value(
    saturation: float, value: float
) -> None:
    """
    RGB access leaves stored hue, saturation, and value unchanged.
    """
    hue_saturation_value = HueSaturationValue(
        hue=0.75, saturation=saturation, value=value
    )
    color = Color.from_hue_saturation_value(hue_saturation_value, opacity=0.5)

    color.to_rgb()
    color.to_rgba()
    color.to_hex()

    assert color.to_hue_saturation_value() is hue_saturation_value
    assert color.hue_saturation_value is hue_saturation_value
    assert color.A == 0.5


def test_existing_constructor_stores_hue_saturation_value() -> None:
    """
    The existing constructor preserves its positional and keyword channels.
    """
    color = Color(0.25, G=0.5, B=0.75, A=0.5)

    assert color.hue_saturation_value == HueSaturationValue(
        hue=7 / 12, saturation=2 / 3, value=0.75
    )
    assert (color.R, color.G, color.B, color.A) == color.to_rgba()
    assert color.to_rgba() == (0.25, 0.5, 0.75, 0.5)


def test_rgb_converts_to_normalized_hue_saturation_value() -> None:
    """
    RGB colors expose normalized hue, saturation, and value.
    """
    assert Color.GREEN().to_hue_saturation_value() == HueSaturationValue(
        hue=1 / 3, saturation=1.0, value=1.0
    )


def test_hue_saturation_value_exposes_rgb_channels() -> None:
    """
    An HSV color exposes converted RGB channels through the existing API.
    """
    color = Color.from_hue_saturation_value(
        HueSaturationValue(hue=0.75, saturation=0.5, value=0.5), opacity=0.25
    )

    assert color.to_rgba() == (0.375, 0.25, 0.5, 0.25)
    assert (color.R, color.G, color.B, color.A) == color.to_rgba()


def test_red_channel_remains_writable() -> None:
    """
    Assigning red updates the stored HSV coordinates.
    """
    color = Color.BLACK()
    color.R = 1.0
    assert color == Color.RED()


def test_green_channel_remains_writable() -> None:
    """
    Assigning green updates the stored HSV coordinates.
    """
    color = Color.BLACK()
    color.G = 1.0
    assert color == Color.GREEN()


def test_blue_channel_remains_writable() -> None:
    """
    Assigning blue updates the stored HSV coordinates.
    """
    color = Color.BLACK()
    color.B = 1.0
    assert color == Color.BLUE()


def test_opacity_remains_writable_without_changing_color_coordinates() -> None:
    """
    Assigning opacity leaves hue, saturation, and value untouched.
    """
    color = Color.RED()
    hue_saturation_value = color.to_hue_saturation_value()
    color.A = 0.25
    assert color.A == 0.25
    assert color.to_hue_saturation_value() is hue_saturation_value


def test_invalid_channel_count_raises_color_exception() -> None:
    """
    A color requires RGB channels and optional opacity.
    """
    with pytest.raises(InvalidColorChannelCount):
        Color.from_list([])


# %% Color identity


def test_rgb_and_hue_saturation_value_construction_share_equality_and_hash() -> None:
    """
    Equivalent colors constructed from either channel format share a dictionary key.
    """
    rgb_color = Color.RED()
    hsv_color = Color.from_hue_saturation_value(
        HueSaturationValue(hue=0.0, saturation=1.0, value=1.0)
    )

    assert hsv_color == rgb_color
    assert hash(hsv_color) == hash(rgb_color)
    assert len({hsv_color, rgb_color}) == 1


def test_equal_gray_colors_can_retain_different_hues() -> None:
    """
    An undefined visible hue does not affect color equality.
    """
    first = Color.from_hue_saturation_value(
        HueSaturationValue(hue=0.25, saturation=0.0, value=0.5)
    )
    second = Color.from_hue_saturation_value(
        HueSaturationValue(hue=0.75, saturation=0.0, value=0.5)
    )

    assert first == second
    assert hash(first) == hash(second)


# %% Color serialization


@pytest.mark.parametrize("saturation, value", [(0.0, 0.5), (0.75, 0.0)])
def test_serialization_preserves_stored_hue_saturation_value(
    saturation: float, value: float
) -> None:
    """
    JSON persistence preserves stored hue, saturation, value, and opacity.
    """
    hue_saturation_value = HueSaturationValue(
        hue=0.75, saturation=saturation, value=value
    )
    color = Color.from_hue_saturation_value(hue_saturation_value, opacity=0.5)

    restored = from_json(to_json(color))

    assert restored.hue_saturation_value == hue_saturation_value
    assert restored.A == color.A


def test_serialization_preserves_rgb_created_color() -> None:
    """
    RGB-created colors retain their stored HSV coordinates and opacity.
    """
    color = Color(0.25, 0.5, 0.75, 0.5)

    restored = from_json(to_json(color))

    assert restored.hue_saturation_value == color.hue_saturation_value
    assert restored.A == color.A
