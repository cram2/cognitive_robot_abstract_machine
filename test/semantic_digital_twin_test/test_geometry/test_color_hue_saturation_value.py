from __future__ import annotations

# %% Imports
from copy import copy

import pytest

from krrood.adapters.json_serializer import from_json, to_json
from semantic_digital_twin.world_description.geometry import Color
from semantic_digital_twin.world_description.exceptions import InvalidColorChannelCount


# %% Stored hue, saturation, and value
@pytest.mark.parametrize("saturation, value", [(0.0, 0.5), (0.75, 0.0), (0.75, 0.5)])
def test_reading_rgb_preserves_stored_hue_saturation_value(
    saturation: float, value: float
) -> None:
    """
    RGB access leaves stored hue, saturation, and value unchanged.
    """
    expected_color = Color(hue=0.75, saturation=saturation, value=value, opacity=0.5)
    color = copy(expected_color)

    color.to_rgb()
    color.to_rgba()
    color.to_hex()

    assert color.hue == expected_color.hue
    assert color.saturation == expected_color.saturation
    assert color.value == expected_color.value
    assert color.opacity == expected_color.opacity


def test_rgba_factory_exposes_rgb_channels() -> None:
    """
    RGB accessors expose the channels supplied to the RGBA factory.
    """
    color = Color.from_rgba(red=0.25, green=0.5, blue=0.75, opacity=0.5)

    assert (color.R, color.G, color.B, color.A) == color.to_rgba()


def test_rgb_converts_to_normalized_hue_saturation_value() -> None:
    """
    RGB colors expose normalized hue, saturation, and value.
    """
    color = Color.GREEN()

    assert color.hue == 1 / 3
    assert color.saturation == 1.0
    assert color.value == 1.0


def test_hue_saturation_value_exposes_rgb_channels() -> None:
    """
    An HSV color exposes converted RGB channels through the existing API.
    """
    color = Color(hue=0.75, saturation=0.5, value=0.5, opacity=0.25)

    assert color.to_rgba() == (0.375, 0.25, 0.5, 0.25)
    assert (color.R, color.G, color.B, color.A) == color.to_rgba()


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
    hsv_color = Color(hue=0.0, saturation=1.0, value=1.0)

    assert hsv_color == rgb_color
    assert hash(hsv_color) == hash(rgb_color)
    assert len({hsv_color, rgb_color}) == 1


@pytest.mark.parametrize(
    "first, second",
    [
        (
            Color(hue=0.25, saturation=0.0, value=0.5),
            Color(hue=0.75, saturation=0.0, value=0.5),
        ),
        (
            Color(hue=0.25, saturation=0.25, value=0.0),
            Color(hue=0.25, saturation=0.75, value=0.0),
        ),
    ],
)
def test_visually_identical_colors_preserve_distinct_coordinates(
    first: Color, second: Color
) -> None:
    """
    Distinct HSV coordinates remain separate dictionary keys even for identical RGB.
    """
    assert first.to_rgba() == second.to_rgba()
    assert first != second
    assert len({first, second}) == 2


def test_color_hash_uses_stored_coordinates() -> None:
    """
    The color hash is determined by its stored HSVA coordinates.
    """
    color = Color(hue=0.25, saturation=0.0, value=0.5, opacity=0.75)

    assert hash(color) == hash(
        (color.hue, color.saturation, color.value, color.opacity)
    )


# %% Color serialization
@pytest.mark.parametrize("saturation, value", [(0.0, 0.5), (0.75, 0.0)])
def test_serialization_preserves_stored_hue_saturation_value(
    saturation: float, value: float
) -> None:
    """
    JSON persistence preserves stored hue, saturation, value, and opacity.
    """
    color = Color(hue=0.75, saturation=saturation, value=value, opacity=0.5)

    restored = from_json(to_json(color))

    assert restored.hue == color.hue
    assert restored.saturation == color.saturation
    assert restored.value == color.value
    assert restored.opacity == color.opacity


def test_serialization_preserves_rgb_created_color() -> None:
    """
    RGB-created colors retain their stored HSV coordinates and opacity.
    """
    color = Color.from_rgba(red=0.25, green=0.5, blue=0.75, opacity=0.5)

    restored = from_json(to_json(color))

    assert restored.hue == color.hue
    assert restored.saturation == color.saturation
    assert restored.value == color.value
    assert restored.opacity == color.opacity
