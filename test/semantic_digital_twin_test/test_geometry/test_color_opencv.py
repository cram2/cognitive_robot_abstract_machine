from __future__ import annotations

# %% Imports

import pytest

from semantic_digital_twin.world_description.geometry import Color

# %% OpenCV channel conversion


def test_last_hue_bin_precedes_wraparound() -> None:
    """
    The largest 8-bit hue stays below the normalized hue wheel's endpoint.
    """
    color = Color.from_opencv_hue_saturation_value_full(
        hue=255, saturation=255, value=255
    )

    assert color.hue == 255 / 256


@pytest.mark.parametrize("channel_value", range(256))
def test_channel_conversion_preserves_each_byte_value(channel_value: int) -> None:
    """
    Normalizing and restoring an 8-bit threshold preserves its exact boundary.
    """
    color = Color.from_opencv_hue_saturation_value_full(
        hue=channel_value, saturation=channel_value, value=channel_value
    )

    assert color.to_opencv_hue_saturation_value_full() == (
        channel_value,
        channel_value,
        channel_value,
    )


def test_half_saturation_rounds_to_nearest_channel_value() -> None:
    """
    Normalized thresholds are rounded to the nearest representable byte.
    """
    color = Color(hue=0.5, saturation=0.5, value=0.5)

    assert color.to_opencv_hue_saturation_value_full() == (128, 128, 128)
