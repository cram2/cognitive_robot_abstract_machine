from __future__ import annotations

# %% Imports

from dataclasses import dataclass

import numpy as np

from krrood.ormatic.utils import classproperty
from semantic_digital_twin.world_description.geometry import HueSaturationValue

# %% OpenCV color channels


@dataclass
class OpenCVHueSaturationValueAdapter:
    """
    Convert normalized HSV values to and from OpenCV's 8-bit HSV_FULL channels.
    """

    @classproperty
    def maximum_channel_value(cls) -> int:
        """
        Largest channel value in an 8-bit HSV_FULL image.
        """
        return np.iinfo(np.uint8).max

    @classproperty
    def hue_channel_count(cls) -> int:
        """
        Number of hue steps in one full turn of an 8-bit HSV_FULL image.
        """
        return cls.maximum_channel_value + 1

    @classmethod
    def from_channels(cls, hue: int, saturation: int, value: int) -> HueSaturationValue:
        """
        Normalize OpenCV HSV_FULL channel values.
        """
        return HueSaturationValue(
            hue=hue / cls.hue_channel_count,
            saturation=saturation / cls.maximum_channel_value,
            value=value / cls.maximum_channel_value,
        )

    @classmethod
    def to_channels(cls, color: HueSaturationValue) -> tuple[int, int, int]:
        """
        Convert normalized HSV values to OpenCV HSV_FULL channel values.
        """
        return (
            int(color.hue * cls.hue_channel_count),
            round(color.saturation * cls.maximum_channel_value),
            round(color.value * cls.maximum_channel_value),
        )
