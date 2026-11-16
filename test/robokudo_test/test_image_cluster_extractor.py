from __future__ import annotations

# %% Imports

import numpy as np
from semantic_digital_twin.world_description.geometry import Color

from robokudo.annotators.image_cluster_extractor import (
    ColorName,
    ColorRangeBound,
    ImageClusterExtractor,
)

# %% Normalized color bounds


def test_threshold_accepts_normalized_hue_saturation_value() -> None:
    """
    Normalized bounds select pixels at the inclusive channel boundaries.
    """
    annotator = ImageClusterExtractor()
    annotator.hue_saturation_value_image = np.array(
        [[[100, 200, 200], [150, 200, 200], [200, 200, 200], [210, 200, 200]]],
        dtype=np.uint8,
    )

    mask = annotator._threshold_color_image(
        Color(hue=150 / 255, saturation=130 / 255, value=85 / 255),
        Color(hue=200 / 255, saturation=1.0, value=1.0),
    )

    assert mask.tolist() == [[0, 255, 255, 0]]


def test_presets_use_normalized_hue_saturation_value() -> None:
    """
    Configured defaults and color presets contain named hue, saturation, and value
    values.
    """
    parameters = ImageClusterExtractor.Descriptor.Parameters()

    assert type(parameters.minimum_hue_saturation_value) is Color
    assert type(parameters.maximum_hue_saturation_value) is Color
    for bounds in parameters.color_ranges.values():
        for bound in bounds.values():
            assert type(bound) is Color


def test_preset_bounds_use_enum_keys() -> None:
    """
    Each preset identifies its lower and upper bounds with enum members.
    """
    parameters = ImageClusterExtractor.Descriptor.Parameters()

    for bounds in parameters.color_ranges.values():
        assert set(bounds) == set(ColorRangeBound)
        assert all(type(bound) is ColorRangeBound for bound in bounds)


# %% Hue wraparound


class TestHueThresholds:
    """
    True red sits at the hue wheel's 0/255 seam, so a naive contiguous ``cv2.inRange``
    only ever matches one side of it.

    Confirmed live on the real
    Stretch demo: a plain ``[215,255]`` hue range matched 124 of 101760 pixels on a
    red object; wrapping the range around the seam recovered 1748.
    """

    def annotator(self) -> ImageClusterExtractor:
        return ImageClusterExtractor("test")

    def image_with_hues(self, hue_values: list[int]) -> np.ndarray:
        """
        A 1-pixel-tall hue, saturation, and value image with one pixel per given hue,
        full saturation/value.
        """
        return np.array(
            [[[hue, 255, 255] for hue in hue_values]],
            dtype=np.uint8,
        )

    def test_contiguous_range_does_not_wrap(self) -> None:
        """
        A normal range (min hue <= max hue, e.g. the "blue" preset) must behave exactly
        like a plain ``cv2.inRange`` call, matching only within the stated bounds.
        """
        annotator = self.annotator()
        annotator.hue_saturation_value_image = self.image_with_hues(
            [100, 150, 170, 210]
        )

        bounds = annotator.descriptor.parameters.color_ranges[ColorName.BLUE]
        mask = annotator._threshold_color_image(
            bounds[ColorRangeBound.LOWER],
            bounds[ColorRangeBound.UPPER],
        )

        assert list(mask[0]) == [0, 255, 255, 0]

    def test_wrapped_range_matches_both_sides_of_the_hue_seam(self) -> None:
        """
        A wrapped range (min hue > max hue) must match hues close to 255 AND hues close
        to 0, not just one side.
        """
        annotator = self.annotator()
        annotator.hue_saturation_value_image = self.image_with_hues(
            [0, 10, 15, 16, 200, 244, 245, 255]
        )

        bounds = annotator.descriptor.parameters.color_ranges[ColorName.RED]
        mask = annotator._threshold_color_image(
            bounds[ColorRangeBound.LOWER],
            bounds[ColorRangeBound.UPPER],
        )

        assert list(mask[0]) == [255, 255, 255, 0, 0, 0, 255, 255]

    def test_wrapped_range_still_respects_saturation_and_value_bounds(self) -> None:
        """
        Wrapping only changes how the hue channel is matched; saturation/value bounds
        must still exclude low-saturation pixels (e.g. gray/white background) that
        happen to default to hue 0, which is exactly what caused this bug to go
        unnoticed: gray background pixels reporting hue 0 look identical to wrapped red
        on the hue channel alone.
        """
        annotator = self.annotator()
        low_saturation_pixel = [0, 10, 255]
        high_saturation_pixel = [0, 200, 255]
        annotator.hue_saturation_value_image = np.array(
            [[low_saturation_pixel, high_saturation_pixel]], dtype=np.uint8
        )

        bounds = annotator.descriptor.parameters.color_ranges[ColorName.RED]
        mask = annotator._threshold_color_image(
            bounds[ColorRangeBound.LOWER],
            bounds[ColorRangeBound.UPPER],
        )

        assert list(mask[0]) == [0, 255]
