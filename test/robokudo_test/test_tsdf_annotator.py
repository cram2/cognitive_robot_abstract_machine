"""
Checks for visualizing posed object hypotheses.
"""

from __future__ import annotations

# %% Imports
import numpy as np
import pytest

from robokudo.annotators.tsdf_annotator import VolumeIntegrationInput, TSDFAnnotator
from robokudo.types.annotation import PoseAnnotation
from robokudo.types.cv import Point2D, Rect
from robokudo.types.scene import ObjectHypothesis


# %% Posed object visualization
@pytest.mark.parametrize("full_image_mask", [False, True])
def test_visualization_places_object_mask_in_its_roi(full_image_mask: bool) -> None:
    """
    Both image-sized and cropped masks highlight the object's image position.
    """
    image = np.zeros((16, 16, 3), dtype=np.uint8)
    roi = Rect(pos=Point2D(x=4, y=4), width=8, height=8)
    volume_integration_input = VolumeIntegrationInput(
        object_hypothesis=ObjectHypothesis(),
        pose=PoseAnnotation(),
        depth_image_region=roi,
    )
    mask = np.zeros(
        image.shape[:2] if full_image_mask else (roi.height, roi.width), dtype=np.uint8
    )
    mask_row = 8 if full_image_mask else 4
    mask_column = 8 if full_image_mask else 4
    mask[mask_row, mask_column] = 1

    visualization = TSDFAnnotator().get_visualization_image(
        image, [volume_integration_input], [mask]
    )

    np.testing.assert_array_equal(visualization[8, 8], [0, 128, 0])
    np.testing.assert_array_equal(visualization[8, 9], [0, 0, 0])
