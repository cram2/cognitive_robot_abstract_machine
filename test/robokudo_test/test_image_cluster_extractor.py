import numpy as np

from robokudo.annotators.image_cluster_extractor import ImageClusterExtractor


class TestImageClusterExtractorHsvThreshold:
    """
    True red sits at the hue wheel's 0/255 seam, so a naive contiguous
    ``cv2.inRange`` only ever matches one side of it. Confirmed live on the real
    Stretch demo: a plain ``[215,255]`` hue range matched 124 of 101760 pixels on a
    red object; wrapping the range around the seam recovered 1748.
    """

    def annotator(self) -> ImageClusterExtractor:
        return ImageClusterExtractor("test")

    def hsv_image_with_hue(self, hue_values: list[int]) -> np.ndarray:
        """
        A 1-pixel-tall HSV image with one pixel per given hue, full saturation/value.
        """
        return np.array(
            [[[hue, 255, 255] for hue in hue_values]],
            dtype=np.uint8,
        )

    def test_contiguous_range_does_not_wrap(self):
        """
        A normal range (min hue <= max hue, e.g. the "blue" preset) must behave exactly
        like a plain ``cv2.inRange`` call, matching only within the stated bounds.
        """
        annotator = self.annotator()
        annotator.hsv = self.hsv_image_with_hue([100, 150, 170, 210])

        mask = annotator._threshold_hsv((150, 130, 85), (200, 255, 255))

        assert list(mask[0]) == [0, 255, 255, 0]

    def test_wrapped_range_matches_both_sides_of_the_hue_seam(self):
        """
        A wrapped range (min hue > max hue) must match hues close to 255 AND hues close
        to 0, not just one side.
        """
        annotator = self.annotator()
        annotator.hsv = self.hsv_image_with_hue([0, 10, 15, 16, 200, 244, 245, 255])

        mask = annotator._threshold_hsv((245, 150, 95), (15, 255, 255))

        assert list(mask[0]) == [255, 255, 255, 0, 0, 0, 255, 255]

    def test_wrapped_range_still_respects_saturation_and_value_bounds(self):
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
        annotator.hsv = np.array(
            [[low_saturation_pixel, high_saturation_pixel]], dtype=np.uint8
        )

        mask = annotator._threshold_hsv((245, 150, 95), (15, 255, 255))

        assert list(mask[0]) == [0, 255]
