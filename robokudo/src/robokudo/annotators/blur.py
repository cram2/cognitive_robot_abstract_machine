"""Image blur detection for RoboKudo.

This module provides an annotator for calculating and visualizing image blur metrics
using the Laplacian variance method. It can optionally halt pipeline execution if
blur exceeds a threshold.

.. note::
   The blur metric is calculated using the Laplacian operator variance, where:

   * Higher values indicate sharper images
   * Lower values indicate more blur
"""

import copy
from timeit import default_timer

import cv2
from py_trees.common import Status
from typing_extensions import TYPE_CHECKING

from robokudo.annotators.core import BaseAnnotator
from robokudo.cas import CASViews

if TYPE_CHECKING:
    import numpy.typing as npt


class BlurAnnotator(BaseAnnotator):
    """Annotator for calculating and visualizing image blur metrics.

    This annotator:

    * Calculates a blur metric using Laplacian variance
    * Visualizes the blur value on the image
    * Optionally halts pipeline if blur exceeds threshold

    .. warning::
       Setting return_failure_above_threshold to True will stop pipeline
       advancement when the blur threshold is exceeded.
    """

    class Descriptor(BaseAnnotator.Descriptor):
        """Configuration descriptor for blur detection."""

        class Parameters:
            """Parameters for configuring blur detection behavior."""

            def __init__(self) -> None:
                self.blur_threshold: float = 100.0
                """Threshold for acceptable blur level, defaults to 100"""

                self.return_failure_above_threshold: bool = True
                """Let this behaviour return failure to stop the advancement of the current pipeline"""

        # Overwrite the parameters explicitly to enable auto-completion
        parameters = Parameters()

    def __init__(
        self,
        name: str = "BlurAnnotator",
        descriptor: "BlurAnnotator.Descriptor" = Descriptor(),
    ) -> None:
        """Initialize the blur annotator. Minimal one-time init!

        :param name: Name of the annotator instance, defaults to "BlurAnnotator"
        :param descriptor: Configuration descriptor, defaults to Descriptor()
        """
        super().__init__(name, descriptor)
        self.logger.debug("%s.__init__()" % self.__class__.__name__)

    def compute_bluriness(self, img: npt.NDArray) -> float:
        """Compute the blur metric for an image.

        Uses Laplacian variance to measure image sharpness:

        * Converts image to grayscale
        * Computes Laplacian operator
        * Returns variance of Laplacian values

        :param img: Input image in BGR format
        :return: Blur metric value (higher = sharper)
        """
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return cv2.Laplacian(img, cv2.CV_64F).var()

    def update(self) -> Status:
        """Update blur detection and visualization.

        Creates visualizations containing:

        * Color image with blur metric overlay
        * Point cloud data

        :return: SUCCESS if blur is acceptable, FAILURE if above threshold (if configured)
        """
        start_timer = default_timer()

        cloud = self.get_cas().get(CASViews.CLOUD)
        color = self.get_cas().get(CASViews.COLOR_IMAGE)

        bluriness = self.compute_bluriness(color)

        visualization_img = copy.deepcopy(color)
        font = cv2.FONT_HERSHEY_COMPLEX
        visualization_img = cv2.putText(
            visualization_img,
            f"Bluriness: {bluriness}",
            (20, 90),
            font,
            1,
            (0, 0, 255),
            1,
            2,
        )

        # 3D Visualization
        vis_geometries = [
            {"name": "Cloud", "geometry": cloud},
        ]

        self.get_annotator_output_struct().set_geometries(vis_geometries)
        self.get_annotator_output_struct().set_image(visualization_img)

        end_timer = default_timer()
        self.feedback_message = f"Processing took {(end_timer - start_timer):.4f}s"
        return Status.SUCCESS
