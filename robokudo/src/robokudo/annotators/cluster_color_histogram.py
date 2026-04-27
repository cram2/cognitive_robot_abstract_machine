"""Color histogram analysis for object hypotheses.

This module provides an annotator for:

* Calculating 2D color histograms for object hypotheses
* Analyzing hue and saturation distributions
* Supporting masked ROI analysis
* Optional visualization of histogram plots

The module uses:

* HSV color space for analysis
* 2D histograms for hue-saturation distributions
* Matplotlib for optional visualization

.. note::
   Histogram plotting is optional and can be disabled for performance.
"""

from __future__ import annotations
from py_trees.common import Status

from robokudo.annotators.core import BaseAnnotator
from robokudo.types.annotation import ColorHistogram
from robokudo.types.scene import ObjectHypothesis
import copy
import math
from timeit import default_timer

import cv2
import numpy as np
from matplotlib import pyplot as plt
from typing_extensions import List, Optional, Tuple, TYPE_CHECKING

from robokudo.annotators.cluster_color import Color
from robokudo.cas import CASViews
from robokudo.types.cv import Rect

if TYPE_CHECKING:
    import numpy.typing as npt


class ClusterColorHistogramAnnotator(BaseAnnotator):
    """Hue and saturation histogram analysis for object hypotheses with RGB ROI and Mask.

    This annotator:

    * Calculates 2D histograms of hue and saturation
    * Processes masked ROIs from object hypotheses
    * Normalizes histogram distributions
    * Optionally generates visualization plots

    .. warning::
       Histogram plotting can significantly impact performance (200-500ms).
    """

    class Descriptor(BaseAnnotator.Descriptor):
        """Configuration descriptor for color histogram analysis."""

        class Parameters:
            """Parameters for configuring histogram calculation."""

            def __init__(self) -> None:
                self.histogram_cols: int = 16
                """Number of histogram columns (hue bins), defaults to 16"""

                self.histogram_rows: int = 16
                """histogram_rows: Number of histogram rows(saturation bins), defaults to 16"""

                self.generate_plot_output: bool = False
                """Whether to generate histogram plots, defaults to False. Plotting takes a lot of time in matplotlib (200-500ms)"""

        # Overwrite the parameters explicitly to enable auto-completion
        parameters = Parameters()

    def __init__(
        self,
        name: str = "ClusterColorHistogramAnnotator",
        descriptor: "ClusterColorHistogramAnnotator.Descriptor" = Descriptor(),
    ) -> None:
        """Initialize the color histogram analyzer. Minimal one-time init!

        :param name: Name of this annotator instance, defaults to "ClusterColorHistogramAnnotator"
        :param descriptor: Configuration descriptor, defaults to Descriptor()
        """
        super().__init__(name, descriptor)
        self.rk_logger.debug("%s.__init__()" % self.__class__.__name__)

    def update(self) -> Status:
        """Process object hypotheses and calculate color histograms.

        The method:

        * Loads point cloud and color image from CAS
        * Creates color histogram annotations
        * Updates visualization if enabled

        :return: SUCCESS after processing
        """
        start_timer = default_timer()

        cloud = self.get_cas().get(CASViews.CLOUD)
        color = self.get_cas().get(CASViews.COLOR_IMAGE)

        # List for visualization purposes
        self.cluster_color_info: List[List[Tuple[Color, int, float]]] = []
        self.cluster_rois: List[Rect] = []

        visualization_img = self.create_color_histogram_annotations(color)

        if visualization_img is None:  # no plots have been generated?
            visualization_img = copy.deepcopy(color)

        # 3D Visualization
        vis_geometries = [
            {"name": "Cloud", "geometry": cloud},
        ]

        self.get_annotator_output_struct().set_geometries(vis_geometries)
        self.get_annotator_output_struct().set_image(visualization_img)

        end_timer = default_timer()
        self.feedback_message = f"Processing took {(end_timer - start_timer):.4f}s"
        return Status.SUCCESS

    def create_color_histogram_annotations(
        self, color: npt.NDArray
    ) -> Optional[npt.NDArray]:
        """Calculate 2D color histograms for object hypotheses.

        For each object hypothesis with a valid ROI mask:

        * Extracts color image region and mask
        * Converts to HSV color space
        * Calculates 2D histogram of hue and saturation
        * Normalizes histogram distribution
        * Creates histogram annotation
        * Optionally generates visualization plot

        :param color: Input color image
        :return: Combined histogram plot image if enabled, None otherwise
        """
        # Iterate over everything that is a Object hypothesis and calculate the colors
        cluster_index = 0

        object_hypotheses = self.get_cas().filter_annotations_by_type(ObjectHypothesis)

        if self.descriptor.parameters.generate_plot_output:
            plot_columns = 3
            cluster_count = len(object_hypotheses)
            fig, axes = plt.subplots(
                nrows=math.ceil(cluster_count / plot_columns), ncols=plot_columns
            )
            fig.tight_layout()
            axes1d = axes.ravel()

        for object_hypothesis in object_hypotheses:
            if object_hypothesis.roi.mask is None:
                continue

            roi = object_hypothesis.roi.roi

            cluster_color_img = color[
                roi.pos.y : roi.pos.y + roi.height, roi.pos.x : roi.pos.x + roi.width
            ]

            cluster_mask = object_hypothesis.roi.mask

            color_height, color_width, color_dim = cluster_color_img.shape
            mask_height, mask_width = cluster_mask.shape

            assert color_height == mask_height
            assert color_width == mask_width

            # Convert to HSV_FULL. FULL is important, # because otherwise you just get H=0...180
            cluster_hsv_color_img = cv2.cvtColor(
                cluster_color_img, cv2.COLOR_BGR2HSV_FULL
            )

            # Generate the actual histogram
            hist_size = [
                self.descriptor.parameters.histogram_cols,
                self.descriptor.parameters.histogram_rows,
            ]
            histogram_2d = cv2.calcHist(
                [cluster_hsv_color_img],
                [0, 1],
                cluster_mask,
                histSize=hist_size,
                ranges=[0, 256, 0, 256],
            )
            normalized_histogram2d = histogram_2d / histogram_2d.sum()

            color_histogram_annotation = ColorHistogram()
            color_histogram_annotation.normalized = True
            color_histogram_annotation.hist = normalized_histogram2d
            object_hypothesis.annotations.append(color_histogram_annotation)

            # Add the histogram to the plot if desired
            if self.descriptor.parameters.generate_plot_output:
                p = axes1d[cluster_index].imshow(
                    normalized_histogram2d, interpolation="nearest"
                )
                axes1d[cluster_index].set_title(
                    f"HS(V) hist cluster={cluster_index}", fontdict={"fontsize": 8}
                )
                axes1d[cluster_index].set_ylabel("sat")
                axes1d[cluster_index].set_xlabel("hue")

            cluster_index += 1

        # Draw the final result containing all plots if desired
        if self.descriptor.parameters.generate_plot_output:
            # https://stackoverflow.com/questions/43099734/combining-cv2-imshow-with-matplotlib-plt-show-in-real-time/43101480
            fig.canvas.draw()
            img = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep="")
            img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            plt.cla()  # cleanup figures
            return img

        return None
