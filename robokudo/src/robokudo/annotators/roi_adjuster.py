"""ROI adjustment for RoboKudo.

This module provides an annotator for adjusting the size of Region of Interest (ROI)
boundaries for object hypotheses. It can grow or shrink ROIs and their associated
masks by a specified pixel offset.
"""

from timeit import default_timer

from py_trees.common import Status
from typing_extensions import Type

from robokudo.annotators.core import BaseAnnotator
from robokudo.cas import CASViews
from robokudo.types.scene import ObjectHypothesis
from robokudo.utils.cv_helper import adjust_image_roi, adjust_mask
from robokudo.utils.error_handling import catch_and_raise_to_blackboard


class ROIAdjusterAnnotator(BaseAnnotator):
    """Annotator for adjusting ROI sizes of object hypotheses.

    This annotator can grow or shrink ROIs (Regions of Interest) on object
    hypotheses by a specified pixel offset. It handles both the ROI boundaries
    and their associated masks, ensuring proper adjustment of both.
    """

    class Descriptor(BaseAnnotator.Descriptor):
        """Configuration descriptor for ROI adjustment."""

        class Parameters:
            """Parameters for configuring ROI adjustment behavior."""

            def __init__(self) -> None:
                self.offset_pixel: int = 20
                """Pixels to add/subtract from ROI sides (positive grows, negative shrinks)"""

                self.analysis_scope: Type = ObjectHypothesis
                """Type of annotations to process."""

                self.fill_value_mask: int = 0
                """Value to fill new mask areas when growing ROIs"""

        # overwrite the parameters explicitly to enable auto-completion
        parameters = Parameters()

    def __init__(
        self,
        name: str = "ROIAdjusterAnnotator",
        descriptor: "ROIAdjusterAnnotator.Descriptor" = Descriptor(),
    ) -> None:
        """Initialize the ROI adjuster.

        :param name: Name of the annotator instance, defaults to "ROIAdjusterAnnotator"
        :param descriptor: Configuration descriptor, defaults to Descriptor()
        """
        super().__init__(name, descriptor)

    @catch_and_raise_to_blackboard
    def update(self) -> Status:
        """Update ROIs by applying the configured pixel offset.

        For each object hypothesis in the analysis scope:
        - Adjusts the ROI boundaries by the specified pixel offset
        - If a mask exists, adjusts it accordingly with the specified fill value

        :return: SUCCESS after adjusting all ROIs
        """
        start_timer = default_timer()
        color = self.get_cas().get(CASViews.COLOR_IMAGE)

        object_hypotheses = self.get_cas().filter_annotations_by_type(
            self.descriptor.parameters.analysis_scope
        )
        for object_hypothesis in object_hypotheses:
            adjust_image_roi(
                color, object_hypothesis.roi, self.descriptor.parameters.offset_pixel
            )

            if object_hypothesis.roi.mask is not None:
                object_hypothesis.roi.mask = adjust_mask(
                    object_hypothesis.roi.mask,
                    self.descriptor.parameters.offset_pixel,
                    fill_value=self.descriptor.parameters.fill_value_mask,
                )

        end_timer = default_timer()
        self.feedback_message = f"Processing took {(end_timer - start_timer):.4f}s"
        return Status.SUCCESS
