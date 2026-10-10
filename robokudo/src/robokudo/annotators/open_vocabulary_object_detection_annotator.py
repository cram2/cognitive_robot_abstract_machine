"""Open-vocabulary object detection with optional Segment Anything masks."""

from __future__ import annotations

from enum import IntEnum, StrEnum
from timeit import default_timer

import cv2
import numpy
import py_trees
import torch
from transformers import Owlv2Processor, Owlv2ForObjectDetection
from typing_extensions import TYPE_CHECKING
from ultralytics import SAM as SegmentAnythingModel

import robokudo.annotators.core
import robokudo.types
import robokudo.types.annotation
import robokudo.types.scene
import robokudo.utils.annotator_helper
import robokudo.utils.cv_helper
from robokudo.cas import CASViews
from robokudo.types.scene import ObjectHypothesis
from robokudo.utils.error_handling import catch_and_raise_to_blackboard
from semantic_digital_twin.world_description.geometry import Color

if TYPE_CHECKING:
    import numpy.typing as numpy_typing


# %% Detection and segmentation values


class DetectionResultField(StrEnum):
    """Fields returned by the object detection postprocessor."""

    BOXES = "boxes"
    """Detected bounding boxes."""

    SCORES = "scores"
    """Detection confidence scores."""

    LABELS = "labels"
    """Indices of the matching text labels."""


class SegmentPromptLabel(IntEnum):
    """Labels for segmentation prompts."""

    FOREGROUND = 1
    """Mark a prompt as belonging to the foreground."""


# %% Bounding-box labels


def get_box_text(object_hypothesis):
    """Label a hypothesis with its most confident classification or ROI identifier.

    :param object_hypothesis: Hypothesis whose classifications determine the label.
    :return: Identifier, class name, and confidence, or an identifier-only label.
    """
    maximum_confidence = -1
    best_classification = None

    for annotation in object_hypothesis.annotations:
        if isinstance(annotation, robokudo.types.annotation.Classification):
            if annotation.confidence > maximum_confidence:
                maximum_confidence = annotation.confidence
                best_classification = annotation

    if best_classification is None:
        return f"ROI-{object_hypothesis.id}"
    else:
        return f"{object_hypothesis.id}: {best_classification.classname}, {best_classification.confidence:.2f}"


# %% Open-vocabulary object detection


class OpenVocabularyObjectDetectionAnnotator(
    robokudo.annotators.core.ThreadedAnnotator
):
    """Detect configured text classes and optionally segment their image regions."""

    class Descriptor(robokudo.annotators.core.BaseAnnotator.Descriptor):
        class Parameters:
            def __init__(self):
                """Set default detection and segmentation settings."""
                self.classes = ["Cat", "Dog"]
                """Text labels to detect in the color image."""
                self.detection_model = "google/owlv2-base-patch16-ensemble"
                """Pretrained OWLv2 model name or local checkpoint directory."""
                self.detection_processor = "google/owlv2-base-patch16-ensemble"
                """Pretrained OWLv2 processor name or local checkpoint directory."""
                self.detection_threshold = 0.2
                """Confidence threshold for retaining detections."""

                self.segment_anything_model_path = "mobile_sam.pt"
                """Checkpoint for the Segment Anything model."""
                self.precision_mode = False
                """Whether to predict object masks with Segment Anything."""
                self.refine_bounding_boxes = False
                """Whether to refine boxes when ``precision_mode`` is enabled."""

        parameters = Parameters()
        """Detection and segmentation configuration defaults."""

    def __init__(
        self,
        name="OpenVocabularyObjectDetectionAnnotator",
        descriptor=Descriptor(),
    ) -> None:
        """Initialize the threaded annotator and load the configured models.

        :param name: Name of the behaviour-tree node.
        :param descriptor: Detection and segmentation configuration.
        """
        super(OpenVocabularyObjectDetectionAnnotator, self).__init__(name, descriptor)

        self.classes = self.descriptor.parameters.classes
        """Text labels supplied to the detection processor."""

        self.model = Owlv2ForObjectDetection.from_pretrained(
            self.descriptor.parameters.detection_model
        )
        """Loaded OWLv2 object detection model."""
        self.processor = Owlv2Processor.from_pretrained(
            self.descriptor.parameters.detection_processor
        )
        """Processor for the detection model's text and image inputs."""

        if self.descriptor.parameters.precision_mode:
            self.segment_anything_model = SegmentAnythingModel(
                self.descriptor.parameters.segment_anything_model_path
            )
            """Segmentation model available when ``precision_mode`` is enabled."""

    @catch_and_raise_to_blackboard
    def compute(self) -> py_trees.common.Status:
        """Main method: Detect objects based on configured classes and update the visualization.

        :return: ``py_trees.common.Status.SUCCESS`` after processing the color image.
        """
        start_timer = default_timer()
        image = self.get_cas().get(CASViews.COLOR_IMAGE)
        object_hypotheses = self.detect_objects(image)
        visualization = self.visualize_objects(image, object_hypotheses)
        self.get_cas().annotations.extend(object_hypotheses)
        self.get_annotator_output_struct().set_image(visualization)
        self.feedback_message = f"Processing took {default_timer() - start_timer:.4f}s"
        return py_trees.common.Status.SUCCESS

    def detect_objects(
        self, image: numpy_typing.NDArray[numpy.uint8]
    ) -> list[ObjectHypothesis]:
        """Detect configured classes with optional masks cropped to their regions.

        :param image: Color image in BGR channel order.
        :return: Classified hypotheses with bounding boxes and optional masks.
        """
        inputs = self.processor(
            text=self.classes,
            images=cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
            return_tensors="pt",
            padding=True,
        )
        with torch.no_grad():
            outputs = self.model(**inputs)

        results = self.processor.post_process_grounded_object_detection(
            outputs=outputs,
            target_sizes=torch.Tensor([image.shape[:2]]),
            threshold=self.descriptor.parameters.detection_threshold,
        )
        result = results[0]
        object_hypotheses = []
        bounding_box_decimal_places = 2
        confidence_decimal_places = 3
        for box, score, label in zip(
            result[DetectionResultField.BOXES],
            result[DetectionResultField.SCORES],
            result[DetectionResultField.LABELS],
        ):
            box = [
                round(coordinate, bounding_box_decimal_places)
                for coordinate in box.tolist()
            ]
            confidence = round(score.item(), confidence_decimal_places)
            self.rk_logger.info(
                f"Detected {self.classes[label]} with confidence {confidence} at location {box}"
            )
            left, top, right, bottom = box
            object_hypothesis = ObjectHypothesis()
            object_hypothesis.roi.roi.pos.x = int(left)
            object_hypothesis.roi.roi.pos.y = int(top)
            object_hypothesis.roi.roi.width = int(right - left)
            object_hypothesis.roi.roi.height = int(bottom - top)
            if self.descriptor.parameters.precision_mode:
                object_hypothesis.roi.mask = self.predict_mask(image, box)
                if self.descriptor.parameters.refine_bounding_boxes:
                    self.refine_bounding_box(object_hypothesis)
                object_hypothesis.roi.mask = robokudo.utils.cv_helper.crop_image_roi(
                    object_hypothesis.roi.mask, object_hypothesis.roi
                )
            object_hypothesis.annotations.append(
                robokudo.types.annotation.Classification(
                    classname=self.classes[label],
                    source=self.get_class_name(),
                    confidence=confidence,
                )
            )
            object_hypotheses.append(object_hypothesis)
        return object_hypotheses

    def predict_mask(
        self, image: numpy_typing.NDArray[numpy.uint8], box: list[float]
    ) -> numpy_typing.NDArray[numpy.uint8]:
        """Predict a mask at the original color-image resolution.

        :param image: Color image in BGR channel order.
        :param box: Bounding box as ``[left, top, right, bottom]`` in image pixels.
        :return: Mask with foreground pixels set to 255 and background pixels to 0.
        """
        result = self.segment_anything_model.predict(
            image, bboxes=[box], labels=[SegmentPromptLabel.FOREGROUND]
        )[0]
        mask = result.masks.data.cpu().numpy()[0].astype(numpy.uint8)
        return mask * numpy.iinfo(numpy.uint8).max

    def refine_bounding_box(self, object_hypothesis: ObjectHypothesis) -> None:
        """Adjust a hypothesis's bounding box to the first contour of its mask.

        Empty masks leave the bounding box unchanged.

        :param object_hypothesis: Object Hypothesis with a full-image mask; updated in place.
        """
        contours, _ = cv2.findContours(
            object_hypothesis.roi.mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            return
        self.rk_logger.info(f"Attempting to fix {object_hypothesis.roi.roi}")
        left, top, width, height = cv2.boundingRect(contours[0])
        object_hypothesis.roi.roi.pos.x = int(left)
        object_hypothesis.roi.roi.pos.y = int(top)
        object_hypothesis.roi.roi.width = int(width)
        object_hypothesis.roi.roi.height = int(height)
        self.rk_logger.info(f"After fix: {object_hypothesis.roi.roi}")

    def visualize_objects(
        self,
        image: numpy_typing.NDArray[numpy.uint8],
        object_hypotheses: list[ObjectHypothesis],
    ) -> numpy_typing.NDArray[numpy.uint8]:
        """Draw detection labels, bounding boxes, and optional colored masks.

        :param image: Color image in BGR channel order.
        :param object_hypotheses: Hypotheses with optional masks cropped to their
            regions.
        :return: Annotated copy of the color image.
        """
        visualization = image.copy()
        robokudo.utils.annotator_helper.draw_bounding_boxes_from_object_hypotheses(
            visualization, object_hypotheses, get_box_text
        )
        if not self.descriptor.parameters.precision_mode:
            return visualization
        palette = (
            Color.BLUE(),
            Color.GREEN(),
            Color.RED(),
            Color.MAGENTA(),
            Color.CYAN(),
        )
        # Convert normalized RGB colors to OpenCV's byte-valued BGR.
        mask_colors = numpy.rint(
            numpy.array([color.to_rgb()[::-1] for color in palette])
            * numpy.iinfo(numpy.uint8).max
        ).astype(numpy.uint8)
        for index, object_hypothesis in enumerate(object_hypotheses):
            image_region = robokudo.utils.cv_helper.crop_image_roi(
                visualization, object_hypothesis.roi
            )
            image_region[object_hypothesis.roi.mask == numpy.iinfo(numpy.uint8).max] = (
                mask_colors[index % len(mask_colors)]
            )
        return visualization
