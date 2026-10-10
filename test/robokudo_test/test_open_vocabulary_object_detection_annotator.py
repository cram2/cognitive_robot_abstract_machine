"""
Verify NumPy image inputs for open-vocabulary detection and mask prediction.
"""

from unittest.mock import MagicMock

import cv2
import numpy
import pytest
import torch
from py_trees.common import Status
from semantic_digital_twin.world_description.geometry import Color

from robokudo.annotators.open_vocabulary_object_detection_annotator import (
    DetectionResultField,
    OpenVocabularyObjectDetectionAnnotator,
    Owlv2ForObjectDetection,
    Owlv2Processor,
)
from robokudo.cas import CASViews
from robokudo.pipeline import Pipeline
from robokudo.types.annotation import Classification
from robokudo.types.scene import ObjectHypothesis
from robokudo.utils.cv_helper import crop_image_roi

# %% NumPy image inputs


@pytest.fixture
def numpy_image_annotator(
    monkeypatch: pytest.MonkeyPatch,
) -> OpenVocabularyObjectDetectionAnnotator:
    """
    Create an annotator with local model mocks and a non-square BGR image.
    """
    processor = MagicMock(return_value={})
    processor.post_process_grounded_object_detection.return_value = [
        {
            DetectionResultField.BOXES: torch.empty((0, 4)),
            DetectionResultField.SCORES: torch.empty(0),
            DetectionResultField.LABELS: torch.empty(0, dtype=torch.long),
        }
    ]
    monkeypatch.setattr(
        Owlv2Processor, "from_pretrained", MagicMock(return_value=processor)
    )
    monkeypatch.setattr(
        Owlv2ForObjectDetection, "from_pretrained", MagicMock(return_value=MagicMock())
    )
    descriptor = OpenVocabularyObjectDetectionAnnotator.Descriptor()
    descriptor.parameters = descriptor.Parameters()
    descriptor.parameters.global_with_depth = False
    annotator = OpenVocabularyObjectDetectionAnnotator(descriptor=descriptor)
    monkeypatch.setattr(annotator, "get_annotator_output_struct", MagicMock())
    pipeline = Pipeline("Detection")
    pipeline.add_child(annotator)
    image = numpy.arange(4 * 6 * 3, dtype=numpy.uint8).reshape(4, 6, 3)
    annotator.get_cas().set(CASViews.COLOR_IMAGE, image)
    return annotator


def test_detector_receives_rgb_numpy_image(
    numpy_image_annotator: OpenVocabularyObjectDetectionAnnotator,
) -> None:
    """
    Convert the stored BGR image to an RGB array for the detector.
    """
    image = numpy_image_annotator.get_cas().get(CASViews.COLOR_IMAGE)
    assert numpy_image_annotator.compute() == Status.SUCCESS
    received_image = numpy_image_annotator.processor.call_args.kwargs["images"]
    assert isinstance(received_image, numpy.ndarray)
    numpy.testing.assert_array_equal(
        received_image, cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    )


def test_target_sizes_use_numpy_image_height_and_width(
    numpy_image_annotator: OpenVocabularyObjectDetectionAnnotator,
) -> None:
    """
    Scale detection boxes to the image height and width, excluding channels.
    """
    image = numpy_image_annotator.get_cas().get(CASViews.COLOR_IMAGE)
    assert numpy_image_annotator.compute() == Status.SUCCESS
    target_sizes = numpy_image_annotator.processor.post_process_grounded_object_detection.call_args.kwargs[
        "target_sizes"
    ]
    torch.testing.assert_close(target_sizes, torch.Tensor([image.shape[:2]]))


def test_mask_predictor_receives_original_bgr_numpy_image(
    numpy_image_annotator: OpenVocabularyObjectDetectionAnnotator,
) -> None:
    """
    Preserve the stored BGR array when requesting a precision-mode mask.
    """
    image = numpy_image_annotator.get_cas().get(CASViews.COLOR_IMAGE)
    height, width = image.shape[:2]
    numpy_image_annotator.descriptor.parameters.precision_mode = True
    numpy_image_annotator.segment_anything_model = MagicMock()
    numpy_image_annotator.segment_anything_model.predict.return_value[
        0
    ].masks.data.cpu.return_value.numpy.return_value = numpy.ones((1, height, width))
    numpy_image_annotator.processor.post_process_grounded_object_detection.return_value = [
        {
            DetectionResultField.BOXES: torch.Tensor([[0, 0, width, height]]),
            DetectionResultField.SCORES: torch.Tensor([1]),
            DetectionResultField.LABELS: torch.tensor([0]),
        }
    ]
    assert numpy_image_annotator.compute() == Status.SUCCESS
    assert (
        numpy_image_annotator.segment_anything_model.predict.call_args.args[0] is image
    )


# %% Precision-mode mask processing


@pytest.fixture
def precision_mask_annotator(
    numpy_image_annotator: OpenVocabularyObjectDetectionAnnotator,
) -> OpenVocabularyObjectDetectionAnnotator:
    """
    Configure a patterned mask and an object box offset from the image origin.
    """
    annotator = numpy_image_annotator
    image = annotator.get_cas().get(CASViews.COLOR_IMAGE)
    height, width = image.shape[:2]
    annotator.descriptor.parameters.precision_mode = True
    annotator.segment_anything_model = MagicMock()
    mask = numpy.arange(height * width, dtype=numpy.uint8).reshape(height, width) % 2
    annotator.segment_anything_model.predict.return_value[
        0
    ].masks.data.cpu.return_value.numpy.return_value = mask[None]
    annotator.processor.post_process_grounded_object_detection.return_value = [
        {
            DetectionResultField.BOXES: torch.Tensor([[1, 1, width - 1, height - 1]]),
            DetectionResultField.SCORES: torch.Tensor([1]),
            DetectionResultField.LABELS: torch.tensor([0]),
        }
    ]
    return annotator


@pytest.mark.parametrize("color_to_depth_ratio", [None, (1, 1), (0.5, 0.5)])
def test_precision_mask_preserves_color_coordinates(
    precision_mask_annotator: OpenVocabularyObjectDetectionAnnotator,
    color_to_depth_ratio: tuple[float, float] | None,
) -> None:
    """
    Crop Segment Anything's color-resolution mask independently of the color-to-depth
    ratio.
    """
    annotator = precision_mask_annotator
    annotator.descriptor.parameters.global_with_depth = color_to_depth_ratio is not None
    annotator.get_cas().set(CASViews.COLOR2DEPTH_RATIO, color_to_depth_ratio)
    mask = annotator.segment_anything_model.predict.return_value[
        0
    ].masks.data.cpu.return_value.numpy.return_value[0]
    expected_mask = mask * numpy.iinfo(numpy.uint8).max

    assert annotator.compute() == Status.SUCCESS

    object_hypothesis = annotator.get_cas().filter_annotations_by_type(
        ObjectHypothesis
    )[0]
    numpy.testing.assert_array_equal(
        object_hypothesis.roi.mask,
        crop_image_roi(expected_mask, object_hypothesis.roi),
    )


def test_precision_mask_does_not_require_color_to_depth_ratio(
    precision_mask_annotator: OpenVocabularyObjectDetectionAnnotator,
) -> None:
    """
    Process color-resolution Segment Anything masks without depth scaling information.
    """
    annotator = precision_mask_annotator
    annotator.descriptor.parameters.global_with_depth = True
    annotator.get_cas().set(CASViews.COLOR2DEPTH_RATIO, None)

    assert annotator.compute() == Status.SUCCESS


def test_precision_bounding_box_uses_segment_anything_mask_color_coordinates(
    precision_mask_annotator: OpenVocabularyObjectDetectionAnnotator,
) -> None:
    """
    Keep mask-derived bounding boxes in color-image coordinates with depth enabled.
    """
    annotator = precision_mask_annotator
    annotator.descriptor.parameters.refine_bounding_boxes = True
    annotator.descriptor.parameters.global_with_depth = True
    annotator.get_cas().set(CASViews.COLOR2DEPTH_RATIO, (0.5, 0.5))
    mask = (
        annotator.segment_anything_model.predict.return_value[
            0
        ].masks.data.cpu.return_value.numpy.return_value[0]
        * numpy.iinfo(numpy.uint8).max
    )
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    expected_rectangle = cv2.boundingRect(contours[0])

    assert annotator.compute() == Status.SUCCESS

    object_hypothesis = annotator.get_cas().filter_annotations_by_type(
        ObjectHypothesis
    )[0]
    assert object_hypothesis.roi.roi.as_tuple() == expected_rectangle


# %% Detection and mask operations


def test_detect_objects_creates_classified_hypotheses_without_publishing(
    numpy_image_annotator: OpenVocabularyObjectDetectionAnnotator,
) -> None:
    """
    Return classified detections while leaving publication to compute.
    """
    annotator = numpy_image_annotator
    image = annotator.get_cas().get(CASViews.COLOR_IMAGE)
    height, width = image.shape[:2]
    result = {
        DetectionResultField.BOXES: torch.Tensor([[1, 1, width - 1, height - 1]]),
        DetectionResultField.SCORES: torch.Tensor([0.875]),
        DetectionResultField.LABELS: torch.tensor([1]),
    }
    annotator.processor.post_process_grounded_object_detection.return_value = [result]

    hypotheses = annotator.detect_objects(image)

    assert len(hypotheses) == 1
    classification = hypotheses[0].annotations[0]
    assert isinstance(classification, Classification)
    assert (
        classification.classname
        == annotator.classes[result[DetectionResultField.LABELS][0]]
    )
    assert classification.confidence == result[DetectionResultField.SCORES][0].item()
    assert classification.source == annotator.get_class_name()
    assert hypotheses[0].roi.roi.get_corner_points() == tuple(
        result[DetectionResultField.BOXES][0].tolist()
    )
    assert annotator.get_cas().annotations == []
    annotator.get_annotator_output_struct().set_image.assert_not_called()


def test_predict_mask_returns_full_color_resolution_byte_mask(
    precision_mask_annotator: OpenVocabularyObjectDetectionAnnotator,
) -> None:
    """
    Return Segment Anything foreground values scaled to bytes without cropping or depth
    resizing.
    """
    annotator = precision_mask_annotator
    image = annotator.get_cas().get(CASViews.COLOR_IMAGE)
    box = annotator.processor.post_process_grounded_object_detection.return_value[0][
        DetectionResultField.BOXES
    ][0].tolist()
    source_mask = annotator.segment_anything_model.predict.return_value[
        0
    ].masks.data.cpu.return_value.numpy.return_value[0]
    expected_mask = source_mask * numpy.iinfo(numpy.uint8).max

    mask = annotator.predict_mask(image, box)

    assert mask.dtype == numpy.uint8
    numpy.testing.assert_array_equal(mask, expected_mask)


@pytest.mark.parametrize(
    "mask_index,color",
    list(
        enumerate(
            (
                Color.BLUE(),
                Color.GREEN(),
                Color.RED(),
                Color.MAGENTA(),
                Color.CYAN(),
                Color.BLUE(),
            )
        )
    ),
)
def test_visualize_objects_draws_roi_mask_without_mutating_source(
    precision_mask_annotator: OpenVocabularyObjectDetectionAnnotator,
    mask_index: int,
    color: Color,
) -> None:
    """
    Render foreground pixels in the object ROI on a separate image.
    """
    annotator = precision_mask_annotator
    image = annotator.get_cas().get(CASViews.COLOR_IMAGE)
    original_image = image.copy()
    hypotheses = annotator.detect_objects(image)
    object_hypothesis = hypotheses[0]

    visualization = annotator.visualize_objects(image, hypotheses * (mask_index + 1))

    foreground = object_hypothesis.roi.mask == numpy.iinfo(numpy.uint8).max
    image_region = crop_image_roi(visualization, object_hypothesis.roi)
    numpy.testing.assert_array_equal(
        image_region[foreground] / numpy.iinfo(numpy.uint8).max,
        numpy.broadcast_to(color.to_rgb()[::-1], image_region[foreground].shape),
    )
    numpy.testing.assert_array_equal(image, original_image)
