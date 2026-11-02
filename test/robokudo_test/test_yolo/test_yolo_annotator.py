"""
Tests for detecting objects with YOLO and annotating them as hypotheses.

The YOLO and SAM models are replaced by mocks reporting predefined results, so no
weights are downloaded.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pytest
from py_trees.common import Status
from ultralytics.utils.plotting import colors

import robokudo
import robokudo.annotators.yolo_annotator
from robokudo.annotators.yolo_annotator import YoloAnnotator
from robokudo.exceptions import DetectionModelNotConfigured
from robokudo.types.annotation import Classification
from robokudo.types.scene import HumanHypothesis, ObjectHypothesis
from robokudo.utils.annotator_helper import draw_bounding_boxes_from_object_hypotheses
from robokudo.utils.file_loader import FileLoader
from ._black_wall_scene import BlackWallScene
from ._yolo_mock import (
    NormalizedBox,
    MockDetection,
    MockDetectionModel,
    MockModelLoader,
)

# %% scene setup


@dataclass
class CupScene(BlackWallScene):
    """
    The black wall scene, in which the mock model detects a cup.
    """

    class_names: dict[int, str] = field(
        default_factory=lambda: {
            0: YoloAnnotator.Descriptor.Parameters().person_class_name,
            41: "cup",
            46: "banana",
        }
    )
    """
    The classes the mock model detects, by class id.
    """

    cup_class_id: int = 41
    """
    Class id of the cup the mock model detects.
    """

    cup_box: NormalizedBox = field(
        default_factory=lambda: NormalizedBox(
            left=0.25, top=0.25, right=0.75, bottom=0.75
        )
    )
    """
    Where the cup is detected in every image.
    """

    pretrained_model: str = "yolov8n.pt"
    """
    The pretrained model the annotators under test load.
    """

    def cup_corners(self) -> list[int]:
        """
        :return: The corners (x1, y1, x2, y2) of the cup in color image pixels.
        """
        return [int(corner) for corner in self.cup_box.corners_in(self.color_image())]

    def descriptor(self) -> YoloAnnotator.Descriptor:
        """
        :return: A descriptor loading the pretrained model and ignoring depth.
        """
        descriptor = YoloAnnotator.Descriptor()
        descriptor.parameters.models_directory = self.models_directory
        descriptor.parameters.pretrained_model = self.pretrained_model
        descriptor.parameters.threshold = 0.5
        descriptor.parameters.global_with_depth = False
        return descriptor

    def annotator_in_pipeline(
        self, descriptor: YoloAnnotator.Descriptor, color_to_depth_ratio: float = 1.0
    ) -> YoloAnnotator:
        """
        :param color_to_depth_ratio: Size of the depth image relative to the color
            image.
        :return: An annotator whose CAS holds a color image and a matching depth image.
        """
        annotator = YoloAnnotator(descriptor=descriptor)
        self.place_in_pipeline(annotator, color_to_depth_ratio)
        return annotator

    def cup_shaped_hypothesis(self) -> ObjectHypothesis:
        """
        :return: A hypothesis covering exactly the region the cup is detected in.
        """
        x1, y1, x2, y2 = self.cup_corners()
        return _hypothesis_at(x=x1, y=y1, width=x2 - x1, height=y2 - y1)


@pytest.fixture
def scene(tmp_path: Path) -> CupScene:
    """
    The camera view the annotators under test detect a cup in.
    """
    return CupScene(models_directory=tmp_path / "models")


@pytest.fixture
def model_loader(monkeypatch, scene: CupScene) -> MockModelLoader:
    """
    Make the annotator load mock models detecting the scene's cup instead of real ones.
    """
    model_loader = MockModelLoader(
        model=MockDetectionModel(
            names=scene.class_names,
            detections=[
                MockDetection(
                    class_id=scene.cup_class_id, confidence=0.8, box=scene.cup_box
                )
            ],
        )
    )
    monkeypatch.setattr(
        robokudo.annotators.yolo_annotator, "YOLO", model_loader.load_model
    )
    monkeypatch.setattr(
        robokudo.annotators.yolo_annotator, "SAM", model_loader.load_segmentation_model
    )
    return model_loader


def _depth_descriptor(scene: CupScene) -> YoloAnnotator.Descriptor:
    descriptor = scene.descriptor()
    descriptor.parameters.global_with_depth = True
    return descriptor


def _hypotheses(annotator: YoloAnnotator) -> list[ObjectHypothesis]:
    return annotator.get_cas().filter_annotations_by_type(ObjectHypothesis)


def _only_hypothesis(annotator: YoloAnnotator) -> ObjectHypothesis:
    (hypothesis,) = _hypotheses(annotator)
    return hypothesis


def _classification(hypothesis: ObjectHypothesis) -> Classification:
    (classification,) = [
        annotation
        for annotation in hypothesis.annotations
        if isinstance(annotation, Classification)
    ]
    return classification


def _hypothesis_at(x: int, y: int, width: int, height: int) -> ObjectHypothesis:
    hypothesis = ObjectHypothesis()
    hypothesis.roi.roi.pos.x = x
    hypothesis.roi.roi.pos.y = y
    hypothesis.roi.roi.width = width
    hypothesis.roi.roi.height = height
    return hypothesis


# %% model loading


def test_models_directory_defaults_to_models_folder_of_robokudo_package():
    parameters = YoloAnnotator.Descriptor.Parameters()

    assert parameters.models_directory == Path(robokudo.__file__).parent / "models"


def test_pretrained_model_is_loaded_from_models_directory(model_loader, scene):
    descriptor = scene.descriptor()

    YoloAnnotator(descriptor=descriptor).setup()

    assert model_loader.loaded_model_locations == [
        str(scene.models_directory / descriptor.parameters.pretrained_model)
    ]


def test_segmentation_model_is_loaded_from_models_directory(model_loader, scene):
    descriptor = scene.descriptor()
    descriptor.parameters.precision_mode = True

    YoloAnnotator(descriptor=descriptor).setup()

    assert model_loader.loaded_segmentation_model_locations == [
        str(scene.models_directory / descriptor.parameters.segmentation_model)
    ]


def test_setup_creates_missing_models_directory(model_loader, scene):
    YoloAnnotator(descriptor=scene.descriptor()).setup()

    assert scene.models_directory.is_dir()


def test_annotator_without_model_raises(model_loader, scene):
    descriptor = scene.descriptor()
    descriptor.parameters.pretrained_model = None

    with pytest.raises(DetectionModelNotConfigured):
        YoloAnnotator(descriptor=descriptor)


def test_custom_weights_and_class_names_are_read_from_ros_package(
    model_loader, scene, monkeypatch, tmp_path
):
    monkeypatch.setattr(
        FileLoader,
        "get_path_to_file_in_ros_package",
        staticmethod(
            lambda ros_pkg_name, relative_path: tmp_path / ros_pkg_name / relative_path
        ),
    )
    descriptor = scene.descriptor()
    parameters = descriptor.parameters
    parameters.pretrained_model = None
    parameters.ros_package_name = "custom_models"
    parameters.weights_path = "models/best.pt"
    class_names_file = (
        tmp_path / parameters.ros_package_name / parameters.class_names_path
    )
    class_names_file.parent.mkdir(parents=True)
    class_names_file.write_text(json.dumps({"3": "mug"}), encoding="utf-8")

    annotator = YoloAnnotator(descriptor=descriptor)
    annotator.setup()

    assert model_loader.loaded_model_locations == [
        str(tmp_path / parameters.ros_package_name / parameters.weights_path)
    ]
    assert annotator.class_names == {3: "mug"}


# %% detection


def test_detection_becomes_hypothesis_classified_as_detected_class(model_loader, scene):
    annotator = scene.annotator_in_pipeline(scene.descriptor())

    status = annotator.compute()

    assert status == Status.SUCCESS
    hypothesis = _only_hypothesis(annotator)
    assert type(hypothesis) is ObjectHypothesis
    classification = _classification(hypothesis)
    assert classification.classname == scene.class_names[scene.cup_class_id]
    assert classification.class_id == scene.cup_class_id
    assert classification.confidence == pytest.approx(
        model_loader.model.detections[0].confidence
    )
    assert classification.source == annotator.name


def test_hypothesis_region_is_detection_bounding_box(model_loader, scene):
    annotator = scene.annotator_in_pipeline(scene.descriptor())

    annotator.compute()

    assert _only_hypothesis(annotator).roi.roi.get_corner_points() == tuple(
        scene.cup_corners()
    )


def test_person_detection_becomes_human_hypothesis(model_loader, scene):
    descriptor = scene.descriptor()
    person_class_id = next(
        class_id
        for class_id, name in scene.class_names.items()
        if name == descriptor.parameters.person_class_name
    )
    model_loader.model.detections[0].class_id = person_class_id
    annotator = scene.annotator_in_pipeline(descriptor)

    annotator.compute()

    assert type(_only_hypothesis(annotator)) is HumanHypothesis


def test_detection_of_configured_person_class_becomes_human_hypothesis(
    model_loader, scene
):
    descriptor = scene.descriptor()
    descriptor.parameters.person_class_name = scene.class_names[scene.cup_class_id]
    annotator = scene.annotator_in_pipeline(descriptor)

    annotator.compute()

    assert type(_only_hypothesis(annotator)) is HumanHypothesis


def test_detection_of_ignored_class_is_not_annotated(model_loader, scene):
    descriptor = scene.descriptor()
    descriptor.parameters.ignored_class_names = [
        f" {scene.class_names[scene.cup_class_id]} "
    ]
    annotator = scene.annotator_in_pipeline(descriptor)

    annotator.compute()

    assert _hypotheses(annotator) == []


def test_model_predicts_with_configured_threshold(model_loader, scene):
    descriptor = scene.descriptor()
    descriptor.parameters.threshold = 0.42
    annotator = scene.annotator_in_pipeline(descriptor)

    annotator.compute()

    assert model_loader.model.received_confidence == descriptor.parameters.threshold


# %% masks


def test_mask_fills_bounding_box_without_precision_mode(model_loader, scene):
    annotator = scene.annotator_in_pipeline(scene.descriptor())

    annotator.compute()

    x1, y1, x2, y2 = scene.cup_corners()
    np.testing.assert_array_equal(
        _only_hypothesis(annotator).roi.mask,
        np.full((y2 - y1, x2 - x1), YoloAnnotator.mask_foreground, dtype=np.uint8),
    )


def _precision_descriptor(scene: CupScene) -> YoloAnnotator.Descriptor:
    descriptor = scene.descriptor()
    descriptor.parameters.precision_mode = True
    return descriptor


def _segmented_cup_mask(model_loader: MockModelLoader, scene: CupScene) -> np.ndarray:
    """
    :return: The segmentation of the cup, cropped to its bounding box.
    """
    image = scene.color_image()
    segmentation = model_loader.segmentation_model.segment(
        image, scene.cup_box.corners_in(image)
    )
    x1, y1, x2, y2 = scene.cup_corners()
    return (segmentation[y1:y2, x1:x2] * YoloAnnotator.mask_foreground).astype(np.uint8)


def test_precision_mode_mask_is_segmentation_of_detection(model_loader, scene):
    annotator = scene.annotator_in_pipeline(_precision_descriptor(scene))

    annotator.compute()

    np.testing.assert_array_equal(
        _only_hypothesis(annotator).roi.mask, _segmented_cup_mask(model_loader, scene)
    )


def test_precision_mode_erosion_shrinks_segmentation(model_loader, scene):
    descriptor = _precision_descriptor(scene)
    descriptor.parameters.precision_mode_apply_erosion = True
    descriptor.parameters.precision_mode_erosion_kernel_size = 3
    annotator = scene.annotator_in_pipeline(descriptor)

    annotator.compute()

    eroded = _only_hypothesis(annotator).roi.mask == YoloAnnotator.mask_foreground
    segmented = (
        _segmented_cup_mask(model_loader, scene) == YoloAnnotator.mask_foreground
    )
    assert np.all(segmented[eroded])
    assert eroded.sum() < segmented.sum()


# %% visualization


def test_visualization_draws_labelled_bounding_box_of_each_detection(
    model_loader, scene
):
    annotator = scene.annotator_in_pipeline(scene.descriptor())

    annotator.compute()

    hypothesis = _only_hypothesis(annotator)
    classification = _classification(hypothesis)
    expected_image = scene.color_image()
    draw_bounding_boxes_from_object_hypotheses(
        expected_image,
        [hypothesis],
        lambda _: f"{classification.classname}, {classification.confidence:.2f}",
    )
    np.testing.assert_array_equal(
        annotator.get_annotator_output_struct().image, expected_image
    )


def test_precision_mode_visualization_paints_mask_in_palette_color(model_loader, scene):
    annotator = scene.annotator_in_pipeline(_precision_descriptor(scene))

    annotator.compute()

    x1, y1, x2, y2 = scene.cup_corners()
    segmented_rows = slice(y1 + 2, (y1 + y2) // 2)
    columns_inside_box = slice(x1 + 2, x2 - 2)
    painted = annotator.get_annotator_output_struct().image[
        segmented_rows, columns_inside_box
    ]
    assert np.all(painted == colors(0, bgr=True))


# %% depth


def test_point_cloud_holds_depth_measured_inside_bounding_box(model_loader, scene):
    annotator = scene.annotator_in_pipeline(_depth_descriptor(scene))

    annotator.compute()

    x1, y1, x2, y2 = scene.cup_corners()
    points = np.asarray(_only_hypothesis(annotator).points.points)
    assert len(points) == (x2 - x1) * (y2 - y1)
    np.testing.assert_allclose(points[:, 2], scene.wall_distance)


def test_region_detected_on_smaller_depth_resolution_is_in_color_pixels(
    model_loader, scene
):
    annotator = scene.annotator_in_pipeline(
        _depth_descriptor(scene), color_to_depth_ratio=0.5
    )

    annotator.compute()

    hypothesis = _only_hypothesis(annotator)
    x1, y1, x2, y2 = scene.cup_corners()
    assert hypothesis.roi.roi.get_corner_points() == (x1, y1, x2, y2)
    assert hypothesis.roi.mask.shape == (y2 - y1, x2 - x1)


# %% existing hypotheses


def _existing_hypothesis_descriptor(scene: CupScene) -> YoloAnnotator.Descriptor:
    descriptor = scene.descriptor()
    descriptor.parameters.classify_existing_hypotheses = True
    return descriptor


def _annotator_with_existing_hypothesis(
    scene: CupScene, descriptor: YoloAnnotator.Descriptor, existing: ObjectHypothesis
) -> YoloAnnotator:
    annotator = scene.annotator_in_pipeline(descriptor)
    annotator.get_cas().annotations.append(existing)
    return annotator


def test_detection_overlapping_existing_hypothesis_classifies_it(model_loader, scene):
    existing = scene.cup_shaped_hypothesis()
    annotator = _annotator_with_existing_hypothesis(
        scene, _existing_hypothesis_descriptor(scene), existing
    )

    annotator.compute()

    assert _hypotheses(annotator) == [existing]
    assert _classification(existing).classname == scene.class_names[scene.cup_class_id]


def test_detection_apart_from_existing_hypothesis_creates_new_one(model_loader, scene):
    existing = _hypothesis_at(x=0, y=0, width=5, height=5)
    annotator = _annotator_with_existing_hypothesis(
        scene, _existing_hypothesis_descriptor(scene), existing
    )

    annotator.compute()

    assert len(_hypotheses(annotator)) == 2
    assert existing.annotations == []


def test_existing_hypotheses_are_left_alone_unless_enabled(model_loader, scene):
    existing = scene.cup_shaped_hypothesis()
    annotator = _annotator_with_existing_hypothesis(scene, scene.descriptor(), existing)

    annotator.compute()

    assert len(_hypotheses(annotator)) == 2
    assert existing.annotations == []
