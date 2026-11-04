"""
Tests for detecting humans and their body keypoints with a YOLO pose model.

The pose model is replaced by a mock reporting predefined detections, so no weights are
downloaded.
"""

from __future__ import annotations

from dataclasses import astuple, dataclass, field
from pathlib import Path

import cv2
import numpy as np
import pytest
from py_trees.common import Status

import robokudo
import robokudo.annotators.human_and_pose
from robokudo.annotators.exceptions import NotAPoseModelException
from robokudo.annotators.human_and_pose import HumanAndPoseAnnotator, PoseKeypoint
from robokudo.cas import CASViews
from robokudo.types.annotation import Classification, CloudAnnotation
from robokudo.types.core import Annotation
from robokudo.types.cv import Rect
from robokudo.types.human import KeypointAnnotation, KeypointKind
from robokudo.types.scene import HumanHypothesis
from ._black_wall_scene import BlackWallScene
from ._yolo_mock import (
    MockDetection,
    MockDetectionModel,
    MockModelLoader,
    NormalizedBox,
    NormalizedKeypoint,
)

# %% scene setup


@dataclass
class PersonScene(BlackWallScene):
    """
    The black wall scene, in which the mock pose model detects a person.
    """

    person_class_name: str = field(
        default_factory=lambda: HumanAndPoseAnnotator.Descriptor.Parameters().person_class_name
    )
    """
    Name of the class the mock pose model reports persons as.
    """

    person_class_id: int = 0
    """
    Class id of persons in the mock pose model.
    """

    person_box: NormalizedBox = field(
        default_factory=lambda: NormalizedBox(
            left=0.25, top=0.25, right=0.75, bottom=0.75
        )
    )
    """
    Where the person is detected in every image.
    """

    person_confidence: float = 0.9
    """
    Confidence of the person detection.
    """

    visible_keypoint_confidence: float = 0.95
    """
    Confidence of the keypoints the pose model sees clearly.
    """

    hidden_keypoint_confidence: float = 0.1
    """
    Confidence of the keypoints the pose model barely sees.
    """

    def person_keypoints(self) -> list[NormalizedKeypoint]:
        """
        :return: One keypoint per body part inside the person's box, with only the
            nose clearly visible.
        """
        return [
            NormalizedKeypoint(
                x=0.3 + 0.02 * keypoint,
                y=0.5,
                confidence=(
                    self.visible_keypoint_confidence
                    if keypoint == PoseKeypoint.NOSE
                    else self.hidden_keypoint_confidence
                ),
            )
            for keypoint in PoseKeypoint
        ]

    def person_detection(self, box: NormalizedBox) -> MockDetection:
        """
        :return: A detection of the person, with its keypoints, at the given box.
        """
        return MockDetection(
            class_id=self.person_class_id,
            confidence=self.person_confidence,
            box=box,
            keypoints=self.person_keypoints(),
        )

    def person_box_in(self, image: np.ndarray) -> Rect:
        """
        :return: The box of the person in pixels of the given image.
        """
        return Rect.from_corner_points(*self.person_box.corners_in(image))

    def person_box_shrunk_by(self, fraction: float) -> Rect:
        """
        :param fraction: Fraction of its width and height the box is shrunk by around
            its center.
        :return: The shrunk box of the person in color image pixels.
        """
        box = self.person_box_in(self.color_image())
        horizontal_margin = box.width * fraction / 2
        vertical_margin = box.height * fraction / 2
        return Rect.from_corner_points(
            x1=box.pos.x + horizontal_margin,
            y1=box.pos.y + vertical_margin,
            x2=box.pos.x + box.width - horizontal_margin,
            y2=box.pos.y + box.height - vertical_margin,
        )

    def descriptor(self) -> HumanAndPoseAnnotator.Descriptor:
        """
        :return: A descriptor loading the model from the scene's models directory,
            ignoring depth and skipping the warmup.
        """
        descriptor = HumanAndPoseAnnotator.Descriptor()
        descriptor.parameters.models_directory = self.models_directory
        descriptor.parameters.warmup_annotator = False
        descriptor.parameters.global_with_depth = False
        return descriptor

    def depth_descriptor(self) -> HumanAndPoseAnnotator.Descriptor:
        """
        :return: A descriptor like :meth:`descriptor` that builds point clouds.
        """
        descriptor = self.descriptor()
        descriptor.parameters.global_with_depth = True
        return descriptor

    def annotator_in_pipeline(
        self,
        descriptor: HumanAndPoseAnnotator.Descriptor,
        color_to_depth_ratio: float = 1.0,
    ) -> HumanAndPoseAnnotator:
        """
        :param color_to_depth_ratio: Size of the depth image relative to the color
            image.
        :return: An annotator whose CAS holds a color image and a matching depth image.
        """
        annotator = HumanAndPoseAnnotator(descriptor=descriptor)
        self.place_in_pipeline(annotator, color_to_depth_ratio)
        return annotator


@pytest.fixture
def scene(tmp_path: Path) -> PersonScene:
    """
    The camera view the annotator under test detects a person in.
    """
    return PersonScene(models_directory=tmp_path / "models")


@pytest.fixture
def model_loader(
    monkeypatch: pytest.MonkeyPatch, scene: PersonScene
) -> MockModelLoader:
    """
    Make the annotator load a mock pose model detecting the scene's person.
    """
    model_loader = MockModelLoader(
        model=MockDetectionModel(
            names={scene.person_class_id: scene.person_class_name},
            detections=[scene.person_detection(scene.person_box)],
        )
    )
    monkeypatch.setattr(
        robokudo.annotators.human_and_pose, "YOLO", model_loader.load_model
    )
    return model_loader


def _only_hypothesis(annotator: HumanAndPoseAnnotator) -> HumanHypothesis:
    (hypothesis,) = annotator.get_cas().filter_annotations_by_type(HumanHypothesis)
    return hypothesis


def _annotations_of_type(
    hypothesis: HumanHypothesis, annotation_type: type[Annotation]
) -> list[Annotation]:
    """
    :return: The annotations of the hypothesis that are instances of the given type.
    """
    return [
        annotation
        for annotation in hypothesis.annotations
        if isinstance(annotation, annotation_type)
    ]


def _keypoints_of_kind(
    hypothesis: HumanHypothesis, kind: KeypointKind
) -> KeypointAnnotation:
    """
    :return: The only keypoint annotation of the hypothesis of the given kind.
    """
    (keypoints,) = [
        annotation
        for annotation in _annotations_of_type(hypothesis, KeypointAnnotation)
        if annotation.kind == kind
    ]
    return keypoints


def _area(box: Rect) -> int:
    """
    :return: The number of pixels inside the box.
    """
    return box.width * box.height


# %% model loading


def test_models_directory_defaults_to_models_folder_of_robokudo_package():
    parameters = HumanAndPoseAnnotator.Descriptor.Parameters()

    assert parameters.models_directory == Path(robokudo.__file__).parent / "models"


def test_pretrained_model_is_loaded_from_models_directory(
    model_loader: MockModelLoader, scene: PersonScene
):
    descriptor = scene.descriptor()

    HumanAndPoseAnnotator(descriptor=descriptor).setup()

    assert model_loader.loaded_model_locations == [
        str(scene.models_directory / descriptor.parameters.pretrained_model)
    ]


def test_setup_creates_models_directory(
    model_loader: MockModelLoader, scene: PersonScene
):
    HumanAndPoseAnnotator(descriptor=scene.descriptor()).setup()

    assert scene.models_directory.is_dir()


def test_setup_warms_model_up_on_blank_image(
    model_loader: MockModelLoader, scene: PersonScene
):
    descriptor = scene.descriptor()
    descriptor.parameters.warmup_annotator = True

    HumanAndPoseAnnotator(descriptor=descriptor).setup()

    assert (
        model_loader.model.received_image_shape
        == descriptor.parameters.warmup_image_shape
    )


def test_construction_loads_no_model(model_loader: MockModelLoader, scene: PersonScene):
    HumanAndPoseAnnotator(descriptor=scene.descriptor())

    assert model_loader.loaded_model_locations == []


# %% detection


def test_person_becomes_human_hypothesis_classified_as_person(
    model_loader: MockModelLoader, scene: PersonScene
):
    annotator = scene.annotator_in_pipeline(scene.descriptor())

    status = annotator.compute()

    assert status == Status.SUCCESS
    hypothesis = _only_hypothesis(annotator)
    (classification,) = _annotations_of_type(hypothesis, Classification)
    assert classification.classname == scene.person_class_name
    assert classification.confidence == pytest.approx(scene.person_confidence)
    assert classification.source == annotator.name


def test_hypothesis_region_is_person_bounding_box(
    model_loader: MockModelLoader, scene: PersonScene
):
    annotator = scene.annotator_in_pipeline(scene.descriptor())

    annotator.compute()

    assert _only_hypothesis(annotator).roi.roi == (
        scene.person_box_in(scene.color_image())
    )


def test_each_person_is_identified_by_its_detection_index(
    model_loader: MockModelLoader, scene: PersonScene
):
    second_person_box = NormalizedBox(left=0.0, top=0.0, right=0.2, bottom=0.5)
    model_loader.model.detections.append(scene.person_detection(second_person_box))
    annotator = scene.annotator_in_pipeline(scene.descriptor())

    annotator.compute()

    hypotheses = annotator.get_cas().filter_annotations_by_type(HumanHypothesis)
    assert [hypothesis.id for hypothesis in hypotheses] == [0, 1]


def test_hypothesis_carries_two_dimensional_keypoints_in_pixels(
    model_loader: MockModelLoader, scene: PersonScene
):
    annotator = scene.annotator_in_pipeline(scene.descriptor())

    annotator.compute()

    keypoints = _keypoints_of_kind(
        _only_hypothesis(annotator), KeypointKind.IMAGE_POSITION
    )
    np.testing.assert_allclose(
        [astuple(keypoint) for keypoint in keypoints.keypoints],
        [
            astuple(keypoint.in_pixels_of(scene.color_image()))
            for keypoint in scene.person_keypoints()
        ],
        rtol=1e-5,
    )


def test_model_predicts_with_configured_threshold(
    model_loader: MockModelLoader, scene: PersonScene
):
    descriptor = scene.descriptor()
    descriptor.parameters.confidence_threshold = 0.42
    annotator = scene.annotator_in_pipeline(descriptor)

    annotator.compute()

    assert (
        model_loader.model.received_confidence
        == descriptor.parameters.confidence_threshold
    )


def test_model_merges_detections_overlapping_more_than_configured(
    model_loader: MockModelLoader, scene: PersonScene
):
    descriptor = scene.descriptor()
    descriptor.parameters.non_maximum_suppression_overlap = 0.3
    annotator = scene.annotator_in_pipeline(descriptor)

    annotator.compute()

    assert (
        model_loader.model.received_prediction_settings["iou"]
        == descriptor.parameters.non_maximum_suppression_overlap
    )


def test_model_without_keypoints_raises(
    model_loader: MockModelLoader, scene: PersonScene
):
    model_loader.model.detections = [
        MockDetection(
            class_id=scene.person_class_id,
            confidence=scene.person_confidence,
            box=scene.person_box,
        )
    ]
    annotator = scene.annotator_in_pipeline(scene.descriptor())

    with pytest.raises(NotAPoseModelException):
        annotator.compute()


def test_shrinking_narrows_hypothesis_region_around_its_center(
    model_loader: MockModelLoader, scene: PersonScene
):
    descriptor = scene.descriptor()
    descriptor.parameters.shrink_bounding_box = 0.5
    annotator = scene.annotator_in_pipeline(descriptor)

    annotator.compute()

    assert _only_hypothesis(annotator).roi.roi == (
        scene.person_box_shrunk_by(descriptor.parameters.shrink_bounding_box)
    )


def test_shrink_as_annotation_without_depth_shrinks_region(
    model_loader: MockModelLoader, scene: PersonScene
):
    descriptor = scene.descriptor()
    descriptor.parameters.shrink_bounding_box = 0.5
    descriptor.parameters.shrink_as_annotation = True
    annotator = scene.annotator_in_pipeline(descriptor)

    annotator.compute()

    assert _only_hypothesis(annotator).roi.roi == (
        scene.person_box_shrunk_by(descriptor.parameters.shrink_bounding_box)
    )


# %% depth


def test_point_cloud_holds_depth_measured_inside_bounding_box(
    model_loader: MockModelLoader, scene: PersonScene
):
    annotator = scene.annotator_in_pipeline(scene.depth_descriptor())

    annotator.compute()

    points = np.asarray(_only_hypothesis(annotator).points.points)
    assert len(points) == _area(scene.person_box_in(scene.color_image()))
    np.testing.assert_allclose(points[:, 2], scene.wall_distance)


def test_point_cloud_is_built_on_smaller_depth_resolution(
    model_loader: MockModelLoader, scene: PersonScene
):
    annotator = scene.annotator_in_pipeline(
        scene.depth_descriptor(), color_to_depth_ratio=0.5
    )

    annotator.compute()

    depth_image = annotator.get_cas().get(CASViews.DEPTH_IMAGE)
    points = np.asarray(_only_hypothesis(annotator).points.points)
    assert len(points) == _area(scene.person_box_in(depth_image))


def test_shrunk_cloud_is_added_as_annotation_when_requested(
    model_loader: MockModelLoader, scene: PersonScene
):
    descriptor = scene.depth_descriptor()
    descriptor.parameters.shrink_bounding_box = 0.5
    descriptor.parameters.shrink_as_annotation = True
    annotator = scene.annotator_in_pipeline(descriptor)

    annotator.compute()

    hypothesis = _only_hypothesis(annotator)
    (shrunk_cloud,) = _annotations_of_type(hypothesis, CloudAnnotation)
    assert hypothesis.roi.roi == (scene.person_box_in(scene.color_image()))
    assert len(shrunk_cloud.points.points) == _area(
        scene.person_box_shrunk_by(descriptor.parameters.shrink_bounding_box)
    )


def test_shrink_as_annotation_without_shrinking_keeps_box_and_adds_no_cloud(
    model_loader: MockModelLoader, scene: PersonScene
):
    descriptor = scene.depth_descriptor()
    descriptor.parameters.shrink_as_annotation = True
    annotator = scene.annotator_in_pipeline(descriptor)

    annotator.compute()

    hypothesis = _only_hypothesis(annotator)
    assert hypothesis.roi.roi == (scene.person_box_in(scene.color_image()))
    assert _annotations_of_type(hypothesis, CloudAnnotation) == []


def test_keypoint_clouds_cover_disc_around_confident_keypoints_only(
    model_loader: MockModelLoader, scene: PersonScene
):
    descriptor = scene.depth_descriptor()
    descriptor.parameters.generate_clouds_for_keypoints = True
    annotator = scene.annotator_in_pipeline(descriptor)

    annotator.compute()

    keypoints = _keypoints_of_kind(
        _only_hypothesis(annotator), KeypointKind.POINT_CLOUD
    )
    point_counts = {
        keypoint: len(keypoint_cloud.points.points)
        for keypoint, keypoint_cloud in zip(PoseKeypoint, keypoints.keypoints)
    }
    nose = scene.person_keypoints()[PoseKeypoint.NOSE].in_pixels_of(scene.color_image())
    nose_disc = cv2.circle(
        np.zeros((scene.resolution.height, scene.resolution.width), dtype=np.uint8),
        (int(nose.x), int(nose.y)),
        descriptor.parameters.keypoint_masking_radius,
        1,
        thickness=cv2.FILLED,
    )
    assert point_counts == {
        keypoint: (np.count_nonzero(nose_disc) if keypoint == PoseKeypoint.NOSE else 0)
        for keypoint in PoseKeypoint
    }


def test_keypoint_clouds_are_built_around_the_two_dimensional_keypoints(
    model_loader: MockModelLoader, scene: PersonScene
):
    descriptor = scene.depth_descriptor()
    descriptor.parameters.generate_clouds_for_keypoints = True
    annotator = scene.annotator_in_pipeline(descriptor)

    annotator.compute()

    hypothesis = _only_hypothesis(annotator)
    keypoint_clouds = _keypoints_of_kind(hypothesis, KeypointKind.POINT_CLOUD)
    keypoints = _keypoints_of_kind(hypothesis, KeypointKind.IMAGE_POSITION)
    assert [
        keypoint_cloud.keypoint for keypoint_cloud in keypoint_clouds.keypoints
    ] == keypoints.keypoints


# %% visualization


def test_visualization_shows_detections_drawn_by_model(
    model_loader: MockModelLoader, scene: PersonScene
):
    descriptor = scene.descriptor()
    annotator = scene.annotator_in_pipeline(descriptor)

    annotator.compute()

    (expected_result,) = model_loader.model.predict(
        scene.color_image(), conf=descriptor.parameters.confidence_threshold
    )
    np.testing.assert_array_equal(
        annotator.get_annotator_output_struct().image, expected_result.plot()
    )


def test_visualization_shows_point_clouds_of_hypotheses(
    model_loader: MockModelLoader, scene: PersonScene
):
    annotator = scene.annotator_in_pipeline(scene.depth_descriptor())

    annotator.compute()

    assert len(annotator.get_annotator_output_struct().geometries.points) == _area(
        scene.person_box_in(scene.color_image())
    )


def test_visualization_without_depth_shows_no_point_clouds(
    model_loader: MockModelLoader, scene: PersonScene
):
    annotator = scene.annotator_in_pipeline(scene.descriptor())

    annotator.compute()

    assert annotator.get_annotator_output_struct().geometries == []
