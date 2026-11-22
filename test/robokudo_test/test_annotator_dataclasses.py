"""
Dataclass construction preserves annotator configuration and tree-node identity.
"""

from __future__ import annotations

# %% Imports
from dataclasses import fields, replace

import pytest
from py_trees.common import Status
from py_trees.composites import Sequence
from robokudo.annotators.blur import BlurAnnotator
from robokudo.annotators.cas_check import CASCheckAnnotationTypeExists
from robokudo.annotators.cluster_position import ClusterPositionAnnotator
from robokudo.annotators.core import BaseAnnotator, Worker
from robokudo.annotators.human_and_pose import HumanAndPoseAnnotator
from robokudo.annotators.image_preprocessor import ImagePreprocessorAnnotator
from robokudo.annotators.plane import PlaneAnnotator
from robokudo.annotators.pointcloud_check import PointcloudCheckAnnotator
from robokudo.annotators.pointcloud_crop import PointcloudCropAnnotator
from robokudo.annotators.roi_adjuster import ROIAdjusterAnnotator
from robokudo.annotators.shape_estimator import ShapeEstimatorAnnotator
from robokudo.annotators.static_camera_transform import StaticCameraTransformAnnotator
from robokudo.annotators.world_visualizer import WorldVisualizer
from robokudo.annotators.yolo_annotator import YoloAnnotator
from robokudo.cas import CAS
from robokudo.types.annotation import Classification


# %% Dataclass construction
@pytest.mark.parametrize(
    "annotator_type",
    [
        BlurAnnotator,
        ClusterPositionAnnotator,
        HumanAndPoseAnnotator,
        ImagePreprocessorAnnotator,
        PlaneAnnotator,
        PointcloudCheckAnnotator,
        PointcloudCropAnnotator,
        ROIAdjusterAnnotator,
        ShapeEstimatorAnnotator,
        StaticCameraTransformAnnotator,
        WorldVisualizer,
        YoloAnnotator,
    ],
)
def test_replacing_annotator_configuration_initializes_a_new_tree_node(
    annotator_type: type[BaseAnnotator],
) -> None:
    """
    A dataclass copy retains its descriptor and obtains independent tree state.
    """
    descriptor = annotator_type.Descriptor()
    if annotator_type is YoloAnnotator:
        descriptor.parameters.pretrained_model = "yolov8n.pt"
    original = annotator_type(descriptor=descriptor)
    renamed = replace(original, name="RenamedAnnotator")

    assert renamed.name == "RenamedAnnotator"
    assert renamed.descriptor is original.descriptor
    assert renamed.status is Status.INVALID
    assert renamed.parent is None
    assert renamed.id != original.id
    assert renamed.rk_logger is original.rk_logger
    assert {"name", "descriptor"} <= {item.name for item in fields(renamed)}


def test_annotators_with_equal_configuration_remain_distinct_tree_nodes() -> None:
    """
    Removing one configured node preserves another with identical settings.
    """
    first = BlurAnnotator()
    second = BlurAnnotator(descriptor=first.descriptor)
    parent = Sequence(name="DataclassNodes", memory=True)
    parent.add_children([first, second])
    parent.remove_child(second)

    assert first != second
    assert len({first, second}) == 2
    assert parent.children == [first]
    assert first.parent is parent
    assert second.parent is None


def test_default_descriptors_and_runtime_collections_belong_to_each_annotator() -> None:
    """
    Constructing a second detector does not share mutable model metadata.
    """
    first_descriptor = YoloAnnotator.Descriptor()
    first_descriptor.parameters.pretrained_model = "yolov8n.pt"
    second_descriptor = YoloAnnotator.Descriptor()
    second_descriptor.parameters.pretrained_model = (
        first_descriptor.parameters.pretrained_model
    )
    first = YoloAnnotator(descriptor=first_descriptor)
    second = YoloAnnotator(descriptor=second_descriptor)
    first.class_names[0] = "FirstDetectorClass"
    first.descriptor.parameters.ignored_class_names.append("IgnoredByFirst")

    assert second.class_names == {}
    assert "IgnoredByFirst" not in second.descriptor.parameters.ignored_class_names


# %% Runtime state and specialized constructors
def test_replacing_threaded_annotator_does_not_reuse_worker() -> None:
    """
    A new tree node starts without the original node's pending computation.
    """
    original = PlaneAnnotator()
    original.compute_worker = Worker(original.compute)
    original.compute_worker_started = True

    replacement = replace(original, name="ReplacementPlane")

    assert replacement.compute_worker is None
    assert replacement.compute_worker_started is False
    assert replacement.compute_worker_thread is None


def test_replacing_annotation_checker_retains_condition() -> None:
    """
    A renamed checker still tests the annotation type supplied positionally.
    """
    original = CASCheckAnnotationTypeExists("ClassificationChecker", Classification)
    replacement = replace(original, name="ReplacementChecker")
    cas = CAS()

    assert replacement.function(cas) is False
    cas.annotations.append(Classification())
    assert replacement.function(cas) is True
    assert replacement.annotation_type is original.annotation_type
