"""
Serializable annotator configuration and construction-time global defaults.
"""

from __future__ import annotations

# %% Imports
import json
from dataclasses import is_dataclass

import pytest
import torch
from krrood.adapters.json_serializer import from_json, to_json
from krrood.utils import get_full_class_name, resolve_class_from_full_name
from robokudo.datastructures.annotator_parameters import AnnotatorPredefinedParameters
from robokudo.annotators.annotation_writer import (
    AnnotationPublisherWriterDescriptor,
    AnnotationStorageWriterDescriptor,
)
from robokudo.annotators.blur import BlurAnnotator, BlurDescriptor, BlurParameters
from robokudo.annotators.clip_annotator import CLIPDescriptor
from robokudo.annotators.cluster_color import ClusterColorDescriptor
from robokudo.annotators.cluster_color_histogram import ClusterColorHistogramDescriptor
from robokudo.annotators.cluster_pose_bounding_box import (
    ClusterPoseBoundingBoxDescriptor,
)
from robokudo.annotators.cluster_position import ClusterPositionDescriptor
from robokudo.annotators.collection_reader import (
    CollectionReaderAnnotator,
    CollectionReaderDescriptor,
    CollectionReaderParameters,
)
from robokudo.annotators.file_writer import FileWriterDescriptor
from robokudo.annotators.filter import FilterDescriptor
from robokudo.annotators.human_and_pose import HumanAndPoseDescriptor
from robokudo.annotators.icp_pose_refinement import ICPPoseRefinementDescriptor
from robokudo.annotators.image_cluster_extractor import ImageClusterExtractorDescriptor
from robokudo.annotators.image_preprocessor import ImagePreprocessorDescriptor
from robokudo.annotators.lambda_function import LambdaFunctionDescriptor
from robokudo.annotators.location import LocationDescriptor
from robokudo.annotators.object_hypothesis_visualizer import (
    ObjectHypothesisVisualizerDescriptor,
)
from robokudo.annotators.open_vocabulary_object_detection_annotator import (
    OpenVocabularyObjectDetectionDescriptor,
)
from robokudo.annotators.outlier_removal_objecthypothesis import (
    OutlierRemovalOnObjectHypothesisDescriptor,
)
from robokudo.annotators.plane import PlaneDescriptor
from robokudo.annotators.pointcloud_check import PointcloudCheckDescriptor
from robokudo.annotators.pointcloud_cluster_extractor import (
    NaivePointCloudClusterExtractorDescriptor,
    PointCloudClusterExtractorDescriptor,
)
from robokudo.annotators.pointcloud_crop import PointcloudCropDescriptor
from robokudo.annotators.query import GenerateQueryResultDescriptor
from robokudo.annotators.region_filter import RegionFilterDescriptor
from robokudo.annotators.roi_adjuster import ROIAdjusterDescriptor
from robokudo.annotators.semantic_world_connector import (
    SemanticDigitalTwinConnectorDescriptor,
)
from robokudo.annotators.shape_estimator import ShapeEstimatorDescriptor
from robokudo.annotators.static_camera_transform import StaticCameraTransformDescriptor
from robokudo.annotators.static_object_detector import StaticObjectDetectorDescriptor
from robokudo.annotators.storage import StorageWriterDescriptor
from robokudo.annotators.testing import ScopedDescriptor
from robokudo.annotators.tsdf_annotator import TSDFDescriptor
from robokudo.annotators.world_descriptor_bootstrap import (
    WorldDescriptorBootstrapDescriptor,
)
from robokudo.annotators.world_visualizer import WorldVisualizerDescriptor
from robokudo.annotators.yolo_annotator import (
    YoloAnnotator,
    YoloDescriptor,
    YoloParameters,
)
from robokudo.annotators.zero_shot_classification_annotator import (
    ZeroShotClassificationDescriptor,
)
from robokudo.datastructures.annotator_descriptor import (
    AnnotatorDescriptor,
    BaseAnnotatorDescriptor,
)
from robokudo.descriptors.camera_configs.base_camera_config import BaseCameraConfig
from robokudo.io.camera_interface import CameraInterface
from robokudo.pipeline import Pipeline
from robokudo.types.annotation import Classification
from typing_extensions import get_type_hints

from .dataset.annotation_filters import ClassificationFilter


# %% Descriptor discovery
def test_descriptor_discovery_returns_module_level_class() -> None:
    """
    The familiar annotator entry point returns an importable dataclass.
    """
    assert YoloAnnotator.Descriptor is YoloDescriptor
    assert (
        resolve_class_from_full_name(get_full_class_name(YoloDescriptor))
        is YoloDescriptor
    )
    assert is_dataclass(YoloAnnotator.Descriptor)
    assert is_dataclass(YoloAnnotator.Descriptor().parameters)


def test_descriptor_parameters_are_concrete_fields() -> None:
    """
    The descriptor declares the annotator's concrete parameter type.
    """
    assert get_type_hints(YoloDescriptor)["parameters"] is YoloParameters


# %% Configuration round trips
def test_descriptor_round_trip_preserves_local_and_global_values() -> None:
    """
    Effective values and concrete configuration classes survive JSON text.
    """
    original = YoloDescriptor()
    original.parameters.threshold = 0.75
    original.parameters.global_with_depth = False
    original.parameters.ignored_class_names.append(
        original.parameters.person_class_name
    )
    restored = from_json(json.loads(json.dumps(to_json(original))))
    assert type(restored) is YoloDescriptor
    assert type(restored.parameters) is YoloParameters
    assert restored == original


def test_mutable_parameter_defaults_are_independent() -> None:
    """
    Separate descriptors never share editable parameter defaults.
    """
    first = YoloDescriptor()
    second = YoloDescriptor()
    first.parameters.ignored_class_names.append(first.parameters.person_class_name)
    assert second.parameters.ignored_class_names == []


def test_image_segmentation_parameters_round_trip() -> None:
    """
    Color-range defaults remain serializable structured configuration.
    """
    original = ImageClusterExtractorDescriptor()
    restored = from_json(json.loads(json.dumps(to_json(original))))
    assert restored == original


# %% Global defaults
def test_global_defaults_are_materialized_at_descriptor_construction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    New descriptors snapshot configured defaults without changing old ones.
    """
    defaults = AnnotatorPredefinedParameters()
    original = BlurDescriptor()
    monkeypatch.setattr(
        defaults,
        "global_with_visualization",
        not original.parameters.global_with_visualization,
    )
    configured = BlurDescriptor()
    assert (
        configured.parameters.global_with_visualization
        is defaults.global_with_visualization
    )
    assert (
        original.parameters.global_with_visualization
        is not defaults.global_with_visualization
    )


def test_local_global_override_does_not_change_other_descriptors() -> None:
    """
    An AnalysisEngine can override one annotator's effective global value.
    """
    first = BlurAnnotator.Descriptor()
    second = BlurAnnotator.Descriptor()
    first.parameters.global_with_depth = not second.parameters.global_with_depth
    assert (
        second.parameters.global_with_depth
        is AnnotatorPredefinedParameters().global_with_depth
    )
    assert first.parameters.global_with_depth is not second.parameters.global_with_depth


def test_explicit_parameters_take_precedence_over_global_defaults(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    A supplied parameter dataclass is retained without reapplying defaults.
    """
    monkeypatch.setattr(AnnotatorPredefinedParameters(), "global_with_depth", False)
    parameters = BlurParameters(global_with_depth=True)
    assert BlurDescriptor(parameters=parameters).parameters is parameters
    assert parameters.global_with_depth is True


def test_global_defaults_apply_across_nested_and_independent_pipelines(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    All pipelines use application defaults while local changes remain isolated.
    """
    monkeypatch.setattr(
        AnnotatorPredefinedParameters(), "global_with_visualization", False
    )
    first = BlurAnnotator()
    nested = BlurAnnotator()
    independent = BlurAnnotator()
    Pipeline(children=[first, Pipeline(children=[nested])])
    Pipeline(children=[independent])
    first.descriptor.parameters.global_with_visualization = True
    assert first.descriptor.parameters.global_with_visualization is True
    assert nested.descriptor.parameters.global_with_visualization is False
    assert independent.descriptor.parameters.global_with_visualization is False


def test_parameter_factory_applies_global_defaults_and_explicit_overrides(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Parameter factories combine shared defaults with caller-supplied choices.
    """
    monkeypatch.setattr(AnnotatorPredefinedParameters(), "global_with_depth", False)
    monkeypatch.setattr(
        AnnotatorPredefinedParameters(), "global_with_visualization", False
    )
    parameters = YoloParameters.from_defaults(global_with_depth=True, threshold=0.75)
    assert parameters.global_with_depth is True
    assert parameters.global_with_visualization is False
    assert parameters.threshold == 0.75


# %% Runtime-independent configuration
def test_collection_reader_descriptor_round_trip() -> None:
    """
    A camera implementation is saved by type without constructing live resources.
    """
    original = CollectionReaderDescriptor(
        parameters=CollectionReaderParameters.from_defaults(
            camera_configuration=BaseCameraConfig(
                interface_type=CameraInterface.__name__
            ),
            camera_interface_type=CameraInterface,
        )
    )
    restored = from_json(json.loads(json.dumps(to_json(original))))
    assert restored == original
    assert restored.parameters.camera_interface_type is CameraInterface


def test_restored_collection_reader_constructs_its_runtime_interface() -> None:
    """
    A restored descriptor constructs its declared camera implementation.
    """
    descriptor = CollectionReaderDescriptor(
        parameters=CollectionReaderParameters.from_defaults(
            camera_configuration=BaseCameraConfig(
                interface_type=CameraInterface.__name__
            ),
            camera_interface_type=CameraInterface,
        )
    )
    restored = from_json(json.loads(json.dumps(to_json(descriptor))))
    reader = CollectionReaderAnnotator(restored)
    assert type(reader.camera_interface) is restored.parameters.camera_interface_type
    assert (
        reader.camera_interface.camera_configuration
        is restored.parameters.camera_configuration
    )


def test_callable_dataclass_parameter_round_trip() -> None:
    """
    Configured callable dataclasses retain both predicate state and behavior.
    """
    predicate = ClassificationFilter(classname=Classification.__name__)
    original = FilterDescriptor()
    original.parameters.function = predicate
    restored = from_json(json.loads(json.dumps(to_json(original))))
    assert restored.parameters.function == predicate
    assert (
        restored.parameters.function(Classification(classname=predicate.classname))
        is True
    )
    assert restored.parameters.function(Classification()) is False


def test_annotator_specific_global_default_is_preserved(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Pose detection retains its explicit opt-out from depth processing.
    """
    monkeypatch.setattr(AnnotatorPredefinedParameters(), "global_with_depth", True)
    assert HumanAndPoseDescriptor().parameters.global_with_depth is False


# %% Device selections
@pytest.mark.parametrize("device_identifier", ["cpu", "cuda:2"])
def test_clip_device_selection_round_trips_without_device_access(
    device_identifier: str,
) -> None:
    """
    Device selections are restored even when the requested hardware is absent.
    """
    original = CLIPDescriptor()
    original.parameters.device = torch.device(device_identifier)
    restored = from_json(json.loads(json.dumps(to_json(original))))
    assert restored.parameters.device == original.parameters.device


# %% All annotator configurations
@pytest.mark.parametrize(
    "descriptor_type",
    [
        BaseAnnotatorDescriptor,
        AnnotationStorageWriterDescriptor,
        AnnotationPublisherWriterDescriptor,
        BlurDescriptor,
        CLIPDescriptor,
        ClusterColorDescriptor,
        ClusterColorHistogramDescriptor,
        ClusterPoseBoundingBoxDescriptor,
        ClusterPositionDescriptor,
        FileWriterDescriptor,
        FilterDescriptor,
        HumanAndPoseDescriptor,
        ICPPoseRefinementDescriptor,
        ImageClusterExtractorDescriptor,
        ImagePreprocessorDescriptor,
        LambdaFunctionDescriptor,
        LocationDescriptor,
        ObjectHypothesisVisualizerDescriptor,
        OpenVocabularyObjectDetectionDescriptor,
        OutlierRemovalOnObjectHypothesisDescriptor,
        PlaneDescriptor,
        PointcloudCheckDescriptor,
        PointCloudClusterExtractorDescriptor,
        NaivePointCloudClusterExtractorDescriptor,
        PointcloudCropDescriptor,
        GenerateQueryResultDescriptor,
        RegionFilterDescriptor,
        ROIAdjusterDescriptor,
        SemanticDigitalTwinConnectorDescriptor,
        ShapeEstimatorDescriptor,
        StaticCameraTransformDescriptor,
        StaticObjectDetectorDescriptor,
        StorageWriterDescriptor,
        ScopedDescriptor,
        TSDFDescriptor,
        WorldDescriptorBootstrapDescriptor,
        WorldVisualizerDescriptor,
        YoloDescriptor,
        ZeroShotClassificationDescriptor,
    ],
)
def test_all_default_descriptors_round_trip(
    descriptor_type: type[AnnotatorDescriptor],
) -> None:
    """
    Every default annotator configuration restores its concrete type and values.
    """
    original = descriptor_type()
    assert descriptor_type.Parameters is type(original.parameters)
    assert resolve_class_from_full_name(
        get_full_class_name(type(original.parameters))
    ) is type(original.parameters)
    encoded = json.loads(json.dumps(to_json(original)))
    restored = from_json(encoded)
    assert type(restored) is descriptor_type
    assert type(restored.parameters) is type(original.parameters)
    assert to_json(restored) == to_json(original)
