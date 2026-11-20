"""
Tests that annotations name the class of the annotator that produced them.
"""

from __future__ import annotations

# %% Imports

import numpy
import open3d
import pytest
from py_trees.common import Status

from robokudo.annotators.cluster_color import ClusterColorAnnotator
from robokudo.annotators.core import BaseAnnotator
from robokudo.annotators.outputs import AnnotatorOutputPerPipelineMap
from robokudo.annotators.pointcloud_cluster_extractor import (
    NaivePointCloudClusterExtractor,
    PointCloudClusterExtractor,
)
from robokudo.annotators.shape_estimator import ShapeEstimatorAnnotator
from robokudo.annotators.static_object_detector import StaticObjectDetectorAnnotator
from robokudo.annotators.testing import EmptyAnnotator
from robokudo.annotators.tsdf_annotator import TSDFAnnotator
from robokudo.cas import CASViews
from robokudo.pipeline import Pipeline
from robokudo.types.annotation import (
    Classification,
    PoseAnnotation,
    SemanticColor,
    Shape,
    Plane,
    TSDFAnnotation,
)
from robokudo.types.camera import CameraObservation
from robokudo.types.cv import ColorToDepthRatio, Rect
from robokudo.types.scene import ObjectHypothesis
from krrood_spatial_math.spatial_types import Point2
from semantic_digital_twin.datastructures.camera_model import PinholeCameraModel
from semantic_digital_twin.datastructures.camera_resolution import CameraResolution
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.robots.robot_parts import Camera
from semantic_digital_twin.world_description.world_entity import Body


# %% Fixtures


def _add_to_pipeline(annotator: BaseAnnotator) -> Pipeline:
    """
    :return: A set up pipeline whose only child is the given annotator.
    """
    pipeline = Pipeline()
    pipeline.add_child(annotator)
    pipeline.setup()
    return pipeline


def _square_camera_model(*, side_length: int) -> PinholeCameraModel:
    """
    :return: A camera model of a square image whose principal point is its center.
    """
    return PinholeCameraModel(
        image_resolution=CameraResolution(width=side_length, height=side_length),
        focal_length_x=float(side_length),
        focal_length_y=float(side_length),
        principal_point=Point2(x=side_length / 2, y=side_length / 2),
    )


def _point_grid(*, center: list[float], extent: float, count: int) -> numpy.ndarray:
    """
    :return: A cube of ``count`` points per axis around the center.
    """
    axis = numpy.linspace(-extent / 2, extent / 2, count)
    offsets = numpy.stack(numpy.meshgrid(axis, axis, axis), axis=-1).reshape(-1, 3)
    return offsets + numpy.asarray(center)


def _place_cloud_in_cas(*, pipeline: Pipeline, points: numpy.ndarray) -> None:
    """
    Put the points, a black color image and a matching projection model into the CAS.
    """
    side_length = 40
    pipeline.cas.set(
        CASViews.CLOUD,
        open3d.geometry.PointCloud(open3d.utility.Vector3dVector(points)),
    )
    pipeline.cas.set(
        CASViews.COLOR_IMAGE,
        numpy.zeros((side_length, side_length, 3), dtype=numpy.uint8),
    )
    pipeline.cas.set(CASViews.COLOR_TO_DEPTH_RATIO, ColorToDepthRatio())
    pipeline.cas.pointcloud_camera_model = _square_camera_model(side_length=side_length)


# %% Class name


def test_class_name_is_readable_from_annotator_class() -> None:
    """
    The class name is available without instantiating the annotator.
    """
    assert EmptyAnnotator.get_class_name() == EmptyAnnotator.__name__


def test_class_name_ignores_custom_annotator_name(
    renamed_annotator_name: str,
) -> None:
    """
    A custom instance name does not change the class name.
    """
    annotator = EmptyAnnotator(name=renamed_annotator_name)

    assert annotator.get_class_name() == EmptyAnnotator.__name__


# %% Annotation source


def test_static_object_detector_classification_names_detector_class(
    annotator_output_buffer: AnnotatorOutputPerPipelineMap,
) -> None:
    """
    Classifications record the detector class as their source.
    """
    hypothesis = ObjectHypothesis()

    StaticObjectDetectorAnnotator.add_classification_annotation(
        object_hypothesis=hypothesis, class_name="cup"
    )

    (classification,) = hypothesis.annotations
    assert isinstance(classification, Classification)
    assert classification.source == StaticObjectDetectorAnnotator.get_class_name()


def test_renamed_cluster_color_annotator_names_its_class(
    renamed_annotator_name: str,
    annotator_output_buffer: AnnotatorOutputPerPipelineMap,
) -> None:
    """
    Colors from a renamed annotator record its class, not its instance name.
    """
    height, width = 10, 10
    annotator = ClusterColorAnnotator(name=renamed_annotator_name)
    pipeline = _add_to_pipeline(annotator)
    color_image = numpy.zeros((height, width, 3), dtype=numpy.uint8)
    hypothesis = ObjectHypothesis()
    hypothesis.roi.roi = Rect(width=width, height=height)
    hypothesis.roi.mask = numpy.full((height, width), 255, dtype=numpy.uint8)
    pipeline.cas.set(CASViews.CLOUD, open3d.geometry.PointCloud())
    pipeline.cas.set(CASViews.COLOR_IMAGE, color_image)
    pipeline.cas.annotations.append(hypothesis)

    assert annotator.update() == Status.SUCCESS

    colors = [
        annotation
        for annotation in hypothesis.annotations
        if isinstance(annotation, SemanticColor)
    ]
    assert len(colors) > 0
    assert all(
        color.source == ClusterColorAnnotator.get_class_name() for color in colors
    )


def test_renamed_shape_estimator_names_its_class(
    renamed_annotator_name: str,
    annotator_output_buffer: AnnotatorOutputPerPipelineMap,
) -> None:
    """
    Shapes from a renamed annotator record its class, not its instance name.
    """
    annotator = ShapeEstimatorAnnotator(name=renamed_annotator_name)
    pipeline = _add_to_pipeline(annotator)
    box_surface = open3d.geometry.TriangleMesh.create_box(
        width=0.1, height=0.2, depth=0.3
    ).sample_points_uniformly(number_of_points=2000)
    hypothesis = ObjectHypothesis(points=box_surface)
    pipeline.cas.annotations.append(hypothesis)

    assert annotator.compute() == Status.SUCCESS

    (shape,) = [
        annotation
        for annotation in hypothesis.annotations
        if isinstance(annotation, Shape)
    ]
    assert shape.source == ShapeEstimatorAnnotator.get_class_name()


def test_renamed_naive_point_cloud_cluster_extractor_names_its_class(
    renamed_annotator_name: str,
    annotator_output_buffer: AnnotatorOutputPerPipelineMap,
) -> None:
    """
    Clusters from a renamed naive extractor record its class as their source.
    """
    annotator = NaivePointCloudClusterExtractor(name=renamed_annotator_name)
    pipeline = _add_to_pipeline(annotator)
    _place_cloud_in_cas(
        pipeline=pipeline,
        points=_point_grid(center=[0.0, 0.0, 1.0], extent=0.1, count=12),
    )

    assert annotator.compute() == Status.SUCCESS

    (hypothesis,) = pipeline.cas.filter_annotations_by_type(ObjectHypothesis)
    assert hypothesis.source == NaivePointCloudClusterExtractor.get_class_name()


def test_renamed_point_cloud_cluster_extractor_names_its_class(
    renamed_annotator_name: str,
    annotator_output_buffer: AnnotatorOutputPerPipelineMap,
) -> None:
    """
    Clusters above a plane from a renamed extractor record its class as their source.
    """
    annotator = PointCloudClusterExtractor(name=renamed_annotator_name)
    pipeline = _add_to_pipeline(annotator)
    table_axis = numpy.linspace(-0.3, 0.3, 61)
    table = numpy.stack(
        numpy.meshgrid(table_axis, table_axis, [1.0, 1.005]), axis=-1
    ).reshape(-1, 3)
    cube = _point_grid(center=[0.0, 0.0, 0.9], extent=0.1, count=12)
    _place_cloud_in_cas(pipeline=pipeline, points=numpy.concatenate([table, cube]))
    pipeline.cas.annotations.append(
        Plane(model=[0.0, 0.0, 1.0, -1.0], inliers=list(range(len(table))))
    )

    assert annotator.compute() == Status.SUCCESS

    (hypothesis,) = pipeline.cas.filter_annotations_by_type(ObjectHypothesis)
    assert hypothesis.source == PointCloudClusterExtractor.get_class_name()


def test_renamed_tsdf_annotator_names_its_class(
    renamed_annotator_name: str,
    annotator_output_buffer: AnnotatorOutputPerPipelineMap,
) -> None:
    """
    Volumes from a renamed annotator record its class, not its instance name.
    """
    side_length = 40
    annotator = TSDFAnnotator(name=renamed_annotator_name)
    pipeline = _add_to_pipeline(annotator)
    camera_model = _square_camera_model(side_length=side_length)
    pipeline.cas.camera_observation = CameraObservation(
        camera=Camera(
            name=PrefixedName("camera"),
            root=Body(name=PrefixedName("optical_frame")),
            camera_model=camera_model,
        ),
        effective_camera_model=camera_model,
        world_T_camera=None,
        timestamp_nanoseconds=0,
    )
    pipeline.cas.set(CASViews.COLOR_TO_DEPTH_RATIO, ColorToDepthRatio())
    pipeline.cas.set(
        CASViews.COLOR_IMAGE,
        numpy.zeros((side_length, side_length, 3), dtype=numpy.uint8),
    )
    pipeline.cas.set(
        CASViews.DEPTH_IMAGE,
        numpy.full((side_length, side_length), 1000, dtype=numpy.uint16),
    )
    hypothesis = ObjectHypothesis()
    hypothesis.roi.roi = Rect(width=side_length, height=side_length)
    hypothesis.roi.mask = numpy.full((side_length, side_length), 255, dtype=numpy.uint8)
    hypothesis.annotations.append(
        PoseAnnotation(translation=[0.0, 0.0, 0.0], rotation=[0.0, 0.0, 0.0, 1.0])
    )
    pipeline.cas.annotations.append(hypothesis)

    assert annotator.compute() == Status.SUCCESS

    (volume,) = [
        annotation
        for annotation in hypothesis.annotations
        if isinstance(annotation, TSDFAnnotation)
    ]
    assert volume.source == TSDFAnnotator.get_class_name()
