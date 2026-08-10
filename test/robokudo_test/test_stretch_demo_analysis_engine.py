"""
Tests for the Stretch apartment demo's perception pipeline configuration.
"""

import numpy as np

from robokudo.annotators.image_cluster_extractor import ImageClusterExtractor
from robokudo.annotators.plane import PlaneAnnotator
from robokudo.annotators.pointcloud_cluster_extractor import PointCloudClusterExtractor
from robokudo.annotators.pointcloud_crop import PointcloudCropAnnotator
from robokudo.descriptors.analysis_engines.stretch_demo import (
    TARGET_SHELF_LAYER_MAX_WORLD_Z,
    TARGET_SHELF_LAYER_MIN_WORLD_Z,
    TARGET_SHELF_MAX_WORLD_X,
    TARGET_SHELF_MAX_WORLD_Y,
    TARGET_SHELF_MIN_WORLD_X,
    TARGET_SHELF_MIN_WORLD_Y,
    AnalysisEngine,
)
from test.robokudo_test.test_analysis_engine_query_composition import (
    bounded_build_time,
)


def test_pointcloud_crop_is_narrowed_to_the_target_shelf_layer():
    """
    Without a height-bounded, world-relative crop, the pipeline's wide RealSense field
    of view sees every shelf layer at once and answers a query with one untyped
    candidate per layer instead of just the object on the targeted one.
    """
    with bounded_build_time():
        pipeline = AnalysisEngine().implementation()

    crop_annotator = next(
        node for node in pipeline.children if isinstance(node, PointcloudCropAnnotator)
    )
    parameters = crop_annotator.descriptor.parameters

    assert parameters.relative_to_world is True
    assert parameters.min_x == TARGET_SHELF_MIN_WORLD_X
    assert parameters.max_x == TARGET_SHELF_MAX_WORLD_X
    assert parameters.min_y == TARGET_SHELF_MIN_WORLD_Y
    assert parameters.max_y == TARGET_SHELF_MAX_WORLD_Y
    assert parameters.min_z == TARGET_SHELF_LAYER_MIN_WORLD_Z
    assert parameters.max_z == TARGET_SHELF_LAYER_MAX_WORLD_Z


def test_pipeline_extracts_objects_by_color_not_depth_clustering():
    """
    The target object is glossy enough that the RealSense returns no depth on its face,
    which starves depth-based clustering of points to work with. Extracting by color
    instead bounds the object's region from its RGB contour, so a depth hole just means
    fewer 3D points survive within an already-correctly-shaped region.
    """
    with bounded_build_time():
        pipeline = AnalysisEngine().implementation()

    assert not any(
        isinstance(node, (PlaneAnnotator, PointCloudClusterExtractor))
        for node in pipeline.children
    )

    cluster_annotator = next(
        node for node in pipeline.children if isinstance(node, ImageClusterExtractor)
    )
    parameters = cluster_annotator.descriptor.parameters
    red_hsv_range = parameters.color_name_to_hsv_range["red"]

    assert parameters.hsv_min == red_hsv_range["hsv_min"]
    assert parameters.hsv_max == red_hsv_range["hsv_max"]


def test_target_shelf_layer_bounds_are_the_midpoints_to_its_neighbours():
    """
    The bounds are derived from the apartment demo's own shelf layer heights (0.283m,
    0.63m, 1.265m, 1.613m), targeting the second layer at 0.63m -- pinned here so a
    change to either side silently drifting out of sync is caught.
    """
    shelf_layer_heights = [0.283, 0.63, 1.265, 1.613]
    target_layer_height = shelf_layer_heights[1]

    assert (
        TARGET_SHELF_LAYER_MIN_WORLD_Z
        == (shelf_layer_heights[0] + target_layer_height) / 2
    )
    assert (
        TARGET_SHELF_LAYER_MAX_WORLD_Z
        == (target_layer_height + shelf_layer_heights[2]) / 2
    )


def shelf_world_footprint_corners() -> np.ndarray:
    """
    The shelf's four outer-footprint corners in world coordinates.

    Reproduces the placement in ``experiments.real_stretch_apartment_demo.demo``: a
    0.305m x 0.85m footprint centred at world (0.88, -0.17) with a -90 degree yaw.
    """
    center = np.array([0.455 + 0.85 / 2, -0.17])
    yaw = -np.pi / 2
    local_x_extent, local_y_extent = 0.305, 0.85
    cos_yaw, sin_yaw = np.cos(yaw), np.sin(yaw)
    local_x_axis_in_world = np.array([cos_yaw, sin_yaw]) * (local_x_extent / 2)
    local_y_axis_in_world = np.array([-sin_yaw, cos_yaw]) * (local_y_extent / 2)
    return np.array(
        [
            center + sign_x * local_x_axis_in_world + sign_y * local_y_axis_in_world
            for sign_x in (-1, 1)
            for sign_y in (-1, 1)
        ]
    )


def test_shelf_lateral_bounds_contain_the_shelfs_actual_world_footprint():
    """
    The lateral crop bounds must not clip the shelf itself: pinned against the shelf's
    real placement rather than duplicating the literal numbers a second time, so a
    change to the shelf's position in the apartment demo is caught here too.
    """
    corners = shelf_world_footprint_corners()

    assert TARGET_SHELF_MIN_WORLD_X < corners[:, 0].min()
    assert TARGET_SHELF_MAX_WORLD_X > corners[:, 0].max()
    assert TARGET_SHELF_MIN_WORLD_Y < corners[:, 1].min()
    assert TARGET_SHELF_MAX_WORLD_Y > corners[:, 1].max()
