"""
Tests for the static object detector.
"""

from __future__ import annotations

import numpy as np
import open3d as o3d
import pytest

from robokudo.annotators.outputs import AnnotatorOutputPerPipelineMap
from robokudo.annotators.static_object_detector import StaticObjectDetectorAnnotator
from robokudo.cas import CASViews
from robokudo.pipeline import Pipeline
from robokudo.types.cv import ColorToDepthRatio


@pytest.fixture
def detector_on_half_resolution_depth(
    annotator_output_buffer: AnnotatorOutputPerPipelineMap,
) -> StaticObjectDetectorAnnotator:
    """
    A detector in a pipeline whose depth image has half the resolution of the color
    image and measures the same distance at every pixel.
    """
    detector = StaticObjectDetectorAnnotator()
    pipeline = Pipeline("Sequence")
    pipeline.add_child(detector)
    pipeline.setup()

    depth_height, depth_width = 20, 40
    detector.get_cas().set(
        CASViews.COLOR_TO_DEPTH_RATIO, ColorToDepthRatio(horizontal=0.5, vertical=0.5)
    )
    detector.depth = np.full((depth_height, depth_width), 1000, dtype=np.uint16)
    detector.camera_intrinsics = o3d.camera.PinholeCameraIntrinsic(
        depth_width, depth_height, 50.0, 50.0, depth_width / 2, depth_height / 2
    )
    return detector


# %% bounding box cloud


def test_bounding_box_cloud_covers_box_on_depth_resolution(
    detector_on_half_resolution_depth: StaticObjectDetectorAnnotator,
):
    detector = detector_on_half_resolution_depth
    parameters = detector.descriptor.parameters
    parameters.bounding_box_x = 4
    parameters.bounding_box_y = 6
    parameters.bounding_box_width = 10
    parameters.bounding_box_height = 8
    color_rgb = np.zeros((*detector.depth.shape, 3), dtype=np.uint8)

    cloud = detector.get_cloud_from_2d_bb_roi(color_rgb)

    depth_box_area = (7 - 2) * (7 - 3)
    assert len(cloud.points) == depth_box_area
