"""
Tests for detecting the dominant plane of a point cloud.

Open3D's plane segmentation is RANSAC, whose random samples decide which of several
nearly equally good planes wins; on a cloud with off-plane points, two runs can disagree
by enough to move points between the plane and the clusters found above it. The
annotator therefore takes a seed, so a pipeline that needs the same plane every time can
have it.
"""

from __future__ import annotations

import numpy as np
import open3d as o3d
from py_trees.blackboard import Blackboard
from py_trees.common import Status

# robokudo.pipeline must be imported before robokudo.annotators.outputs: importing
# outputs first trips a circular import between it and robokudo.annotators.core.
import robokudo.pipeline
from robokudo.annotators.outputs import AnnotatorOutputPerPipelineMap, AnnotatorOutputs
from robokudo.annotators.plane import PlaneAnnotator
from robokudo.cas import CAS, CASViews
from robokudo.pipeline import Pipeline
from robokudo.types.annotation import Plane

SEED = 7
"""
The seed the seeded annotators are given.
"""


def cas_with_a_noisy_plane() -> CAS:
    """
    :return: A CAS whose cloud is a plane at ``z = 1`` with noise, under a scatter of
        points above it, so that RANSAC has several nearly equally good planes to choose
        from; built afresh each time, since an annotator writes its plane into it.
    """
    random_state = np.random.default_rng(0)
    on_the_plane = np.column_stack(
        [
            random_state.uniform(-1.0, 1.0, 3000),
            random_state.uniform(-1.0, 1.0, 3000),
            1.0 + random_state.normal(0.0, 0.01, 3000),
        ]
    )
    above_the_plane = np.column_stack(
        [
            random_state.uniform(-1.0, 1.0, 600),
            random_state.uniform(-1.0, 1.0, 600),
            random_state.uniform(0.5, 1.0, 600),
        ]
    )
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(
        np.vstack([on_the_plane, above_the_plane])
    )
    cas = CAS()
    cas.set_ref(CASViews.CLOUD, cloud)
    cas.set(CASViews.COLOR_IMAGE, np.zeros((200, 200, 3), dtype=np.uint8))
    cas.set(CASViews.DEPTH_IMAGE, np.zeros((200, 200), dtype=np.uint16))
    cas.set(CASViews.CAMERA_INTRINSIC, o3d.camera.PinholeCameraIntrinsic())
    return cas


def _plane_annotator_in_pipeline(random_seed: int | None) -> PlaneAnnotator:
    """
    :param random_seed: The seed to give the annotator, if any.
    :return: A plane annotator over a fresh noisy cloud, wired up with just enough
        pipeline state to run.
    """
    descriptor = PlaneAnnotator.Descriptor()
    descriptor.parameters.visualize_plane_model = False
    descriptor.parameters.random_seed = random_seed
    pipeline = Pipeline("TestPipeline")
    pipeline.cas = cas_with_a_noisy_plane()
    annotator = PlaneAnnotator(descriptor=descriptor)
    pipeline.add_child(annotator)
    output_map = AnnotatorOutputPerPipelineMap()
    output_map.map[pipeline.name] = AnnotatorOutputs()
    Blackboard().set("annotator_output_pipeline_map_buffer", output_map)
    return annotator


def _detected_plane(annotator: PlaneAnnotator) -> Plane:
    """
    :param annotator: A plane annotator to run once.
    :return: The plane it annotated.
    """
    assert annotator.compute() == Status.SUCCESS
    [plane] = [
        annotation
        for annotation in annotator.get_cas().annotations
        if isinstance(annotation, Plane)
    ]
    return plane


def test_a_seeded_annotator_finds_the_same_plane_every_time():
    first = _detected_plane(_plane_annotator_in_pipeline(SEED))
    second = _detected_plane(_plane_annotator_in_pipeline(SEED))

    assert np.array_equal(first.model, second.model)
    assert first.inliers == second.inliers


def test_the_seed_is_off_by_default():
    assert PlaneAnnotator.Descriptor().parameters.random_seed is None
