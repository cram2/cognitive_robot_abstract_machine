"""
Tests for cropping a point cloud relative to the world frame.

``PointcloudCropAnnotator``'s own docstring documents two supported modes: cropping in
sensor coordinates (the default) or relative to the world frame. The world-relative mode
transforms the cloud into world coordinates, crops it there, then transforms the result
back into camera coordinates -- but an early, unconditional ``return Status.FAILURE``
made every step after the transform unreachable, so the mode never actually cropped
anything even when the camera-to-world transform was available.
"""

import numpy as np
import open3d as o3d
import pytest
from py_trees.blackboard import Blackboard
from py_trees.common import Status

# robokudo.pipeline must be imported before robokudo.annotators.outputs: importing
# outputs first trips a circular import between it and robokudo.annotators.core.
import robokudo.pipeline
from robokudo.annotators.outputs import AnnotatorOutputPerPipelineMap, AnnotatorOutputs
from robokudo.annotators.pointcloud_crop import PointcloudCropAnnotator
from robokudo.cas import CAS, CASViews
from robokudo.pipeline import Pipeline
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix


@pytest.fixture()
def camera_intrinsics() -> o3d.camera.PinholeCameraIntrinsic:
    """
    Intrinsics sized so the test cloud's points project well inside the color image.
    """
    intrinsics = o3d.camera.PinholeCameraIntrinsic()
    intrinsics.set_intrinsics(
        width=200, height=200, fx=100.0, fy=100.0, cx=100.0, cy=100.0
    )
    return intrinsics


@pytest.fixture()
def cas_with_cloud_at_world_origin(
    camera_intrinsics: o3d.camera.PinholeCameraIntrinsic,
) -> CAS:
    """
    A CAS whose cloud sits at the world origin, one metre in front of the camera, with
    an identity camera-to-world transform (so world and camera coordinates coincide
    here, keeping the fixture's geometry simple).
    """
    cas = CAS()
    cas.camera_to_world_transform = HomogeneousTransformationMatrix.from_xyz_quaternion(
        pos_x=0.0,
        pos_y=0.0,
        pos_z=0.0,
        quat_x=0.0,
        quat_y=0.0,
        quat_z=0.0,
        quat_w=1.0,
    )

    points_inside_the_crop = np.array([[0.0, 0.0, 1.0], [0.05, 0.05, 1.0]])
    points_outside_the_crop = np.array([[5.0, 5.0, 1.0], [-5.0, -5.0, 1.0]])
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(
        np.vstack([points_inside_the_crop, points_outside_the_crop])
    )

    cas.set_ref(CASViews.CLOUD, cloud)
    cas.set(CASViews.COLOR_IMAGE, np.zeros((200, 200, 3), dtype=np.uint8))
    cas.set(CASViews.POINTCLOUD_CAMERA_INTRINSIC, camera_intrinsics)
    cas.set(CASViews.COLOR2DEPTH_RATIO, (1.0, 1.0))
    return cas


@pytest.fixture()
def crop_annotator_in_pipeline(cas_with_cloud_at_world_origin: CAS):
    """
    A ``PointcloudCropAnnotator`` configured to crop relative to the world frame, wired
    up with just enough pipeline/blackboard state for ``update()`` to run.
    """
    descriptor = PointcloudCropAnnotator.Descriptor()
    descriptor.parameters.relative_to_world = True
    descriptor.parameters.min_x = -1.0
    descriptor.parameters.max_x = 1.0
    descriptor.parameters.min_y = -1.0
    descriptor.parameters.max_y = 1.0
    descriptor.parameters.min_z = 0.0
    descriptor.parameters.max_z = 2.0

    pipeline = Pipeline("TestPipeline")
    pipeline.cas = cas_with_cloud_at_world_origin
    annotator = PointcloudCropAnnotator(descriptor=descriptor)
    pipeline.add_child(annotator)

    output_map = AnnotatorOutputPerPipelineMap()
    output_map.map[pipeline.name] = AnnotatorOutputs()
    Blackboard().set("annotator_output_pipeline_map_buffer", output_map)

    return annotator


def test_world_relative_crop_succeeds_when_the_transform_is_available(
    crop_annotator_in_pipeline: PointcloudCropAnnotator,
):
    status = crop_annotator_in_pipeline.update()

    assert status == Status.SUCCESS


def test_world_relative_crop_actually_crops_the_cloud(
    crop_annotator_in_pipeline: PointcloudCropAnnotator,
):
    crop_annotator_in_pipeline.update()

    cropped_cloud = crop_annotator_in_pipeline.get_cas().get(CASViews.CLOUD)

    assert len(cropped_cloud.points) == 2


def test_world_relative_crop_fails_without_a_camera_to_world_transform(
    crop_annotator_in_pipeline: PointcloudCropAnnotator,
):
    crop_annotator_in_pipeline.get_cas().camera_to_world_transform = None

    status = crop_annotator_in_pipeline.update()

    assert status == Status.FAILURE
