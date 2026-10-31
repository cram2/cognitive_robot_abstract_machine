"""
Tests for cropping a point cloud relative to the world frame.

``PointcloudCropAnnotator``'s own docstring documents two supported modes: cropping in
sensor coordinates (the default) or relative to the world frame. The world-relative mode
transforms the cloud into world coordinates, crops it there, then transforms the result
back into camera coordinates -- but an early, unconditional ``return Status.FAILURE``
made every step after the transform unreachable, so the mode never actually cropped
anything even when the camera-to-world transform was available.
"""

# %% Imports

import numpy as np
import open3d
import pytest
from dataclasses import replace
from py_trees.blackboard import Blackboard
from py_trees.common import Status

# robokudo.pipeline must be imported before robokudo.annotators.outputs: importing
# outputs first trips a circular import between it and robokudo.annotators.core.
from robokudo.types.cv import ColorToDepthRatio
import robokudo.pipeline
from robokudo.annotators.outputs import AnnotatorOutputPerPipelineMap, AnnotatorOutputs
from robokudo.annotators.pointcloud_crop import PointcloudCropAnnotator
from robokudo.cas import CAS, CASViews
from robokudo.pipeline import Pipeline
from robokudo.types.camera import CameraObservation
from semantic_digital_twin.spatial_types import Point2
from semantic_digital_twin.datastructures.camera_model import PinholeCameraModel
from semantic_digital_twin.datastructures.camera_resolution import CameraResolution
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.robots.robot_parts import Camera
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.world_description.world_entity import Body

# %% Camera behavior


@pytest.fixture()
def camera_model() -> PinholeCameraModel:
    """
    Intrinsics sized so the test cloud's points project well inside the color image.
    """
    return PinholeCameraModel(
        image_resolution=CameraResolution(width=200, height=200),
        focal_length_x=100.0,
        focal_length_y=100.0,
        principal_point=Point2(x=100.0, y=100.0),
    )


@pytest.fixture()
def cas_with_cloud_at_world_origin(
    camera_model: PinholeCameraModel,
) -> CAS:
    """
    A CAS whose cloud sits at the world origin, one metre in front of the camera, with
    an identity camera-to-world transform (so world and camera coordinates coincide
    here, keeping the fixture's geometry simple).
    """
    cas = CAS()
    world_T_camera = HomogeneousTransformationMatrix.from_xyz_quaternion(
        pos_x=0.0,
        pos_y=0.0,
        pos_z=0.0,
        quat_x=0.0,
        quat_y=0.0,
        quat_z=0.0,
        quat_w=1.0,
    )
    camera_body = Body(name=PrefixedName(name="camera"))
    world_frame = Body(name=PrefixedName(name="world"))
    world_T_camera = HomogeneousTransformationMatrix(
        data=world_T_camera,
        reference_frame=world_frame,
        child_frame=camera_body,
    )
    cas.camera_observation = CameraObservation(
        camera=Camera(
            name=PrefixedName(name="camera"),
            root=camera_body,
            camera_model=camera_model,
        ),
        effective_camera_model=camera_model,
        world_T_camera=world_T_camera,
        timestamp_nanoseconds=123,
    )

    points_inside_the_crop = np.array([[0.0, 0.0, 1.0], [0.05, 0.05, 1.0]])
    points_outside_the_crop = np.array([[5.0, 5.0, 1.0], [-5.0, -5.0, 1.0]])
    cloud = open3d.geometry.PointCloud()
    cloud.points = open3d.utility.Vector3dVector(
        np.vstack([points_inside_the_crop, points_outside_the_crop])
    )

    cas.set_ref(CASViews.CLOUD, cloud)
    cas.set(CASViews.COLOR_IMAGE, np.zeros((200, 200, 3), dtype=np.uint8))
    cas.pointcloud_camera_model = camera_model
    cas.set(CASViews.COLOR_TO_DEPTH_RATIO, ColorToDepthRatio())
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
    cas = crop_annotator_in_pipeline.get_cas()
    cas.camera_observation = replace(
        cas.require_camera_observation(), world_T_camera=None
    )

    status = crop_annotator_in_pipeline.update()

    assert status == Status.FAILURE
