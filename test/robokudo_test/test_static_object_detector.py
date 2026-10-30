"""
Tests for the static object detector.
"""

from __future__ import annotations

# %% Imports
import numpy as np
import open3d
import pytest
from typing_extensions import TYPE_CHECKING

from robokudo.annotators.outputs import AnnotatorOutputPerPipelineMap
from robokudo.annotators.static_object_detector import StaticObjectDetectorAnnotator
from robokudo.cas import CASViews
from robokudo.pipeline import Pipeline
from robokudo.types.cv import ColorToDepthRatio
from robokudo.types.annotation import BoundingBox3DAnnotation, PoseAnnotation
from robokudo.types.camera import CameraObservation
from robokudo.types.scene import ObjectHypothesis
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix, Point2
from semantic_digital_twin.datastructures.camera_model import PinholeCameraModel
from semantic_digital_twin.datastructures.camera_resolution import CameraResolution
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.robots.robot_parts import Camera
from semantic_digital_twin.world_description.connections import FixedConnection
from semantic_digital_twin.world_description.geometry import Box, Scale
from semantic_digital_twin.world_description.shape_collection import ShapeCollection
from semantic_digital_twin.world_description.world_entity import Body

if TYPE_CHECKING:
    from semantic_digital_twin.world import World


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
    detector.camera_model = PinholeCameraModel(
        image_resolution=CameraResolution(width=depth_width, height=depth_height),
        focal_length_x=50.0,
        focal_length_y=50.0,
        principal_point=Point2(x=depth_width / 2, y=depth_height / 2),
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


# %% Body-relative bounding boxes
@pytest.mark.parametrize("explicit_camera_transform", [False, True])
def test_body_detection_transforms_offset_bounding_box_into_camera_frame(
    detector_on_half_resolution_depth: StaticObjectDetectorAnnotator,
    world_with_two_bodies: tuple[World, Body, Body],
    explicit_camera_transform: bool,
) -> None:
    """
    Camera and body poses place an offset bounding box in the optical frame.
    """
    detector = detector_on_half_resolution_depth
    world, root, body = world_with_two_bodies
    shape = Box(
        origin=HomogeneousTransformationMatrix.from_xyz_rpy(
            x=1.0, reference_frame=body
        ),
        scale=Scale(x=0.4, y=0.8, z=1.2),
    )
    body.collision = ShapeCollection([shape], reference_frame=body)
    with world.modify_world():
        world.add_connection(
            FixedConnection(
                parent=root,
                child=body,
                parent_T_connection_expression=HomogeneousTransformationMatrix.from_xyz_rpy(
                    x=4.0, y=3.0, z=6.0, yaw=np.pi
                ),
            )
        )
    camera_model = detector.camera_model.scaled(scale_x=2.0, scale_y=2.0)
    camera = Camera(
        name=PrefixedName("camera"),
        root=Body(name=PrefixedName("optical_frame")),
        camera_model=camera_model,
    )
    world_T_camera = HomogeneousTransformationMatrix.from_xyz_rpy(
        x=2.0,
        y=2.0,
        yaw=np.pi / 2,
        reference_frame=root,
        child_frame=camera.root,
    )
    detector.get_cas().camera_observation = CameraObservation(
        camera=camera,
        effective_camera_model=camera_model,
        world_T_camera=world_T_camera,
        timestamp_nanoseconds=0,
    )
    detector.get_cas().set(CASViews.COLOR_IMAGE, np.zeros((40, 80, 3), dtype=np.uint8))
    detector.get_cas().pointcloud_camera_model = detector.camera_model
    detector.cloud = open3d.geometry.PointCloud(
        open3d.utility.Vector3dVector([[1.0, -1.0, 6.0], [1.0, -1.0, 8.0]])
    )
    detector.descriptor.parameters.create_pose_annotation = True
    detector.descriptor.parameters.create_bounding_box_annotation = True

    hypothesis = (
        detector.detect_from_body(body, camera_T_world=world_T_camera.inverse())
        if explicit_camera_transform
        else detector.detect_from_body(body)
    )

    assert isinstance(hypothesis, ObjectHypothesis)
    pose = next(
        annotation
        for annotation in hypothesis.annotations
        if isinstance(annotation, PoseAnnotation)
    )
    bounding_box = next(
        annotation
        for annotation in hypothesis.annotations
        if isinstance(annotation, BoundingBox3DAnnotation)
    )
    np.testing.assert_allclose(pose.translation, [1.0, -1.0, 6.0], atol=1e-12)
    np.testing.assert_allclose(
        pose.rotation, [0.0, 0.0, np.sqrt(0.5), np.sqrt(0.5)], atol=1e-12
    )
    np.testing.assert_allclose(bounding_box.pose.translation, pose.translation)
    np.testing.assert_allclose(bounding_box.pose.rotation, pose.rotation)
    np.testing.assert_allclose(
        [bounding_box.x_length, bounding_box.y_length, bounding_box.z_length],
        shape.scale.to_np(),
    )
    np.testing.assert_allclose(np.asarray(hypothesis.points.points), [[1.0, -1.0, 6.0]])
