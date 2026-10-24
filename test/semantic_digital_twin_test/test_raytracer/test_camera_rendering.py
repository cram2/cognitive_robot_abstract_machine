"""
Aligned camera rendering from shared ray intersections.
"""

from __future__ import annotations

# %% Imports

import io
from dataclasses import dataclass, replace
from unittest.mock import Mock
import numpy as np
import pytest
from PIL import Image, UnidentifiedImageError
from pyglet.canvas.xlib import NoSuchDisplayException
from semantic_digital_twin.spatial_types import Point2
from semantic_digital_twin.datastructures.camera_model import (
    CameraRange,
    PlumbBobCameraDistortion,
    FieldOfViewCameraModel,
    PinholeCameraModel,
)
from semantic_digital_twin.datastructures.camera_resolution import CameraResolution
from semantic_digital_twin.datastructures.field_of_view import FieldOfView
from semantic_digital_twin.datastructures.camera_rendering import (
    CameraRays,
    RayIntersections,
    DepthMode,
    ColorRenderingMode,
)
from semantic_digital_twin.robots.robot_parts import Camera
from semantic_digital_twin.spatial_computations.raytracer import RayTracer
from semantic_digital_twin.spatial_computations.exceptions import (
    RayPointShapeMismatch,
    UnsupportedCameraProjection,
)
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.testing import ray_test_world
from semantic_digital_twin.world_description.geometry import Color
from semantic_digital_twin.world_description.world_entity import Body
from semantic_digital_twin.world import World


# %% Controlled camera intersections
@dataclass
class CameraIntersections:
    """
    A rectangular camera with two hits and remaining background pixels.
    """

    ray_tracer: RayTracer
    """
    Ray tracer operating on the fixture world.
    """
    camera: Mock
    """
    Camera providing a projection model, range, and pose.
    """

    body: Body
    """
    World body hit by both viewing rays.
    """
    query: Mock
    """
    Controlled intersection query used to verify shared rendering.
    """

    resolution: CameraResolution
    """
    Rectangular image dimensions.
    """


@pytest.fixture
def camera_intersections(
    ray_test_world: tuple[World, Body, Body, Body, Body],
    monkeypatch: pytest.MonkeyPatch,
) -> CameraIntersections:
    """
    Provide on-axis and off-axis hits at equal optical depth.
    """
    world, body, *_ = ray_test_world
    ray_tracer = RayTracer(world)
    resolution = CameraResolution(width=3, height=2)
    origins = np.zeros((2, 3))
    directions = np.array([[0.0, 0.0, -1.0], [0.6, 0.0, -0.8]])
    pixels = np.array([[0, 0], [1, 0]])
    query = Mock(
        return_value=RayIntersections(
            hit_points=np.array([[0.0, 0.0, -2.0], [1.5, 0.0, -2.0]]),
            ray_indices=np.array([0, 1]),
            bodies=[body, body],
        )
    )
    monkeypatch.setattr(
        ray_tracer,
        "create_camera_rays",
        Mock(
            return_value=CameraRays(
                origin_points=origins,
                direction_vectors=directions,
                pixel_coordinates=pixels,
            )
        ),
    )
    monkeypatch.setattr(ray_tracer, "ray_test", query)
    ray_tracer.scene.camera_transform = np.eye(4)
    camera = Mock(spec=Camera)
    camera.camera_model = PinholeCameraModel.from_field_of_view(
        resolution=resolution, field_of_view=FieldOfView()
    )
    camera.camera_model = replace(
        camera.camera_model, focal_length_x=camera.camera_model.focal_length_y
    )
    camera.camera_range = CameraRange(minimum_distance=0.1, maximum_distance=8.0)
    camera.world_T_forward_view = HomogeneousTransformationMatrix()
    camera.root = Mock(global_transform=HomogeneousTransformationMatrix())
    camera.optical_R_forward_view = Camera.optical_R_forward_view
    return CameraIntersections(ray_tracer, camera, body, query, resolution)


# %% Depth conventions
@pytest.mark.parametrize("depth_mode", list(DepthMode))
def test_combined_render_uses_requested_depth_convention(
    camera_intersections: CameraIntersections, depth_mode: DepthMode
) -> None:
    """
    One query produces segmentation and the requested physical depth quantity.
    """
    fixture = camera_intersections
    frame = fixture.ray_tracer.create_segmentation_and_depth(
        fixture.camera.world_T_forward_view,
        resolution=fixture.resolution,
        min_distance=fixture.camera.camera_range.minimum_distance,
        max_distance=fixture.camera.camera_range.maximum_distance,
        depth_mode=depth_mode,
    )
    expected_depth = np.full(
        (fixture.resolution.width, fixture.resolution.height), -1.0, dtype=np.float32
    )
    expected_depth[0, 0] = 2.0
    expected_depth[1, 0] = 2.5 if depth_mode == DepthMode.RAY_DISTANCE else 2.0
    expected_segmentation = np.full(
        (fixture.resolution.width, fixture.resolution.height), -1, dtype=np.int32
    )
    expected_segmentation[:2, 0] = fixture.body.index
    np.testing.assert_array_equal(frame.depth, expected_depth)
    np.testing.assert_array_equal(frame.segmentation, expected_segmentation)
    fixture.query.assert_called_once()
    assert (
        fixture.query.call_args.kwargs["min_distance"]
        == fixture.camera.camera_range.minimum_distance
    )
    assert (
        fixture.query.call_args.kwargs["max_distance"]
        == fixture.camera.camera_range.maximum_distance
    )


@pytest.mark.parametrize("depth_mode", list(DepthMode))
def test_combined_render_handles_no_hits(
    camera_intersections: CameraIntersections, depth_mode: DepthMode
) -> None:
    """
    An empty intersection query produces only background pixels.
    """
    fixture = camera_intersections
    fixture.query.return_value = RayIntersections(
        hit_points=np.empty((0, 3)), ray_indices=np.array([], dtype=int), bodies=[]
    )
    frame = fixture.ray_tracer.create_segmentation_and_depth(
        fixture.camera.world_T_forward_view,
        resolution=fixture.resolution,
        depth_mode=depth_mode,
    )
    np.testing.assert_array_equal(
        frame.segmentation,
        np.full(
            (fixture.resolution.width, fixture.resolution.height), -1, dtype=np.int32
        ),
    )
    np.testing.assert_array_equal(
        frame.depth,
        np.full(
            (fixture.resolution.width, fixture.resolution.height),
            -1.0,
            dtype=np.float32,
        ),
    )


# %% RGB-D products


def test_semantic_rgbd_render_preserves_encodings(
    camera_intersections: CameraIntersections,
) -> None:
    """
    Semantic colors, optical depth, and body labels share the same hit query.
    """
    fixture = camera_intersections
    fixture.body.collision[0].color = Color(R=1.0, G=0.0, B=0.0)
    frame = fixture.ray_tracer.render(
        fixture.camera, rgb_mode=ColorRenderingMode.SEMANTIC
    )
    expected_color = np.zeros(
        (fixture.resolution.width, fixture.resolution.height, 3), dtype=np.uint8
    )
    expected_color[:2, 0] = [0, 0, 255]
    np.testing.assert_array_equal(frame.color_bgr, expected_color)
    expected_depth = np.zeros(
        (fixture.resolution.width, fixture.resolution.height), dtype=np.uint16
    )
    expected_depth[:2, 0] = RayTracer.convert_depth_to_millimeters(
        -fixture.query.return_value.hit_points[:, 2]
    )
    np.testing.assert_array_equal(frame.depth_in_millimeters, expected_depth)
    assert frame.object_color_map == {"255,0,0": fixture.body.name.name}
    assert frame.camera_model is fixture.camera.camera_model
    fixture.query.assert_called_once()


def test_render_derives_pinhole_calibration(
    camera_intersections: CameraIntersections,
) -> None:
    """
    A field-of-view camera renders with an equivalent effective pinhole model.
    """
    fixture = camera_intersections
    model = FieldOfViewCameraModel(
        angular_extent=FieldOfView(), image_resolution=fixture.resolution
    )
    fixture.camera.camera_model = model
    frame = fixture.ray_tracer.render(fixture.camera)
    expected = PinholeCameraModel.from_field_of_view(
        resolution=model.resolution, field_of_view=model.field_of_view
    )
    assert frame.camera_model == expected


def test_depth_conversion_preserves_millimeter_encoding() -> None:
    """
    Misses become zero while meter depths are rounded and saturated to unsigned
    millimeters.
    """
    depths = np.array([[-1.0, 0.0, 1.2346, 100.0]], dtype=np.float32)
    np.testing.assert_array_equal(
        RayTracer.convert_depth_to_millimeters(depths),
        np.array([[0, 0, 1235, 65535]], dtype=np.uint16),
    )


def test_mesh_render_preserves_rgb_channels(
    camera_intersections: CameraIntersections, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    Mesh images are published in BGR order with no semantic color map.
    """
    fixture = camera_intersections
    image = Image.new(
        "RGB", (fixture.resolution.width, fixture.resolution.height), (255, 0, 0)
    )
    image_bytes = io.BytesIO()
    image.save(image_bytes, format="PNG")
    monkeypatch.setattr(
        fixture.ray_tracer.scene,
        "save_image",
        Mock(return_value=image_bytes.getvalue()),
    )
    frame = fixture.ray_tracer.render(
        fixture.camera, rgb_mode=ColorRenderingMode.TEXTURED
    )
    expected = np.zeros(
        (fixture.resolution.width, fixture.resolution.height, 3), dtype=np.uint8
    )
    expected[:] = [0, 0, 255]
    np.testing.assert_array_equal(frame.color_bgr, expected)
    assert frame.object_color_map == {}
    assert (
        frame.color_bgr.shape[:2]
        == frame.depth_in_millimeters.shape
        == frame.segmentation.shape
    )


@pytest.mark.parametrize(
    "rendering_error",
    [NoSuchDisplayException, UnidentifiedImageError, OSError, ValueError],
)
def test_textured_render_propagates_backend_errors(
    camera_intersections: CameraIntersections,
    monkeypatch: pytest.MonkeyPatch,
    rendering_error: type[Exception],
) -> None:
    """
    Textured rendering reports backend failures without replacing the requested output.
    """
    fixture = camera_intersections
    monkeypatch.setattr(
        fixture.ray_tracer.scene, "save_image", Mock(side_effect=rendering_error)
    )
    with pytest.raises(rendering_error):
        fixture.ray_tracer.render(fixture.camera, rgb_mode=ColorRenderingMode.TEXTURED)


# %% Shared hit selection
def test_textured_render_generates_camera_rays_once(
    camera_intersections: CameraIntersections,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Textured rendering reuses the camera setup from the shared depth and label query.
    """
    fixture = camera_intersections
    monkeypatch.setattr(
        fixture.ray_tracer.scene, "save_image", Mock(side_effect=OSError)
    )
    with pytest.raises(OSError):
        fixture.ray_tracer.render(fixture.camera, rgb_mode=ColorRenderingMode.TEXTURED)
    fixture.ray_tracer.create_camera_rays.assert_called_once()
    fixture.query.assert_called_once()


@pytest.mark.parametrize("depth_mode", list(DepthMode))
def test_combined_render_uses_same_first_hit_for_depth_and_segmentation(
    camera_intersections: CameraIntersections,
    depth_mode: DepthMode,
) -> None:
    """
    Later hits on a viewing ray do not replace its visible body or depth.
    """
    fixture = camera_intersections
    other_body = next(
        body for body in fixture.ray_tracer.world.bodies if body is not fixture.body
    )
    fixture.query.return_value = RayIntersections(
        hit_points=np.array([[0.0, 0.0, -2.0], [0.0, 0.0, -4.0], [1.5, 0.0, -2.0]]),
        ray_indices=np.array([0, 0, 1]),
        bodies=[fixture.body, other_body, fixture.body],
    )
    frame = fixture.ray_tracer.create_segmentation_and_depth(
        fixture.camera.world_T_forward_view,
        resolution=fixture.resolution,
        depth_mode=depth_mode,
    )
    assert frame.segmentation[0, 0] == fixture.body.index
    assert frame.depth[0, 0] == 2.0


@pytest.mark.parametrize("depth_mode", list(DepthMode))
def test_depth_map_supports_explicit_depth_mode(
    camera_intersections: CameraIntersections,
    depth_mode: DepthMode,
) -> None:
    """
    The standalone depth API exposes both conventions through the shared renderer.
    """
    fixture = camera_intersections
    depth = fixture.ray_tracer.create_depth_map(
        fixture.camera.world_T_forward_view,
        resolution=fixture.resolution,
        depth_mode=depth_mode,
    )
    assert depth[1, 0] == (2.5 if depth_mode == DepthMode.RAY_DISTANCE else 2.0)


@pytest.mark.parametrize("depth_mode", list(DepthMode))
def test_rgbd_render_supports_explicit_depth_mode(
    camera_intersections: CameraIntersections,
    depth_mode: DepthMode,
) -> None:
    """
    RGB-D rendering applies the selected depth convention before millimeter encoding.
    """
    fixture = camera_intersections
    frame = fixture.ray_tracer.render(fixture.camera, depth_mode=depth_mode)
    assert frame.depth_in_millimeters[1, 0] == (
        RayTracer.convert_depth_to_millimeters(
            np.array(2.5 if depth_mode == DepthMode.RAY_DISTANCE else 2.0)
        )
    )


# %% Semantic body colors
@pytest.mark.parametrize("collision_color", [True, False])
def test_semantic_body_color_uses_available_geometry(
    camera_intersections: CameraIntersections, collision_color: bool
) -> None:
    """
    Collision or visual geometry supplies the same unsigned-byte semantic color.
    """
    body = Mock(spec=Body)
    geometry = Mock()
    geometry.color = Color(R=1.0, G=0.5, B=0.0)
    body.collision = [geometry] if collision_color else []
    body.visual = [] if collision_color else [geometry]
    np.testing.assert_array_equal(
        camera_intersections.ray_tracer.get_body_color(body),
        np.array([255, 128, 0], dtype=np.uint8),
    )


def test_semantic_body_color_defaults_to_gray(
    camera_intersections: CameraIntersections,
) -> None:
    """
    A body without geometry retains the neutral fallback color.
    """
    body = Mock(spec=Body)
    body.collision = []
    body.visual = []
    np.testing.assert_array_equal(
        camera_intersections.ray_tracer.get_body_color(body),
        np.round(
            np.asarray(camera_intersections.ray_tracer.default_body_color.to_rgb())
            * np.iinfo(np.uint8).max
        ).astype(np.uint8),
    )


# %% Intersection ordering


def test_camera_intersections_return_nearest_surface_first(
    ray_test_world: tuple[World, Body, Body, Body, Body],
) -> None:
    """
    The backend orders camera-ray intersections with the visible surface first.
    """
    world, front_body, back_body, first_sphere, second_sphere = ray_test_world
    for body, position in (
        (back_body, [1.0, 0.0, 0.0]),
        (first_sphere, [0.0, 10.0, 0.0]),
        (second_sphere, [0.0, -10.0, 0.0]),
    ):
        connection = world.get_connection(world.root, body)
        connection.origin = HomogeneousTransformationMatrix.from_xyz_rpy(
            x=position[0],
            y=position[1],
            z=position[2],
            reference_frame=world.root,
            child_frame=body,
        )
    ray_origin = np.array([[-2.5, 0.0, 0.0]])
    ray_intersections = world.ray_tracer.ray_test(
        ray_origin,
        np.array([[2.5, 0.0, 0.0]]),
        multiple_hits=True,
    )
    hit_points = ray_intersections.hit_points
    ray_indices = ray_intersections.ray_indices
    bodies = ray_intersections.bodies
    distances = np.linalg.norm(hit_points - ray_origin[ray_indices], axis=1)
    assert bodies[0] is front_body
    assert distances[0] == distances.min()


# %% Invalid ray queries


def test_ray_query_rejects_mismatched_point_counts(
    camera_intersections: CameraIntersections,
) -> None:
    """
    Mismatched ray origins and targets raise the specific spatial-input error.
    """
    fixture = camera_intersections
    del fixture.ray_tracer.ray_test
    origins = np.zeros((1, 3))
    targets = np.zeros((2, 3))
    with pytest.raises(RayPointShapeMismatch) as error:
        fixture.ray_tracer.ray_test(origins, targets)
    assert error.value.origin_shape == origins.shape
    assert error.value.target_shape == targets.shape


# %% Unsupported lens projection
@pytest.mark.parametrize("radial_coefficient", [0.0, 0.1])
def test_render_rejects_distorted_calibration(
    camera_intersections: CameraIntersections, radial_coefficient: float
) -> None:
    """
    Distortion metadata is never advertised on an undistorted rendered image.
    """
    fixture = camera_intersections
    fixture.camera.camera_model = replace(
        fixture.camera.camera_model,
        distortion=PlumbBobCameraDistortion(radial_coefficient_1=radial_coefficient),
    )
    with pytest.raises(UnsupportedCameraProjection):
        fixture.ray_tracer.render(fixture.camera)
    fixture.query.assert_not_called()


# %% Textured projection constraints
@pytest.mark.parametrize("off_center", [True, False])
def test_textured_render_rejects_incompatible_projection(
    camera_intersections: CameraIntersections,
    off_center: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    The centered textured viewer rejects off-center or unequal-focal calibration.
    """
    fixture = camera_intersections
    original = fixture.camera.camera_model
    monkeypatch.setattr(
        fixture.ray_tracer,
        "_render_textured_color",
        Mock(
            return_value=np.zeros(
                (fixture.resolution.width, fixture.resolution.height, 3), dtype=np.uint8
            )
        ),
    )
    fixture.camera.camera_model = replace(
        original,
        focal_length_y=(
            original.focal_length_x if off_center else original.focal_length_x * 2
        ),
        principal_point=Point2(
            x=float(original.principal_point.x) + (0.25 if off_center else 0.0),
            y=original.principal_point.y,
        ),
    )
    with pytest.raises(UnsupportedCameraProjection):
        fixture.ray_tracer.render(fixture.camera, rgb_mode=ColorRenderingMode.TEXTURED)
    fixture.query.assert_not_called()
