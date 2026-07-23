import numpy as np
import pytest

from semantic_digital_twin.exceptions import InvalidCameraResolutionError
from semantic_digital_twin.spatial_computations.raytracer import (
    CameraResolution,
    RayTracer,
)
from semantic_digital_twin.spatial_types.spatial_types import (
    HomogeneousTransformationMatrix,
)
from semantic_digital_twin.testing import ray_test_world


def test_create_segmentation_mask(ray_test_world):
    world, body1, body2, body3, body4 = ray_test_world
    rt = RayTracer(world)
    rt.update_scene()

    camera_pose = np.array(
        [
            [1.0, 0.0, 0.0, -2.5],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.2],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )

    world.get_connection(world.root, body2).origin = np.array(
        [[1, 0, 0, -1], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
    )

    seg = rt.create_segmentation_mask(
        HomogeneousTransformationMatrix(camera_pose, reference_frame=world.root),
        resolution=256,
    )
    assert seg.shape == (256, 256)  # Assuming a standard resolution

    hit, index, body = rt.ray_test(np.array([1, 0, 1]), np.array([-1, 0, 1]))
    assert hit is not None
    assert index is not None
    assert body is not None
    assert body1.index in seg
    assert body2.index in seg


def test_create_depth_map(ray_test_world):
    world, body1, body2, body3, body4 = ray_test_world
    rt = RayTracer(world)
    rt.update_scene()

    camera_pose = np.array(
        [
            [1.0, 0.0, 0.0, -2.5],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )

    depth_map = rt.create_depth_map(
        HomogeneousTransformationMatrix(camera_pose, reference_frame=world.root),
        resolution=512,
    )
    assert depth_map is not None
    assert depth_map[0, 0] == -1  # Assuming no objects are hit at the upper left corner
    assert depth_map.shape == (512, 512)
    assert depth_map.max() <= 2.5
    assert depth_map[depth_map > 0].min() >= 2.375


def test_create_camera_rays_with_rectangular_resolution(ray_test_world):
    world, *_ = ray_test_world
    ray_tracer = RayTracer(world)

    camera_pose = np.eye(4, dtype=float)
    camera_pose[:3, 3] = [-2.5, 0.0, 0.0]

    resolution = CameraResolution(width=128, height=64)
    ray_origins, ray_directions, pixels = ray_tracer.create_camera_rays(
        HomogeneousTransformationMatrix(camera_pose, reference_frame=world.root),
        resolution=resolution,
    )

    assert resolution.shape == (resolution.width, resolution.height)
    assert ray_origins.shape == (resolution.width * resolution.height, 3)
    assert ray_directions.shape == (resolution.width * resolution.height, 3)
    assert pixels.shape == (resolution.width * resolution.height, 2)
    assert set(pixels[:, 0]) == set(range(resolution.width))
    assert set(pixels[:, 1]) == set(range(resolution.height))


def test_create_segmentation_mask_with_rectangular_resolution(ray_test_world):
    world, *_ = ray_test_world
    ray_tracer = RayTracer(world)

    camera_pose = np.array(
        [
            [1.0, 0.0, 0.0, -2.5],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.2],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )

    segmentation_mask = ray_tracer.create_segmentation_mask(
        HomogeneousTransformationMatrix(camera_pose, reference_frame=world.root),
        resolution=(128, 64),
    )

    assert segmentation_mask.shape == (128, 64)


def test_create_depth_map_with_rectangular_resolution(ray_test_world):
    world, *_ = ray_test_world
    ray_tracer = RayTracer(world)

    camera_pose = np.array(
        [
            [1.0, 0.0, 0.0, -2.5],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )

    depth_map = ray_tracer.create_depth_map(
        HomogeneousTransformationMatrix(camera_pose, reference_frame=world.root),
        resolution=(128, 64),
    )

    assert depth_map.shape == (128, 64)


def test_camera_resolution_can_be_created_from_public_resolution_values():
    square_resolution = CameraResolution.from_integral(128)
    rectangular_resolution = CameraResolution.from_iterable((128, 64))

    assert square_resolution == CameraResolution(width=128, height=128)
    assert rectangular_resolution == CameraResolution(width=128, height=64)


@pytest.mark.parametrize("resolution", [0, (128,), (128.0, 64), "128x64"])
def test_invalid_camera_resolution_raises_domain_exception(resolution):
    with pytest.raises(InvalidCameraResolutionError):
        CameraResolution.from_value(resolution)


def test_ray_test(ray_test_world):
    world, body1, body2, body3, body4 = ray_test_world
    rt = RayTracer(world)
    rt.update_scene()

    hit, index, body = rt.ray_test(np.array([1, 0, 0.1]), np.array([-1, 0, 0.1]))
    assert hit is not None
    assert index is not None
    assert body is not None
    assert body1 in body

    # Test with a ray that does not hit any object
    hit, index, body = rt.ray_test(np.array([10, 10, 10]), np.array([20, 20, 20]))
    assert not np.any(hit)
    assert not np.any(index)
    assert not np.any(body)


def test_ray_test_batch(ray_test_world):
    world, body1, body2, body3, body4 = ray_test_world
    world.get_connection(world.root, body1).origin = np.array(
        [[1, 0, 0, 1], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
    )
    world.get_connection(world.root, body2).origin = np.array(
        [[1, 0, 0, -1], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
    )

    rt = RayTracer(world)
    rt.update_scene()

    rays = np.array([[1, 1, 0.1], [-1, 1, 0.1]])
    targets = np.array([[1, -1, 0.1], [-1, -1, 0.1]])

    hits, indices, bodies = rt.ray_test(rays, targets)
    assert hits is not None
    assert indices is not None
    assert bodies is not None
    assert len(hits) == len(rays)
    assert len(indices) == len(rays)
    assert len(bodies) == len(rays)
    # Test return
    assert bodies[0] == body1
    assert bodies[1] == body2


def test_min_distance(ray_test_world):
    world, body1, body2, body3, body4 = ray_test_world
    world.get_connection(world.root, body1).origin = np.array(
        [[1, 0, 0, 0.5], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
    )
    world.get_connection(world.root, body2).origin = np.array(
        [[1, 0, 0, -1], [0, 1, 0, 0], [0, 0, 1, 10], [0, 0, 0, 1]]
    )

    rt = RayTracer(world)
    rt.update_scene()

    rays = np.array([[0, 0, 0.1], [-1, 0, 0.1]])
    targets = np.array([[1, 0, 0.1], [1, 0, 0.1]])

    hits, indices, bodies = rt.ray_test(rays, targets, min_distance=1)

    assert len(hits) == 1
    assert bodies[0] == body1


def test_max_distance(ray_test_world):
    world, body1, body2, body3, body4 = ray_test_world
    world.get_connection(world.root, body1).origin = np.array(
        [[1, 0, 0, 1.5], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
    )
    world.get_connection(world.root, body2).origin = np.array(
        [[1, 0, 0, -1], [0, 1, 0, 0], [0, 0, 1, 10], [0, 0, 0, 1]]
    )

    rt = RayTracer(world)
    rt.update_scene()

    rays = np.array([[0, 0, 0.1], [-1, 0, 0.1]])
    targets = np.array([[2, 0, 0.1], [2, 0, 0.1]])

    hits, indices, bodies = rt.ray_test(rays, targets)

    assert len(hits) == 2

    hits, indices, bodies = rt.ray_test(rays, targets, max_distance=1)

    assert len(hits) == 0
