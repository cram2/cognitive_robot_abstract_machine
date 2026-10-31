import tracemalloc

import numpy as np
import pytest
import trimesh
from PIL import Image
from trimesh.visual import TextureVisuals
from trimesh.visual.material import SimpleMaterial

from semantic_digital_twin.datastructures.camera_resolution import CameraResolution
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.exceptions import InvalidCameraResolutionError
from semantic_digital_twin.spatial_computations.raytracer import RayTracer
from semantic_digital_twin.spatial_types.spatial_types import (
    HomogeneousTransformationMatrix,
)
from semantic_digital_twin.testing import ray_test_world
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import FixedConnection
from semantic_digital_twin.world_description.geometry import Mesh
from semantic_digital_twin.world_description.shape_collection import ShapeCollection
from semantic_digital_twin.world_description.world_entity import Body


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

    world.get_connection(world.root, body2).origin = HomogeneousTransformationMatrix(
        np.array([[1, 0, 0, -1], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]),
        reference_frame=world.root,
        child_frame=body2,
    )

    resolution = CameraResolution(width=256, height=256)
    seg = rt.create_segmentation_mask(
        HomogeneousTransformationMatrix(camera_pose, reference_frame=world.root),
        resolution=resolution,
    )
    assert seg.shape == (resolution.width, resolution.height)

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

    resolution = CameraResolution(width=512, height=512)
    depth_map = rt.create_depth_map(
        HomogeneousTransformationMatrix(camera_pose, reference_frame=world.root),
        resolution=resolution,
    )
    assert depth_map is not None
    assert depth_map[0, 0] == -1  # Assuming no objects are hit at the upper left corner
    assert depth_map.shape == (resolution.width, resolution.height)
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

    resolution = CameraResolution(width=128, height=64)
    segmentation_mask = ray_tracer.create_segmentation_mask(
        HomogeneousTransformationMatrix(camera_pose, reference_frame=world.root),
        resolution=resolution,
    )

    assert segmentation_mask.shape == (resolution.width, resolution.height)


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

    resolution = CameraResolution(width=128, height=64)
    depth_map = ray_tracer.create_depth_map(
        HomogeneousTransformationMatrix(camera_pose, reference_frame=world.root),
        resolution=resolution,
    )

    assert depth_map.shape == (resolution.width, resolution.height)


def test_camera_resolution_defaults_to_square_resolution():
    resolution = CameraResolution()

    assert resolution == CameraResolution(width=512, height=512)


@pytest.mark.parametrize(
    ("width", "height"),
    [(0, 64), (-1, 64), (64, 0), (64, -1)],
)
def test_non_positive_camera_resolution_raises_domain_exception(width, height):
    with pytest.raises(InvalidCameraResolutionError):
        CameraResolution(width=width, height=height)


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
    world.get_connection(world.root, body1).origin = HomogeneousTransformationMatrix(
        np.array([[1, 0, 0, 1], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]),
        reference_frame=world.root,
        child_frame=body1,
    )
    world.get_connection(world.root, body2).origin = HomogeneousTransformationMatrix(
        np.array([[1, 0, 0, -1], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]),
        reference_frame=world.root,
        child_frame=body2,
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
    world.get_connection(world.root, body1).origin = HomogeneousTransformationMatrix(
        np.array([[1, 0, 0, 0.5], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]),
        reference_frame=world.root,
        child_frame=body1,
    )
    world.get_connection(world.root, body2).origin = HomogeneousTransformationMatrix(
        np.array([[1, 0, 0, -1], [0, 1, 0, 0], [0, 0, 1, 10], [0, 0, 0, 1]]),
        reference_frame=world.root,
        child_frame=body2,
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
    world.get_connection(world.root, body1).origin = HomogeneousTransformationMatrix(
        np.array([[1, 0, 0, 1.5], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]),
        reference_frame=world.root,
        child_frame=body1,
    )
    world.get_connection(world.root, body2).origin = HomogeneousTransformationMatrix(
        np.array([[1, 0, 0, -1], [0, 1, 0, 0], [0, 0, 1, 10], [0, 0, 0, 1]]),
        reference_frame=world.root,
        child_frame=body2,
    )

    rt = RayTracer(world)
    rt.update_scene()

    rays = np.array([[0, 0, 0.1], [-1, 0, 0.1]])
    targets = np.array([[2, 0, 0.1], [2, 0, 0.1]])

    hits, indices, bodies = rt.ray_test(rays, targets)

    assert len(hits) == 2

    hits, indices, bodies = rt.ray_test(rays, targets, max_distance=1)

    assert len(hits) == 0


# %% casting rays into a textured world

TEXTURE_SIZE = 1024
"""
The edge length, in pixels, of the texture painted on each box of a textured world.
"""


def _textured_box_body(name: str, seed: int) -> Body:
    """
    :return: A body whose collision mesh is a unit box painted with a texture of its
        own.
    """
    body = Body(name=PrefixedName(name))
    box = trimesh.creation.box(extents=(1.0, 1.0, 1.0))
    generator = np.random.default_rng(seed)
    box.visual = TextureVisuals(
        uv=generator.random((len(box.vertices), 2)),
        material=SimpleMaterial(
            image=Image.fromarray(
                generator.integers(
                    0, 255, (TEXTURE_SIZE, TEXTURE_SIZE, 3), dtype=np.uint8
                )
            )
        ),
    )
    body.collision = ShapeCollection(
        [
            Mesh.from_trimesh(
                mesh=box,
                origin=HomogeneousTransformationMatrix.from_xyz_rpy(
                    reference_frame=body
                ),
            )
        ],
        reference_frame=body,
    )
    return body


@pytest.fixture
def textured_world() -> World:
    """
    :return: A world of two textured boxes standing 2 m apart along x.
    """
    world = World()
    root = Body(name=PrefixedName("root"))
    with world.modify_world():
        world.add_kinematic_structure_entity(root)
        for index in range(2):
            box = _textured_box_body(f"box_{index}", seed=index)
            world.add_connection(
                FixedConnection(
                    parent=root,
                    child=box,
                    parent_T_connection_expression=HomogeneousTransformationMatrix.from_xyz_rpy(
                        x=2.0 * index, reference_frame=root
                    ),
                )
            )
    return world


def test_a_ray_hits_the_textured_box_it_points_at(textured_world):
    ray_tracer = RayTracer(textured_world)

    _, _, bodies = ray_tracer.ray_test(
        np.array([2.0, 0.0, 2.0]), np.array([2.0, 0.0, -2.0])
    )

    assert bodies == [textured_world.get_body_by_name("box_1")]


def test_casting_rays_does_not_decode_the_textures_of_the_world(textured_world):
    """
    A ray only meets geometry, so casting it must not unpack a texture.
    """
    ray_tracer = RayTracer(textured_world)
    origins = np.array([[x, 0.0, 2.0] for x in np.linspace(-1.0, 3.0, 50)])
    targets = origins - np.array([0.0, 0.0, 4.0])

    tracemalloc.start()
    ray_tracer.ray_test(origins, targets)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    assert peak < TEXTURE_SIZE * TEXTURE_SIZE * 3
