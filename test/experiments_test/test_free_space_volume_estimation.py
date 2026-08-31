import numpy as np
import pytest
import trimesh

from experiments.experiment_definitions import VolumeBound
from experiments.free_space_volume_estimation import (
    MonteCarloFreeSpaceSampler,
    ObstacleContainmentChecker,
)
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import FixedConnection
from semantic_digital_twin.world_description.geometry import (
    VolumetricBoundingBox,
    Box,
    Mesh,
    Scale,
    Shape,
)
from semantic_digital_twin.world_description.world_entity import Body


def _search_space_box(world: World, half_extent: float = 5.0) -> VolumetricBoundingBox:
    return VolumetricBoundingBox(
        min_x=-half_extent,
        min_y=-half_extent,
        min_z=-half_extent,
        max_x=half_extent,
        max_y=half_extent,
        max_z=half_extent,
        origin=HomogeneousTransformationMatrix(reference_frame=world.root),
    )


def _world_with_box_obstacle(scale: Scale) -> tuple[World, Shape]:
    world = World()
    with world.modify_world():
        root = Body(name=PrefixedName("map"))
        world.add_kinematic_structure_entity(root)
        obstacle = Body(name=PrefixedName("obstacle"))
        world.add_connection(FixedConnection.create_with_dofs(world, root, obstacle))
        obstacle.collision.append(Box(scale=scale))
    return world, obstacle.collision.shapes[0]


# %% ObstacleContainmentChecker


def test_watertight_mesh_containment_matches_the_real_geometry():
    world, shape = _world_with_box_obstacle(Scale(2.0, 2.0, 2.0))
    containment_checker = ObstacleContainmentChecker.for_shape(shape, world.root)

    points = np.array([[0.0, 0.0, 0.0], [10.0, 10.0, 10.0]])

    np.testing.assert_array_equal(containment_checker.contains(points), [True, False])


def _non_watertight_ring_mesh() -> trimesh.Trimesh:
    """
    A ring with one face removed: still spans the same bounding box (including its hole)
    as a proper annulus, but is not a closed manifold, so
    :meth:`trimesh.Trimesh.contains` cannot be trusted on it.
    """
    ring = trimesh.creation.annulus(r_min=0.3, r_max=0.5, height=0.2)
    open_ring = ring.copy()
    open_ring.update_faces(np.arange(len(open_ring.faces) - 2))
    open_ring.remove_unreferenced_vertices()
    assert not open_ring.is_watertight
    return open_ring


def test_non_watertight_mesh_is_convex_decomposed_instead_of_using_its_bounding_box(
    tmp_path,
):
    world = World()
    with world.modify_world():
        root = Body(name=PrefixedName("map"))
        world.add_kinematic_structure_entity(root)
        obstacle = Body(name=PrefixedName("ring"))
        world.add_connection(FixedConnection.create_with_dofs(world, root, obstacle))
        obstacle.collision.append(
            Mesh.from_trimesh(
                mesh=_non_watertight_ring_mesh(), directory=tmp_path, file_type="stl"
            )
        )
    shape = obstacle.collision.shapes[0]
    containment_checker = ObstacleContainmentChecker.for_shape(shape, world.root)

    # Unlike a bounding-box fallback, convex decomposition does not fill in the ring's
    # own hole at the center, so a point there is correctly reported as free.
    center_hole_and_outside = np.array([[0.0, 0.0, 0.0], [10.0, 10.0, 10.0]])

    np.testing.assert_array_equal(
        containment_checker.contains(center_hole_and_outside), [False, False]
    )

    # A point actually inside the ring's material is still reported as occupied.
    inside_ring_material = np.array([[0.4, 0.0, 0.0]])
    assert containment_checker.contains(inside_ring_material)[0]


def test_decomposition_yielding_no_pieces_falls_back_to_the_bounding_box(tmp_path):
    class _EmptyMeshDecomposer:
        def apply_to_shape(self, shape):
            return []

    world = World()
    with world.modify_world():
        root = Body(name=PrefixedName("map"))
        world.add_kinematic_structure_entity(root)
        obstacle = Body(name=PrefixedName("ring"))
        world.add_connection(FixedConnection.create_with_dofs(world, root, obstacle))
        obstacle.collision.append(
            Mesh.from_trimesh(
                mesh=_non_watertight_ring_mesh(), directory=tmp_path, file_type="stl"
            )
        )
    shape = obstacle.collision.shapes[0]
    containment_checker = ObstacleContainmentChecker.for_shape(
        shape, world.root, mesh_decomposer=_EmptyMeshDecomposer()
    )

    center_and_outside = np.array([[0.0, 0.0, 0.0], [10.0, 10.0, 10.0]])

    np.testing.assert_array_equal(
        containment_checker.contains(center_and_outside), [True, False]
    )


# %% MonteCarloFreeSpaceSampler.wilson_score_interval


def test_wilson_score_interval_matches_known_reference_values():
    lower, upper = MonteCarloFreeSpaceSampler.wilson_score_interval(
        success_count=50, total=100, confidence_level=0.95
    )

    assert lower == pytest.approx(0.4038, abs=1e-3)
    assert upper == pytest.approx(0.5963, abs=1e-3)


def test_wilson_score_interval_clips_zero_successes_at_zero():
    lower, _ = MonteCarloFreeSpaceSampler.wilson_score_interval(
        success_count=0, total=100, confidence_level=0.95
    )

    assert lower == 0.0


def test_wilson_score_interval_clips_all_successes_at_one():
    _, upper = MonteCarloFreeSpaceSampler.wilson_score_interval(
        success_count=100, total=100, confidence_level=0.95
    )

    assert upper == 1.0


# %% MonteCarloFreeSpaceSampler.free_volume_bound


def test_free_volume_bound_of_an_empty_scene_brackets_the_full_search_space():
    world = World()
    with world.modify_world():
        root = Body(name=PrefixedName("map"))
        world.add_kinematic_structure_entity(root)
    sampler = MonteCarloFreeSpaceSampler(
        search_space_bounding_box=_search_space_box(world),
        obstacle_containment_checkers=[],
        sample_count=2000,
        random_seed=0,
    )

    bound = sampler.free_volume_bound()

    assert isinstance(bound, VolumeBound)
    assert bound.upper == pytest.approx(sampler.search_space_volume)
    assert bound.lower > 0.9 * sampler.search_space_volume


def test_free_volume_bound_of_a_fully_occupied_scene_is_zero_at_the_lower_end():
    world, shape = _world_with_box_obstacle(Scale(10.0, 10.0, 10.0))
    sampler = MonteCarloFreeSpaceSampler(
        search_space_bounding_box=_search_space_box(world),
        obstacle_containment_checkers=[
            ObstacleContainmentChecker.for_shape(shape, world.root)
        ],
        sample_count=2000,
        random_seed=0,
    )

    bound = sampler.free_volume_bound()

    assert bound.lower == pytest.approx(0.0, abs=1e-8)
    assert bound.upper < 0.1 * sampler.search_space_volume


def test_sample_points_are_deterministic_for_a_fixed_seed():
    world = World()
    with world.modify_world():
        root = Body(name=PrefixedName("map"))
        world.add_kinematic_structure_entity(root)
    search_space = _search_space_box(world)
    first = MonteCarloFreeSpaceSampler(
        search_space_bounding_box=search_space,
        obstacle_containment_checkers=[],
        sample_count=50,
        random_seed=42,
    )
    second = MonteCarloFreeSpaceSampler(
        search_space_bounding_box=search_space,
        obstacle_containment_checkers=[],
        sample_count=50,
        random_seed=42,
    )

    np.testing.assert_array_equal(first.sample_points, second.sample_points)
