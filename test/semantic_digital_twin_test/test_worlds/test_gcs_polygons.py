import numpy as np
import pytest
import trimesh
from pydrake.geometry.optimization import HPolyhedron, VPolytope

from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.exceptions import PointOccupiedError
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix, Point3
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import FixedConnection
from semantic_digital_twin.world_description.geometry import (
    VolumetricBoundingBox,
    Box,
    Mesh,
    Scale,
)
from semantic_digital_twin.world_description.graph_of_convex_sets.base import (
    GraphOfConvexSets,
)
from semantic_digital_twin.world_description.graph_of_convex_sets.polygons import (
    GraphOfConvexPolygons,
    IrisSeedingSettings,
)
from semantic_digital_twin.world_description.graph_of_convex_sets.exceptions import (
    UnboundedSearchSpaceError,
)
from semantic_digital_twin.world_description.shape_collection import (
    BoundingBoxCollection,
)
from semantic_digital_twin.world_description.world_entity import Body


@pytest.fixture
def unit_box_world() -> World:
    world = World()
    with world.modify_world():
        root = Body(name=PrefixedName("map"))
        world.add_kinematic_structure_entity(root)
        obstacle = Body(name=PrefixedName("unit_cube"))
        world.add_connection(FixedConnection.create_with_dofs(world, root, obstacle))
        obstacle.collision.append(Box(scale=Scale(1.0, 1.0, 1.0)))
    return world


@pytest.fixture
def unit_box_search_space(unit_box_world: World) -> BoundingBoxCollection:
    return BoundingBoxCollection(
        [
            VolumetricBoundingBox(
                min_x=-2,
                max_x=3,
                min_y=-2,
                max_y=3,
                min_z=-0.5,
                max_z=0.5,
                origin=HomogeneousTransformationMatrix(
                    reference_frame=unit_box_world.root
                ),
            )
        ],
        unit_box_world.root,
    )


@pytest.fixture
def unit_box_graph_of_convex_polygons(
    unit_box_world: World, unit_box_search_space: BoundingBoxCollection
) -> GraphOfConvexPolygons:
    return GraphOfConvexPolygons.from_world(unit_box_world, unit_box_search_space)


@pytest.fixture
def non_convex_ring_world(tmp_path) -> World:
    """
    A world whose single obstacle is a non-convex ring (an annulus has a hole, so its
    convex hull is not itself), used to exercise the bounding-box fallback in :func:`~se
    mantic_digital_twin.world_description.graph_of_convex_sets.polygons._shape_to_convex
    _set`.
    """
    world = World()
    with world.modify_world():
        root = Body(name=PrefixedName("map"))
        world.add_kinematic_structure_entity(root)
        obstacle = Body(name=PrefixedName("ring"))
        world.add_connection(FixedConnection.create_with_dofs(world, root, obstacle))
        ring_mesh = trimesh.creation.annulus(r_min=0.3, r_max=0.5, height=0.2)
        assert not ring_mesh.is_convex
        obstacle.collision.append(
            Mesh.from_trimesh(mesh=ring_mesh, directory=tmp_path, file_type="stl")
        )
    return world


@pytest.fixture
def non_convex_ring_search_space(non_convex_ring_world: World) -> BoundingBoxCollection:
    return BoundingBoxCollection(
        [
            VolumetricBoundingBox(
                min_x=-2,
                max_x=3,
                min_y=-2,
                max_y=3,
                min_z=-0.5,
                max_z=0.5,
                origin=HomogeneousTransformationMatrix(
                    reference_frame=non_convex_ring_world.root
                ),
            )
        ],
        non_convex_ring_world.root,
    )


class TestGraphOfConvexPolygons:
    def test_is_a_graph_of_convex_sets(
        self, unit_box_graph_of_convex_polygons: GraphOfConvexPolygons
    ):
        assert isinstance(unit_box_graph_of_convex_polygons, GraphOfConvexSets)

    def test_from_world_grows_at_least_one_region(
        self, unit_box_graph_of_convex_polygons: GraphOfConvexPolygons
    ):
        assert unit_box_graph_of_convex_polygons.region_count > 0
        assert (
            len(unit_box_graph_of_convex_polygons.regions)
            == unit_box_graph_of_convex_polygons.region_count
        )

    def test_no_region_contains_the_obstacle(
        self, unit_box_graph_of_convex_polygons: GraphOfConvexPolygons
    ):
        for region in unit_box_graph_of_convex_polygons.regions:
            assert not region.PointInSet(np.array([0.0, 0.0, 0.0]))

    def test_path_from_to_finds_a_path_around_the_cube(
        self,
        unit_box_world: World,
        unit_box_graph_of_convex_polygons: GraphOfConvexPolygons,
    ):
        start = Point3(-1.5, -1.5, 0.0, reference_frame=unit_box_world.root)
        goal = Point3(2.5, 2.5, 0.0, reference_frame=unit_box_world.root)

        path = unit_box_graph_of_convex_polygons.path_from_to(start, goal)

        assert path is not None
        assert path[0] is start
        assert path[-1] is goal
        assert len(path) >= 2

    def test_path_from_to_raises_for_an_occupied_start(
        self,
        unit_box_world: World,
        unit_box_graph_of_convex_polygons: GraphOfConvexPolygons,
    ):
        occupied = Point3(0.0, 0.0, 0.0, reference_frame=unit_box_world.root)
        free = Point3(2.5, 2.5, 0.0, reference_frame=unit_box_world.root)
        with pytest.raises(PointOccupiedError):
            unit_box_graph_of_convex_polygons.path_from_to(occupied, free)

    def test_path_from_to_raises_for_an_occupied_goal(
        self,
        unit_box_world: World,
        unit_box_graph_of_convex_polygons: GraphOfConvexPolygons,
    ):
        occupied = Point3(0.0, 0.0, 0.0, reference_frame=unit_box_world.root)
        free = Point3(-1.5, -1.5, 0.0, reference_frame=unit_box_world.root)
        with pytest.raises(PointOccupiedError):
            unit_box_graph_of_convex_polygons.path_from_to(free, occupied)

    def test_repeated_queries_reuse_the_persistent_region_subgraph(
        self,
        unit_box_world: World,
        unit_box_graph_of_convex_polygons: GraphOfConvexPolygons,
    ):
        start = Point3(-1.5, -1.5, 0.0, reference_frame=unit_box_world.root)
        goal = Point3(2.5, 2.5, 0.0, reference_frame=unit_box_world.root)

        for start, goal in [(start, goal), (goal, start), (start, goal)]:
            path = unit_box_graph_of_convex_polygons.path_from_to(start, goal)
            assert path is not None
            # region subgraph + one source + one target subgraph, regardless of how
            # many queries have already been solved.
            assert (
                len(
                    unit_box_graph_of_convex_polygons._trajectory_optimization.GetSubgraphs()
                )
                == 3
            )

    def test_from_world_rejects_a_default_unbounded_search_space(
        self, unit_box_world: World
    ):
        with pytest.raises(UnboundedSearchSpaceError):
            GraphOfConvexPolygons.from_world(unit_box_world, None)

    def test_from_world_rejects_a_multi_box_search_space(self, unit_box_world: World):
        origin = HomogeneousTransformationMatrix(reference_frame=unit_box_world.root)
        two_boxes = BoundingBoxCollection(
            [
                VolumetricBoundingBox(-2, -2, -0.5, -1, -1, 0.5, origin),
                VolumetricBoundingBox(1, 1, -0.5, 2, 2, 0.5, origin),
            ],
            unit_box_world.root,
        )
        with pytest.raises(UnboundedSearchSpaceError):
            GraphOfConvexPolygons.from_world(unit_box_world, two_boxes)

    def test_extra_seed_points_are_seeded_before_the_coverage_grid(
        self, unit_box_world: World, unit_box_search_space: BoundingBoxCollection
    ):
        far_corner = Point3(-1.9, -1.9, 0.0, reference_frame=unit_box_world.root)
        graph_of_convex_polygons = GraphOfConvexPolygons.from_world(
            unit_box_world,
            unit_box_search_space,
            seeding_settings=IrisSeedingSettings(grid_resolution=1, max_regions=1),
            extra_seed_points=[far_corner],
        )
        assert graph_of_convex_polygons.region_count == 1
        assert graph_of_convex_polygons.regions[0].PointInSet(
            np.array([-1.9, -1.9, 0.0])
        )

    def test_convex_obstacle_uses_an_exact_vpolytope(
        self, unit_box_graph_of_convex_polygons: GraphOfConvexPolygons
    ):
        assert len(unit_box_graph_of_convex_polygons.obstacles) == 1
        assert isinstance(unit_box_graph_of_convex_polygons.obstacles[0], VPolytope)

    def test_non_convex_obstacle_falls_back_to_a_bounding_box(
        self,
        non_convex_ring_world: World,
        non_convex_ring_search_space: BoundingBoxCollection,
    ):
        graph_of_convex_polygons = GraphOfConvexPolygons.from_world(
            non_convex_ring_world, non_convex_ring_search_space
        )
        assert len(graph_of_convex_polygons.obstacles) == 1
        assert isinstance(graph_of_convex_polygons.obstacles[0], HPolyhedron)
        # The ring's bounding box spans roughly [-0.5, 0.5] in x/y; its own hole at the
        # center is not carved out by the bounding-box fallback, so the center must
        # still be treated as occupied.
        center = Point3(0.0, 0.0, 0.0, reference_frame=non_convex_ring_world.root)
        with pytest.raises(PointOccupiedError):
            graph_of_convex_polygons.path_from_to(
                center,
                Point3(2.5, 2.5, 0.0, reference_frame=non_convex_ring_world.root),
            )
