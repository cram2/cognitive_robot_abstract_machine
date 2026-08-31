import os.path

import numpy as np
import plotly.graph_objects as go
import pytest
from dataclasses import dataclass

from numpy import allclose

from krrood.entity_query_language.exceptions import NotNumberLikeFieldError
from krrood.entity_query_language.factories import a, entity, set_of, variable
from random_events.interval import SimpleInterval
from random_events.product_algebra import SimpleEvent
from semantic_digital_twin.adapters.mjcf import MJCFParser
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.datastructures.variables import SpatialVariables
from semantic_digital_twin.exceptions import PointOccupiedError
from semantic_digital_twin.semantic_annotations.semantic_annotations import Floor
from semantic_digital_twin.spatial_types import Point2, Point3, Pose
from semantic_digital_twin.spatial_types.spatial_types import (
    HomogeneousTransformationMatrix,
)
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import FixedConnection
from semantic_digital_twin.world_description.geometry import (
    VolumetricBoundingBox,
    PlanarBoundingBox,
    Box,
    Scale,
)
from semantic_digital_twin.world_description.graph_of_convex_sets.base import (
    create_reference_frame_with_only_yaw_from_body,
)
from semantic_digital_twin.world_description.graph_of_convex_sets.boxes import (
    PlanarGraphOfBoundingBoxes,
    VolumetricGraphOfBoundingBoxes,
)
from semantic_digital_twin.world_description.graph_of_convex_sets.exceptions import (
    AmbiguousSelectedVariableError,
)
from semantic_digital_twin.world_description.shape_collection import (
    BoundingBoxCollection,
    ShapeCollection,
)
from semantic_digital_twin.world_description.world_entity import Body, Region

MJCF_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..",
    "..",
    "..",
    "semantic_digital_twin",
    "resources",
    "mjcf",
)


@dataclass
class VolumetricGraphOfBoundingBoxesFixture:
    """
    Data class for Graph of Convex Sets test fixture.
    """

    world: World
    graph_of_convex_sets: VolumetricGraphOfBoundingBoxes


@pytest.fixture
def graph_of_convex_sets_unit_box() -> VolumetricGraphOfBoundingBoxesFixture:
    """
    Create a VolumetricGraphOfBoundingBoxes for navigation around a unit box.
    """
    world = World()
    with world.modify_world():
        world.add_kinematic_structure_entity(Body())

    graph_of_convex_sets = VolumetricGraphOfBoundingBoxes(world)

    obstacle = VolumetricBoundingBox(0, 0, 0, 1, 1, 1, world.root.global_pose)

    z_lim = SimpleInterval.from_data(0.45, 0.55)
    x_lim = SimpleInterval.from_data(-2, 3)
    y_lim = SimpleInterval.from_data(-2, 3)
    limiting_event = SimpleEvent.from_data(
        {
            SpatialVariables.x.value: x_lim,
            SpatialVariables.y.value: y_lim,
            SpatialVariables.z.value: z_lim,
        }
    )
    obstacles = BoundingBoxCollection.from_event(
        VolumetricBoundingBox,
        world.root,
        ~obstacle.simple_event.as_composite_set() & limiting_event.as_composite_set(),
    )
    for bounding_box in obstacles:
        graph_of_convex_sets.add_node(bounding_box)

    graph_of_convex_sets.calculate_connectivity()
    return VolumetricGraphOfBoundingBoxesFixture(world, graph_of_convex_sets)


def test_reachability(
    graph_of_convex_sets_unit_box: VolumetricGraphOfBoundingBoxesFixture,
):
    """
    Verify if a path can be found around the unit box.
    """
    start_point = Point3(
        -1, -1, 0.5, reference_frame=graph_of_convex_sets_unit_box.world.root
    )
    target_point = Point3(
        2, 2, 0.5, reference_frame=graph_of_convex_sets_unit_box.world.root
    )

    path = graph_of_convex_sets_unit_box.graph_of_convex_sets.path_from_to(
        start_point, target_point
    )
    # Shortcutting collapses the two waypoints hugging the box's corner into one,
    # since a straight line from start_point to (0, 2, 0.5) already clears the box.
    assert len(path) == 3


def test_plot(graph_of_convex_sets_unit_box: VolumetricGraphOfBoundingBoxesFixture):
    """
    Verify if the free and occupied space can be plotted.
    """
    free_space_plot = go.Figure(
        graph_of_convex_sets_unit_box.graph_of_convex_sets.plot_free_space()
    )
    assert free_space_plot is not None
    occupied_space_plot = go.Figure(
        graph_of_convex_sets_unit_box.graph_of_convex_sets.plot_occupied_space()
    )
    assert occupied_space_plot is not None


def test_from_world(table_world: World):
    """
    Verify the generation of a connectivity graph from a world.
    """
    search_space = BoundingBoxCollection(
        [
            VolumetricBoundingBox(
                min_x=-5,
                max_x=-2,
                min_y=-1,
                max_y=2,
                min_z=0,
                max_z=2,
                origin=HomogeneousTransformationMatrix(
                    reference_frame=table_world.root
                ),
            )
        ],
        table_world.root,
    )
    graph_of_convex_sets = VolumetricGraphOfBoundingBoxes.free_space_from_world(
        table_world, search_space=search_space
    )
    assert graph_of_convex_sets is not None
    assert len(graph_of_convex_sets.graph.nodes()) > 0
    assert len(graph_of_convex_sets.graph.edges()) > 0

    start = Point3(-4.5, -0.5, 0.4, reference_frame=table_world.root)
    target = Point3(-2.5, 1.5, 0.9, reference_frame=table_world.root)

    path = graph_of_convex_sets.path_from_to(start, target)

    assert path is not None
    assert len(path) > 1

    with pytest.raises(PointOccupiedError):
        start_occupied = Point3(-10, -10, -10, reference_frame=table_world.root)
        target_occupied = Point3(10, 10, 10, reference_frame=table_world.root)
        graph_of_convex_sets.path_from_to(start_occupied, target_occupied)


def test_constrain_to_free_space_requires_floatlike_fields(table_world: World):
    """
    constrain_to_free_space must check that the given variable has a float-like field
    for every spatial coordinate its free space is expressed over, rather than silently
    building a nonsensical condition -- EQL's own attribute access never raises for a
    field that does not actually exist on the queried type.
    """
    search_space = BoundingBoxCollection(
        [
            VolumetricBoundingBox(
                -1,
                -1,
                0,
                1,
                1,
                1,
                HomogeneousTransformationMatrix(reference_frame=table_world.root),
            )
        ],
        table_world.root,
    )
    graph_of_convex_sets = VolumetricGraphOfBoundingBoxes.free_space_from_world(
        table_world, search_space=search_space
    )

    query = entity(variable(str))

    with pytest.raises(NotNumberLikeFieldError):
        graph_of_convex_sets.constrain_to_free_space(query)


def test_constrain_to_free_space_adds_a_where_condition(table_world: World):
    """
    constrain_to_free_space attaches its condition to the variable's own query, rather
    than merely returning it for the caller to attach.

    This test only checks that a condition ends up attached, without depending on
    ``UnderspecifiedParameters``' behaviour for a bare top-level selection.
    """
    search_space = BoundingBoxCollection(
        [
            VolumetricBoundingBox(
                -1,
                -1,
                0,
                1,
                1,
                1,
                HomogeneousTransformationMatrix(reference_frame=table_world.root),
            )
        ],
        table_world.root,
    )
    graph_of_convex_sets = VolumetricGraphOfBoundingBoxes.free_space_from_world(
        table_world, search_space=search_space
    )

    query = a(Point3)(x=..., y=..., z=..., reference_frame=None)

    condition = graph_of_convex_sets.constrain_to_free_space(query.expression)

    assert condition is not None
    assert condition._children_


def test_constrain_to_free_space_rejects_a_query_selecting_multiple_variables(
    table_world: World,
):
    """
    A query over more than one variable has no single variable to constrain, so
    constrain_to_free_space must reject it rather than resolving one of its selected
    variables' fields regardless.
    """
    search_space = BoundingBoxCollection(
        [
            VolumetricBoundingBox(
                -1,
                -1,
                0,
                1,
                1,
                1,
                HomogeneousTransformationMatrix(reference_frame=table_world.root),
            )
        ],
        table_world.root,
    )
    graph_of_convex_sets = VolumetricGraphOfBoundingBoxes.free_space_from_world(
        table_world, search_space=search_space
    )

    query = set_of(variable(Point3), variable(Point3))

    with pytest.raises(AmbiguousSelectedVariableError):
        graph_of_convex_sets.constrain_to_free_space(query)


def test_navigation_map_from_world(table_world: World):
    """
    Verify the generation of a navigation map from a world.
    """
    search_space = BoundingBoxCollection(
        [
            VolumetricBoundingBox(
                min_x=-5,
                max_x=-2,
                min_y=-1,
                max_y=2,
                min_z=0,
                max_z=2,
                origin=HomogeneousTransformationMatrix(
                    reference_frame=table_world.root
                ),
            )
        ],
        table_world.root,
    )
    graph_of_convex_sets = PlanarGraphOfBoundingBoxes.navigation_map_from_world(
        table_world, search_space=search_space
    )
    assert len(graph_of_convex_sets.graph.nodes()) > 0
    assert len(graph_of_convex_sets.graph.edges()) > 0

    # Regression test for the z-reinflation hack this class replaces: the
    # decomposition never touches z, so its nodes and search space are genuinely
    # two-dimensional rather than 3-D boxes with a fake z=reals() extent.
    assert all(
        isinstance(node, PlanarBoundingBox)
        for node in graph_of_convex_sets.graph.nodes()
    )
    assert all(
        isinstance(box, PlanarBoundingBox)
        for box in graph_of_convex_sets.search_space.bounding_boxes
    )


def test_navigation_map_path_returns_point2_waypoints(table_world: World):
    """
    A planar GCS's path is expressed in Point2, not Point3: there is no z to report, and
    interior waypoints (portals between boxes) carry no meaningful orientation.
    """
    search_space = BoundingBoxCollection(
        [
            VolumetricBoundingBox(
                min_x=-5,
                max_x=-2,
                min_y=-1,
                max_y=2,
                min_z=0,
                max_z=2,
                origin=HomogeneousTransformationMatrix(
                    reference_frame=table_world.root
                ),
            )
        ],
        table_world.root,
    )
    graph_of_convex_sets = PlanarGraphOfBoundingBoxes.navigation_map_from_world(
        table_world, search_space=search_space
    )

    start = Point2(-4.5, -0.5, reference_frame=table_world.root)
    goal = Point2(-2.5, 1.5, reference_frame=table_world.root)
    path = graph_of_convex_sets.path_from_to(start, goal)

    assert path is not None
    assert len(path) > 1
    for waypoint in path:
        assert isinstance(waypoint, Point2)


def test_from_world_with_rotated_box():
    """
    Verify if a path can be found in a world with two boxes, where one is rotated and a
    gcs is calculated for the rotated box.
    """
    world = World()
    with world.modify_world():
        root_body = Body(name=PrefixedName("map"))
        world.add_kinematic_structure_entity(root_body)

        # Box 1: at origin
        axis_aligned_box_body = Body(name=PrefixedName("box_one"))
        world.add_connection(
            FixedConnection.create_with_dofs(world, root_body, axis_aligned_box_body)
        )
        axis_aligned_box = Box(scale=Scale(1, 1, 1))
        axis_aligned_box_body.collision.append(axis_aligned_box)

        # Box 2: at (2, 0, 0) rotated 45 deg around Z
        rotated_box_body = Body(name=PrefixedName("box_two"))
        rotated_box_body_pose = HomogeneousTransformationMatrix.from_xyz_rpy(
            2, 0, 0, 0, np.pi / 4, np.pi / 4, reference_frame=root_body
        )
        world.add_connection(
            FixedConnection.create_with_dofs(
                world,
                root_body,
                rotated_box_body,
                parent_T_connection_expression=rotated_box_body_pose,
            )
        )
        rotated_box = Box(scale=Scale(1, 1, 1))
        rotated_box_body.collision.append(rotated_box)

    vertical_stabilized_base = create_reference_frame_with_only_yaw_from_body(
        rotated_box_body
    )

    search_space = BoundingBoxCollection(
        [
            VolumetricBoundingBox(
                min_x=-5,
                max_x=5,
                min_y=-5,
                max_y=5,
                min_z=0,
                max_z=2,
                origin=HomogeneousTransformationMatrix(
                    reference_frame=vertical_stabilized_base
                ),
            )
        ],
        reference_frame=vertical_stabilized_base,
    )

    graph_of_convex_sets = VolumetricGraphOfBoundingBoxes.free_space_from_world(
        world, search_space=search_space
    )

    assert len(graph_of_convex_sets.graph.nodes()) > 0

    start = Point3(-1, 0, 0.5, reference_frame=world.root)
    target = Point3(4, 0, 0.5, reference_frame=world.root)

    path = graph_of_convex_sets.path_from_to(start, target)
    assert path is not None

    for bounding_box in graph_of_convex_sets.graph.nodes():
        bounding_box_T_world: Pose = world.transform(
            bounding_box.as_shape().origin, world.root
        ).to_pose()

        assert bounding_box_T_world.roll == 0
        assert bounding_box_T_world.pitch == 0
        assert allclose(bounding_box_T_world.yaw, rotated_box_body_pose.to_pose().yaw)


def test_path_from_to_prefers_shorter_distance_over_fewer_hops():
    """
    A detour with many small hops that is geometrically short must be preferred over a
    detour with few large hops that is geometrically long.

    Regression test: ``path_from_to`` used to run an unweighted (hop-count) Dijkstra
    search, so a four-hop detour through three huge boxes could beat an eleven-hop
    detour through small boxes even though the four-hop detour is far longer in real
    distance. There is no direct line of sight between start and goal (a "wall" blocks
    it), so this keeps being true even after shortcutting simplifies whichever detour
    was chosen.
    """
    world = World()
    with world.modify_world():
        world.add_kinematic_structure_entity(Body())

    graph_of_convex_sets = VolumetricGraphOfBoundingBoxes(world)

    origin = world.root.global_pose
    start_box = VolumetricBoundingBox(0, 0, 0, 1, 1, 1, origin)
    goal_box = VolumetricBoundingBox(10, 0, 0, 11, 1, 1, origin)
    # No boxes exist at y=[0, 1] between start_box and goal_box - a "wall" that rules
    # out a direct line of sight, so reaching goal_box always requires a detour.

    # Short way: hop over the wall through a chain of ten small boxes.
    over_the_wall = [
        VolumetricBoundingBox(i, 1, 0, i + 1, 2, 1, origin) for i in range(10)
    ]

    # Long way: dip far below the wall and back up through three huge boxes. Costs
    # only four hops overall, but each hop's center-to-center distance is huge.
    below_the_wall = [
        VolumetricBoundingBox(0, -100, 0, 1, 1, 1, origin),
        VolumetricBoundingBox(0, -100, 0, 11, -99, 1, origin),
        VolumetricBoundingBox(10, -100, 0, 11, 1, 1, origin),
    ]

    for box in [start_box, goal_box, *over_the_wall, *below_the_wall]:
        graph_of_convex_sets.add_node(box)
    graph_of_convex_sets.calculate_connectivity()

    start = Point3(0.5, 0.5, 0.5, reference_frame=world.root)
    goal = Point3(10.5, 0.5, 0.5, reference_frame=world.root)

    path = graph_of_convex_sets.path_from_to(start, goal)

    total_length = sum(
        float(np.linalg.norm(a.to_np()[:3] - b.to_np()[:3]))
        for a, b in zip(path, path[1:])
    )
    # The short way over the wall totals roughly 10-11 units; the long way below it
    # totals roughly 200 (a ~100-unit dip, there and back). 30 comfortably separates
    # the two, so this only passes if the short way was chosen.
    assert total_length < 30


def test_path_from_to_shortcuts_redundant_waypoints():
    """
    Waypoints that a straight line can bypass without leaving free space must be dropped
    from the returned path.
    """
    world = World()
    with world.modify_world():
        world.add_kinematic_structure_entity(Body())

    graph_of_convex_sets = VolumetricGraphOfBoundingBoxes(world)

    origin = world.root.global_pose
    start_box = VolumetricBoundingBox(0, 0, 0, 1, 1, 1, origin)
    # Ten boxes tiling one straight, unobstructed corridor from start_box to goal_box.
    chain = [VolumetricBoundingBox(i, 0, 0, i + 1, 1, 1, origin) for i in range(1, 10)]
    goal_box = VolumetricBoundingBox(10, 0, 0, 11, 1, 1, origin)

    for box in [start_box, *chain, goal_box]:
        graph_of_convex_sets.add_node(box)
    graph_of_convex_sets.calculate_connectivity()

    start = Point3(0.5, 0.5, 0.5, reference_frame=world.root)
    goal = Point3(10.5, 0.5, 0.5, reference_frame=world.root)

    path = graph_of_convex_sets.path_from_to(start, goal)

    # Every intermediate waypoint is redundant: the straight line from start to goal
    # never leaves the corridor, so it should be shortcut down to just the endpoints.
    assert path == [start, goal]


def test_path_from_to_scales_to_a_real_apartment_scene():
    """
    Regression test for a real scalability bug: ``path_from_to`` used to enumerate *all*
    shortest (by hop count) paths via ``rustworkx.all_shortest_paths`` before picking
    the first one, which never finished once the free-space decomposition of a real-
    sized scene reached a few thousand nodes (there can be exponentially many equally-
    short paths).

    It now runs a single-source Dijkstra search instead.
    """
    world = MJCFParser(os.path.join(MJCF_DIR, "iai_apartment.xml")).parse()
    search_space = BoundingBoxCollection(
        [
            VolumetricBoundingBox(
                min_x=-0.2,
                max_x=3.0,
                min_y=1.0,
                max_y=2.4,
                min_z=0.0,
                max_z=1.2,
                origin=HomogeneousTransformationMatrix(reference_frame=world.root),
            )
        ],
        world.root,
    )
    graph_of_convex_sets = VolumetricGraphOfBoundingBoxes.free_space_from_world(
        world, search_space=search_space, bloat_obstacles=0.06
    )
    # Confirms this is a genuinely large graph, not an accidentally-trivial one.
    assert len(graph_of_convex_sets.graph.nodes()) > 1000

    start = Point3(0.8, 1.0, 0.6, reference_frame=world.root)
    goal = Point3(0.8, 2.3, 0.6, reference_frame=world.root)
    path = graph_of_convex_sets.path_from_to(start, goal)

    assert path is not None
    assert len(path) > 1


def test_planar_free_space_uses_the_supporting_surface_as_search_space():
    """
    HasSupportingSurface.planar_free_space must derive its search space from the
    supporting surface's own area, rather than requiring the caller to build one.
    """
    world = World.create_with_root_body("root")
    with world.modify_world():
        floor = Floor.create_with_new_body_in_world(
            name="floor", world=world, scale=Scale(4, 4, 0.01)
        )
        surface = Region.from_shape_collection(
            PrefixedName("floor_surface"),
            ShapeCollection(
                [Box(scale=Scale(4, 4, 0.001))], reference_frame=floor.root
            ),
        )
        world.add_region(surface)
        world.add_connection(FixedConnection(parent=floor.root, child=surface))
        floor.supporting_surface = surface

        obstacle = Body(name=PrefixedName("obstacle"))
        world.add_connection(
            FixedConnection.create_with_dofs(
                world,
                world.root,
                obstacle,
                parent_T_connection_expression=HomogeneousTransformationMatrix.from_xyz_rpy(
                    0.0, 0.0, 0.5, reference_frame=world.root
                ),
            )
        )
        obstacle.collision.append(Box(scale=Scale(0.4, 0.4, 1.0)))

    graph = floor.planar_free_space(max_height=2.0)

    assert isinstance(graph, PlanarGraphOfBoundingBoxes)
    search_box = graph.search_space.bounding_box()
    assert search_box.min_x == pytest.approx(-2.0)
    assert search_box.max_x == pytest.approx(2.0)
    assert search_box.min_y == pytest.approx(-2.0)
    assert search_box.max_y == pytest.approx(2.0)
    assert len(graph.graph.nodes()) > 0
