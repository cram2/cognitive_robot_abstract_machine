"""
Tests for the three-panel Graph of Convex Sets figure.

The scenes are built from a purpose-made room world rather than from a URDF, so every
obstacle position the assertions depend on is stated here.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest
from matplotlib.colors import to_rgba

from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.semantic_annotations.semantic_annotations import (
    SemanticEnvironmentAnnotation,
)
from semantic_digital_twin.spatial_types import Point3
from semantic_digital_twin.spatial_types.spatial_types import (
    HomogeneousTransformationMatrix,
)
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import FixedConnection
from semantic_digital_twin.world_description.geometry import (
    Box,
    VolumetricBoundingBox,
    PlanarBoundingBox,
    Scale,
)
from semantic_digital_twin.world_description.graph_of_convex_sets.boxes import (
    PlanarGraphOfBoundingBoxes,
    VolumetricGraphOfBoundingBoxes,
)
from semantic_digital_twin.world_description.graph_of_convex_sets.exceptions import (
    UnconnectedGraphError,
)
from semantic_digital_twin.world_description.graph_of_convex_sets.plotting import (
    ConvexSetAdjacency,
    ConvexSetsPanel,
    EnvironmentPanel,
    Footprint,
    GraphOfConvexSetsFigure,
    NavigationPath,
    NavigationScene,
    OptimalPathPanel,
    PanelArrangement,
    Theme,
)
from semantic_digital_twin.world_description.shape_collection import (
    BoundingBoxCollection,
)
from semantic_digital_twin.world_description.world_entity import Body

from .gcs_test_helpers import hardest_path_query

# %% the world every scene in this module is built from

ROOM_HALF_EXTENT = 3.0
"""
Half the side length of the square room, so its walls span -3 m to 3 m in x and y.
"""

WALL_THICKNESS = 0.2
"""
Thickness of each of the room's four walls.
"""

PILLAR_HALF_EXTENT = 0.5
"""
Half the side length of the pillar standing in the middle of the room.
"""

ROOM_HEIGHT = 2.0
"""
Height of the room's walls and of its pillar.
"""

CLEARANCE = 0.2
"""
Clearance the scenes under test are built with.
"""


def add_box_body(world: World, name: str, scale: Scale, x: float, y: float) -> Body:
    """
    Add a box-shaped obstacle standing on the floor of a world.

    :param world: The world to add the obstacle to.
    :param name: The name of the body carrying the obstacle.
    :param scale: The extent of the box.
    :param x: The x coordinate of the box's center.
    :param y: The y coordinate of the box's center.
    :return: The added body.
    """
    body = Body(name=PrefixedName(name))
    world.add_connection(
        FixedConnection.create_with_dofs(
            world,
            world.root,
            body,
            parent_T_connection_expression=HomogeneousTransformationMatrix.from_xyz_rpy(
                x, y, ROOM_HEIGHT / 2, reference_frame=world.root
            ),
        )
    )
    body.collision.append(Box(scale=scale))
    return body


@pytest.fixture
def room_world() -> World:
    """
    A square room walled on all four sides, with a single square pillar in its middle.

    The pillar sits on the line between any two opposite corners, so a path across the
    room has to detour around it.
    """
    world = World()
    with world.modify_world():
        world.add_kinematic_structure_entity(Body(name=PrefixedName("map")))

        wall_center = ROOM_HALF_EXTENT - WALL_THICKNESS / 2
        wall_length = 2 * ROOM_HALF_EXTENT
        horizontal_wall = Scale(wall_length, WALL_THICKNESS, ROOM_HEIGHT)
        vertical_wall = Scale(WALL_THICKNESS, wall_length, ROOM_HEIGHT)

        add_box_body(world, "south_wall", horizontal_wall, 0.0, -wall_center)
        add_box_body(world, "north_wall", horizontal_wall, 0.0, wall_center)
        add_box_body(world, "west_wall", vertical_wall, -wall_center, 0.0)
        add_box_body(world, "east_wall", vertical_wall, wall_center, 0.0)
        add_box_body(
            world,
            "pillar",
            Scale(2 * PILLAR_HALF_EXTENT, 2 * PILLAR_HALF_EXTENT, ROOM_HEIGHT),
            0.0,
            0.0,
        )
    return world


def room_scene_of(world: World, clearance: float = CLEARANCE) -> NavigationScene:
    """
    Build the scene of a room world, planning between the two convex sets it forces the
    longest detour between.

    :param world: The room world to build the scene of.
    :param clearance: Amount obstacles are bloated by while building the graph.
    :return: The scene.
    """
    origin = HomogeneousTransformationMatrix(reference_frame=world.root)
    search_space = BoundingBoxCollection(
        [
            VolumetricBoundingBox(
                -ROOM_HALF_EXTENT,
                -ROOM_HALF_EXTENT,
                0.0,
                ROOM_HALF_EXTENT,
                ROOM_HALF_EXTENT,
                ROOM_HEIGHT,
                origin,
            )
        ],
        world.root,
    )
    graph = PlanarGraphOfBoundingBoxes.navigation_map_from_world(
        world=world, search_space=search_space, bloat_obstacles=clearance
    )
    query = hardest_path_query(graph)
    waypoints = graph.path_from_to(query.start, query.goal)
    obstacles = SemanticEnvironmentAnnotation(
        root=world.root, _world=world
    ).as_bounding_box_collection_at_origin(origin)
    return NavigationScene(
        graph_of_convex_sets=graph,
        environment_name="room",
        path=NavigationPath(waypoints),
        obstacles=obstacles,
    )


@pytest.fixture
def room_scene(room_world: World) -> NavigationScene:
    """
    The scene of the room world, planned with :data:`CLEARANCE`.
    """
    return room_scene_of(room_world)


@dataclass
class BoxRowFixture:
    """
    A graph of three convex sets laid out in a row along x, whose farthest pair is known
    by construction.
    """

    graph: VolumetricGraphOfBoundingBoxes
    boxes: list[VolumetricBoundingBox]


@pytest.fixture
def box_row() -> BoxRowFixture:
    """
    Three unit boxes touching along x, spanning x from 0 m to 3 m.
    """
    world = World()
    with world.modify_world():
        world.add_kinematic_structure_entity(Body(name=PrefixedName("map")))

    graph = VolumetricGraphOfBoundingBoxes(world)
    origin = HomogeneousTransformationMatrix(reference_frame=world.root)
    boxes = [
        VolumetricBoundingBox(offset, 0.0, 0.0, offset + 1.0, 1.0, 1.0, origin)
        for offset in (0.0, 1.0, 2.0)
    ]
    for box in boxes:
        graph.add_node(box)
    graph.calculate_connectivity()
    return BoxRowFixture(graph, boxes)


# %% the pieces a scene is made of


def test_footprint_projects_a_bounding_box_including_its_origin():
    """
    A footprint is the box's x-y extent in the frame its origin is expressed in, not the
    extent relative to that origin.
    """
    origin = HomogeneousTransformationMatrix.from_xyz_rpy(2.0, -1.0, 5.0)
    footprint = Footprint.of(
        VolumetricBoundingBox(-0.5, -0.25, 0.0, 0.5, 0.25, 1.0, origin)
    )

    assert (footprint.min_x, footprint.max_x) == (1.5, 2.5)
    assert (footprint.min_y, footprint.max_y) == (-1.25, -0.75)
    assert (footprint.width, footprint.height) == (1.0, 0.5)


def test_navigation_path_length_sums_its_segments():
    """
    The reported length is the distance travelled, not the distance between the
    endpoints.
    """
    world = World()
    with world.modify_world():
        world.add_kinematic_structure_entity(Body(name=PrefixedName("map")))

    corner_path = NavigationPath(
        [
            Point3(0.0, 0.0, 0.0, reference_frame=world.root),
            Point3(3.0, 0.0, 0.0, reference_frame=world.root),
            Point3(3.0, 4.0, 0.0, reference_frame=world.root),
        ]
    )

    assert corner_path.length == pytest.approx(7.0)


def test_navigation_path_reports_how_much_of_it_is_vertical():
    """
    The height a path gains and loses is counted whichever way it goes, and is zero for
    a path that stays level.
    """
    world = World()
    with world.modify_world():
        world.add_kinematic_structure_entity(Body(name=PrefixedName("map")))

    climbing_path = NavigationPath(
        [
            Point3(0.0, 0.0, 0.0, reference_frame=world.root),
            Point3(0.0, 0.0, 2.0, reference_frame=world.root),
            Point3(1.0, 0.0, 1.5, reference_frame=world.root),
        ]
    )
    level_path = NavigationPath(
        [
            Point3(0.0, 0.0, 1.0, reference_frame=world.root),
            Point3(5.0, 0.0, 1.0, reference_frame=world.root),
        ]
    )

    assert climbing_path.vertical_travel == pytest.approx(2.5)
    assert level_path.vertical_travel == pytest.approx(0.0)


def test_a_graph_with_no_connected_pair_has_nothing_to_plan_between():
    """
    Two convex sets that do not touch leave no query to pose, which is reported rather
    than failing inside the search for the most distant pair.
    """
    world = World()
    with world.modify_world():
        world.add_kinematic_structure_entity(Body(name=PrefixedName("map")))

    graph = VolumetricGraphOfBoundingBoxes(world)
    origin = HomogeneousTransformationMatrix(reference_frame=world.root)
    for offset in (0.0, 10.0):
        graph.add_node(
            VolumetricBoundingBox(offset, 0.0, 0.0, offset + 1.0, 1.0, 1.0, origin)
        )
    graph.calculate_connectivity()

    with pytest.raises(UnconnectedGraphError):
        hardest_path_query(graph)


def test_adjacency_is_drawn_through_the_portal_the_sets_share(
    box_row: BoxRowFixture,
):
    """
    An edge is drawn from one set's center through the overlap the two sets share, so it
    never cuts a corner the planner cannot cut.
    """
    adjacency = ConvexSetAdjacency(
        source_center=box_row.boxes[0].center,
        portal_center=box_row.graph.graph.get_edge_data(
            box_row.graph.box_to_index_map[box_row.boxes[0]],
            box_row.graph.box_to_index_map[box_row.boxes[1]],
        ).intersection.center,
        target_center=box_row.boxes[1].center,
    )

    assert adjacency.coordinates.tolist() == [[0.5, 0.5], [1.0, 0.5], [1.5, 0.5]]


def test_query_spans_the_graph_in_coordinate_order(box_row: BoxRowFixture):
    """
    The query connects the two ends of the longest shortest path, oriented so that the
    same graph always produces the same start, whatever order its nodes are held in.
    """
    query = hardest_path_query(box_row.graph)

    assert (float(query.start.x), float(query.start.y)) == (0.5, 0.5)
    assert (float(query.goal.x), float(query.goal.y)) == (2.5, 0.5)


def test_query_measures_distance_along_the_graph_not_across_it():
    """
    Two sets facing each other across a barrier are far apart to travel between but
    close together in space, and it is the travelling distance that decides the query.
    """
    world = World()
    with world.modify_world():
        world.add_kinematic_structure_entity(Body(name=PrefixedName("map")))

    graph = VolumetricGraphOfBoundingBoxes(world)
    origin = HomogeneousTransformationMatrix(reference_frame=world.root)
    # A U of unit boxes: the two tips face each other one meter apart, but reaching one
    # tip from the other means walking around the whole U.
    corner_offsets = [
        (0.0, 0.0),
        (0.0, 1.0),
        (0.0, 2.0),
        (1.0, 2.0),
        (2.0, 2.0),
        (2.0, 1.0),
        (2.0, 0.0),
    ]
    for offset_x, offset_y in corner_offsets:
        graph.add_node(
            VolumetricBoundingBox(
                offset_x, offset_y, 0.0, offset_x + 1.0, offset_y + 1.0, 1.0, origin
            )
        )
    graph.calculate_connectivity()

    query = hardest_path_query(graph)

    assert (float(query.start.x), float(query.start.y)) == (0.5, 0.5)
    assert (float(query.goal.x), float(query.goal.y)) == (2.5, 0.5)
    assert float(query.start.euclidean_distance(query.goal)) == pytest.approx(2.0)


# %% the scene


def test_convex_sets_keep_the_clearance_away_from_every_obstacle(
    room_scene: NavigationScene,
):
    """
    No convex set reaches into an obstacle bloated by the clearance, which is what makes
    a path through the sets passable for a robot of that radius.
    """
    bloated_obstacles = [
        Footprint.of(obstacle.bloat(CLEARANCE, CLEARANCE, 0.0))
        for obstacle in room_scene.obstacles
    ]

    overlaps = [
        (convex_set, obstacle)
        for convex_set in map(Footprint.of, room_scene.convex_sets)
        for obstacle in bloated_obstacles
        if convex_set.min_x < obstacle.max_x
        and obstacle.min_x < convex_set.max_x
        and convex_set.min_y < obstacle.max_y
        and obstacle.min_y < convex_set.max_y
    ]

    assert overlaps == []


def test_path_detours_around_the_pillar(room_scene: NavigationScene):
    """
    The path runs from its start to its goal without entering the pillar, and is
    therefore longer than the straight line between them.
    """
    pillar_footprint = Footprint(
        min_x=-PILLAR_HALF_EXTENT - CLEARANCE,
        min_y=-PILLAR_HALF_EXTENT - CLEARANCE,
        max_x=PILLAR_HALF_EXTENT + CLEARANCE,
        max_y=PILLAR_HALF_EXTENT + CLEARANCE,
    )
    waypoints = room_scene.path.waypoints

    assert not [
        waypoint
        for waypoint in waypoints
        if pillar_footprint.min_x < float(waypoint.x) < pillar_footprint.max_x
        and pillar_footprint.min_y < float(waypoint.y) < pillar_footprint.max_y
    ]
    straight_line_distance = (
        (float(waypoints[-1].x) - float(waypoints[0].x)) ** 2
        + (float(waypoints[-1].y) - float(waypoints[0].y)) ** 2
    ) ** 0.5
    assert room_scene.path.length > straight_line_distance


def test_navigation_scene_derives_a_planar_obstacle_collection_for_a_planar_graph(
    room_world: World,
):
    """
    A scene built with no obstacles collection derives one from the graph's own search
    space and free space; for a planar graph that derivation must stay two-dimensional
    rather than reaching for the volumetric bounding-box collection unconditionally.
    """
    origin = HomogeneousTransformationMatrix(reference_frame=room_world.root)
    search_space = BoundingBoxCollection(
        [
            VolumetricBoundingBox(
                -ROOM_HALF_EXTENT,
                -ROOM_HALF_EXTENT,
                0.0,
                ROOM_HALF_EXTENT,
                ROOM_HALF_EXTENT,
                ROOM_HEIGHT,
                origin,
            )
        ],
        room_world.root,
    )
    graph = PlanarGraphOfBoundingBoxes.navigation_map_from_world(
        world=room_world, search_space=search_space, bloat_obstacles=CLEARANCE
    )
    query = hardest_path_query(graph)
    waypoints = graph.path_from_to(query.start, query.goal)

    scene = NavigationScene(
        graph_of_convex_sets=graph,
        environment_name="room",
        path=NavigationPath(waypoints),
    )

    assert len(scene.obstacles.bounding_boxes) > 0
    assert all(
        isinstance(box, PlanarBoundingBox) for box in scene.obstacles.bounding_boxes
    )


# %% the figure


def test_wide_environments_are_stacked_and_compact_ones_are_placed_side_by_side():
    """
    The arrangement follows the environment's aspect ratio, so panels never end up as
    thin slivers.
    """
    wide = Footprint(min_x=0.0, min_y=0.0, max_x=20.0, max_y=5.0)
    compact = Footprint(min_x=0.0, min_y=0.0, max_x=6.0, max_y=6.0)

    assert PanelArrangement.for_extent(wide, 3) == PanelArrangement(rows=3, columns=1)
    assert PanelArrangement.for_extent(compact, 3) == PanelArrangement(
        rows=1, columns=3
    )


def test_a_meter_is_the_same_length_in_every_panel():
    """
    Panels are framed to a shared extent and sized from a single scale, so the three can
    be read against each other.
    """
    extent = Footprint(min_x=0.0, min_y=0.0, max_x=4.0, max_y=4.0)
    arrangement = PanelArrangement(rows=1, columns=3)

    width, _ = arrangement.figure_size(extent)

    assert width == pytest.approx(3 * 4.0 * arrangement.preferred_inches_per_meter)


def test_a_legend_too_wide_for_the_figure_wraps_into_even_rows():
    """
    A stacked figure is too narrow to lay its legend out on one row, and wrapping it
    spreads the entries rather than leaving a row with a single entry in it.
    """
    narrow = Footprint(min_x=0.0, min_y=0.0, max_x=12.0, max_y=8.0)
    arrangement = PanelArrangement(rows=3, columns=1)

    width, _ = arrangement.figure_size(narrow)
    entries_that_fit = int(width / arrangement.legend_entry_inches)

    assert entries_that_fit == 3
    assert arrangement.legend_columns(narrow, 7) == 3
    assert arrangement.legend_columns(narrow, 4) == 2


def test_a_legend_that_fits_stays_on_one_row():
    """
    A legend with room for all of its entries is laid out in a single row.
    """
    wide = Footprint(min_x=0.0, min_y=0.0, max_x=6.0, max_y=6.0)
    arrangement = PanelArrangement(rows=1, columns=3)

    width, _ = arrangement.figure_size(wide)
    entry_count = int(width / arrangement.legend_entry_inches)

    assert arrangement.legend_columns(wide, entry_count) == entry_count


def test_figure_draws_one_titled_panel_per_stage(room_scene: NavigationScene):
    """
    The figure walks through what the planner is given, what it builds and what it
    returns, each panel labelled and quantified.
    """
    figure = GraphOfConvexSetsFigure(room_scene)

    rendered = figure.render()
    titles = [axes.get_title(loc="left") for axes in rendered.axes]

    assert titles == [
        f"(a) {EnvironmentPanel().name}",
        f"(b) {ConvexSetsPanel().name}",
        f"(c) {OptimalPathPanel().name}",
    ]
    assert ConvexSetsPanel().subtitle(room_scene) == (
        f"{len(room_scene.convex_sets)} convex sets, "
        f"{len(room_scene.adjacencies)} adjacencies"
    )


def test_figure_is_painted_on_its_theme_surface(room_scene: NavigationScene):
    """
    A theme's palette is applied to the figure itself, not only to the marks in it.
    """
    rendered = GraphOfConvexSetsFigure(room_scene, theme=Theme.DARK).render()

    assert rendered.get_facecolor() == pytest.approx(to_rgba(Theme.DARK.value.surface))


def test_figure_is_saved_as_vector_and_raster(room_scene, tmp_path):
    """
    Saving produces both a file to typeset and a file to look at, named after the
    environment and the theme.
    """
    written = GraphOfConvexSetsFigure(room_scene).save(tmp_path / "figures")

    assert [path.name for path in written] == [
        "graph_of_convex_sets_room_light.pdf",
        "graph_of_convex_sets_room_light.png",
    ]
    assert all(path.stat().st_size > 0 for path in written)
