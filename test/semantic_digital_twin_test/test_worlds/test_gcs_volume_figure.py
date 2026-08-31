"""
Tests for the volume variant of the three-panel Graph of Convex Sets figure.

The scene is built from a purpose-made world whose only way from one end to the other is
over a barrier, so that a path is forced to use the third dimension the figure is about.
"""

from __future__ import annotations

import numpy as np
import rustworkx as rx
import pytest

from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.semantic_annotations.semantic_annotations import (
    SemanticEnvironmentAnnotation,
)
from semantic_digital_twin.spatial_types.spatial_types import (
    HomogeneousTransformationMatrix,
)
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import FixedConnection
from semantic_digital_twin.world_description.geometry import (
    Box,
    VolumetricBoundingBox,
    Scale,
)
from semantic_digital_twin.world_description.graph_of_convex_sets.boxes import (
    VolumetricGraphOfBoundingBoxes,
)
from semantic_digital_twin.world_description.graph_of_convex_sets.plotting import (
    NavigationPath,
    NavigationScene,
    Theme,
)
from semantic_digital_twin.world_description.graph_of_convex_sets.volume_figure import (
    ConvexSetsVolumePanel,
    EnvironmentVolumePanel,
    GraphOfConvexSetsVolumeFigure,
    ObstacleEmphasis,
    ObstacleVolumeLayer,
    OptimalPathVolumePanel,
    polyline_trace,
)
from semantic_digital_twin.world_description.shape_collection import (
    BoundingBoxCollection,
)

from .gcs_test_helpers import hardest_path_query
from semantic_digital_twin.world_description.world_entity import Body

# %% the world the scene is built from

FLOOR_SIZE = (4.0, 2.0)
"""
The x and y extent of the floor, centred on the origin.
"""

FLOOR_THICKNESS = 0.02
"""
Thickness of the floor slab, thin enough to count as ground.
"""

BARRIER_HEIGHT = 0.5
"""
Height of the barrier crossing the room, low enough to pass over and wide enough that
there is no way around it.
"""

SIDE_WALL_HEIGHT = 1.5
"""
Height of the walls at either end, which is what gives the world a volume above the
barrier to plan through.
"""

CLEARANCE = 0.05
"""
Clearance the scene under test is built with.
"""


def add_box_body(
    world: World, name: str, scale: Scale, x: float, y: float, z: float
) -> Body:
    """
    Add a box-shaped body to a world.

    :param world: The world to add the body to.
    :param name: The name of the body.
    :param scale: The extent of the box.
    :param x: The x coordinate of the box's center.
    :param y: The y coordinate of the box's center.
    :param z: The z coordinate of the box's center.
    :return: The added body.
    """
    body = Body(name=PrefixedName(name))
    world.add_connection(
        FixedConnection.create_with_dofs(
            world,
            world.root,
            body,
            parent_T_connection_expression=HomogeneousTransformationMatrix.from_xyz_rpy(
                x, y, z, reference_frame=world.root
            ),
        )
    )
    body.collision.append(Box(scale=scale))
    return body


@pytest.fixture
def barrier_world() -> World:
    """
    A floor with a low barrier across its full width and a tall wall at either end.

    The barrier cannot be passed beside, only over, and the walls give the world enough
    height above it to do so.
    """
    world = World()
    with world.modify_world():
        world.add_kinematic_structure_entity(Body(name=PrefixedName("map")))

        add_box_body(
            world,
            "floor",
            Scale(FLOOR_SIZE[0], FLOOR_SIZE[1], FLOOR_THICKNESS),
            0.0,
            0.0,
            -FLOOR_THICKNESS / 2,
        )
        add_box_body(
            world,
            "barrier",
            Scale(0.2, FLOOR_SIZE[1], BARRIER_HEIGHT),
            0.0,
            0.0,
            BARRIER_HEIGHT / 2,
        )
        for name, x in (
            ("west_wall", -FLOOR_SIZE[0] / 2),
            ("east_wall", FLOOR_SIZE[0] / 2),
        ):
            add_box_body(
                world,
                name,
                Scale(0.1, FLOOR_SIZE[1], SIDE_WALL_HEIGHT),
                x,
                0.0,
                SIDE_WALL_HEIGHT / 2,
            )
    return world


def barrier_search_space_of(world: World) -> BoundingBoxCollection:
    """
    :param world: The barrier world to cover.
    :return: The floor's footprint, raised to the world's full height.
    """
    origin = HomogeneousTransformationMatrix(reference_frame=world.root)
    return BoundingBoxCollection(
        [
            VolumetricBoundingBox(
                -FLOOR_SIZE[0] / 2,
                -FLOOR_SIZE[1] / 2,
                0.0,
                FLOOR_SIZE[0] / 2,
                FLOOR_SIZE[1] / 2,
                SIDE_WALL_HEIGHT,
                origin,
            )
        ],
        world.root,
    )


def navigation_scene_of(
    graph: VolumetricGraphOfBoundingBoxes, world: World
) -> NavigationScene:
    """
    :param graph: The already-decomposed graph of convex sets.
    :param world: The world the graph was built from, to draw the true obstacles of.
    :return: The scene, planned between the two convex sets the graph forces the
        longest detour between.
    """
    origin = HomogeneousTransformationMatrix(reference_frame=world.root)
    query = hardest_path_query(graph)
    waypoints = graph.path_from_to(query.start, query.goal)
    obstacles = SemanticEnvironmentAnnotation(
        root=world.root, _world=world
    ).as_bounding_box_collection_at_origin(origin)
    return NavigationScene(
        graph_of_convex_sets=graph,
        environment_name="barrier",
        path=NavigationPath(waypoints),
        obstacles=obstacles,
    )


@pytest.fixture
def barrier_scene(barrier_world: World) -> NavigationScene:
    """
    The volumetric scene of the barrier world.
    """
    graph = VolumetricGraphOfBoundingBoxes.free_space_from_world(
        world=barrier_world,
        search_space=barrier_search_space_of(barrier_world),
        bloat_obstacles=CLEARANCE,
    )
    return navigation_scene_of(graph, barrier_world)


def test_separate_polylines_are_not_joined_to_each_other():
    """
    Polylines share one trace and must not be connected end to end.
    """
    first = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    second = np.array([[0.0, 5.0, 0.0], [1.0, 5.0, 0.0]])

    trace = polyline_trace([first, second], "#000000", 1.0, "lines")
    points = np.stack([trace.x, trace.y, trace.z], axis=-1)

    assert len(points) == 6
    assert np.isnan(points[2]).all()
    assert np.isnan(points[5]).all()


# %% what the panels compose


def test_the_path_panel_draws_obstacles_it_can_be_seen_through():
    """
    A room is seen from outside, so the panel whose subject is the path has to let it
    show through whatever stands in front of it.
    """
    layers = OptimalPathVolumePanel().layers()

    obstacle_layers = [
        layer for layer in layers if isinstance(layer, ObstacleVolumeLayer)
    ]

    assert [layer.emphasis for layer in obstacle_layers] == [ObstacleEmphasis.CONTEXT]
    assert ObstacleEmphasis.CONTEXT.value < ObstacleEmphasis.SUBJECT.value < 1.0


def test_the_third_dimension_connects_what_a_shallow_search_space_leaves_apart(
    barrier_world: World, barrier_scene: NavigationScene
):
    """
    A search space too shallow to clear the barrier is blocked by it, cutting the room
    in two.

    A search space tall enough to rise over the barrier leaves the space above it, and
    with it a single connected graph.
    """
    origin = HomogeneousTransformationMatrix(reference_frame=barrier_world.root)
    shallow_search_space = BoundingBoxCollection(
        [
            VolumetricBoundingBox(
                -FLOOR_SIZE[0] / 2,
                -FLOOR_SIZE[1] / 2,
                0.0,
                FLOOR_SIZE[0] / 2,
                FLOOR_SIZE[1] / 2,
                BARRIER_HEIGHT - 0.1,
                origin,
            )
        ],
        barrier_world.root,
    )
    shallow_graph = VolumetricGraphOfBoundingBoxes.free_space_from_world(
        world=barrier_world,
        search_space=shallow_search_space,
        bloat_obstacles=CLEARANCE,
    )

    assert len(rx.connected_components(shallow_graph.graph)) == 2
    assert len(rx.connected_components(barrier_scene.graph_of_convex_sets.graph)) == 1


def test_a_volumetric_path_crosses_the_barrier_above_it(
    barrier_scene: NavigationScene,
):
    """
    The path runs from one side of the barrier to the other, and does so at a height
    that clears it.
    """
    waypoints = barrier_scene.path.spatial_coordinates

    assert waypoints[0, 0] < 0.0 < waypoints[-1, 0]
    assert waypoints[:, 2].min() > BARRIER_HEIGHT


def test_the_path_panel_reports_how_much_of_the_path_is_vertical(
    barrier_scene: NavigationScene,
):
    """
    The subtitle names the height the path changes, which is the quantity a floor plan
    has no way to report.
    """
    assert OptimalPathVolumePanel().subtitle(barrier_scene) == (
        f"{barrier_scene.path.length:.2f} m over "
        f"{len(barrier_scene.path.waypoints)} waypoints, "
        f"{barrier_scene.path.vertical_travel:.2f} m of it vertical"
    )


def test_the_convex_sets_panel_counts_what_it_draws(barrier_scene: NavigationScene):
    """
    The subtitle reports the size of the decomposition the panel shows.
    """
    subtitle = ConvexSetsVolumePanel().subtitle(barrier_scene)

    assert subtitle == (
        f"{len(barrier_scene.convex_sets)} convex sets, "
        f"{len(barrier_scene.adjacencies)} adjacencies"
    )


def test_the_environment_panel_reports_the_volume_it_frames(
    barrier_scene: NavigationScene,
):
    """
    The subtitle reports all three extents, since a volume is what this figure is about.
    """
    subtitle = EnvironmentVolumePanel().subtitle(barrier_scene)

    assert subtitle == (
        f"{len(barrier_scene.obstacles)} obstacle boxes in "
        f"{FLOOR_SIZE[0]:.1f} × {FLOOR_SIZE[1]:.1f} × {SIDE_WALL_HEIGHT:.1f} m"
    )


# %% the figure


def test_every_panel_is_framed_to_the_same_volume(barrier_scene: NavigationScene):
    """
    Panels are pinned to the search space rather than auto-scaled to their own traces,
    so a metre is the same length in all three and they can be read against each other.
    """
    figure = GraphOfConvexSetsVolumeFigure(barrier_scene).render()
    bounds = barrier_scene.search_space.bounding_box().to_array_bounds()

    ranges = [
        [
            figure.layout[f"scene{index}"][axis].range
            for axis in ("xaxis", "yaxis", "zaxis")
        ]
        for index in range(1, 4)
    ]

    assert ranges[0] == ranges[1] == ranges[2]
    assert np.allclose(ranges[0], np.stack([bounds.lower, bounds.upper], axis=-1))


def test_every_panel_is_looked_at_from_the_same_place(barrier_scene: NavigationScene):
    """
    A shared camera is the other half of making the panels comparable.
    """
    figure = GraphOfConvexSetsVolumeFigure(barrier_scene).render()

    cameras = [figure.layout[f"scene{index}"].camera for index in range(1, 4)]

    assert cameras[0] == cameras[1] == cameras[2]
    assert cameras[0].up.z == 1.0


def test_each_legend_entry_is_shown_once_across_the_whole_figure(
    barrier_scene: NavigationScene,
):
    """
    Obstacles appear in two panels and the legend names them once.
    """
    figure = GraphOfConvexSetsVolumeFigure(barrier_scene).render()

    shown = [trace.legendgroup for trace in figure.data if trace.showlegend]

    assert sorted(shown) == [
        "adjacency graph",
        "convex set",
        "goal",
        "obstacle",
        "optimal path",
        "search space",
        "start",
    ]


def test_a_panel_title_is_generated_for_every_panel_not_just_the_first_three(
    barrier_scene: NavigationScene,
):
    """
    A panel's label is derived from its position among however many panels the figure
    holds, rather than zipped against a fixed three-letter alphabet that would silently
    drop the title of a fourth panel instead of labelling it.
    """
    panels = (
        EnvironmentVolumePanel(),
        ConvexSetsVolumePanel(),
        OptimalPathVolumePanel(),
        EnvironmentVolumePanel(),
    )
    figure = GraphOfConvexSetsVolumeFigure(barrier_scene, panels=panels).render()

    labels = [
        annotation.text.split("<br>")[0] for annotation in figure.layout.annotations
    ]

    assert labels == [
        f"<b>({letter}) {panel.name}</b>" for letter, panel in zip("abcd", panels)
    ]


def test_the_figure_is_saved_as_an_interactive_page(barrier_scene, tmp_path):
    """
    A single camera angle hides whatever it projects behind something else, so a page
    that can be turned is always written.
    """
    written = GraphOfConvexSetsVolumeFigure(
        barrier_scene, theme=Theme.DARK, image_formats=()
    ).save(tmp_path / "figures")

    assert [path.name for path in written] == [
        "graph_of_convex_sets_volume_barrier_dark.html"
    ]
    assert "plotly" in written[0].read_text()
