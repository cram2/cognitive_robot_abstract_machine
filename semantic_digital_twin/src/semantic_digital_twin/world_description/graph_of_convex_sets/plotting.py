"""
Renders a three-panel figure explaining what a Graph of Convex Sets (GCS) does with an
environment.

The panels answer three questions in order, all drawn as a top-down view of the same
x-y extent so they can be read against each other:

  * **Environment** -- what the planner is given: the obstacle footprints of a world and
    the search space bounding them.

  * **Graph of convex sets** -- what the planner builds: the exact partition of free
    space into axis-aligned convex sets, and the adjacency graph over them.

  * **Optimal path** -- what the planner returns: the minimum-distance sequence of
    waypoints between the two convex sets the environment forces the longest detour
    between.

The figure only draws an already-built
:class:`~semantic_digital_twin.world_description.graph_of_convex_sets.boxes.GraphOfBoundingBoxes`
and an already-solved path over it, wrapped by :class:`NavigationScene`; building either
is not this module's concern. :class:`GraphOfConvexSetsFigure` draws the scene.
"""

from __future__ import annotations

import enum
import itertools
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.artist import Artist
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from typing_extensions import Iterable, List, Optional, Sequence, Self

from semantic_digital_twin.spatial_types import Point, Point3
from semantic_digital_twin.world_description.geometry import (
    AxisAlignedBox,
)
from semantic_digital_twin.world_description.graph_of_convex_sets.boxes import (
    GraphOfBoundingBoxes,
)
from semantic_digital_twin.world_description.shape_collection import (
    BoundingBoxCollection,
)

# %% palette


@dataclass(frozen=True)
class FigurePalette:
    """
    The colors a figure draws with, one field per role rather than per hue, so the same
    entity keeps its color across all panels.
    """

    surface: str
    """
    The color the figure and its panels are painted on, and the color separating
    adjacent fills from each other.
    """

    text_primary: str
    """
    The color of titles and axis labels.
    """

    text_secondary: str
    """
    The color of panel subtitles, tick labels and the search-space outline.
    """

    obstacle: str
    """
    The color of obstacle footprints.
    """

    obstacle_edge: str
    """
    A darker step of :attr:`obstacle`, separating adjacent obstacles from each other.
    """

    convex_set: str
    """
    The color filling a convex set of free space.
    """

    convex_set_edge: str
    """
    A darker step of :attr:`convex_set`, separating neighbouring sets from each other.
    """

    receded_convex_set: str
    """
    A lighter step of :attr:`convex_set`, used where the sets are context rather than
    subject.
    """

    adjacency: str
    """
    The color of the adjacency graph's nodes and edges.
    """

    path: str
    """
    The color of the optimal path.
    """

    start: str
    """
    The color of the start marker.
    """

    goal: str
    """
    The color of the goal marker.
    """

    @classmethod
    def light(cls) -> Self:
        """
        :return: The palette stepped for a light surface.
        """
        return cls(
            surface="#fcfcfb",
            text_primary="#0b0b0b",
            text_secondary="#52514e",
            obstacle="#c9c8c2",
            obstacle_edge="#a8a7a1",
            convex_set="#b7d3f6",
            convex_set_edge="#86b6ef",
            receded_convex_set="#cde2fb",
            adjacency="#184f95",
            path="#eb6834",
            start="#1baf7a",
            goal="#e34948",
        )

    @classmethod
    def dark(cls) -> Self:
        """
        :return: The palette stepped for a dark surface, chosen against that surface
            rather than inverted from :meth:`light`.
        """
        return cls(
            surface="#1a1a19",
            text_primary="#ffffff",
            text_secondary="#c3c2b7",
            obstacle="#4d4c48",
            obstacle_edge="#6b6a65",
            convex_set="#1c5cab",
            convex_set_edge="#3987e5",
            receded_convex_set="#123a6b",
            adjacency="#86b6ef",
            path="#d95926",
            start="#199e70",
            goal="#e66767",
        )


class Theme(enum.Enum):
    """
    The surface a figure is rendered for.

    Its value is the palette stepped for that surface, so that a new theme is added as a member rather than as a branch.

    Kept distinct from :class:`FigurePalette` rather than exposing the palette itself as
    the picker: a theme is a closed choice between exactly two designed variants, where
    a :class:`FigurePalette` is an open dataclass of a dozen individual colors that was
    never meant to be picked from directly. Its filename-safe identity is
    :attr:`~enum.Enum.name` (``"LIGHT"``/``"DARK"``), lowercased, since :attr:`value` is
    now the palette rather than a string.
    """

    LIGHT = FigurePalette.light()
    DARK = FigurePalette.dark()


# %% the scene a figure draws


@dataclass(frozen=True)
class Footprint:
    """
    The x-y projection of a bounding box, which is the shape a top-down panel draws for
    it.
    """

    min_x: float
    """
    The lower x bound, in the search space's reference frame.
    """

    min_y: float
    """
    The lower y bound, in the search space's reference frame.
    """

    max_x: float
    """
    The upper x bound, in the search space's reference frame.
    """

    max_y: float
    """
    The upper y bound, in the search space's reference frame.
    """

    @classmethod
    def of(cls, bounding_box: AxisAlignedBox) -> Self:
        """
        :param bounding_box: The bounding box to project.
        :return: The projection of that box onto the x-y plane.
        """
        bounds = bounding_box.to_array_bounds()
        return cls(
            min_x=float(bounds.lower[0]),
            min_y=float(bounds.lower[1]),
            max_x=float(bounds.upper[0]),
            max_y=float(bounds.upper[1]),
        )

    @property
    def width(self) -> float:
        """
        :return: The extent along x.
        """
        return self.max_x - self.min_x

    @property
    def height(self) -> float:
        """
        :return: The extent along y.
        """
        return self.max_y - self.min_y

    def as_rectangle(
        self, facecolor: str, edgecolor: str, linewidth: float, zorder: float
    ) -> Rectangle:
        """
        :param facecolor: The color filling the rectangle.
        :param edgecolor: The color of its outline.
        :param linewidth: The width of its outline in points.
        :param zorder: The drawing order of the rectangle.
        :return: A matplotlib rectangle covering this footprint.
        """
        return Rectangle(
            (self.min_x, self.min_y),
            self.width,
            self.height,
            facecolor=facecolor,
            edgecolor=edgecolor,
            linewidth=linewidth,
            zorder=zorder,
        )


@dataclass(frozen=True)
class NavigationPath:
    """
    A solved path through a graph of convex sets.

    Kept as a class rather than a plain ``List[Point]``: :attr:`length`,
    :attr:`vertical_travel`, :attr:`coordinates` and :attr:`spatial_coordinates` are each
    read from panels in both this module and
    :mod:`~semantic_digital_twin.world_description.graph_of_convex_sets.volume_figure`,
    so wrapping the waypoints once means every reader shares the same computation
    instead of recomputing it from a bare list.
    """

    waypoints: List[Point]
    """
    The points to navigate to, starting at the query's start and ending at its goal.
    """

    @property
    def _is_volumetric(self) -> bool:
        """
        :return: Whether these waypoints carry a real z-coordinate (:class:`Point3`,
            a 3D path) rather than lying flat on a floor plan (:class:`Point2`, which
            has no ``z`` at all).
        """
        return isinstance(self.waypoints[0], Point3)

    @property
    def length(self) -> float:
        """
        :return: The distance travelled along the path.
        """
        if self._is_volumetric:
            return sum(
                float(
                    (
                        (following.x - current.x) ** 2
                        + (following.y - current.y) ** 2
                        + (following.z - current.z) ** 2
                    )
                    ** 0.5
                )
                for current, following in zip(self.waypoints, self.waypoints[1:])
            )
        return sum(
            float(
                ((following.x - current.x) ** 2 + (following.y - current.y) ** 2)
                ** 0.5
            )
            for current, following in zip(self.waypoints, self.waypoints[1:])
        )

    @property
    def vertical_travel(self) -> float:
        """
        :return: The height gained and lost along the path, which is zero for a path
            planned on a floor plan.
        """
        if not self._is_volumetric:
            return 0.0
        return sum(
            abs(float(following.z) - float(current.z))
            for current, following in zip(self.waypoints, self.waypoints[1:])
        )

    @property
    def coordinates(self) -> np.ndarray:
        """
        :return: The waypoints as an array of x-y rows, in the order they are visited.
        """
        return np.array(
            [[float(waypoint.x), float(waypoint.y)] for waypoint in self.waypoints]
        )

    @property
    def spatial_coordinates(self) -> np.ndarray:
        """
        :return: The waypoints as an array of x-y-z rows, in the order they are visited.
        """
        return np.array(
            [
                [float(waypoint.x), float(waypoint.y), float(waypoint.z)]
                for waypoint in self.waypoints
            ]
        )


@dataclass(frozen=True)
class ConvexSetAdjacency:
    """
    One edge of the graph, drawn the way the planner crosses it: from the center of one
    convex set, through the portal the two sets share, to the center of the other.
    """

    source_center: Point
    """
    The center of the convex set the edge starts at.
    """

    portal_center: Point
    """
    The center of the region where the two convex sets overlap.
    """

    target_center: Point
    """
    The center of the convex set the edge ends at.
    """

    @property
    def spatial_coordinates(self) -> np.ndarray:
        """
        :return: The edge as an array of x-y-z rows, from source through portal to
            target.
        """
        return np.array(
            [
                [float(point.x), float(point.y), float(point.z)]
                for point in (
                    self.source_center,
                    self.portal_center,
                    self.target_center,
                )
            ]
        )

    @property
    def coordinates(self) -> np.ndarray:
        """
        :return: The edge as an array of x-y rows, from source through portal to target.
        """
        return np.array(
            [
                [float(point.x), float(point.y)]
                for point in (
                    self.source_center,
                    self.portal_center,
                    self.target_center,
                )
            ]
        )


@dataclass
class NavigationScene:
    """
    An already-built graph of convex sets and the path drawn over it, presented to the
    panels that draw them.

    Building the graph (from a world or otherwise) and solving the path are both the
    caller's job: see
    :class:`~semantic_digital_twin.world_description.graph_of_convex_sets.boxes.GraphOfBoundingBoxes`'s
    ``navigation_map_from_world``/``free_space_from_world`` and ``path_from_to``.
    """

    graph_of_convex_sets: GraphOfBoundingBoxes
    """
    The decomposition of free space into convex sets, and their adjacency graph.
    """

    environment_name: str
    """
    The label the figure is titled with.
    """

    path: NavigationPath
    """
    The path to draw, from its start to its goal.
    """

    obstacles: Optional[BoundingBoxCollection] = None
    """
    The environment's true collision geometry, unbloated.

    Pass ``None`` to derive it as whatever the search space is not free space;
    ``__post_init__`` resolves that default, so this attribute is never ``None`` once
    the object exists. The graph was built from *bloated* obstacles, though, so the
    derived region already includes the clearance margin and is larger than the true
    geometry. Pass the real geometry explicitly (already at hand while building the
    graph) wherever the difference matters, such as showing how far a convex set
    actually stays from an obstacle.
    """

    def __post_init__(self):
        if self.obstacles is None:
            occupied_space = (
                ~self.graph_of_convex_sets.free_space_event & self.search_space.event
            )
            box_type = type(self.search_space.bounding_boxes[0])
            self.obstacles = BoundingBoxCollection.from_event(
                box_type,
                reference_frame=self.search_space.reference_frame,
                event=occupied_space,
            )

    @property
    def search_space(self) -> BoundingBoxCollection:
        """
        :return: The volume the graph of convex sets was built in, and the extent every
            panel is framed to.
        """
        return self.graph_of_convex_sets.search_space

    @property
    def extent(self) -> Footprint:
        """
        :return: The x-y extent every panel is framed to.
        """
        return Footprint.of(self.search_space.bounding_box())

    @property
    def convex_sets(self) -> List[AxisAlignedBox]:
        """
        :return: The convex sets partitioning free space.
        """
        return list(self.graph_of_convex_sets.graph.nodes())

    @property
    def adjacencies(self) -> List[ConvexSetAdjacency]:
        """
        :return: One adjacency per edge of the graph.
        """
        graph = self.graph_of_convex_sets.graph
        return [
            ConvexSetAdjacency(
                source_center=graph[source].center,
                portal_center=adjacency.intersection.center,
                target_center=graph[target].center,
            )
            for source, target, adjacency in graph.weighted_edge_list()
        ]


# %% the layers a panel is drawn from


@dataclass(frozen=True)
class LegendEntry:
    """
    One row of the figure's shared legend.
    """

    handle: Artist
    """
    The sample mark drawn beside the label.
    """

    label: str
    """
    The name of the entity the mark stands for.
    """


@dataclass(frozen=True)
class SceneLayer(ABC):
    """
    One kind of mark a panel draws, so that panels differ in what they compose rather
    than in how any one entity is drawn.
    """

    @abstractmethod
    def draw(
        self, axes: Axes, scene: NavigationScene, palette: FigurePalette
    ) -> Sequence[LegendEntry]:
        """
        Draw this layer's marks.

        :param axes: The panel to draw into.
        :param scene: The scene to draw.
        :param palette: The colors to draw with.
        :return: The legend entries this layer contributes.
        """
        raise NotImplementedError


@dataclass(frozen=True)
class SearchSpaceLayer(SceneLayer):
    """
    Draws the outline of every part of the volume the graph of convex sets was built in,
    which in a world with floor geometry is one outline per floor.
    """

    def draw(
        self, axes: Axes, scene: NavigationScene, palette: FigurePalette
    ) -> Sequence[LegendEntry]:
        outlines = [
            Footprint.of(part).as_rectangle(
                facecolor="none",
                edgecolor=palette.text_secondary,
                linewidth=0.8,
                zorder=10,
            )
            for part in scene.search_space
        ]
        for outline in outlines:
            outline.set_linestyle((0, (4, 3)))
            axes.add_patch(outline)
        return [LegendEntry(outlines[0], "search space")]


@dataclass(frozen=True)
class ObstacleLayer(SceneLayer):
    """
    Draws the environment's obstacle footprints, outlined in a darker tint of their own
    fill so that abutting obstacles stay countable.
    """

    def draw(
        self, axes: Axes, scene: NavigationScene, palette: FigurePalette
    ) -> Sequence[LegendEntry]:
        rectangles = [
            Footprint.of(obstacle).as_rectangle(
                facecolor=palette.obstacle,
                edgecolor=palette.obstacle_edge,
                linewidth=0.4,
                zorder=2,
            )
            for obstacle in scene.obstacles
        ]
        for rectangle in rectangles:
            axes.add_patch(rectangle)
        return [LegendEntry(rectangles[0], "obstacle")] if rectangles else []


class ConvexSetEmphasis(enum.Enum):
    """
    Whether the convex sets are the subject of a panel or its context.
    """

    SUBJECT = enum.auto()
    CONTEXT = enum.auto()

    def fill(self, palette: FigurePalette) -> str:
        """
        :param palette: The colors of the figure.
        :return: The fill the sets take at this emphasis.
        """
        return (
            palette.convex_set
            if self is ConvexSetEmphasis.SUBJECT
            else palette.receded_convex_set
        )

    def edge(self, palette: FigurePalette) -> str:
        """
        :param palette: The colors of the figure.
        :return: The color separating neighbouring sets at this emphasis.
        """
        return (
            palette.convex_set_edge
            if self is ConvexSetEmphasis.SUBJECT
            else palette.convex_set
        )


@dataclass(frozen=True)
class ConvexSetsLayer(SceneLayer):
    """
    Draws the convex sets partitioning free space, outlined in a darker tint of their
    own fill so that the partition is visible without slicing free space into stripes
    where the sets are narrow.
    """

    emphasis: ConvexSetEmphasis = ConvexSetEmphasis.SUBJECT
    """
    Whether the sets are the panel's subject or the background its subject sits on.
    """

    def draw(
        self, axes: Axes, scene: NavigationScene, palette: FigurePalette
    ) -> Sequence[LegendEntry]:
        rectangles = [
            Footprint.of(convex_set).as_rectangle(
                facecolor=self.emphasis.fill(palette),
                edgecolor=self.emphasis.edge(palette),
                linewidth=0.4,
                zorder=3,
            )
            for convex_set in scene.convex_sets
        ]
        for rectangle in rectangles:
            axes.add_patch(rectangle)
        if self.emphasis is ConvexSetEmphasis.CONTEXT:
            return []
        return [LegendEntry(rectangles[0], "convex set")] if rectangles else []


@dataclass(frozen=True)
class AdjacencyLayer(SceneLayer):
    """
    Draws the graph over the convex sets: a node at every set's center and an edge
    between every pair of adjacent sets.
    """

    def draw(
        self, axes: Axes, scene: NavigationScene, palette: FigurePalette
    ) -> Sequence[LegendEntry]:
        for adjacency in scene.adjacencies:
            coordinates = adjacency.coordinates
            axes.plot(
                coordinates[:, 0],
                coordinates[:, 1],
                color=palette.adjacency,
                linewidth=1.0,
                solid_capstyle="round",
                solid_joinstyle="round",
                zorder=4,
            )
        centers = np.array(
            [
                [float(convex_set.center.x), float(convex_set.center.y)]
                for convex_set in scene.convex_sets
            ]
        )
        axes.plot(
            centers[:, 0],
            centers[:, 1],
            linestyle="none",
            marker="o",
            markersize=3.0,
            color=palette.adjacency,
            markeredgecolor=palette.surface,
            markeredgewidth=0.5,
            zorder=5,
        )
        sample = Line2D(
            [],
            [],
            color=palette.adjacency,
            linewidth=1.0,
            marker="o",
            markersize=3.0,
        )
        return [LegendEntry(sample, "adjacency graph")]


@dataclass(frozen=True)
class PathLayer(SceneLayer):
    """
    Draws the optimal path and the waypoints it turns at.
    """

    def draw(
        self, axes: Axes, scene: NavigationScene, palette: FigurePalette
    ) -> Sequence[LegendEntry]:
        coordinates = scene.path.coordinates
        line = axes.plot(
            coordinates[:, 0],
            coordinates[:, 1],
            color=palette.path,
            linewidth=2.0,
            solid_capstyle="round",
            solid_joinstyle="round",
            zorder=6,
        )
        axes.plot(
            coordinates[1:-1, 0],
            coordinates[1:-1, 1],
            linestyle="none",
            marker="o",
            markersize=4.0,
            color=palette.path,
            markeredgecolor=palette.surface,
            markeredgewidth=0.6,
            zorder=7,
        )
        return [LegendEntry(line[0], "optimal path")]


@dataclass(frozen=True)
class EndpointsLayer(SceneLayer):
    """
    Draws the start and the goal of the path, each labelled beside its marker so that
    neither is identified by color alone.
    """

    label_offset: float = 0.18
    """
    Distance in meters between an endpoint's marker and its label.
    """

    def draw(
        self, axes: Axes, scene: NavigationScene, palette: FigurePalette
    ) -> Sequence[LegendEntry]:
        waypoints = scene.path.waypoints
        return [
            self._draw_endpoint(
                axes, waypoints[0], "start", "o", palette.start, palette
            ),
            self._draw_endpoint(
                axes, waypoints[-1], "goal", "s", palette.goal, palette
            ),
        ]

    def _draw_endpoint(
        self,
        axes: Axes,
        endpoint: Point,
        label: str,
        marker: str,
        color: str,
        palette: FigurePalette,
    ) -> LegendEntry:
        """
        :param axes: The panel to draw into.
        :param endpoint: The point to mark.
        :param label: The name written beside the marker.
        :param marker: The matplotlib marker to draw.
        :param color: The color of the marker.
        :param palette: The colors of the figure.
        :return: The legend entry of this endpoint.
        """
        x, y = float(endpoint.x), float(endpoint.y)
        drawn = axes.plot(
            [x],
            [y],
            linestyle="none",
            marker=marker,
            markersize=8.0,
            color=color,
            markeredgecolor=palette.surface,
            markeredgewidth=1.0,
            zorder=8,
        )
        axes.annotate(
            label,
            (x, y),
            xytext=(self.label_offset, self.label_offset),
            textcoords="offset fontsize",
            color=palette.text_primary,
            fontsize=8.0,
            zorder=9,
        )
        return LegendEntry(drawn[0], label)


# %% the panels of the figure


@dataclass(frozen=True)
class ScenePanel(ABC):
    """
    One sub-plot of the figure, defined by the layers it composes and the statistic it
    reports about the scene.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """
        :return: The panel's title.
        """
        raise NotImplementedError

    @abstractmethod
    def subtitle(self, scene: NavigationScene) -> str:
        """
        :param scene: The scene the panel draws.
        :return: The line under the title, quantifying what the panel shows.
        """
        raise NotImplementedError

    @abstractmethod
    def layers(self) -> Sequence[SceneLayer]:
        """
        :return: The layers to draw, in the order they are stacked.
        """
        raise NotImplementedError

    def draw(
        self,
        axes: Axes,
        scene: NavigationScene,
        palette: FigurePalette,
        panel_label: str,
    ) -> Sequence[LegendEntry]:
        """
        Draw the whole panel.

        :param axes: The sub-plot to draw into.
        :param scene: The scene to draw.
        :param palette: The colors to draw with.
        :param panel_label: The enumerating label the title is prefixed with.
        :return: The legend entries the panel's layers contribute.
        """
        self._frame(axes, scene, palette)
        entries = [
            entry
            for layer in self.layers()
            for entry in layer.draw(axes, scene, palette)
        ]
        axes.set_title(
            f"{panel_label} {self.name}",
            loc="left",
            pad=14.0,
            fontsize=11.0,
            fontweight="semibold",
            color=palette.text_primary,
        )
        axes.text(
            0.0,
            1.01,
            self.subtitle(scene),
            transform=axes.transAxes,
            fontsize=8.5,
            color=palette.text_secondary,
        )
        return entries

    def _frame(
        self, axes: Axes, scene: NavigationScene, palette: FigurePalette
    ) -> None:
        """
        Give the panel the metric frame every panel shares, so the three read against
        each other.

        :param axes: The sub-plot to frame.
        :param scene: The scene whose extent the panel is framed to.
        :param palette: The colors to draw with.
        """
        extent = scene.extent
        margin = 0.02 * max(extent.width, extent.height)
        axes.set_xlim(extent.min_x - margin, extent.max_x + margin)
        axes.set_ylim(extent.min_y - margin, extent.max_y + margin)
        axes.set_aspect("equal")
        axes.set_facecolor(palette.surface)
        axes.set_xlabel("x [m]", fontsize=9.0, color=palette.text_secondary)
        axes.set_ylabel("y [m]", fontsize=9.0, color=palette.text_secondary)
        axes.tick_params(
            labelsize=8.0, colors=palette.text_secondary, length=3.0, width=0.6
        )
        for spine in axes.spines.values():
            spine.set_visible(False)


@dataclass(frozen=True)
class EnvironmentPanel(ScenePanel):
    """
    Shows what the planner is given: the obstacles of the environment and the search
    space bounding them.
    """

    @property
    def name(self) -> str:
        return "Environment"

    def subtitle(self, scene: NavigationScene) -> str:
        extent = scene.extent
        return (
            f"{len(scene.obstacles)} obstacle footprints in "
            f"{extent.width:.1f} × {extent.height:.1f} m"
        )

    def layers(self) -> Sequence[SceneLayer]:
        return (SearchSpaceLayer(), ObstacleLayer())


@dataclass(frozen=True)
class ConvexSetsPanel(ScenePanel):
    """
    Shows what the planner builds: free space partitioned into convex sets, and the
    graph connecting the sets that touch.
    """

    @property
    def name(self) -> str:
        return "Graph of convex sets"

    def subtitle(self, scene: NavigationScene) -> str:
        return (
            f"{len(scene.convex_sets)} convex sets, "
            f"{len(scene.adjacencies)} adjacencies"
        )

    def layers(self) -> Sequence[SceneLayer]:
        return (ObstacleLayer(), ConvexSetsLayer(), AdjacencyLayer())


@dataclass(frozen=True)
class OptimalPathPanel(ScenePanel):
    """
    Shows what the planner returns: the minimum-distance path between the queried start
    and goal, threading through the convex sets.
    """

    @property
    def name(self) -> str:
        return "Optimal path"

    def subtitle(self, scene: NavigationScene) -> str:
        return f"{scene.path.length:.2f} m over {len(scene.path.waypoints)} waypoints"

    def layers(self) -> Sequence[SceneLayer]:
        return (
            ObstacleLayer(),
            ConvexSetsLayer(ConvexSetEmphasis.CONTEXT),
            PathLayer(),
            EndpointsLayer(),
        )


# %% the figure


@dataclass(frozen=True)
class PanelArrangement:
    """
    How the panels are tiled, chosen so that a wide environment is stacked and a compact
    one is placed side by side.
    """

    rows: int
    """
    Number of panel rows.
    """

    columns: int
    """
    Number of panel columns.
    """

    wide_aspect_ratio: float = 1.4
    """
    Width-to-height ratio above which an environment is considered wide.
    """

    maximum_width_inches: float = 10.0
    """
    Largest figure width the arrangement will produce.
    """

    preferred_inches_per_meter: float = 0.5
    """
    Scale used unless it would exceed :attr:`maximum_width_inches`.
    """

    title_inches_per_row: float = 0.75
    """
    Vertical space reserved per row for its panel's title and axis labels.
    """

    legend_inches: float = 1.0
    """
    Vertical space reserved for the figure title and the shared legend.
    """

    legend_entry_inches: float = 1.6
    """
    Width one legend entry takes, used to decide how many fit on a row.
    """

    @classmethod
    def for_extent(cls, extent: Footprint, panel_count: int) -> Self:
        """
        :param extent: The x-y extent one panel spans.
        :param panel_count: Number of panels to tile.
        :return: The arrangement to tile them with.
        """
        if extent.width > cls.wide_aspect_ratio * extent.height:
            return cls(rows=panel_count, columns=1)
        return cls(rows=1, columns=panel_count)

    def figure_size(self, extent: Footprint) -> tuple[float, float]:
        """
        :param extent: The x-y extent one panel spans.
        :return: The figure's width and height in inches, scaled so that a meter is the
            same length in every panel.
        """
        scale = min(
            self.preferred_inches_per_meter,
            self.maximum_width_inches / (self.columns * extent.width),
        )
        width = self.columns * extent.width * scale
        height = (
            self.rows * (extent.height * scale + self.title_inches_per_row)
            + self.legend_inches
        )
        return width, height

    def legend_columns(self, extent: Footprint, entry_count: int) -> int:
        """
        Wrap the legend so that a narrow figure does not run it off the page, and spread
        the entries evenly over however many rows that takes rather than leaving a last
        row with one entry in it.

        :param extent: The x-y extent one panel spans.
        :param entry_count: Number of legend entries to lay out.
        :return: How many entries to put on one legend row.
        """
        width, _ = self.figure_size(extent)
        entries_that_fit = max(1, int(width / self.legend_entry_inches))
        rows = math.ceil(entry_count / entries_that_fit)
        return math.ceil(entry_count / rows)


@dataclass
class GraphOfConvexSetsFigure:
    """
    The three-panel figure of a scene: its environment, its graph of convex sets, and
    the optimal path over that graph.
    """

    scene: NavigationScene
    """
    The scene every panel draws.
    """

    theme: Theme = Theme.LIGHT
    """
    The surface the figure is rendered for.
    """

    panels: tuple[ScenePanel, ...] = field(
        default_factory=lambda: (
            EnvironmentPanel(),
            ConvexSetsPanel(),
            OptimalPathPanel(),
        )
    )
    """
    The panels to draw, left to right or top to bottom.
    """

    dots_per_inch: int = 300
    """
    Resolution of the rendered raster image.
    """

    def render(self) -> Figure:
        """
        Draw the figure.

        :return: The rendered figure, which the caller owns and has to close.
        """
        palette = self.theme.value
        arrangement = PanelArrangement.for_extent(self.scene.extent, len(self.panels))
        with plt.rc_context(self._rc_parameters(palette)):
            figure, axes_grid = plt.subplots(
                arrangement.rows,
                arrangement.columns,
                figsize=arrangement.figure_size(self.scene.extent),
                layout="constrained",
            )
            entries = self._draw_panels(np.atleast_1d(axes_grid).ravel(), palette)
            figure.suptitle(
                f"Graph of convex sets navigation in {self.scene.environment_name}",
                fontsize=13.0,
                fontweight="semibold",
                color=palette.text_primary,
                x=0.02,
                horizontalalignment="left",
            )
            figure.legend(
                handles=[entry.handle for entry in entries],
                labels=[entry.label for entry in entries],
                loc="outside lower left",
                ncols=arrangement.legend_columns(self.scene.extent, len(entries)),
                frameon=False,
                fontsize=9.0,
                labelcolor=palette.text_secondary,
            )
        return figure

    def save(self, output_directory: Path) -> List[Path]:
        """
        Render the figure and write it as both a vector and a raster file.

        :param output_directory: Directory the files are written to; created if missing.
        :return: The written paths.
        """
        output_directory.mkdir(parents=True, exist_ok=True)
        stem = (
            f"graph_of_convex_sets_{self.scene.environment_name}_"
            f"{self.theme.name.lower()}"
        )
        figure = self.render()
        written = []
        for suffix in (".pdf", ".png"):
            path = output_directory / f"{stem}{suffix}"
            figure.savefig(path, dpi=self.dots_per_inch)
            written.append(path)
        plt.close(figure)
        return written

    def _draw_panels(
        self, all_axes: Iterable[Axes], palette: FigurePalette
    ) -> List[LegendEntry]:
        """
        Draw every panel and collect one legend entry per distinct label.

        :param all_axes: The sub-plots, in the order the panels are drawn into them.
        :param palette: The colors to draw with.
        :return: The legend entries of the whole figure.
        """
        entries: List[LegendEntry] = []
        seen_labels = set()
        for panel, axes, label in zip(self.panels, all_axes, self._panel_labels()):
            for entry in panel.draw(axes, self.scene, palette, label):
                if entry.label in seen_labels:
                    continue
                seen_labels.add(entry.label)
                entries.append(entry)
        return entries

    def _panel_labels(self) -> List[str]:
        """
        :return: The enumerating labels the panel titles are prefixed with.
        """
        return [
            f"({letter})"
            for letter in itertools.islice(
                (chr(ordinal) for ordinal in range(ord("a"), ord("z") + 1)),
                len(self.panels),
            )
        ]

    def _rc_parameters(self, palette: FigurePalette) -> dict[str, object]:
        """
        :param palette: The colors to draw with.
        :return: The matplotlib settings the figure is rendered under, applied in a
            context so that no global state is changed.
        """
        return {
            "figure.facecolor": palette.surface,
            "savefig.facecolor": palette.surface,
            "axes.facecolor": palette.surface,
            "text.color": palette.text_primary,
            "axes.labelcolor": palette.text_secondary,
            "axes.grid": False,
            "font.size": 9.0,
        }
