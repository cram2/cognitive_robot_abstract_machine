"""
Renders the three-panel Graph of Convex Sets figure as a volume rather than a floor
plan.

The panels answer the same three questions as the two-dimensional figure -- what the
planner is given, what it builds, what it returns -- but over a
:class:`~semantic_digital_twin.world_description.graph_of_convex_sets.plotting.VolumetricDecomposition`,
so free space is partitioned in all three dimensions and a path may change height to
pass over what it cannot pass beside.

Drawn with plotly, which the graph of convex sets already uses for its own
three-dimensional plots. Every run writes an interactive page next to the static image,
since a single camera angle hides whatever it projects behind something else.
"""

from __future__ import annotations

import enum
import itertools
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import numpy.typing as npt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from random_events.plotting import EventPlotter
from random_events.product_algebra import Event
from typing_extensions import List, Sequence

from semantic_digital_twin.spatial_types import Point3
from semantic_digital_twin.world_description.geometry import Bounds
from semantic_digital_twin.world_description.graph_of_convex_sets.plotting import (
    FigurePalette,
    NavigationScene,
    Theme,
)

# %% shared trace helpers


def polyline_trace(
    polylines: Sequence[npt.NDArray[np.float64]], color: str, width: float, name: str
) -> go.Scatter3d:
    """
    Join separate polylines into a single trace, separated by gaps.

    :param polylines: The polylines, each an array of x-y-z rows.
    :param color: The color of the lines.
    :param width: The width of the lines in pixels.
    :param name: The name the trace carries into the legend.
    :return: The trace.
    """
    separator = np.full((1, 3), np.nan)
    points = np.vstack([row for polyline in polylines for row in (polyline, separator)])
    return go.Scatter3d(
        x=points[:, 0],
        y=points[:, 1],
        z=points[:, 2],
        mode="lines",
        line=dict(color=color, width=width),
        hoverinfo="skip",
        name=name,
        legendgroup=name,
    )


# %% the layers a panel is drawn from


@dataclass(frozen=True)
class VolumeLayer(ABC):
    """
    One kind of geometry a panel draws, so that panels differ in what they compose
    rather than in how any one entity is drawn.
    """

    @abstractmethod
    def traces(
        self, scene: NavigationScene, palette: FigurePalette
    ) -> Sequence[go.BaseTraceType]:
        """
        :param scene: The scene to draw.
        :param palette: The colors to draw with.
        :return: The traces to add to the panel.
        """
        raise NotImplementedError


@dataclass(frozen=True)
class SearchSpaceVolumeLayer(VolumeLayer):
    """
    Draws the volume the graph of convex sets was built in, as a wireframe so that
    everything inside it stays visible.
    """

    def traces(
        self, scene: NavigationScene, palette: FigurePalette
    ) -> Sequence[go.BaseTraceType]:
        event = Event.from_simple_sets(
            *(box.simple_event for box in scene.search_space)
        )
        plotter = EventPlotter(event)
        return [plotter.plot_3d_wireframe(palette.text_secondary, 2.0, "search space")]


class ObstacleEmphasis(float, enum.Enum):
    """
    Whether the obstacles are the subject of a panel or the context something else is
    read against. Its value is how opaque the obstacles are at that emphasis.

    Neither is opaque: a room is seen from outside, so solid walls would hide every
    thing the figure is about. Both steps let what is behind them show through, and the
    wireframe edges are what keep the geometry readable.
    """

    SUBJECT = 0.45
    CONTEXT = 0.2


@dataclass(frozen=True)
class ObstacleVolumeLayer(VolumeLayer):
    """
    Draws the environment's collision geometry as boxes.
    """

    emphasis: ObstacleEmphasis = ObstacleEmphasis.SUBJECT
    """
    Whether the obstacles are the panel's subject or the context its subject sits in.
    """

    def traces(
        self, scene: NavigationScene, palette: FigurePalette
    ) -> Sequence[go.BaseTraceType]:
        event = Event.from_simple_sets(*(box.simple_event for box in scene.obstacles))
        plotter = EventPlotter(event)
        return [
            plotter.plot_3d_mesh(palette.obstacle, self.emphasis.value, "obstacle"),
            plotter.plot_3d_wireframe(palette.obstacle_edge, 1.0, "obstacle"),
        ]


@dataclass(frozen=True)
class ConvexSetVolumeLayer(VolumeLayer):
    """
    Draws the convex sets partitioning free space, translucent enough to see the sets
    behind them and wireframed so the partition itself is legible.
    """

    fill_opacity: float = 0.08
    """
    How opaque a convex set's fill is; low enough that a stack of them does not turn
    into a solid block.
    """

    def traces(
        self, scene: NavigationScene, palette: FigurePalette
    ) -> Sequence[go.BaseTraceType]:
        event = Event.from_simple_sets(*(box.simple_event for box in scene.convex_sets))
        plotter = EventPlotter(event)
        return [
            plotter.plot_3d_mesh(palette.convex_set, self.fill_opacity, "convex set"),
            plotter.plot_3d_wireframe(palette.convex_set_edge, 1.0, "convex set"),
        ]


@dataclass(frozen=True)
class AdjacencyVolumeLayer(VolumeLayer):
    """
    Draws the graph over the convex sets: a node at every set's center and an edge
    through the portal every pair of adjacent sets shares.
    """

    def traces(
        self, scene: NavigationScene, palette: FigurePalette
    ) -> Sequence[go.BaseTraceType]:
        centers = np.array(
            [
                [
                    float(convex_set.center.x),
                    float(convex_set.center.y),
                    float(convex_set.center.z),
                ]
                for convex_set in scene.convex_sets
            ]
        )
        return [
            polyline_trace(
                [adjacency.spatial_coordinates for adjacency in scene.adjacencies],
                palette.adjacency,
                2.0,
                "adjacency graph",
            ),
            go.Scatter3d(
                x=centers[:, 0],
                y=centers[:, 1],
                z=centers[:, 2],
                mode="markers",
                marker=dict(size=2.5, color=palette.adjacency),
                hoverinfo="skip",
                name="adjacency graph",
                legendgroup="adjacency graph",
            ),
        ]


@dataclass(frozen=True)
class PathVolumeLayer(VolumeLayer):
    """
    Draws the optimal path and the waypoints it turns at.
    """

    def traces(
        self, scene: NavigationScene, palette: FigurePalette
    ) -> Sequence[go.BaseTraceType]:
        coordinates = scene.path.spatial_coordinates
        return [
            go.Scatter3d(
                x=coordinates[:, 0],
                y=coordinates[:, 1],
                z=coordinates[:, 2],
                mode="lines+markers",
                line=dict(color=palette.path, width=8.0),
                marker=dict(size=4.0, color=palette.path),
                hoverinfo="skip",
                name="optimal path",
                legendgroup="optimal path",
            )
        ]


@dataclass(frozen=True)
class EndpointsVolumeLayer(VolumeLayer):
    """
    Draws the start and the goal of the path, each labelled beside its marker so that
    neither is identified by color alone.
    """

    def traces(
        self, scene: NavigationScene, palette: FigurePalette
    ) -> Sequence[go.BaseTraceType]:
        waypoints = scene.path.waypoints
        return [
            self._endpoint_trace(waypoints[0], "start", "circle", palette.start),
            self._endpoint_trace(waypoints[-1], "goal", "square", palette.goal),
        ]

    @staticmethod
    def _endpoint_trace(
        endpoint: Point3, label: str, symbol: str, color: str
    ) -> go.Scatter3d:
        """
        :param endpoint: The point to mark.
        :param label: The name written beside the marker.
        :param symbol: The plotly marker symbol to draw.
        :param color: The color of the marker.
        :return: The trace marking the endpoint.
        """
        return go.Scatter3d(
            x=[float(endpoint.x)],
            y=[float(endpoint.y)],
            z=[float(endpoint.z)],
            mode="markers+text",
            marker=dict(size=7.0, color=color, symbol=symbol),
            text=[label],
            textposition="top center",
            hoverinfo="skip",
            name=label,
            legendgroup=label,
        )


# %% the panels of the figure


@dataclass(frozen=True)
class VolumePanel(ABC):
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
    def layers(self) -> Sequence[VolumeLayer]:
        """
        :return: The layers to draw, in the order they are stacked.
        """
        raise NotImplementedError

    def traces(
        self, scene: NavigationScene, palette: FigurePalette
    ) -> List[go.BaseTraceType]:
        """
        :param scene: The scene to draw.
        :param palette: The colors to draw with.
        :return: Every trace the panel's layers contribute.
        """
        return [
            trace for layer in self.layers() for trace in layer.traces(scene, palette)
        ]


@dataclass(frozen=True)
class EnvironmentVolumePanel(VolumePanel):
    """
    Shows what the planner is given: the collision geometry of a world and the volume
    bounding it.
    """

    @property
    def name(self) -> str:
        return "Environment"

    def subtitle(self, scene: NavigationScene) -> str:
        bounds = scene.search_space.bounding_box().to_array_bounds()
        extent = bounds.upper - bounds.lower
        return (
            f"{len(scene.obstacles)} obstacle boxes in "
            f"{extent[0]:.1f} × {extent[1]:.1f} × {extent[2]:.1f} m"
        )

    def layers(self) -> Sequence[VolumeLayer]:
        return (SearchSpaceVolumeLayer(), ObstacleVolumeLayer())


@dataclass(frozen=True)
class ConvexSetsVolumePanel(VolumePanel):
    """
    Shows what the planner builds: free space partitioned into convex sets in all three
    dimensions, and the graph connecting the sets that touch.
    """

    @property
    def name(self) -> str:
        return "Graph of convex sets"

    def subtitle(self, scene: NavigationScene) -> str:
        return (
            f"{len(scene.convex_sets)} convex sets, "
            f"{len(scene.adjacencies)} adjacencies"
        )

    def layers(self) -> Sequence[VolumeLayer]:
        return (ConvexSetVolumeLayer(), AdjacencyVolumeLayer())


@dataclass(frozen=True)
class OptimalPathVolumePanel(VolumePanel):
    """
    Shows what the planner returns: the minimum-distance path between the queried start
    and goal, including the height it changes on the way.

    The convex sets are left out here; translucent boxes filling the whole volume would
    hide the one line the panel is about.
    """

    @property
    def name(self) -> str:
        return "Optimal path"

    def subtitle(self, scene: NavigationScene) -> str:
        return (
            f"{scene.path.length:.2f} m over {len(scene.path.waypoints)} waypoints, "
            f"{scene.path.vertical_travel:.2f} m of it vertical"
        )

    def layers(self) -> Sequence[VolumeLayer]:
        return (
            ObstacleVolumeLayer(ObstacleEmphasis.CONTEXT),
            PathVolumeLayer(),
            EndpointsVolumeLayer(),
        )


# %% the figure


@dataclass(frozen=True)
class SceneCamera:
    """
    Where every panel is looked at from, shared by all of them so the three read against
    each other.
    """

    eye: Point3 = field(default_factory=lambda: Point3(1.25, -1.25, 1.15))
    """
    The viewpoint, in units of the scene's own size.
    """

    def as_plotly(self) -> dict[str, object]:
        """
        :return: The camera as plotly expects it, with z up.
        """
        return dict(
            eye=dict(x=float(self.eye.x), y=float(self.eye.y), z=float(self.eye.z)),
            up=dict(x=0.0, y=0.0, z=1.0),
        )


@dataclass
class GraphOfConvexSetsVolumeFigure:
    """
    The three-panel volume figure of a scene: its environment, its graph of convex sets,
    and the optimal path over that graph.
    """

    scene: NavigationScene
    """
    The scene every panel draws.
    """

    theme: Theme = Theme.LIGHT
    """
    The surface the figure is rendered for.
    """

    camera: SceneCamera = field(default_factory=SceneCamera)
    """
    Where the panels are looked at from.
    """

    panels: tuple[VolumePanel, ...] = field(
        default_factory=lambda: (
            EnvironmentVolumePanel(),
            ConvexSetsVolumePanel(),
            OptimalPathVolumePanel(),
        )
    )
    """
    The panels to draw, left to right.
    """

    image_formats: tuple[str, ...] = (".pdf", ".png")
    """
    The static formats written beside the interactive page.

    Rendering these drives a headless browser, so pass an empty sequence where only the
    page is wanted.
    """

    panel_width_pixels: int = 620
    """
    Width one panel is rendered at.
    """

    height_pixels: int = 520
    """
    Height the figure is rendered at.
    """

    image_scale: int = 2
    """
    Factor the raster image is rendered at above the nominal size.
    """

    def render(self) -> go.Figure:
        """
        Draw the figure.

        :return: The rendered figure.
        """
        palette = self.theme.value
        figure = make_subplots(
            rows=1,
            cols=len(self.panels),
            specs=[[{"type": "scene"}] * len(self.panels)],
            horizontal_spacing=0.02,
        )
        self._add_panels(figure, palette)
        self._style(figure, palette)
        return figure

    def save(self, output_directory: Path) -> List[Path]:
        """
        Render the figure and write it as an interactive page, a vector file and a
        raster file.

        :param output_directory: Directory the files are written to; created if missing.
        :return: The written paths.
        """
        output_directory.mkdir(parents=True, exist_ok=True)
        stem = (
            f"graph_of_convex_sets_volume_{self.scene.environment_name}_"
            f"{self.theme.name.lower()}"
        )
        figure = self.render()

        written = [output_directory / f"{stem}.html"]
        figure.write_html(str(written[0]), include_plotlyjs=True)
        for suffix in self.image_formats:
            path = output_directory / f"{stem}{suffix}"
            figure.write_image(
                str(path),
                width=self.panel_width_pixels * len(self.panels),
                height=self.height_pixels,
                scale=self.image_scale,
            )
            written.append(path)
        return written

    def _add_panels(self, figure: go.Figure, palette: FigurePalette) -> None:
        """
        Add every panel's traces, showing each legend entry once across the figure.

        :param figure: The figure to add to.
        :param palette: The colors to draw with.
        """
        shown_groups = set()
        for column, panel in enumerate(self.panels, start=1):
            for trace in panel.traces(self.scene, palette):
                trace.showlegend = trace.legendgroup not in shown_groups
                shown_groups.add(trace.legendgroup)
                figure.add_trace(trace, row=1, col=column)

    def _style(self, figure: go.Figure, palette: FigurePalette) -> None:
        """
        Give the figure its titles, its shared camera and its theme.

        :param figure: The figure to style.
        :param palette: The colors to draw with.
        """
        figure.update_layout(
            title=dict(
                text=(
                    "Graph of convex sets navigation in "
                    f"{self.scene.environment_name}"
                ),
                x=0.01,
                font=dict(size=20, color=palette.text_primary),
            ),
            paper_bgcolor=palette.surface,
            font=dict(color=palette.text_primary, size=12),
            legend=dict(
                orientation="h",
                x=0.0,
                y=-0.02,
                bgcolor="rgba(0,0,0,0)",
                font=dict(color=palette.text_secondary),
            ),
            margin=dict(l=0, r=0, t=110, b=10),
            annotations=self._panel_titles(palette),
        )
        # Every panel is pinned to the same box rather than auto-scaled to its own
        # traces, so a metre is the same length in all three and they can be read
        # against each other.
        bounds = self.scene.search_space.bounding_box().to_array_bounds()
        for index in range(1, len(self.panels) + 1):
            figure.update_layout(
                {
                    f"scene{index}": dict(
                        aspectmode="data",
                        camera=self.camera.as_plotly(),
                        xaxis=self._axis("x [m]", palette, bounds, 0),
                        yaxis=self._axis("y [m]", palette, bounds, 1),
                        zaxis=self._axis("z [m]", palette, bounds, 2),
                    )
                }
            )

    def _panel_titles(self, palette: FigurePalette) -> List[dict[str, object]]:
        """
        :param palette: The colors to draw with.
        :return: One annotation per panel, holding its label, its name and the statistic
            it reports.
        """
        width = 1.0 / len(self.panels)
        labels = itertools.islice(
            (chr(ordinal) for ordinal in range(ord("a"), ord("z") + 1)),
            len(self.panels),
        )
        return [
            dict(
                text=(
                    f"<b>({label}) {panel.name}</b><br>"
                    f'<span style="color:{palette.text_secondary}">'
                    f"{panel.subtitle(self.scene)}</span>"
                ),
                x=index * width,
                y=1.0,
                xref="paper",
                yref="paper",
                xanchor="left",
                yanchor="bottom",
                showarrow=False,
                align="left",
                font=dict(size=14, color=palette.text_primary),
            )
            for index, (label, panel) in enumerate(zip(labels, self.panels))
        ]

    @staticmethod
    def _axis(
        title: str, palette: FigurePalette, bounds: Bounds[np.ndarray], axis: int
    ) -> dict[str, object]:
        """
        :param title: The axis label.
        :param palette: The colors to draw with.
        :param bounds: The corners of the box every panel is framed to.
        :param axis: Which of the three axes this is.
        :return: The styling and range shared by this axis across every panel.
        """
        return dict(
            title=dict(text=title, font=dict(size=11, color=palette.text_secondary)),
            range=[float(bounds.lower[axis]), float(bounds.upper[axis])],
            backgroundcolor=palette.surface,
            gridcolor=palette.obstacle_edge,
            zerolinecolor=palette.obstacle_edge,
            tickfont=dict(size=9, color=palette.text_secondary),
        )
