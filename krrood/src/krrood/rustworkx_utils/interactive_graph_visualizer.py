from __future__ import annotations

import threading
import webbrowser
from dataclasses import dataclass, field

import rustworkx as rx
from typing_extensions import Any, ClassVar, Dict, List, Optional, Tuple, TYPE_CHECKING

from krrood.rustworkx_utils.graph_visualizer_base import (
    DEFAULT_NODE_COLOR,
    GraphLayout,
    GraphVisualizerBase,
)

if TYPE_CHECKING:
    import plotly.graph_objects as go

# Re-exported so callers and tests can import the layout enum from this module.
__all__ = ["GraphLayout", "InteractiveGraphVisualizer"]

NodePosition = Tuple[float, float]
"""A node's two-dimensional position in the drawing."""


@dataclass
class InteractiveGraphVisualizer(GraphVisualizerBase):
    """Render a live rustworkx graph as an interactive Dash application backed by Plotly.

    Hovering a node shows its label and clicking a node shows the details produced by
    :attr:`info_getter`. The figure is rebuilt on every refresh tick, so nodes and edges added
    while the graph is being built appear without restarting the application.

    .. note::
        Dash and Plotly are imported lazily so that importing this module does not require them.
        Install them with ``pip install krrood[visualization]``.
    """

    required_modules: ClassVar[Tuple[str, ...]] = ("dash", "plotly")
    """The libraries needed to render with Dash and Plotly."""

    _selected_details: List[str] = field(default_factory=list, init=False, repr=False)
    """The details of the most recently clicked node, empty when nothing is selected."""

    def compute_positions(self) -> Dict[int, NodePosition]:
        """
        :return: A position for every node, keyed by its rustworkx node index.
        """
        if self.layout is GraphLayout.LAYERED:
            return self._layered_positions()
        return self._spring_positions()

    def build_figure(self) -> go.Figure:
        """
        :return: A Plotly figure with one trace for the edges and one for the nodes.
        """
        import plotly.graph_objects as go

        positions = self.compute_positions()
        return go.Figure(
            data=[self._edge_trace(positions), self._node_trace(positions)],
            layout=self._figure_layout(),
        )

    def run(self) -> None:
        """
        Start the Dash application on a daemon thread and optionally open it in a browser.

        The server runs on a daemon thread, so the calling process must stay alive for the
        visualization to remain reachable.
        """
        self.check_dependencies()
        application = self._build_application()
        threading.Thread(
            target=lambda: application.run(port=self.port),
            daemon=True,
            name="InteractiveGraphVisualizerServer",
        ).start()
        if self.open_browser:
            webbrowser.open(f"http://127.0.0.1:{self.port}")

    def _spring_positions(self) -> Dict[int, NodePosition]:
        """Force-directed positions for every node."""
        if self.graph.num_nodes() == 0:
            return {}
        layout = rx.spring_layout(self.graph)
        return {index: tuple(layout[index]) for index in self.graph.node_indices()}

    def _layered_positions(self) -> Dict[int, NodePosition]:
        """Top-to-bottom positions grouped by distance from the source nodes."""
        source_indices = [
            index for index in self.graph.node_indices() if self.graph.in_degree(index) == 0
        ]
        if not source_indices:
            return self._spring_positions()

        positions: Dict[int, NodePosition] = {}
        for depth, layer in enumerate(rx.layers(self.graph, source_indices, index_output=True)):
            for position_in_layer, index in enumerate(layer):
                positions[index] = (float(position_in_layer), float(-depth))
        return positions

    def _node_trace(self, positions: Dict[int, NodePosition]) -> go.Scatter:
        """The scatter trace holding one clickable marker per node."""
        import plotly.graph_objects as go

        indices = list(positions)
        labels = [self.node_label(index) for index in indices]
        return go.Scatter(
            name="nodes",
            x=[positions[index][0] for index in indices],
            y=[positions[index][1] for index in indices],
            mode="markers+text",
            text=labels,
            textposition="top center",
            hovertext=labels,
            hoverinfo="text",
            customdata=indices,
            marker=dict(size=18, color=[self.node_color(index) for index in indices]),
        )

    def _edge_trace(self, positions: Dict[int, NodePosition]) -> go.Scatter:
        """The line trace connecting adjacent nodes."""
        import plotly.graph_objects as go

        x_coordinates: List[Optional[float]] = []
        y_coordinates: List[Optional[float]] = []
        for source, target in self.graph.edge_list():
            if source not in positions or target not in positions:
                continue
            x_coordinates += [positions[source][0], positions[target][0], None]
            y_coordinates += [positions[source][1], positions[target][1], None]
        return go.Scatter(
            name="edges",
            x=x_coordinates,
            y=y_coordinates,
            mode="lines",
            line=dict(color="#888888"),
            hoverinfo="none",
        )

    def _figure_layout(self) -> go.Layout:
        """The axis-free layout used for the drawing."""
        import plotly.graph_objects as go

        hidden_axis = dict(showgrid=False, zeroline=False, showticklabels=False)
        return go.Layout(
            title=self.title,
            showlegend=False,
            xaxis=hidden_axis,
            yaxis=hidden_axis,
            margin=dict(l=20, r=20, t=40, b=20),
        )

    def _build_application(self):
        """Assemble the Dash application, its layout and its callbacks."""
        from dash import Dash, Input, Output, dcc, html

        application = Dash(__name__)
        application.layout = html.Div(
            style={"display": "flex"},
            children=[
                dcc.Graph(
                    id="graph",
                    figure=self.build_figure(),
                    style={"flex": "3", "height": "90vh"},
                ),
                html.Div(
                    id="details",
                    style={"flex": "1", "padding": "1rem"},
                    children=self._render_details(),
                ),
                dcc.Interval(
                    id="refresh",
                    interval=int(self.refresh_interval_seconds * 1000),
                ),
            ],
        )

        @application.callback(Output("graph", "figure"), Input("refresh", "n_intervals"))
        def refresh_figure(_):
            return self.build_figure()

        @application.callback(Output("details", "children"), Input("graph", "clickData"))
        def show_details(click_data):
            if click_data is None:
                return self._render_details()
            self._selected_details = self.node_details(
                click_data["points"][0]["customdata"]
            )
            return self._render_details()

        return application

    def _render_details(self) -> List[Any]:
        """The children of the details panel for the current selection."""
        from dash import html

        if not self._selected_details:
            return [html.I("Click a node to see its details")]
        return [html.P(detail) for detail in self._selected_details]
