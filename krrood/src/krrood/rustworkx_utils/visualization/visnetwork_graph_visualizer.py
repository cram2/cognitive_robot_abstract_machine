from __future__ import annotations

import json
import threading
import webbrowser
from dataclasses import dataclass

from typing_extensions import Any, ClassVar, Dict, List, TYPE_CHECKING, Tuple

from krrood.rustworkx_utils.graph_visualizer_base import (
    GraphLayout,
    GraphVisualizerBase,
    GraphLayoutOptions,
)

if TYPE_CHECKING:
    from flask import Flask

VisNode = Dict[str, Any]
"""A single vis-network node."""

VisEdge = Dict[str, Any]
"""A single vis-network edge."""


@dataclass
class VisNetworkGraphVisualizer(GraphVisualizerBase):
    """Render a live rustworkx graph as an interactive Flask application backed by vis-network.

    The browser lays out and draws the graph; it polls the server for the current nodes and edges
    so that additions made while the graph is being built appear without a restart, and requests a
    node's details from :attr:`info_getter` when the node is clicked. With
    :attr:`~krrood.rustworkx_utils.graph_visualizer_base.GraphLayout.PHYSICS` the layout keeps
    simulating, so nodes self-organize and bounce when dragged or when new nodes appear.

    .. note::
        Flask is imported lazily so that importing this module does not require it; install it with
        ``pip install krrood[visualization]``. vis-network is loaded in the browser from a content
        delivery network, so viewing the page needs internet access.
    """

    required_modules: ClassVar[Tuple[str, ...]] = ("flask",)
    """The libraries needed to serve the application."""

    layout_options: ClassVar[GraphLayoutOptions] = GraphLayoutOptions(
        layout_options={
            GraphLayout.SPRING: {
                "physics": {
                    "enabled": True,
                    "stabilization": {"enabled": True, "iterations": 200},
                },
            },
            GraphLayout.LAYERED: {
                "layout": {
                    "hierarchical": {
                        "direction": "UD",
                        "sortMethod": "directed",
                        "nodeSpacing": 150,
                    }
                },
                "physics": {"enabled": False},
            },
            GraphLayout.PHYSICS: {
                "physics": {"enabled": True, "stabilization": {"enabled": False}},
            },
        }
    )
    """The vis-network layout options for each :class:`GraphLayout`."""

    def graph_nodes(self) -> List[VisNode]:
        """
        :return: The vis-network nodes for the current graph.
        """
        return [
            {
                "id": str(index),
                "label": self.node_label(index),
                "color": {
                    "background": self.node_color(index),
                    "border": self.node_border_color(index) or self.default_border_color,
                },
            }
            for index in self.graph.node_indices()
        ]

    def graph_edges(self) -> List[VisEdge]:
        """
        :return: The vis-network edges for the current graph.
        """
        return [
            {"id": f"{source}-{target}", "from": str(source), "to": str(target)}
            for source, target in self.graph.edge_list()
        ]

    def build_application(self) -> Flask:
        """
        :return: The Flask application serving the page and the graph and node endpoints.
        """
        from flask import Flask, jsonify, render_template

        application = Flask(__name__)

        @application.route("/")
        def index() -> str:
            """
            Index page of the flask app that renders the jinja template and presents it
            """
            return render_template(
                "visnetwork_graph_visualizer.jinja",
                title=self.title,
                layout_options_json=json.dumps(self.layout_options.get_options(self.layout)),
                interval_ms=int(self.refresh_interval_seconds * 1000),
            )

        @application.route("/graph")
        def graph():
            """
            Entry function of the flask app serving the graph and node endpoints.
            """
            return jsonify(nodes=self.graph_nodes(), edges=self.graph_edges())

        @application.route("/node/<int:node_index>")
        def node(node_index: int):
            """
            Called for each node to parse the node details to json
            """
            return jsonify(details=self.node_details(node_index))

        return application

    def run(self) -> None:
        """
        Start the Flask application on a daemon thread and optionally open it in a browser.

        The server runs on a daemon thread, so the calling process must stay alive for the
        visualization to remain reachable.
        """
        self.check_dependencies()
        application = self.build_application()
        threading.Thread(
            target=lambda: application.run(
                port=self.port, threaded=True, use_reloader=False
            ),
            daemon=True,
            name="VisNetworkGraphVisualizerServer",
        ).start()
        if self.open_browser:
            webbrowser.open(f"http://127.0.0.1:{self.port}")
