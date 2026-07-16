from __future__ import annotations

import json
import threading
import webbrowser
from dataclasses import dataclass

from typing_extensions import Any, ClassVar, Dict, List, TYPE_CHECKING, Tuple

from krrood.rustworkx_utils.graph_visualizer_base import (
    GraphLayout,
    GraphVisualizerBase,
)

if TYPE_CHECKING:
    from flask import Flask

VisNode = Dict[str, Any]
"""A single vis-network node."""

VisEdge = Dict[str, Any]
"""A single vis-network edge."""

LayoutOptions = Dict[str, Any]
"""The option mapping passed to vis-network's ``Network`` constructor."""

DEFAULT_BORDER_COLOR = "#888888"
"""The border color used for nodes and edges when no border color is given."""

_LAYOUT_OPTIONS: Dict[GraphLayout, LayoutOptions] = {
    GraphLayout.SPRING: {
        "physics": {"enabled": True, "stabilization": {"enabled": True, "iterations": 200}},
    },
    GraphLayout.LAYERED: {
        "layout": {
            "hierarchical": {"direction": "UD", "sortMethod": "directed", "nodeSpacing": 150}
        },
        "physics": {"enabled": False},
    },
    GraphLayout.PHYSICS: {
        "physics": {"enabled": True, "stabilization": {"enabled": False}},
    },
}
"""The vis-network layout and physics options used for each :class:`GraphLayout`."""

_PAGE_TEMPLATE = """<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>{{ title }}</title>
<script src="https://unpkg.com/vis-network@9.1.9/standalone/umd/vis-network.min.js"></script>
<style>
  html, body { margin: 0; height: 100%; font-family: sans-serif; }
  #container { display: flex; height: 100vh; }
  #network { flex: 3; background: #000000; }
  #details { flex: 1; padding: 1rem; overflow: auto; border-left: 1px solid #ccc; }
  #details h3 { margin-top: 0; }
</style>
</head>
<body>
<div id="container">
  <div id="network"></div>
  <div id="details"><i>Click a node to see its details</i></div>
</div>
<script>
const layoutOptions = {{ layout_options_json | safe }};
const nodes = new vis.DataSet([]);
const edges = new vis.DataSet([]);
const network = new vis.Network(
  document.getElementById("network"),
  { nodes: nodes, edges: edges },
  Object.assign(
    {
      nodes: { shape: "dot", size: 15, font: { color: "#ffffff" } },
      edges: { color: "#888888", arrows: "to" },
    },
    layoutOptions
  )
);

function escapeHtml(text) {
  const holder = document.createElement("div");
  holder.textContent = text;
  return holder.innerHTML;
}

async function refresh() {
  const response = await fetch("graph");
  const payload = await response.json();

  const currentNodeIds = new Set(payload.nodes.map((node) => node.id));
  const currentEdgeIds = new Set(payload.edges.map((edge) => edge.id));

  nodes.update(payload.nodes);
  edges.update(payload.edges);

  nodes.getIds().forEach((id) => { if (!currentNodeIds.has(id)) { nodes.remove(id); } });
  edges.getIds().forEach((id) => { if (!currentEdgeIds.has(id)) { edges.remove(id); } });
}

network.on("click", async (params) => {
  if (params.nodes.length === 0) { return; }
  const nodeId = params.nodes[0];
  const response = await fetch("node/" + nodeId);
  const payload = await response.json();
  const lines = payload.details.map((line) => "<p>" + escapeHtml(line) + "</p>").join("");
  document.getElementById("details").innerHTML =
    "<h3>" + escapeHtml(nodes.get(nodeId).label) + "</h3>" + lines;
});

refresh();
setInterval(refresh, {{ interval_ms }});
</script>
</body>
</html>
"""
"""The single-page vis-network front-end that polls the graph and shows node details on click."""


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
                    "border": self.node_border_color(index) or DEFAULT_BORDER_COLOR,
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

    def visnetwork_layout_options(self) -> LayoutOptions:
        """
        :return: The vis-network layout and physics options for the configured layout.
        """
        return _LAYOUT_OPTIONS[self.layout]

    def build_application(self) -> Flask:
        """
        :return: The Flask application serving the page and the graph and node endpoints.
        """
        from flask import Flask, jsonify, render_template_string

        application = Flask(__name__)

        @application.route("/")
        def index() -> str:
            return render_template_string(
                _PAGE_TEMPLATE,
                title=self.title,
                layout_options_json=json.dumps(self.visnetwork_layout_options()),
                interval_ms=int(self.refresh_interval_seconds * 1000),
            )

        @application.route("/graph")
        def graph():
            return jsonify(nodes=self.graph_nodes(), edges=self.graph_edges())

        @application.route("/node/<int:node_index>")
        def node(node_index: int):
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
