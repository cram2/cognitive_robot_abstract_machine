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

CytoscapeElement = Dict[str, Dict[str, Any]]
"""A single Cytoscape.js node or edge, wrapping its data in a ``data`` mapping."""

LayoutOptions = Dict[str, Any]
"""The option mapping passed to a Cytoscape.js layout engine."""

_LAYOUT_OPTIONS: Dict[GraphLayout, LayoutOptions] = {
    GraphLayout.SPRING: {"name": "cose", "animate": True},
    GraphLayout.LAYERED: {"name": "breadthfirst", "directed": True},
    GraphLayout.PHYSICS: {
        "name": "cola",
        "infinite": True,
        "fit": False,
        "edgeLength": 120,
        "nodeSpacing": 10,
    },
}
"""The Cytoscape.js layout options used for each :class:`GraphLayout`."""

_PAGE_TEMPLATE = """<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>{{ title }}</title>
<script src="https://unpkg.com/cytoscape@3.30.2/dist/cytoscape.min.js"></script>
<script src="https://unpkg.com/webcola@3.4.0/WebCola/cola.min.js"></script>
<script src="https://unpkg.com/cytoscape-cola@2.5.1/cytoscape-cola.js"></script>
<style>
  html, body { margin: 0; height: 100%; font-family: sans-serif; }
  #container { display: flex; height: 100vh; }
  #cy { flex: 3; background: #000000; }
  #details { flex: 1; padding: 1rem; overflow: auto; border-left: 1px solid #ccc; }
  #details h3 { margin-top: 0; }
</style>
</head>
<body>
<div id="container">
  <div id="cy"></div>
  <div id="details"><i>Click a node to see its details</i></div>
</div>
<script>
const layoutOptions = {{ layout_options_json | safe }};
const cy = cytoscape({
  container: document.getElementById("cy"),
  style: [
    { selector: "node", style: {
        "label": "data(label)", "background-color": "data(color)",
        "text-valign": "top", "text-halign": "center", "color": "#ffffff",
        "font-size": 12, "width": 30, "height": 30 } },
    { selector: "edge", style: {
        "width": 2, "line-color": "#888", "target-arrow-color": "#888",
        "target-arrow-shape": "triangle", "curve-style": "bezier" } },
    { selector: "node:selected", style: { "border-width": 4, "border-color": "#ffffff" } }
  ]
});

const knownNodes = new Set();
const knownEdges = new Set();
let runningLayout = null;

function relayout() {
  if (runningLayout) { runningLayout.stop(); }
  runningLayout = cy.layout(layoutOptions);
  runningLayout.run();
}

function escapeHtml(text) {
  const holder = document.createElement("div");
  holder.textContent = text;
  return holder.innerHTML;
}

async function refresh() {
  const response = await fetch("graph");
  const payload = await response.json();
  const currentNodes = new Set();
  const currentEdges = new Set();
  let changed = false;
  for (const element of payload.elements) {
    const data = element.data;
    if (data.source !== undefined) {
      currentEdges.add(data.id);
      if (!knownEdges.has(data.id)) { cy.add(element); knownEdges.add(data.id); changed = true; }
    } else {
      currentNodes.add(data.id);
      if (!knownNodes.has(data.id)) { cy.add(element); knownNodes.add(data.id); changed = true; }
      else { cy.getElementById(data.id).data(data); }
    }
  }
  for (const id of Array.from(knownNodes)) {
    if (!currentNodes.has(id)) { cy.remove(cy.getElementById(id)); knownNodes.delete(id); changed = true; }
  }
  for (const id of Array.from(knownEdges)) {
    if (!currentEdges.has(id)) { cy.remove(cy.getElementById(id)); knownEdges.delete(id); changed = true; }
  }
  if (changed) { relayout(); }
}

cy.on("tap", "node", async (event) => {
  const node = event.target;
  const response = await fetch("node/" + node.id());
  const payload = await response.json();
  const lines = payload.details.map((line) => "<p>" + escapeHtml(line) + "</p>").join("");
  document.getElementById("details").innerHTML =
    "<h3>" + escapeHtml(node.data("label")) + "</h3>" + lines;
});

refresh();
setInterval(refresh, {{ interval_ms }});
</script>
</body>
</html>
"""
"""The single-page Cytoscape.js front-end that polls the graph and shows node details on click."""


@dataclass
class CytoscapeGraphVisualizer(GraphVisualizerBase):
    """Render a live rustworkx graph as an interactive Flask application backed by Cytoscape.js.

    The browser lays out and draws the graph; it polls the server for the current nodes and edges
    so that additions made while the graph is being built appear without a restart, and requests a
    node's details from :attr:`info_getter` when the node is clicked. With
    :attr:`~krrood.rustworkx_utils.graph_visualizer_base.GraphLayout.PHYSICS` the layout keeps
    simulating, so nodes self-organize and bounce when dragged or when new nodes appear.

    .. note::
        Flask is imported lazily so that importing this module does not require it; install it with
        ``pip install krrood[visualization]``. Cytoscape.js and its layout engines are loaded in the
        browser from a content delivery network, so viewing the page needs internet access.
    """

    required_modules: ClassVar[Tuple[str, ...]] = ("flask",)
    """The libraries needed to serve the application."""

    def graph_elements(self) -> List[CytoscapeElement]:
        """
        :return: The Cytoscape.js node and edge elements for the current graph.
        """
        elements: List[CytoscapeElement] = [
            {
                "data": {
                    "id": str(index),
                    "label": self.node_label(index),
                    "color": self.node_color(index),
                }
            }
            for index in self.graph.node_indices()
        ]
        elements += [
            {"data": {"id": f"{source}-{target}", "source": str(source), "target": str(target)}}
            for source, target in self.graph.edge_list()
        ]
        return elements

    def cytoscape_layout_options(self) -> LayoutOptions:
        """
        :return: The Cytoscape.js layout options for the configured layout.
        """
        return _LAYOUT_OPTIONS[self.layout]

    def cytoscape_layout_name(self) -> str:
        """
        :return: The name of the Cytoscape.js layout engine for the configured layout.
        """
        return self.cytoscape_layout_options()["name"]

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
                layout_options_json=json.dumps(self.cytoscape_layout_options()),
                interval_ms=int(self.refresh_interval_seconds * 1000),
            )

        @application.route("/graph")
        def graph():
            return jsonify(elements=self.graph_elements())

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
            name="CytoscapeGraphVisualizerServer",
        ).start()
        if self.open_browser:
            webbrowser.open(f"http://127.0.0.1:{self.port}")
