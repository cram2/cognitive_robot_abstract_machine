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

ThreeNode = Dict[str, Any]
"""A single 3d-force-graph node."""

ThreeLink = Dict[str, Any]
"""A single 3d-force-graph link."""

LayoutOptions = Dict[str, Any]
"""Fluent-setter options applied to the 3d-force-graph instance, keyed by method name."""

_LAYOUT_OPTIONS: Dict[GraphLayout, LayoutOptions] = {
    GraphLayout.SPRING: {"cooldownTicks": 100},
    GraphLayout.LAYERED: {"dagMode": "td", "cooldownTicks": 100},
    GraphLayout.PHYSICS: {"cooldownTicks": float("inf")},
    # A handful of ticks rather than 0: d3-force applies fx/fy/fz to x/y/z during its normal tick
    # loop, so the simulation still needs to run briefly for fixed positions to actually take
    # effect, even though there's nothing left to iteratively settle.
    GraphLayout.FIXED: {"cooldownTicks": 10},
}
"""The 3d-force-graph layout options used for each :class:`GraphLayout`."""

_PAGE_TEMPLATE = """<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>{{ title }}</title>
<script src="https://unpkg.com/3d-force-graph"></script>
{{ extra_head | safe }}
<style>
  html, body { margin: 0; height: 100%; font-family: "Segoe UI", sans-serif; background: #0b0b0f; }
  #container { display: flex; height: 100vh; }
  #graph { flex: 3; background: #000000; }
  #details { flex: 1; padding: 1.25rem; overflow: auto; background: #121218;
    color: #d6d6e0; border-left: 1px solid #2a2a35; }
  #details h3 { margin-top: 0; color: #ffffff; border-bottom: 1px solid #2a2a35; padding-bottom: 0.5rem; }
  #details p { color: #b5b5c4; line-height: 1.5; }
  #details i { color: #7a7a8c; }
  #popup { position: fixed; display: none; max-width: 280px; background: #121218;
    color: #d6d6e0; border: 1px solid #2a2a35; border-radius: 6px; padding: 0.75rem 1rem;
    box-shadow: 0 4px 16px rgba(0, 0, 0, 0.5); z-index: 10; }
  #popup h3 { margin: 0 0 0.5rem 0; color: #ffffff; font-size: 14px; padding-right: 1.25rem; }
  #popup p { margin: 0.25rem 0; color: #b5b5c4; font-size: 12px; line-height: 1.4; }
  #popup .close-popup { position: absolute; top: 4px; right: 8px; cursor: pointer;
    color: #7a7a8c; background: none; border: none; font-size: 14px; line-height: 1; }
  .node-label { position: fixed; display: none; pointer-events: none; color: #ffffff;
    font-size: 12px; white-space: nowrap; transform: translate(-50%, -100%);
    text-shadow: 0 0 4px #000000, 0 0 4px #000000, 0 0 4px #000000; }
</style>
</head>
<body>
<div id="container">
  <div id="graph"></div>
  <div id="details"><i>Click a node to see its details</i></div>
</div>
<div id="popup"></div>
<script>
const layoutOptions = {{ layout_options_json | safe }};

function escapeHtml(text) {
  const holder = document.createElement("div");
  holder.textContent = text;
  return holder.innerHTML;
}

function hidePopup() {
  document.getElementById("popup").style.display = "none";
}

function showPopup(label, lines, clientX, clientY) {
  const popup = document.getElementById("popup");
  popup.innerHTML =
    '<button class="close-popup" onclick="hidePopup()">×</button>' +
    "<h3>" + escapeHtml(label) + "</h3>" +
    lines.map((line) => "<p>" + escapeHtml(line) + "</p>").join("");
  popup.style.display = "block";
  const maxLeft = window.innerWidth - popup.offsetWidth - 12;
  const maxTop = window.innerHeight - popup.offsetHeight - 12;
  popup.style.left = Math.max(12, Math.min(clientX + 12, maxLeft)) + "px";
  popup.style.top = Math.max(12, Math.min(clientY + 12, maxTop)) + "px";
}

let lastMouseX = 0;
let lastMouseY = 0;
document.getElementById("graph").addEventListener("mousemove", (event) => {
  lastMouseX = event.clientX;
  lastMouseY = event.clientY;
});

let hoveredNodeId = null;

const graph = ForceGraph3D()(document.getElementById("graph"))
  .backgroundColor("#000000")
  .nodeLabel(null)
  .nodeColor((node) => node.color)
  .linkColor(() => "#ffffff")
  .linkWidth(2)
  .onNodeClick(async (node) => {
    const response = await fetch("node/" + node.id);
    const payload = await response.json();
    const lines = payload.details.map((line) => "<p>" + escapeHtml(line) + "</p>").join("");
    document.getElementById("details").innerHTML =
      "<h3>" + escapeHtml(node.label) + "</h3>" + lines;
  })
  .onNodeHover(async (node) => {
    hoveredNodeId = node ? node.id : null;
    if (!node) { hidePopup(); return; }
    const response = await fetch("node/" + node.id);
    const payload = await response.json();
    if (hoveredNodeId !== node.id) { return; } // hover moved on before this resolved
    showPopup(node.label, payload.details, lastMouseX, lastMouseY);
  })
  .onBackgroundClick(hidePopup);

Object.keys(layoutOptions).forEach((key) => { graph[key](layoutOptions[key]); });

// Always-visible node labels, positioned every frame by projecting each node's 3D position to
// screen space. This borrows 3d-force-graph's own internal Vector3 class (via an existing
// Object3D's .position, which was constructed by that same internal copy) instead of constructing
// one from a separately loaded three.js: 3d-force-graph bundles its own three.js without exposing
// it globally, and a separately loaded, mismatched copy can be incompatible with what it renders
// (confirmed the hard way with a Sprite-based label - the shaders didn't match and nothing drew).
const ProjectionVector3 = graph.scene().position.constructor;
const labelElements = new Map();

function updateLabels() {
  const graphElement = document.getElementById("graph");
  const rect = graphElement.getBoundingClientRect();
  const camera = graph.camera();
  const currentIds = new Set();

  graph.graphData().nodes.forEach((node) => {
    if (node.x === undefined) { return; }
    currentIds.add(node.id);
    let element = labelElements.get(node.id);
    if (!element) {
      element = document.createElement("div");
      element.className = "node-label";
      document.body.appendChild(element);
      labelElements.set(node.id, element);
    }
    element.textContent = node.label;

    const projected = new ProjectionVector3(node.x, node.y, node.z).project(camera);
    if (projected.z > 1) {
      element.style.display = "none";
      return;
    }
    element.style.display = "block";
    element.style.left = (rect.left + (projected.x * 0.5 + 0.5) * rect.width) + "px";
    element.style.top = (rect.top + (-projected.y * 0.5 + 0.5) * rect.height - 10) + "px";
  });

  labelElements.forEach((element, id) => {
    if (!currentIds.has(id)) { element.remove(); labelElements.delete(id); }
  });

  requestAnimationFrame(updateLabels);
}
requestAnimationFrame(updateLabels);

const knownNodeIds = new Set();
const knownLinkIds = new Set();

function linkId(link) { return link.source + "->" + link.target; }

async function refresh() {
  let response;
  try {
    response = await fetch("graph");
  } catch (error) {
    console.error("Failed to reach the graph endpoint:", error);
    return;
  }
  if (!response.ok) {
    console.error("Graph endpoint returned status", response.status);
    if (knownNodeIds.size === 0) {
      document.getElementById("details").innerHTML =
        "<i>Failed to load the graph (server returned " + response.status + "). Retrying...</i>";
    }
    return;
  }
  const payload = await response.json();
  const currentNodeIds = new Set(payload.nodes.map((node) => node.id));
  const currentLinkIds = new Set(payload.links.map(linkId));
  const structureChanged =
    currentNodeIds.size !== knownNodeIds.size ||
    currentLinkIds.size !== knownLinkIds.size ||
    [...currentNodeIds].some((id) => !knownNodeIds.has(id)) ||
    [...currentLinkIds].some((id) => !knownLinkIds.has(id));

  if (structureChanged) {
    // Only rebuild the graph (which restarts the force simulation) when nodes or links were
    // actually added or removed; calling graphData() on every poll regardless of whether anything
    // changed made the whole graph visibly "pop" and resettle once a second.
    graph.graphData(payload);
    knownNodeIds.clear();
    currentNodeIds.forEach((id) => knownNodeIds.add(id));
    knownLinkIds.clear();
    currentLinkIds.forEach((id) => knownLinkIds.add(id));
  } else {
    // Update fields (color, label, ...) on the existing, already-simulated node objects in place,
    // without touching their position/velocity or restarting the simulation.
    const nodesById = new Map(graph.graphData().nodes.map((node) => [node.id, node]));
    payload.nodes.forEach((updatedNode) => {
      const existingNode = nodesById.get(updatedNode.id);
      if (existingNode) { Object.assign(existingNode, updatedNode); }
    });
  }
}

refresh();
setInterval(refresh, {{ interval_ms }});
{{ extra_script | safe }}
</script>
</body>
</html>
"""
"""The single-page 3d-force-graph front-end that polls the graph and shows node details on click."""


@dataclass
class ThreeGraphVisualizer(GraphVisualizerBase):
    """Render a live rustworkx graph as an interactive Flask application backed by 3d-force-graph
    (Three.js + d3-force-3d), with a real orbitable WebGL 3D camera over the whole graph.

    The browser lays out and draws the graph; it polls the server for the current nodes and links
    so that additions made while the graph is being built appear without a restart. Every node
    shows its label as a permanent, screen-projected text overlay; hovering a node shows
    :attr:`info_getter`'s details in a popup next to the cursor, and clicking a node shows the same
    details in the side panel. With
    :attr:`~krrood.rustworkx_utils.graph_visualizer_base.GraphLayout.PHYSICS` the layout keeps
    simulating, so nodes self-organize and drift when new nodes appear. With
    :attr:`~krrood.rustworkx_utils.graph_visualizer_base.GraphLayout.FIXED`, nodes are pinned at
    :attr:`~krrood.rustworkx_utils.graph_visualizer_base.GraphVisualizerBase.position_getter`'s
    result instead of being simulated — for example, placing each node at its payload's real
    spatial position rather than an arbitrary force-directed layout.

    .. note::
        Flask is imported lazily so that importing this module does not require it; install it with
        ``pip install krrood[visualization]``. 3d-force-graph is loaded in the browser from a
        content delivery network, so viewing the page needs internet access.
    """

    required_modules: ClassVar[Tuple[str, ...]] = ("flask",)
    """The libraries needed to serve the application."""

    def node_extra_data(self, node_index: int) -> Dict[str, Any]:
        """
        :param node_index: The rustworkx index of the node.
        :return: Extra fields merged into the node's JSON payload, letting subclasses attach data
            (for example a mesh URL) without overriding node construction entirely.
        """
        return {}

    def graph_nodes(self) -> List[ThreeNode]:
        """
        :return: The 3d-force-graph nodes for the current graph.
        """
        nodes = []
        for index in self.graph.node_indices():
            node: ThreeNode = {
                "id": str(index),
                "label": self.node_label(index),
                "color": self.node_color(index),
            }
            node.update(self.node_extra_data(index))
            node.update(self._fixed_position_fields(index))
            nodes.append(node)
        return nodes

    def _fixed_position_fields(self, node_index: int) -> Dict[str, float]:
        """
        :param node_index: The rustworkx index of the node.
        :return: ``fx``/``fy``/(``fz``) fields pinning the node at :meth:`node_position`'s result,
            the d3-force convention for a fixed (non-simulated) position; empty outside
            :attr:`~krrood.rustworkx_utils.graph_visualizer_base.GraphLayout.FIXED` or for nodes
            :meth:`node_position` doesn't cover.
        """
        if self.layout is not GraphLayout.FIXED:
            return {}
        position = self.node_position(node_index)
        if position is None:
            return {}
        fields = {"fx": float(position[0]), "fy": float(position[1])}
        if len(position) > 2:
            fields["fz"] = float(position[2])
        return fields

    def graph_links(self) -> List[ThreeLink]:
        """
        :return: The 3d-force-graph links for the current graph.
        """
        return [
            {"source": str(source), "target": str(target)}
            for source, target in self.graph.edge_list()
        ]

    def three_layout_options(self) -> LayoutOptions:
        """
        :return: The 3d-force-graph fluent-setter options for the configured layout.
        """
        return _LAYOUT_OPTIONS[self.layout]

    def register_additional_routes(self, application: Flask) -> None:
        """
        Register extra Flask routes on ``application``, called once from :meth:`build_application`.

        The default implementation does nothing; subclasses override this to serve extra per-node
        resources (for example meshes referenced by :meth:`node_extra_data`).

        :param application: The Flask application to add routes to.
        """

    def extra_script(self) -> str:
        """
        :return: Extra JavaScript appended to the page after the built-in refresh/interaction
            wiring, letting subclasses add their own behaviour (for example custom node visuals)
            without overriding the whole page template. Empty by default.
        """
        return ""

    def extra_head(self) -> str:
        """
        :return: Extra HTML appended to the page's ``<head>``, letting subclasses load additional
            libraries (for example loaders needed by :meth:`extra_script`) as blocking ``<script>``
            tags that finish loading before the main script runs. Empty by default.

            Use this rather than loading a library asynchronously from :meth:`extra_script` when
            the loaded library must be available to callbacks that run on the very first graph
            update, since libraries like 3d-force-graph only ever resolve a given node's rendered
            object once — a callback that isn't ready yet on that first resolution won't get a
            second chance later.
        """
        return ""

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
                layout_options_json=json.dumps(self.three_layout_options()),
                interval_ms=int(self.refresh_interval_seconds * 1000),
                extra_head=self.extra_head(),
                extra_script=self.extra_script(),
            )

        @application.route("/graph")
        def graph():
            return jsonify(nodes=self.graph_nodes(), links=self.graph_links())

        @application.route("/node/<int:node_index>")
        def node(node_index: int):
            return jsonify(details=self.node_details(node_index))

        self.register_additional_routes(application)
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
            name="ThreeGraphVisualizerServer",
        ).start()
        if self.open_browser:
            webbrowser.open(f"http://127.0.0.1:{self.port}")
