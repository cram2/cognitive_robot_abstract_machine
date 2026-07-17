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
from rustworkx_utils.graph_visualizer_base import GraphLayoutOptions

if TYPE_CHECKING:
    from flask import Flask

ThreeNode = Dict[str, Any]
"""A single 3d-force-graph node."""

ThreeLink = Dict[str, Any]
"""A single 3d-force-graph link."""

LAYOUT_OPTIONS = GraphLayoutOptions(
    layout_options={
        GraphLayout.SPRING: {"cooldownTicks": 100},
        GraphLayout.LAYERED: {"dagMode": "td", "cooldownTicks": 100},
        GraphLayout.PHYSICS: {"cooldownTicks": float("inf")},
        # A handful of ticks rather than 0: d3-force applies fx/fy/fz to x/y/z during its normal tick
        # loop, so the simulation still needs to run briefly for fixed positions to actually take
        # effect, even though there's nothing left to iteratively settle.
        GraphLayout.FIXED: {"cooldownTicks": 10},
    }
)


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
        from flask import Flask, jsonify, render_template

        application = Flask(__name__)

        @application.route("/")
        def index() -> str:
            return render_template(
                "three_graph_visualizer.jinja",
                title=self.title,
                layout_options_json=json.dumps(LAYOUT_OPTIONS.get_options(self.layout)),
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
