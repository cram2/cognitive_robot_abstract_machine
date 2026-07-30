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

CytoscapeElement = Dict[str, Dict[str, Any]]


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

    layout_options: ClassVar[GraphLayoutOptions] = GraphLayoutOptions(
        layout_options={
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
    )
    """The Cytoscape.js layout options for each :class:`GraphLayout`."""

    def _node_element(self, node_index: int) -> CytoscapeElement:
        """
        :param node_index: The rustworkx index of the node.
        :return: The Cytoscape.js element for the node, with its border color included only if
            :attr:`~krrood.rustworkx_utils.graph_visualizer_base.GraphVisualizerBase.border_color_getter`
            supplies one.
        """
        data: Dict[str, Any] = {
            "id": str(node_index),
            "label": self.node_label(node_index),
            "color": self.node_color(node_index),
        }
        border_color = self.node_border_color(node_index)
        if border_color is not None:
            data["borderColor"] = border_color
        return {"data": data}

    def graph_elements(self) -> List[CytoscapeElement]:
        """
        :return: The Cytoscape.js node and edge elements for the current graph.
        """
        elements = [self._node_element(index) for index in self.graph.node_indices()]
        elements += [
            {
                "data": {
                    "id": f"{source}-{target}",
                    "source": str(source),
                    "target": str(target),
                }
            }
            for source, target in self.graph.edge_list()
        ]
        return elements

    def extra_node_styles(self) -> List[Dict[str, Any]]:
        """
        :return: Additional Cytoscape.js style selectors spread in after the built-in node styles,
            letting subclasses layer extra styling (for example an image fill) onto specific nodes
            without overriding the whole page template.
        """
        return []

    def extra_script(self) -> str:
        """
        :return: Extra JavaScript appended to the page after the built-in refresh/interaction
            wiring, letting subclasses add their own behaviour (for example animation loops)
            without overriding the whole page template. Empty by default.
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
            """
            Entry function for the flask app, renders the jinja template and presents it in the browser window.
            """
            return render_template(
                "cytoscape_graph_visualizer.jinja",
                title=self.title,
                layout_options_json=json.dumps(self.layout_options.get_options(self.layout)),
                extra_node_styles_json=json.dumps(self.extra_node_styles()),
                interval_ms=int(self.refresh_interval_seconds * 1000),
                extra_script=self.extra_script(),
            )

        @application.route("/graph")
        def graph():
            """
            Graph function that parses the graph to json
            """
            return jsonify(elements=self.graph_elements())

        @application.route("/node/<int:node_index>")
        def node(node_index: int):
            """
            Is called for each node to parse the node details to json
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
            name="CytoscapeGraphVisualizerServer",
        ).start()
        if self.open_browser:
            webbrowser.open(f"http://127.0.0.1:{self.port}")
