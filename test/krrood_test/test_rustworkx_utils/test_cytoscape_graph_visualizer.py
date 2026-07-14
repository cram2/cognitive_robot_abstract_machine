from __future__ import annotations

import importlib.util
from dataclasses import dataclass

import pytest
import rustworkx as rx

from krrood.rustworkx_utils.cytoscape_graph_visualizer import CytoscapeGraphVisualizer
from krrood.rustworkx_utils.graph_visualizer_base import GraphLayout


@dataclass
class NamedPayload:
    """A payload whose name is shown as the node label and detail line."""

    name: str
    """The identifier rendered for this node."""


def chain_graph(names: list[str]) -> rx.PyDiGraph:
    """Build a directed chain of :class:`NamedPayload` nodes."""
    graph = rx.PyDiGraph(multigraph=False)
    indices = [graph.add_node(NamedPayload(name)) for name in names]
    for parent, child in zip(indices, indices[1:]):
        graph.add_edge(parent, child, None)
    return graph


def named_visualizer(graph: rx.PyDiGraph, **overrides) -> CytoscapeGraphVisualizer:
    """A visualizer that labels and describes nodes by their payload name."""
    return CytoscapeGraphVisualizer(
        graph=graph,
        label_getter=lambda payload: payload.name,
        info_getter=lambda payload: [f"name: {payload.name}"],
        **overrides,
    )


def node_elements(elements):
    """The node elements of a Cytoscape element list."""
    return [element for element in elements if "source" not in element["data"]]


def edge_elements(elements):
    """The edge elements of a Cytoscape element list."""
    return [element for element in elements if "source" in element["data"]]


class TestGraphElements:
    def test_one_element_per_node(self):
        visualizer = named_visualizer(chain_graph(["a", "b", "c"]))

        assert len(node_elements(visualizer.graph_elements())) == 3

    def test_node_labels_come_from_the_label_getter(self):
        visualizer = named_visualizer(chain_graph(["pick", "place"]))

        labels = [node["data"]["label"] for node in node_elements(visualizer.graph_elements())]
        assert labels == ["pick", "place"]

    def test_node_colors_come_from_the_color_getter(self):
        visualizer = named_visualizer(
            chain_graph(["a", "b"]),
            color_getter=lambda payload: "red" if payload.name == "a" else "green",
        )

        colors = [node["data"]["color"] for node in node_elements(visualizer.graph_elements())]
        assert colors == ["red", "green"]

    def test_edges_connect_node_ids(self):
        visualizer = named_visualizer(chain_graph(["a", "b"]))

        edges = edge_elements(visualizer.graph_elements())
        assert [(edge["data"]["source"], edge["data"]["target"]) for edge in edges] == [("0", "1")]

    def test_non_contiguous_indices_after_removal_are_handled(self):
        graph = chain_graph(["a", "b", "c"])
        graph.remove_node(1)
        visualizer = named_visualizer(graph)

        ids = [node["data"]["id"] for node in node_elements(visualizer.graph_elements())]
        assert ids == ["0", "2"]


class TestFlaskEndpoints:
    def test_index_page_serves_cytoscape(self):
        client = named_visualizer(chain_graph(["a"])).build_application().test_client()

        response = client.get("/")

        assert response.status_code == 200
        assert b"cytoscape" in response.data.lower()

    def test_graph_endpoint_returns_nodes_and_edges(self):
        client = named_visualizer(chain_graph(["a", "b"])).build_application().test_client()

        elements = client.get("/graph").get_json()["elements"]

        assert len(node_elements(elements)) == 2
        assert len(edge_elements(elements)) == 1

    def test_node_endpoint_returns_details(self):
        client = named_visualizer(chain_graph(["a", "b", "c"])).build_application().test_client()

        assert client.get("/node/2").get_json()["details"] == ["name: c"]


class TestLayout:
    def test_layered_layout_maps_to_a_hierarchical_cytoscape_layout(self):
        visualizer = named_visualizer(chain_graph(["a"]), layout=GraphLayout.LAYERED)

        assert visualizer.cytoscape_layout_name() == "breadthfirst"

    def test_physics_layout_runs_a_continuous_force_simulation(self):
        visualizer = named_visualizer(chain_graph(["a"]), layout=GraphLayout.PHYSICS)

        options = visualizer.cytoscape_layout_options()
        assert options["name"] == "cola"
        assert options["infinite"] is True

    def test_physics_libraries_are_loaded_in_the_page(self):
        client = named_visualizer(
            chain_graph(["a"]), layout=GraphLayout.PHYSICS
        ).build_application().test_client()

        assert b"cytoscape-cola" in client.get("/").data


class TestAppearance:
    def test_graph_canvas_uses_a_black_background_with_white_labels(self):
        client = named_visualizer(chain_graph(["a"])).build_application().test_client()

        page = client.get("/").data
        assert b"#000000" in page
        assert b'"color": "#ffffff"' in page


class TestCheckDependencies:
    def test_raises_when_flask_is_missing(self, monkeypatch):
        original_find_spec = importlib.util.find_spec

        def find_spec_without_flask(name, *args, **kwargs):
            if name == "flask":
                return None
            return original_find_spec(name, *args, **kwargs)

        monkeypatch.setattr(importlib.util, "find_spec", find_spec_without_flask)

        with pytest.raises(ModuleNotFoundError):
            CytoscapeGraphVisualizer.check_dependencies()
