from __future__ import annotations

import importlib.util
from dataclasses import dataclass

import pytest
import rustworkx as rx

from krrood.rustworkx_utils.graph_visualizer_base import GraphLayout
from rustworkx_utils.visualization.visnetwork_graph_visualizer import (
    VisNetworkGraphVisualizer,
)


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


def named_visualizer(graph: rx.PyDiGraph, **overrides) -> VisNetworkGraphVisualizer:
    """A visualizer that labels and describes nodes by their payload name."""
    return VisNetworkGraphVisualizer(
        graph=graph,
        label_getter=lambda payload: payload.name,
        info_getter=lambda payload: [f"name: {payload.name}"],
        **overrides,
    )


class TestGraphNodes:
    def test_one_node_per_graph_node(self):
        visualizer = named_visualizer(chain_graph(["a", "b", "c"]))

        assert len(visualizer.graph_nodes()) == 3

    def test_node_labels_come_from_the_label_getter(self):
        visualizer = named_visualizer(chain_graph(["pick", "place"]))

        labels = [node["label"] for node in visualizer.graph_nodes()]
        assert labels == ["pick", "place"]

    def test_node_colors_come_from_the_color_getter(self):
        visualizer = named_visualizer(
            chain_graph(["a", "b"]),
            color_getter=lambda payload: "red" if payload.name == "a" else "green",
        )

        colors = [node["color"]["background"] for node in visualizer.graph_nodes()]
        assert colors == ["red", "green"]

    def test_node_border_colors_come_from_the_border_color_getter(self):
        visualizer = named_visualizer(
            chain_graph(["a", "b"]),
            border_color_getter=lambda payload: (
                "black" if payload.name == "a" else "white"
            ),
        )

        border_colors = [node["color"]["border"] for node in visualizer.graph_nodes()]
        assert border_colors == ["black", "white"]

    def test_node_border_color_defaults_without_a_border_color_getter(self):
        visualizer = named_visualizer(chain_graph(["a"]))

        assert visualizer.graph_nodes()[0]["color"]["border"] == "#888888"

    def test_non_contiguous_indices_after_removal_are_handled(self):
        graph = chain_graph(["a", "b", "c"])
        graph.remove_node(1)
        visualizer = named_visualizer(graph)

        ids = [node["id"] for node in visualizer.graph_nodes()]
        assert ids == ["0", "2"]


class TestGraphEdges:
    def test_edges_connect_node_ids(self):
        visualizer = named_visualizer(chain_graph(["a", "b"]))

        edges = visualizer.graph_edges()
        assert [(edge["from"], edge["to"]) for edge in edges] == [("0", "1")]

    def test_one_edge_per_graph_edge(self):
        visualizer = named_visualizer(chain_graph(["a", "b", "c"]))

        assert len(visualizer.graph_edges()) == 2


class TestFlaskEndpoints:
    def test_index_page_serves_vis_network(self):
        client = named_visualizer(chain_graph(["a"])).build_application().test_client()

        response = client.get("/")

        assert response.status_code == 200
        assert b"vis-network" in response.data.lower()

    def test_graph_endpoint_returns_nodes_and_edges(self):
        client = (
            named_visualizer(chain_graph(["a", "b"])).build_application().test_client()
        )

        payload = client.get("/graph").get_json()

        assert len(payload["nodes"]) == 2
        assert len(payload["edges"]) == 1

    def test_node_endpoint_returns_details(self):
        client = (
            named_visualizer(chain_graph(["a", "b", "c"]))
            .build_application()
            .test_client()
        )

        assert client.get("/node/2").get_json()["details"] == ["name: c"]


class TestLayout:
    def test_layered_layout_maps_to_a_hierarchical_layout(self):
        visualizer = named_visualizer(chain_graph(["a"]), layout=GraphLayout.LAYERED)

        options = visualizer.visnetwork_layout_options()
        assert options["layout"]["hierarchical"]["direction"] == "UD"
        assert options["physics"]["enabled"] is False

    def test_physics_layout_keeps_the_simulation_running(self):
        visualizer = named_visualizer(chain_graph(["a"]), layout=GraphLayout.PHYSICS)

        options = visualizer.visnetwork_layout_options()
        assert options["physics"]["enabled"] is True
        assert options["physics"]["stabilization"]["enabled"] is False

    def test_spring_layout_stabilizes_and_stops(self):
        visualizer = named_visualizer(chain_graph(["a"]), layout=GraphLayout.SPRING)

        options = visualizer.visnetwork_layout_options()
        assert options["physics"]["enabled"] is True
        assert options["physics"]["stabilization"]["enabled"] is True


class TestCheckDependencies:
    def test_raises_when_flask_is_missing(self, monkeypatch):
        original_find_spec = importlib.util.find_spec

        def find_spec_without_flask(name, *args, **kwargs):
            if name == "flask":
                return None
            return original_find_spec(name, *args, **kwargs)

        monkeypatch.setattr(importlib.util, "find_spec", find_spec_without_flask)

        with pytest.raises(ModuleNotFoundError):
            VisNetworkGraphVisualizer.check_dependencies()
