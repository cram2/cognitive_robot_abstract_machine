from __future__ import annotations

import importlib.util
from dataclasses import dataclass

import pytest
import rustworkx as rx

from krrood.rustworkx_utils.interactive_graph_visualizer import (
    GraphLayout,
    InteractiveGraphVisualizer,
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


def named_visualizer(
    graph: rx.PyDiGraph, **overrides
) -> InteractiveGraphVisualizer:
    """A visualizer that labels and describes nodes by their payload name."""
    return InteractiveGraphVisualizer(
        graph=graph,
        label_getter=lambda payload: payload.name,
        info_getter=lambda payload: [f"name: {payload.name}"],
        **overrides,
    )


class TestComputePositions:
    @pytest.mark.parametrize("layout", [GraphLayout.SPRING, GraphLayout.LAYERED])
    def test_every_node_gets_a_two_dimensional_position(self, layout):
        graph = chain_graph(["a", "b", "c"])
        visualizer = named_visualizer(graph, layout=layout)

        positions = visualizer.compute_positions()

        assert set(positions) == set(graph.node_indices())
        assert all(len(position) == 2 for position in positions.values())

    @pytest.mark.parametrize("layout", [GraphLayout.SPRING, GraphLayout.LAYERED])
    def test_non_contiguous_indices_after_removal_are_handled(self, layout):
        graph = chain_graph(["a", "b", "c"])
        graph.remove_node(1)
        visualizer = named_visualizer(graph, layout=layout)

        positions = visualizer.compute_positions()

        assert set(positions) == {0, 2}

    def test_layered_layout_orders_nodes_top_to_bottom(self):
        graph = chain_graph(["root", "child"])
        visualizer = named_visualizer(graph, layout=GraphLayout.LAYERED)

        positions = visualizer.compute_positions()

        assert positions[0][1] > positions[1][1]

    def test_empty_graph_yields_empty_positions(self):
        visualizer = named_visualizer(rx.PyDiGraph())

        assert visualizer.compute_positions() == {}


class TestBuildFigure:
    def _node_trace(self, visualizer: InteractiveGraphVisualizer):
        figure = visualizer.build_figure()
        return next(trace for trace in figure.data if trace.name == "nodes")

    def test_node_trace_has_one_point_per_node(self):
        visualizer = named_visualizer(chain_graph(["a", "b", "c"]))

        assert len(self._node_trace(visualizer).x) == 3

    def test_node_labels_come_from_the_label_getter(self):
        visualizer = named_visualizer(chain_graph(["pick", "place"]))

        assert list(self._node_trace(visualizer).text) == ["pick", "place"]

    def test_customdata_carries_the_rustworkx_indices(self):
        graph = chain_graph(["a", "b", "c"])
        graph.remove_node(1)
        visualizer = named_visualizer(graph)

        assert list(self._node_trace(visualizer).customdata) == [0, 2]

    def test_marker_colors_come_from_the_color_getter(self):
        visualizer = named_visualizer(
            chain_graph(["a", "b"]),
            color_getter=lambda payload: "red" if payload.name == "a" else "green",
        )

        assert list(self._node_trace(visualizer).marker.color) == ["red", "green"]

    def test_edges_are_rendered_as_a_separate_trace(self):
        visualizer = named_visualizer(chain_graph(["a", "b"]))

        figure = visualizer.build_figure()

        assert any(trace.name == "edges" for trace in figure.data)


class TestNodeDetails:
    def test_returns_info_getter_output_for_the_clicked_node(self):
        graph = chain_graph(["a", "b", "c"])
        visualizer = named_visualizer(graph)

        assert visualizer.node_details(2) == ["name: c"]


class TestCheckDependencies:
    def test_raises_when_a_visualization_library_is_missing(self, monkeypatch):
        original_find_spec = importlib.util.find_spec

        def find_spec_without_dash(name, *args, **kwargs):
            if name == "dash":
                return None
            return original_find_spec(name, *args, **kwargs)

        monkeypatch.setattr(importlib.util, "find_spec", find_spec_without_dash)

        with pytest.raises(ModuleNotFoundError):
            InteractiveGraphVisualizer.check_dependencies()
