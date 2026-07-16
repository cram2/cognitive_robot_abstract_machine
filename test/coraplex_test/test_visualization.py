from __future__ import annotations

from dataclasses import dataclass, field

import networkx as nx
import pytest
from rustworkx import PyDiGraph, PyGraph

from coraplex.visualization import (
    GraphVisualizer,
    build_nx_graph,
    calculate_layout_positions,
    create_ordered_graph,
    _collect_properties,
    _escape_html,
    _format_params,
    _graph_snapshot,
    _object_params_with_properties,
)


@dataclass
class FakePlanNode:
    """A minimal stand-in for a plan node used to exercise graph construction."""

    index: int
    """The index of the node inside the originating plan graph."""

    children: list = field(default_factory=list)
    """The direct successors of this node."""


@dataclass
class FakePlan:
    """A minimal stand-in for a plan exposing only what the visualizer reads."""

    nodes: list
    """The nodes contained in the plan, in arbitrary order."""


@dataclass
class PayloadWithAttributes:
    """A payload whose public instance attributes should become parameters."""

    name: str
    """A public attribute that must be displayed."""

    count: int
    """A second public attribute that must be displayed."""

    _hidden: str = "secret"
    """A private attribute that must be ignored."""


class PayloadWithProperties:
    """A payload exposing properties with different access behaviours."""

    @property
    def doubled(self) -> int:
        """A readable property whose value must be collected."""
        return 21 * 2

    @property
    def label(self) -> str:
        """A property named ``label`` that must never be collected as a parameter."""
        return "ignored"

    @property
    def broken(self) -> int:
        """A property that raises on access and must be skipped silently."""
        raise ValueError("cannot read this property")


def directed_graph_with_payloads(payloads) -> PyDiGraph:
    """Build a directed rustworkx graph holding the given payloads as a chain."""
    graph = PyDiGraph(multigraph=False)
    indices = [graph.add_node(payload) for payload in payloads]
    for parent, child in zip(indices, indices[1:]):
        graph.add_edge(parent, child, None)
    return graph


class TestCreateOrderedGraph:
    def test_remaps_indices_to_contiguous_range(self):
        root = FakePlanNode(index=10)
        child = FakePlanNode(index=25)
        root.children = [child]
        plan = FakePlan(nodes=[root, child])

        ordered_graph, mapping = create_ordered_graph(plan)

        assert mapping == {10: 0, 25: 1}
        assert ordered_graph.num_nodes() == 2

    def test_preserves_edges_under_remapping(self):
        root = FakePlanNode(index=3)
        first_child = FakePlanNode(index=7)
        second_child = FakePlanNode(index=9)
        root.children = [first_child, second_child]
        plan = FakePlan(nodes=[root, first_child, second_child])

        ordered_graph, mapping = create_ordered_graph(plan)

        expected_edges = {(mapping[3], mapping[7]), (mapping[3], mapping[9])}
        assert set(ordered_graph.edge_list()) == expected_edges

    def test_empty_plan_yields_empty_graph(self):
        ordered_graph, mapping = create_ordered_graph(FakePlan(nodes=[]))

        assert mapping == {}
        assert ordered_graph.num_nodes() == 0


class TestCalculateLayoutPositions:
    def test_empty_graph_returns_empty_mapping(self):
        assert calculate_layout_positions("bfs", nx.Graph()) == {}

    @pytest.mark.parametrize("layout", ["spring", "kamada_kawai", "bfs"])
    def test_layout_assigns_position_to_every_node(self, layout):
        graph = nx.path_graph(4)

        positions = calculate_layout_positions(layout, graph)

        assert set(positions) == set(graph.nodes)
        assert all(len(position) == 2 for position in positions.values())

    def test_unknown_layout_falls_back_to_spring(self):
        graph = nx.path_graph(3)

        positions = calculate_layout_positions("does_not_exist", graph)
        spring_positions = calculate_layout_positions("spring", graph)

        assert set(positions) == set(spring_positions)
        assert all(
            tuple(positions[node]) == tuple(spring_positions[node])
            for node in spring_positions
        )

    def test_bfs_with_invalid_start_uses_existing_node(self):
        graph = nx.path_graph(3)

        positions = calculate_layout_positions("bfs", graph, start=999)

        assert set(positions) == set(graph.nodes)


class TestBuildNxGraph:
    def test_default_label_uses_payload_label_key(self):
        graph = directed_graph_with_payloads([{"label": "pick"}])

        nx_graph = build_nx_graph(graph, None, None, None)

        assert nx_graph.nodes[0]["label"] == "pick"

    def test_custom_node_label_callable_is_used(self):
        graph = directed_graph_with_payloads([{"value": 5}])

        nx_graph = build_nx_graph(
            graph, None, None, lambda index, payload: f"node-{index}"
        )

        assert nx_graph.nodes[0]["label"] == "node-0"

    def test_explicit_node_params_are_rendered(self):
        graph = directed_graph_with_payloads([{"label": "a"}])

        nx_graph = build_nx_graph(graph, {0: {"speed": 3}}, None, None)

        assert "speed" in nx_graph.nodes[0]["param_text"]
        assert "3" in nx_graph.nodes[0]["param_text"]

    def test_attribute_filter_limits_displayed_params(self):
        graph = directed_graph_with_payloads([{"label": "a"}])

        nx_graph = build_nx_graph(graph, {0: {"speed": 3, "force": 7}}, ["speed"], None)

        assert "speed" in nx_graph.nodes[0]["param_text"]
        assert "force" not in nx_graph.nodes[0]["param_text"]

    def test_directed_graph_produces_directed_networkx_graph(self):
        graph = directed_graph_with_payloads([{"label": "a"}, {"label": "b"}])

        nx_graph = build_nx_graph(graph, None, None, None)

        assert nx_graph.is_directed()
        assert list(nx_graph.edges()) == [(0, 1)]

    def test_undirected_graph_produces_undirected_networkx_graph(self):
        graph = PyGraph(multigraph=False)
        first = graph.add_node({"label": "a"})
        second = graph.add_node({"label": "b"})
        graph.add_edge(first, second, None)

        nx_graph = build_nx_graph(graph, None, None, None)

        assert not nx_graph.is_directed()

    def test_non_contiguous_indices_after_removal_are_handled(self):
        graph = directed_graph_with_payloads(
            [{"label": "a"}, {"label": "b"}, {"label": "c"}]
        )
        graph.remove_node(1)

        nx_graph = build_nx_graph(graph, None, None, None)

        assert set(nx_graph.nodes) == {0, 2}


class TestGraphSnapshot:
    def test_snapshot_is_stable_for_equal_graphs(self):
        first = build_nx_graph(
            directed_graph_with_payloads([{"label": "a"}]), None, None, None
        )
        second = build_nx_graph(
            directed_graph_with_payloads([{"label": "a"}]), None, None, None
        )

        assert _graph_snapshot(first) == _graph_snapshot(second)

    def test_snapshot_changes_when_label_changes(self):
        before = build_nx_graph(
            directed_graph_with_payloads([{"label": "a"}]), None, None, None
        )
        after = build_nx_graph(
            directed_graph_with_payloads([{"label": "b"}]), None, None, None
        )

        assert _graph_snapshot(before) != _graph_snapshot(after)

    def test_snapshot_changes_when_edges_change(self):
        single = build_nx_graph(
            directed_graph_with_payloads([{"label": "a"}]), None, None, None
        )
        connected = build_nx_graph(
            directed_graph_with_payloads([{"label": "a"}, {"label": "a"}]),
            None,
            None,
            None,
        )

        assert _graph_snapshot(single) != _graph_snapshot(connected)


class TestObjectParamsWithProperties:
    def test_dict_payload_drops_label_key(self):
        params = _object_params_with_properties({"label": "a", "speed": 4})

        assert params == {"speed": 4}

    def test_none_payload_returns_none(self):
        assert _object_params_with_properties(None) is None

    def test_public_attributes_are_collected_without_private(self):
        params = _object_params_with_properties(
            PayloadWithAttributes(name="cup", count=2)
        )

        assert params == {"name": "cup", "count": 2}

    def test_properties_are_collected_and_failures_skipped(self):
        params = _object_params_with_properties(PayloadWithProperties())

        assert params == {"doubled": 42}


class TestCollectProperties:
    def test_collects_readable_properties_and_skips_broken_and_label(self):
        properties = _collect_properties(PayloadWithProperties())

        assert properties == {"doubled": 42}


class TestFormatParams:
    def test_empty_params_render_placeholder(self):
        assert _format_params(None) == "<i>No parameters</i>"
        assert _format_params({}) == "<i>No parameters</i>"

    def test_params_render_as_escaped_table(self):
        rendered = _format_params({"name": "<b>x</b>"})

        assert "<table>" in rendered
        assert "&lt;b&gt;x&lt;/b&gt;" in rendered
        assert "<b>x</b>" not in rendered.replace("<b>name</b>", "")


class TestEscapeHtml:
    def test_escapes_all_special_characters(self):
        escaped = _escape_html("<a href=\"x\">'&'</a>")

        assert "<" not in escaped
        assert ">" not in escaped
        assert "&lt;a href=&quot;x&quot;&gt;&#39;&amp;&#39;&lt;/a&gt;" == escaped


class TestGraphVisualizer:
    def test_build_current_nx_graph_polls_graph_source(self):
        first = directed_graph_with_payloads([{"label": "a"}])
        second = directed_graph_with_payloads([{"label": "a"}, {"label": "b"}])
        graphs = iter([first, second])
        visualizer = GraphVisualizer(graph=first, graph_source=lambda: next(graphs))

        first_result = visualizer._build_current_nx_graph()
        second_result = visualizer._build_current_nx_graph()

        assert first_result.number_of_nodes() == 1
        assert second_result.number_of_nodes() == 2

    def test_build_current_nx_graph_uses_static_graph_without_source(self):
        graph = directed_graph_with_payloads([{"label": "a"}])
        visualizer = GraphVisualizer(graph=graph)

        nx_graph = visualizer._build_current_nx_graph()

        assert nx_graph.number_of_nodes() == 1
