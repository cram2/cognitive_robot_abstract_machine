from __future__ import annotations

import importlib.util
import math
from dataclasses import dataclass

import pytest
import rustworkx as rx

from krrood.rustworkx_utils.graph_visualizer_base import GraphLayout
from krrood.rustworkx_utils.visualization.three_graph_visualizer import (
    ThreeGraphVisualizer,
    LAYOUT_OPTIONS,
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


def named_visualizer(graph: rx.PyDiGraph, **overrides) -> ThreeGraphVisualizer:
    """A visualizer that labels and describes nodes by their payload name."""
    return ThreeGraphVisualizer(
        graph=graph,
        label_getter=lambda payload: payload.name,
        information_getter=lambda payload: [f"name: {payload.name}"],
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

        colors = [node["color"] for node in visualizer.graph_nodes()]
        assert colors == ["red", "green"]

    def test_non_contiguous_indices_after_removal_are_handled(self):
        graph = chain_graph(["a", "b", "c"])
        graph.remove_node(1)
        visualizer = named_visualizer(graph)

        ids = [node["id"] for node in visualizer.graph_nodes()]
        assert ids == ["0", "2"]

    def test_node_extra_data_defaults_to_empty(self):
        visualizer = named_visualizer(chain_graph(["a"]))

        assert visualizer.node_extra_data(0) == {}

    def test_node_extra_data_is_merged_into_the_node(self):
        class TaggedVisualizer(ThreeGraphVisualizer):
            def node_extra_data(self, node_index):
                return {"tag": "extra"}

        visualizer = TaggedVisualizer(
            graph=chain_graph(["a"]), label_getter=lambda p: p.name
        )

        assert visualizer.graph_nodes()[0]["tag"] == "extra"

    def test_fixed_layout_pins_nodes_at_the_position_getters_3d_result(self):
        visualizer = named_visualizer(
            chain_graph(["a"]),
            layout=GraphLayout.FIXED,
            position_getter=lambda payload: (1.0, 2.0, 3.0),
        )

        node = visualizer.graph_nodes()[0]
        assert (node["fx"], node["fy"], node["fz"]) == (1.0, 2.0, 3.0)

    def test_fixed_layout_supports_a_2d_position(self):
        visualizer = named_visualizer(
            chain_graph(["a"]),
            layout=GraphLayout.FIXED,
            position_getter=lambda payload: (1.0, 2.0),
        )

        node = visualizer.graph_nodes()[0]
        assert (node["fx"], node["fy"]) == (1.0, 2.0)
        assert "fz" not in node

    def test_non_fixed_layout_ignores_the_position_getter(self):
        visualizer = named_visualizer(
            chain_graph(["a"]),
            layout=GraphLayout.SPRING,
            position_getter=lambda payload: (1.0, 2.0, 3.0),
        )

        node = visualizer.graph_nodes()[0]
        assert "fx" not in node and "fy" not in node and "fz" not in node

    def test_fixed_layout_without_a_position_getter_leaves_the_node_unpinned(self):
        visualizer = named_visualizer(chain_graph(["a"]), layout=GraphLayout.FIXED)

        node = visualizer.graph_nodes()[0]
        assert "fx" not in node and "fy" not in node and "fz" not in node

    def test_fixed_position_values_are_plain_json_safe_floats(self):
        import numpy as np

        visualizer = named_visualizer(
            chain_graph(["a"]),
            layout=GraphLayout.FIXED,
            position_getter=lambda payload: (
                np.float64(1.0),
                np.float64(2.0),
                np.float64(3.0),
            ),
        )

        node = visualizer.graph_nodes()[0]
        assert type(node["fx"]) is float
        assert type(node["fy"]) is float
        assert type(node["fz"]) is float


class TestGraphLinks:
    def test_links_connect_node_ids(self):
        visualizer = named_visualizer(chain_graph(["a", "b"]))

        links = visualizer.graph_links()
        assert [(link["source"], link["target"]) for link in links] == [("0", "1")]

    def test_one_link_per_graph_edge(self):
        visualizer = named_visualizer(chain_graph(["a", "b", "c"]))

        assert len(visualizer.graph_links()) == 2


class TestFlaskEndpoints:
    def test_index_page_serves_3d_force_graph(self):
        client = named_visualizer(chain_graph(["a"])).build_application().test_client()

        response = client.get("/")

        assert response.status_code == 200
        assert b"3d-force-graph" in response.data.lower()

    def test_graph_endpoint_returns_nodes_and_links(self):
        client = (
            named_visualizer(chain_graph(["a", "b"])).build_application().test_client()
        )

        payload = client.get("/graph").get_json()

        assert len(payload["nodes"]) == 2
        assert len(payload["links"]) == 1

    def test_node_endpoint_returns_details(self):
        client = (
            named_visualizer(chain_graph(["a", "b", "c"]))
            .build_application()
            .test_client()
        )

        assert client.get("/node/2").get_json()["details"] == ["name: c"]


class TestAppearance:
    def test_refresh_only_rebuilds_the_graph_when_the_structure_changes(self):
        client = named_visualizer(chain_graph(["a"])).build_application().test_client()

        page = client.get("/").data
        assert b"structureChanged" in page
        assert b"graph.graphData(payload)" in page

    def test_links_are_rendered_with_a_nonzero_width(self):
        client = named_visualizer(chain_graph(["a"])).build_application().test_client()

        page = client.get("/").data
        assert b".linkWidth(" in page
        assert b".linkWidth(0)" not in page

    def test_hovering_a_node_shows_a_popup_with_its_details(self):
        client = named_visualizer(chain_graph(["a"])).build_application().test_client()

        page = client.get("/").data
        assert b'id="popup"' in page
        assert b".onNodeHover(" in page
        assert b"showPopup(node.label, payload.details" in page

    def test_clicking_a_node_no_longer_triggers_the_popup(self):
        client = named_visualizer(chain_graph(["a"])).build_application().test_client()

        page = client.get("/").data.decode()
        click_handler = page[page.index(".onNodeClick(") : page.index(".onNodeHover(")]
        assert "showPopup" not in click_handler

    def test_every_node_gets_an_always_visible_label(self):
        client = named_visualizer(chain_graph(["a"])).build_application().test_client()

        page = client.get("/").data
        assert b".node-label {" in page
        assert b'element.className = "node-label"' in page
        assert b"updateLabels" in page
        assert b"onBackgroundClick(hidePopup)" in page


class TestLayout:
    def test_layered_layout_uses_the_built_in_dag_mode(self):
        visualizer = named_visualizer(chain_graph(["a"]), layout=GraphLayout.LAYERED)

        options = LAYOUT_OPTIONS.get_options(visualizer.layout)
        assert options["dagMode"] == "td"

    def test_physics_layout_keeps_the_simulation_running_forever(self):
        visualizer = named_visualizer(chain_graph(["a"]), layout=GraphLayout.PHYSICS)

        options = LAYOUT_OPTIONS.get_options(visualizer.layout)
        assert math.isinf(options["cooldownTicks"])

    def test_spring_layout_settles_and_stops(self):
        visualizer = named_visualizer(chain_graph(["a"]), layout=GraphLayout.SPRING)

        options = LAYOUT_OPTIONS.get_options(visualizer.layout)
        assert options["cooldownTicks"] < math.inf

    def test_fixed_layout_runs_just_enough_ticks_to_apply_pinned_positions(self):
        visualizer = named_visualizer(chain_graph(["a"]), layout=GraphLayout.FIXED)

        options = LAYOUT_OPTIONS.get_options(visualizer.layout)
        assert 0 < options["cooldownTicks"] < math.inf


class TestExtensionHooks:
    def test_register_additional_routes_is_called_with_the_application(self):
        calls = []

        class RoutedVisualizer(ThreeGraphVisualizer):
            def register_additional_routes(self, application):
                calls.append(application)

                @application.route("/extra")
                def extra():
                    return "extra"

        client = (
            RoutedVisualizer(graph=chain_graph(["a"])).build_application().test_client()
        )

        assert client.get("/extra").data == b"extra"
        assert len(calls) == 1

    def test_extra_script_defaults_to_empty(self):
        visualizer = named_visualizer(chain_graph(["a"]))

        assert visualizer.extra_script() == ""

    def test_extra_script_is_appended_to_the_rendered_page(self):
        class ScriptedVisualizer(ThreeGraphVisualizer):
            def extra_script(self):
                return "console.log('marker-from-subclass');"

        client = (
            ScriptedVisualizer(graph=chain_graph(["a"]))
            .build_application()
            .test_client()
        )

        page = client.get("/").data
        assert b"marker-from-subclass" in page

    def test_extra_head_defaults_to_empty(self):
        visualizer = named_visualizer(chain_graph(["a"]))

        assert visualizer.extra_head() == ""

    def test_extra_head_is_rendered_before_the_main_script(self):
        class HeadedVisualizer(ThreeGraphVisualizer):
            def extra_head(self):
                return '<script src="https://example.com/marker-lib.js"></script>'

        client = (
            HeadedVisualizer(graph=chain_graph(["a"])).build_application().test_client()
        )

        page = client.get("/").data.decode()
        assert "marker-lib.js" in page
        assert page.index("marker-lib.js") < page.index("ForceGraph3D()")


class TestCheckDependencies:
    def test_raises_when_flask_is_missing(self, monkeypatch):
        original_find_spec = importlib.util.find_spec

        def find_spec_without_flask(name, *args, **kwargs):
            if name == "flask":
                return None
            return original_find_spec(name, *args, **kwargs)

        monkeypatch.setattr(importlib.util, "find_spec", find_spec_without_flask)

        with pytest.raises(ModuleNotFoundError):
            ThreeGraphVisualizer.check_dependencies()
