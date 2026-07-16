from __future__ import annotations

import importlib.util
import threading
import time

import pytest
import rustworkx as rx

from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.visualization.mesh_cytoscape_graph_visualizer import (
    MESH_ANIMATION_FRAME_COUNT,
    MeshCytoscapeGraphVisualizer,
)
from semantic_digital_twin.world_description.geometry import Box, Scale, Sphere
from semantic_digital_twin.world_description.shape_collection import ShapeCollection
from semantic_digital_twin.world_description.world_entity import Body


def body_graph(*bodies: Body) -> rx.PyDiGraph:
    """Build a chain graph out of the given bodies (or other payloads)."""
    graph = rx.PyDiGraph(multigraph=False)
    indices = [graph.add_node(body) for body in bodies]
    for parent, child in zip(indices, indices[1:]):
        graph.add_edge(parent, child, None)
    return graph


def named_visualizer(graph: rx.PyDiGraph, **overrides) -> MeshCytoscapeGraphVisualizer:
    """A visualizer that labels bodies by their name."""
    return MeshCytoscapeGraphVisualizer(
        graph=graph,
        label_getter=lambda payload: getattr(payload, "name", payload),
        **overrides,
    )


def body_with_visual_sphere(name: str) -> Body:
    return Body(name=PrefixedName(name), visual=ShapeCollection([Sphere(radius=0.3)]))


def body_without_visual(name: str) -> Body:
    return Body(name=PrefixedName(name))


def body_with_asymmetric_visual_box(name: str) -> Body:
    """A body whose visual shape has no rotational symmetry around its vertical axis, so
    different rotation frames are guaranteed to render differently."""
    return Body(
        name=PrefixedName(name),
        visual=ShapeCollection([Box(scale=Scale(x=1.0, y=2.0, z=0.5))]),
    )


class TestNodeMesh:
    def test_body_with_visual_shapes_has_a_mesh(self):
        visualizer = named_visualizer(body_graph(body_with_visual_sphere("a")))

        assert visualizer._node_mesh(0) is not None

    def test_body_without_visual_shapes_has_no_mesh(self):
        visualizer = named_visualizer(body_graph(body_without_visual("a")))

        assert visualizer._node_mesh(0) is None

    def test_non_body_payload_has_no_mesh(self):
        graph = rx.PyDiGraph()
        graph.add_node("not a body")
        visualizer = named_visualizer(graph)

        assert visualizer._node_mesh(0) is None


class TestGraphElements:
    def test_node_with_visual_shapes_gets_an_image_and_frame_count(self):
        visualizer = named_visualizer(body_graph(body_with_visual_sphere("a")))

        data = visualizer.graph_elements()[0]["data"]
        assert data["image"] == "mesh/0/0.png"
        assert data["frameCount"] == MESH_ANIMATION_FRAME_COUNT

    def test_node_without_visual_shapes_has_no_image(self):
        visualizer = named_visualizer(body_graph(body_without_visual("a")))

        element = visualizer.graph_elements()[0]
        assert "image" not in element["data"]
        assert "frameCount" not in element["data"]

    def test_still_falls_back_to_the_base_circle_color(self):
        visualizer = named_visualizer(
            body_graph(body_without_visual("a")), color_getter=lambda body: "red"
        )

        element = visualizer.graph_elements()[0]
        assert element["data"]["color"] == "red"

    def test_building_elements_does_not_render_any_frame(self):
        visualizer = named_visualizer(body_graph(body_with_visual_sphere("a")))

        visualizer.graph_elements()

        assert visualizer._mesh_frame_cache == {}


class TestMeshFrameCaching:
    def test_frame_is_only_rendered_once(self):
        visualizer = named_visualizer(body_graph(body_with_visual_sphere("a")))

        first = visualizer._mesh_frame_image(0, 0)
        second = visualizer._mesh_frame_image(0, 0)

        assert first is second

    def test_different_frames_render_different_rotations(self):
        visualizer = named_visualizer(body_graph(body_with_asymmetric_visual_box("a")))

        first_frame = visualizer._mesh_frame_image(0, 0)
        later_frame = visualizer._mesh_frame_image(0, MESH_ANIMATION_FRAME_COUNT // 4)

        assert first_frame != later_frame

    def test_frame_index_wraps_around(self):
        visualizer = named_visualizer(body_graph(body_with_visual_sphere("a")))

        in_range = visualizer._mesh_frame_image(0, 2)
        wrapped = visualizer._mesh_frame_image(0, 2 + MESH_ANIMATION_FRAME_COUNT)

        assert in_range == wrapped

    def test_missing_mesh_is_cached_as_none(self):
        visualizer = named_visualizer(body_graph(body_without_visual("a")))

        assert visualizer._mesh_frame_image(0, 0) is None
        assert (0, 0) in visualizer._mesh_frame_cache

    def test_concurrent_requests_render_the_same_frame_only_once(self):
        visualizer = named_visualizer(body_graph(body_with_visual_sphere("a")))
        render_calls = []
        original_render = visualizer._render_mesh_thumbnail

        def counting_render(mesh, rotation_angle=0.0):
            render_calls.append(1)
            time.sleep(0.05)  # widen the race window so overlapping calls are likely
            return original_render(mesh, rotation_angle)

        visualizer._render_mesh_thumbnail = counting_render

        threads = [
            threading.Thread(target=visualizer._mesh_frame_image, args=(0, 0)) for _ in range(5)
        ]
        for started_thread in threads:
            started_thread.start()
        for finished_thread in threads:
            finished_thread.join()

        assert len(render_calls) == 1


class TestExtraNodeStyles:
    def test_adds_an_image_fill_selector(self):
        visualizer = named_visualizer(body_graph(body_with_visual_sphere("a")))

        selectors = [style["selector"] for style in visualizer.extra_node_styles()]
        assert "node[image]" in selectors

    def test_mesh_thumbnails_are_shown_uncropped(self):
        visualizer = named_visualizer(body_graph(body_with_visual_sphere("a")))

        image_style = next(
            style["style"]
            for style in visualizer.extra_node_styles()
            if style["selector"] == "node[image]"
        )
        assert image_style["shape"] == "round-rectangle"
        assert image_style["background-fit"] == "contain"

    def test_nodes_are_bigger_than_the_base_default(self):
        visualizer = named_visualizer(body_graph(body_with_visual_sphere("a")))

        node_style = next(
            style["style"]
            for style in visualizer.extra_node_styles()
            if style["selector"] == "node"
        )
        assert node_style["width"] > 30
        assert node_style["width"] == node_style["height"]

    def test_hover_size_grows_from_the_bigger_default(self):
        visualizer = named_visualizer(body_graph(body_with_visual_sphere("a")))

        styles = {style["selector"]: style["style"] for style in visualizer.extra_node_styles()}

        assert styles["node.hovered"]["width"] > styles["node"]["width"]


class TestMeshEndpoint:
    def test_returns_a_png_for_a_node_with_a_mesh(self):
        client = named_visualizer(
            body_graph(body_with_visual_sphere("a"))
        ).build_application().test_client()

        response = client.get("/mesh/0/0.png")

        assert response.status_code == 200
        assert response.mimetype == "image/png"
        assert len(response.data) > 0

    def test_returns_a_different_png_for_a_later_frame(self):
        client = named_visualizer(
            body_graph(body_with_asymmetric_visual_box("a"))
        ).build_application().test_client()

        first = client.get("/mesh/0/0.png").data
        later = client.get(f"/mesh/0/{MESH_ANIMATION_FRAME_COUNT // 4}.png").data

        assert first != later

    def test_returns_404_for_a_node_without_a_mesh(self):
        client = named_visualizer(
            body_graph(body_without_visual("a"))
        ).build_application().test_client()

        response = client.get("/mesh/0/0.png")

        assert response.status_code == 404


class TestExtraScript:
    def test_animation_loop_cycles_frames_by_frame_count(self):
        visualizer = named_visualizer(body_graph(body_with_visual_sphere("a")))

        script = visualizer.extra_script()
        assert "frameCount" in script
        assert "mesh/" in script

    def test_animation_loop_is_appended_to_the_rendered_page(self):
        client = named_visualizer(
            body_graph(body_with_visual_sphere("a"))
        ).build_application().test_client()

        page = client.get("/").data
        assert b"frameCount" in page


class TestCheckDependencies:
    @pytest.mark.parametrize("missing_module", ["flask", "matplotlib"])
    def test_raises_when_a_required_module_is_missing(self, monkeypatch, missing_module):
        original_find_spec = importlib.util.find_spec

        def find_spec_without_module(name, *args, **kwargs):
            if name == missing_module:
                return None
            return original_find_spec(name, *args, **kwargs)

        monkeypatch.setattr(importlib.util, "find_spec", find_spec_without_module)

        with pytest.raises(ModuleNotFoundError):
            MeshCytoscapeGraphVisualizer.check_dependencies()
