from __future__ import annotations

import importlib.util
import threading
import time

import pytest
import rustworkx as rx

from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from krrood.rustworkx_utils.visualization.mesh_three_graph_visualizer import (
    MeshThreeGraphVisualizer,
)
from semantic_digital_twin.world_description.geometry import Sphere
from semantic_digital_twin.world_description.shape_collection import ShapeCollection
from semantic_digital_twin.world_description.world_entity import Body


def body_graph(*bodies: Body) -> rx.PyDiGraph:
    """Build a chain graph out of the given bodies (or other payloads)."""
    graph = rx.PyDiGraph(multigraph=False)
    indices = [graph.add_node(body) for body in bodies]
    for parent, child in zip(indices, indices[1:]):
        graph.add_edge(parent, child, None)
    return graph


def body_visual_mesh(payload):
    """The combined visual mesh of a ``Body`` payload, or ``None`` otherwise."""
    if not isinstance(payload, Body) or not payload.visual:
        return None
    return payload.visual.combined_mesh


def named_visualizer(graph: rx.PyDiGraph, **overrides) -> MeshThreeGraphVisualizer:
    """A visualizer that labels bodies by their name."""
    overrides.setdefault("mesh_getter", body_visual_mesh)
    return MeshThreeGraphVisualizer(
        graph=graph,
        label_getter=lambda payload: getattr(payload, "name", payload),
        **overrides,
    )


def body_with_visual_sphere(name: str) -> Body:
    return Body(name=PrefixedName(name), visual=ShapeCollection([Sphere(radius=0.3)]))


def body_without_visual(name: str) -> Body:
    return Body(name=PrefixedName(name))


class TestNodeMesh:
    def test_body_with_visual_shapes_has_a_mesh(self):
        visualizer = named_visualizer(body_graph(body_with_visual_sphere("a")))

        assert visualizer.mesh_getter(visualizer.graph[0]) is not None

    def test_body_without_visual_shapes_has_no_mesh(self):
        visualizer = named_visualizer(body_graph(body_without_visual("a")))

        assert visualizer.mesh_getter(visualizer.graph[0]) is None

    def test_non_body_payload_has_no_mesh(self):
        graph = rx.PyDiGraph()
        graph.add_node("not a body")
        visualizer = named_visualizer(graph)

        assert visualizer.mesh_getter(visualizer.graph[0]) is None


class TestGraphNodes:
    def test_node_with_visual_shapes_gets_a_mesh_url(self):
        visualizer = named_visualizer(body_graph(body_with_visual_sphere("a")))

        data = visualizer.graph_nodes()[0]
        assert data["meshUrl"] == "mesh/0.glb"

    def test_node_without_visual_shapes_has_no_mesh_url(self):
        visualizer = named_visualizer(body_graph(body_without_visual("a")))

        assert "meshUrl" not in visualizer.graph_nodes()[0]

    def test_still_falls_back_to_the_base_circle_color(self):
        visualizer = named_visualizer(
            body_graph(body_without_visual("a")), color_getter=lambda body: "red"
        )

        assert visualizer.graph_nodes()[0]["color"] == "red"

    def test_building_nodes_does_not_export_any_mesh(self):
        visualizer = named_visualizer(body_graph(body_with_visual_sphere("a")))

        visualizer.graph_nodes()

        assert visualizer._mesh_glb_cache == {}


class TestMeshGlbCaching:
    def test_mesh_is_only_exported_once(self):
        visualizer = named_visualizer(body_graph(body_with_visual_sphere("a")))

        first = visualizer._mesh_glb(0)
        second = visualizer._mesh_glb(0)

        assert first is second

    def test_exported_bytes_are_a_valid_glb_header(self):
        visualizer = named_visualizer(body_graph(body_with_visual_sphere("a")))

        glb_bytes = visualizer._mesh_glb(0)

        assert glb_bytes[:4] == b"glTF"

    def test_missing_mesh_is_cached_as_none(self):
        visualizer = named_visualizer(body_graph(body_without_visual("a")))

        assert visualizer._mesh_glb(0) is None
        assert 0 in visualizer._mesh_glb_cache

    def test_concurrent_requests_export_the_same_mesh_only_once(self):
        visualizer = named_visualizer(body_graph(body_with_visual_sphere("a")))
        export_calls = []
        mesh = visualizer.mesh_getter(visualizer.graph[0])
        original_export = mesh.export

        def counting_export(*args, **kwargs):
            export_calls.append(1)
            time.sleep(0.05)  # widen the race window so overlapping calls are likely
            return original_export(*args, **kwargs)

        visualizer.mesh_getter = lambda payload: mesh
        mesh.export = counting_export

        threads = [
            threading.Thread(target=visualizer._mesh_glb, args=(0,)) for _ in range(5)
        ]
        for started_thread in threads:
            started_thread.start()
        for finished_thread in threads:
            finished_thread.join()

        assert len(export_calls) == 1


class TestMeshEndpoint:
    def test_returns_a_glb_for_a_node_with_a_mesh(self):
        client = (
            named_visualizer(body_graph(body_with_visual_sphere("a")))
            .build_application()
            .test_client()
        )

        response = client.get("/mesh/0.glb")

        assert response.status_code == 200
        assert response.mimetype == "model/gltf-binary"
        assert response.data[:4] == b"glTF"

    def test_returns_404_for_a_node_without_a_mesh(self):
        client = (
            named_visualizer(body_graph(body_without_visual("a")))
            .build_application()
            .test_client()
        )

        response = client.get("/mesh/0.glb")

        assert response.status_code == 404


class TestExtraHead:
    def test_loads_three_and_gltf_loader_as_blocking_scripts(self):
        visualizer = named_visualizer(body_graph(body_with_visual_sphere("a")))

        head = visualizer.extra_head()
        assert "three@" in head
        assert "GLTFLoader" in head

    def test_head_scripts_are_rendered_before_the_main_script(self):
        client = (
            named_visualizer(body_graph(body_with_visual_sphere("a")))
            .build_application()
            .test_client()
        )

        page = client.get("/").data.decode()
        assert page.index("GLTFLoader") < page.index("nodeThreeObject")


class TestExtraScript:
    def test_animation_loop_references_mesh_url(self):
        visualizer = named_visualizer(body_graph(body_with_visual_sphere("a")))

        script = visualizer.extra_script()
        assert "meshUrl" in script
        assert "nodeThreeObject" in script

    def test_animation_loop_is_appended_to_the_rendered_page(self):
        client = (
            named_visualizer(body_graph(body_with_visual_sphere("a")))
            .build_application()
            .test_client()
        )

        page = client.get("/").data
        assert b"meshUrl" in page

    def test_no_mesh_nodes_get_a_plain_sphere_matching_their_color(self):
        visualizer = named_visualizer(body_graph(body_with_visual_sphere("a")))

        script = visualizer.extra_script()
        assert "SphereGeometry" in script
        assert "MeshLambertMaterial({ color: node.color })" in script

    def test_does_not_redefine_label_handling_left_to_the_base_classs_dom_overlay(self):
        visualizer = named_visualizer(body_graph(body_with_visual_sphere("a")))

        script = visualizer.extra_script()
        assert "SpriteText" not in script
        assert "node-label" not in script

    def test_loaded_meshes_render_from_both_sides(self):
        visualizer = named_visualizer(body_graph(body_with_visual_sphere("a")))

        script = visualizer.extra_script()
        loader_callback = script[
            script.index("loader.load(") : script.index("function spin(")
        ]
        assert "THREE.DoubleSide" in loader_callback

    def test_loaded_meshes_are_not_forced_fully_metallic(self):
        visualizer = named_visualizer(body_graph(body_with_visual_sphere("a")))

        script = visualizer.extra_script()
        loader_callback = script[
            script.index("loader.load(") : script.index("function spin(")
        ]
        assert "material.metalness = 0" in loader_callback


class TestCheckDependencies:
    @pytest.mark.parametrize("missing_module", ["flask", "trimesh"])
    def test_raises_when_a_required_module_is_missing(
        self, monkeypatch, missing_module
    ):
        original_find_spec = importlib.util.find_spec

        def find_spec_without_module(name, *args, **kwargs):
            if name == missing_module:
                return None
            return original_find_spec(name, *args, **kwargs)

        monkeypatch.setattr(importlib.util, "find_spec", find_spec_without_module)

        with pytest.raises(ModuleNotFoundError):
            MeshThreeGraphVisualizer.check_dependencies()
