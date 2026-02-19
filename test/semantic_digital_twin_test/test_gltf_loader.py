"""
Test file for GLTFLoader.

Run this to verify the implementation works correctly.
"""

import pytest
import trimesh
import numpy as np
from pathlib import Path

from semantic_digital_twin.world import World
from semantic_digital_twin.pipeline.gltf_loader import GLTFLoader
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.world_description.world_entity import Body
from semantic_digital_twin.world_description.connections import FixedConnection

import tempfile
import os


class TestGLTFLoader:
    """Test suite for GLTFLoader."""

    def test_invalid_file_path(self):
        """Test handling of non-existent file."""
        world = World()
        loader = GLTFLoader(file_path="/nonexistent/path/file.gltf")

        with pytest.raises(ValueError, match="Failed to load file"):
            world = loader.apply(world)

    def test_malformed_gltf(self):
        """Test handling of malformed GLTF data."""
        with tempfile.NamedTemporaryFile(suffix=".gltf", delete=False, mode='w') as f:
            f.write("{ invalid json without closing brace")
            temp_path = f.name

        try:
            world = World()
            loader = GLTFLoader(file_path=temp_path)

            with pytest.raises(ValueError, match="Failed to load file"):
                world = loader.apply(world)
        finally:
            os.unlink(temp_path)
    def test_empty_gltf_file(self):
        """Test handling of GLTF file with only a root node (no geometry)."""
        # Create minimal scene with empty root mesh
        scene = trimesh.Scene()
        empty_mesh = trimesh.Trimesh()  # Empty mesh with 0 vertices
        scene.add_geometry(empty_mesh, node_name="root", geom_name="empty_geom")

        with tempfile.NamedTemporaryFile(suffix=".glb", delete=False) as f:
            temp_path = f.name

        scene.export(temp_path, file_type="glb")

        try:
            world = World()
            loader = GLTFLoader(file_path=temp_path)
            world = loader.apply(world)

            # Should create body with empty collision shapes
            assert world.root is not None
            assert len(world.root.collision.shapes) == 0
        finally:
            os.unlink(temp_path)

    def test_non_geometry_nodes_only(self):
        """Test GLTF with transform nodes that have no actual mesh data."""
        scene = trimesh.Scene()

        # Root with geometry, child without geometry
        mesh = trimesh.creation.box(extents=[1.0, 1.0, 1.0])
        scene.add_geometry(mesh, node_name="root", geom_name="root_geom")

        # Add transform-only node (will be skipped during processing)
        scene.graph.update(
            frame_to="transform_node",
            frame_from="root",
            transform=trimesh.transformations.translation_matrix([1, 0, 0])
        )

        # Add leaf with geometry as child of transform node
        leaf_mesh = trimesh.creation.icosphere(radius=0.3)
        scene.add_geometry(
            leaf_mesh,
            node_name="leaf",
            geom_name="leaf_geom",
            parent_node_name="transform_node",
            transform=trimesh.transformations.translation_matrix([0, 1, 0])
        )

        with tempfile.NamedTemporaryFile(suffix=".glb", delete=False) as f:
            temp_path = f.name

        scene.export(temp_path, file_type="glb")

        try:
            world = World()
            loader = GLTFLoader(file_path=temp_path)
            world = loader.apply(world)

            # Should skip transform_node and connect leaf directly to root
            assert len(list(world.kinematic_structure_entities)) == 2

            # Verify transform_node was skipped
            body_names = {str(b.name) for b in world.kinematic_structure_entities}
            assert "root" in body_names
            assert "leaf" in body_names
            assert "transform_node" not in body_names
        finally:
            os.unlink(temp_path)

    def test_trimesh_to_body(self):
        """Test converting a trimesh to Body object."""
        mesh = trimesh.creation.box(extents=[1.0, 1.0, 1.0])
        loader = GLTFLoader(file_path="/dummy/path.gltf")
        body = loader._trimesh_to_body(mesh, "test_cube")

        assert isinstance(body, Body)
        assert body.name == PrefixedName("test_cube")
        assert body.collision is not None
        assert body.visual is not None
        assert len(body.collision.shapes) == 1
        assert body.collision.combined_mesh is not None

    def test_fusion_meshes_single(self):
        """Test fusing a single mesh."""
        mesh = trimesh.creation.box(extents=[1.0, 1.0, 1.0])
        scene = trimesh.Scene()
        scene.add_geometry(mesh, node_name="box", geom_name="box_geom")

        loader = GLTFLoader(file_path="/dummy/path.gltf")
        loader.scene = scene

        fused = loader._fusion_meshes(["box"])

        assert isinstance(fused, trimesh.Trimesh)
        assert len(fused.vertices) > 0
        assert len(fused.faces) > 0

    def test_fusion_meshes_multiple(self):
        """Test fusing multiple meshes."""
        mesh1 = trimesh.creation.box(extents=[1.0, 1.0, 1.0])
        mesh2 = trimesh.creation.cylinder(radius=0.5, height=2.0)

        scene = trimesh.Scene()
        scene.add_geometry(mesh1, node_name="box", geom_name="box_geom")
        scene.add_geometry(mesh2, node_name="cylinder", geom_name="cyl_geom",
                          transform=trimesh.transformations.translation_matrix([2, 0, 0]))

        loader = GLTFLoader(file_path="/dummy/path.gltf")
        loader.scene = scene

        fused = loader._fusion_meshes(["box", "cylinder"])

        assert isinstance(fused, trimesh.Trimesh)
        assert len(fused.vertices) == len(mesh1.vertices) + len(mesh2.vertices)

    def test_get_root_node_single(self):
        """Test getting root node with single root."""
        mesh = trimesh.creation.box(extents=[1.0, 1.0, 1.0])
        scene = trimesh.Scene()
        scene.add_geometry(mesh, node_name="root_object", geom_name="geom")

        loader = GLTFLoader(file_path="/dummy/path.gltf")
        loader.scene = scene

        root = loader._get_root_node()
        assert root == "root_object"

    def test_get_root_node_multiple_fails(self):
        """Test that multiple roots raises error."""
        mesh1 = trimesh.creation.box(extents=[1.0, 1.0, 1.0])
        mesh2 = trimesh.creation.icosphere(radius=0.5)

        scene = trimesh.Scene()
        scene.add_geometry(mesh1, node_name="root1", geom_name="geom1")
        scene.add_geometry(mesh2, node_name="root2", geom_name="geom2")

        loader = GLTFLoader(file_path="/dummy/path.gltf")
        loader.scene = scene

        with pytest.raises(ValueError, match="More than one root"):
            loader._get_root_node()

    def test_build_world_single_body(self):
        """Test building world with single body."""
        mesh = trimesh.creation.box(extents=[1.0, 1.0, 1.0])
        loader = GLTFLoader(file_path="/dummy/path.gltf")
        body = loader._trimesh_to_body(mesh, "root_body")

        scene = trimesh.Scene()
        scene.add_geometry(mesh, node_name="root_body", geom_name="geom")
        loader.scene = scene

        world = World()
        world_elements = {"root_body": body}
        connection = {"root_body": []}

        # Fixed: wrap in modify_world context
        with world.modify_world():
            world = loader._build_world_from_elements(world_elements, connection, world)

        assert world.root is not None
        assert world.root.name == PrefixedName("root_body")
        assert len(list(world.kinematic_structure_entities)) == 1

    def test_build_world_with_connections(self):
        """Test building world with parent-child connections."""
        mesh1 = trimesh.creation.box(extents=[1.0, 1.0, 1.0])
        mesh2 = trimesh.creation.icosphere(radius=0.5)

        loader = GLTFLoader(file_path="/dummy/path.gltf")
        parent_body = loader._trimesh_to_body(mesh1, "parent")
        child_body = loader._trimesh_to_body(mesh2, "child")

        scene = trimesh.Scene()
        scene.add_geometry(mesh1, node_name="parent", geom_name="geom1")
        scene.add_geometry(mesh2, node_name="child", geom_name="geom2",
                          parent_node_name="parent")
        loader.scene = scene

        world = World()
        world_elements = {"parent": parent_body, "child": child_body}
        connection = {"parent": ["child"], "child": []}

        # Wrap in modify_world context
        with world.modify_world():
            world = loader._build_world_from_elements(world_elements, connection, world)

        assert world.root is not None
        assert world.root.name == PrefixedName("parent")
        assert len(list(world.kinematic_structure_entities)) == 2
        assert len(list(world.connections)) == 1

        conn = list(world.connections)[0]
        assert isinstance(conn, FixedConnection)
        assert conn.parent.name == PrefixedName("parent")
        assert conn.child.name == PrefixedName("child")

    def test_grouping_similar_meshes_basic(self):
        """Test grouping meshes with similar names."""
        scene = trimesh.Scene()
        mesh = trimesh.creation.box(extents=[1.0, 1.0, 1.0])

        scene.add_geometry(mesh, node_name="Bolt_001", geom_name="geom1")
        scene.add_geometry(mesh, node_name="Bolt_002", geom_name="geom2",
                          parent_node_name="Bolt_001")
        scene.add_geometry(mesh, node_name="Nut_001", geom_name="geom3",
                          parent_node_name="Bolt_001")

        loader = GLTFLoader(file_path="/dummy/path.gltf")
        loader.scene = scene

        object_nodes, new_objects = loader._grouping_similar_meshes("Bolt_001")

        assert "Bolt_001" in object_nodes
        assert len(object_nodes) >= 1

    def test_get_relative_transform(self):
        """Test computing relative transform between nodes."""
        mesh = trimesh.creation.box(extents=[1.0, 1.0, 1.0])
        scene = trimesh.Scene()

        parent_transform = trimesh.transformations.translation_matrix([1, 0, 0])
        child_transform = trimesh.transformations.translation_matrix([1, 2, 0])

        scene.add_geometry(mesh, node_name="parent", geom_name="geom1", transform=parent_transform)
        scene.add_geometry(mesh, node_name="child", geom_name="geom2",
                          parent_node_name="parent", transform=child_transform)

        loader = GLTFLoader(file_path="/dummy/path.gltf")
        loader.scene = scene

        relative = loader._get_relative_transform("parent", "child")

        assert relative is not None


def test_readme_example():
    """Test the basic example from README/docs."""
    import tempfile
    import os

    # Create scene with hierarchy
    box1 = trimesh.creation.box(extents=[1.0, 1.0, 1.0])
    box2 = trimesh.creation.box(extents=[0.5, 0.5, 0.5])

    scene = trimesh.Scene()
    scene.add_geometry(box1, node_name="base", geom_name="base_geom")
    scene.add_geometry(box2, node_name="top", geom_name="top_geom",
                      parent_node_name="base",
                      transform=trimesh.transformations.translation_matrix([0, 0, 1]))

    with tempfile.NamedTemporaryFile(suffix=".glb", delete=False) as f:
        temp_path = f.name

    # Export as GLB (binary format preserves hierarchy better)
    scene.export(temp_path, file_type="glb")

    try:
        world = World()
        loader = GLTFLoader(file_path=temp_path)
        world = loader.apply(world)

        assert world.root is not None
        assert len(list(world.kinematic_structure_entities)) >= 1

        print(f"✓ Successfully loaded scene with {len(list(world.kinematic_structure_entities))} bodies")

    finally:
        os.unlink(temp_path)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
