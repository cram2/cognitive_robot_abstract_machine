from typing import List, Tuple, Dict, Set
import re

import numpy as np
import trimesh

from .pipeline import Step
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.world_entity import Body
from semantic_digital_twin.world_description.connections import FixedConnection
from semantic_digital_twin.world_description.shape_collection import ShapeCollection
from semantic_digital_twin.world_description.geometry import TriangleMesh, Scale
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName

class GLTFLoader(Step):
    """Load GLTF/GLB files into a World.

    This loader parses GLTF/GLB files (including FreeCAD exports) and creates
    Body objects with FixedConnection relationships matching the scene hierarchy.

    Features:
    - Handles FreeCAD naming conventions (e.g., Bolt_001, Bolt_002 are fused)
    - Applies node transformations correctly
    - Skips non-geometry nodes while preserving hierarchy
    - Creates proper parent-child connections

    Example:
        >>> world = World()
        >>> loader = GLTFLoader(file_path="model.gltf")
        >>> world = loader.apply(world)

    Limitations:
    - Only creates FixedConnection (no joints/articulations)
    - Does not handle GLTF extensions for physics/joints

    Attributes:
        file_path: Path to the GLTF/GLB file
        scene: The loaded trimesh Scene (set after _apply is called)
    """

    file_path: str
    scene: trimesh.Scene = None


    def __init__(self, file_path: str,):
        self.file_path = file_path


    def _get_root_node(self) -> str:
        base_frame = self.scene.graph.base_frame
        root_children = self.scene.graph.transforms.children.get(base_frame, [])
        if len(root_children) > 1 or len(root_children) == 0:
            raise ValueError("More than one root node found in the scene, or no root node found.")
        return root_children[0]

    def _get_relative_transform(self, parent_node: str, child_node: str) -> HomogeneousTransformationMatrix:
        """Get the relative transform from parent to child node.

        Computes the transform that converts from parent frame to child frame.

        Args:
            parent_node: Name of the parent node
            child_node: Name of the child node

        Returns:
            The relative transformation matrix from parent to child
        """
        parent_transform, _ = self.scene.graph.get(parent_node)
        child_transform, _ = self.scene.graph.get(child_node)

        # Compute relative transform: parent_inv @ child
        parent_inv = np.linalg.inv(parent_transform)
        relative = parent_inv @ child_transform

        return HomogeneousTransformationMatrix(relative)

    def _trimesh_to_body(self, mesh: trimesh.Trimesh, name: str) -> Body:
        """Convert a trimesh.Trimesh to a Body object."""
        # Create TriangleMesh geometry from trimesh
        triangle_mesh = TriangleMesh(
            mesh=mesh,
            origin=HomogeneousTransformationMatrix.from_xyz_rpy(),  # Identity transform
            scale=Scale(1.0, 1.0, 1.0)  # No scaling
        )

        # Create ShapeCollection for collision and visual
        shape_collection = ShapeCollection([triangle_mesh])

        # Create Body
        body = Body(
            name=PrefixedName(name),
            collision=shape_collection,
            visual=shape_collection  # Use same for both collision and visual
        )

        return body

    def _grouping_similar_meshes(self, base_node: str) -> Tuple[Set[str], Set[str]]:
        """Group meshes with similar names (e.g., Bolt_001, Bolt_002 -> Bolt).

        FreeCAD exports parts with suffixes like _001, _002, etc.
        This method groups them for fusion.
        """
        # Extract base name by removing trailing _XXX suffix (numbers or short alphanumeric)
        base_name_match = re.match(r"^(.+?)(?:_\d+|_[A-Za-z]\d*)?$", str(base_node))
        if base_name_match:
            base_name = base_name_match.group(1)
        else:
            base_name = str(base_node)

        object_nodes = set()
        object_nodes.add(base_node)
        new_object_notes = set()
        to_search = [base_node]
        max_iterations = 10000  # Safety limit to prevent infinite loops
        iterations = 0

        while to_search and iterations < max_iterations:
            iterations += 1
            node = to_search.pop()
            children = self.scene.graph.transforms.children.get(node, [])
            for child in children:
                if child in object_nodes:
                    continue
                # Check if child has the same base name
                child_str = str(child)
                child_match = re.match(r"^(.+?)(?:_\d+|_[A-Za-z]\d*)?$", child_str)
                child_base = child_match.group(1) if child_match else child_str

                if child_base == base_name:
                    object_nodes.add(child)
                    to_search.append(child)
                else:
                    new_object_notes.add(child)

        if iterations >= max_iterations:
            print(f"Warning: Hit max iterations in _grouping_similar_meshes for {base_node}")

        return object_nodes, new_object_notes

    def _fusion_meshes(self, object_nodes: Set[str]) -> trimesh.Trimesh:
        """Fuse multiple mesh nodes into a single mesh.

        Applies the world transform to each mesh before concatenating them.

        Args:
            object_nodes: Set of node names to fuse

        Returns:
            A single concatenated mesh, or empty Trimesh if no geometry found
        """
        meshes: List[trimesh.Trimesh] = []
        for node in object_nodes:
            transform, geometry_name = self.scene.graph.get(node)
            if geometry_name is None:
                continue
            geometry = self.scene.geometry.get(geometry_name)
            if geometry is None:
                continue
            mesh = geometry.copy()
            mesh.apply_transform(transform)
            meshes.append(mesh)
        if meshes:
            return trimesh.util.concatenate(meshes)
        return trimesh.Trimesh()  # Empty mesh if no geometry found

    def _build_world_from_elements(self, world_elements: Dict[str, Body], connection: Dict[str, List[str]],
                                   world: World) -> World:
        """Build the world from parsed elements and their connections.

        Args:
            world_elements: Dictionary mapping node names to Body objects
            connection: Dictionary mapping parent node names to list of child node names
            world: The world to add entities to

        Returns:
            The modified world
        """
        object_root = self._get_root_node()
        if object_root not in world_elements:
            raise ValueError(f"Root node '{object_root}' not found in world_elements")
        object_root_body = world_elements[object_root]
        world.add_kinematic_structure_entity(object_root_body)
        if world.root is not None and world.root != object_root_body:
            root_transform, _ = self.scene.graph.get(object_root)
            conn = FixedConnection(
                parent=world.root,
                child=object_root_body,
                parent_T_connection_expression=HomogeneousTransformationMatrix(root_transform),
                name=PrefixedName(f"object_root_{object_root}")
            )
            world.add_connection(conn)
        to_add_nodes = [object_root]
        while to_add_nodes:
            node = to_add_nodes.pop()
            children = connection.get(node, [])
            for child in children:
                to_add_nodes.append(child)
                if child not in world_elements or node not in world_elements:
                    continue
                parent_body = world_elements[node]
                child_body = world_elements[child]
                world.add_kinematic_structure_entity(child_body)
                relative_transform = self._get_relative_transform(node, child)
                conn = FixedConnection(
                    parent=parent_body,
                    child=child_body,
                    parent_T_connection_expression=relative_transform,
                    name=PrefixedName(f"{node}_{child}")
                )
                world.add_connection(conn)
        return world

    def _create_world_objects(self, world: World) -> World:
        """Parse the scene graph and create world objects with connections.

        This method traverses the scene graph, groups similar meshes (e.g., Bolt_001, Bolt_002),
        fuses them, and creates Body objects with parent-child connections.

        Non-geometry nodes (like transforms/sketches) are skipped but their children
        are still processed with the correct parent body.
        """
        root = self._get_root_node()
        world_elements = {}
        connection = {}
        visited_nodes = set()
        to_visit_new_node = set()

        # Root is special case, no geometry can't be ignored.
        trans, root_geometry = self.scene.graph.get(root)
        if root_geometry is None:
            root_body = Body(
                name=PrefixedName(root),
                collision=ShapeCollection([]),
                visual=ShapeCollection([])
            )
            world_elements[root] = root_body
            root_children = self.scene.graph.transforms.children.get(root, [])
            connection[root] = []
            for child in root_children:
                to_visit_new_node.add((child, root))
            visited_nodes.add(root)
        else:
            object_nodes, new_object_notes = self._grouping_similar_meshes(root)
            node_fusion_mesh = self._fusion_meshes(object_nodes)
            visited_nodes = visited_nodes.union(object_nodes)
            for child in new_object_notes:
                to_visit_new_node.add((child, root))
            root_body = self._trimesh_to_body(node_fusion_mesh, root)
            world_elements[root] = root_body
            connection[root] = []


        while to_visit_new_node:
            # Check if we've hit the limit

            node, body_parent = to_visit_new_node.pop()
            if node in visited_nodes:
                continue

            _, geometry_name = self.scene.graph.get(node)

            if geometry_name is None:
                # Non-geometry node, just track connections and pass through
                new_nodes = self.scene.graph.transforms.children.get(node, [])
                for child in new_nodes:
                    if child not in visited_nodes:
                        to_visit_new_node.add((child, body_parent))
                visited_nodes.add(node)
                continue


            object_nodes, new_object_notes = self._grouping_similar_meshes(node)
            node_fusion_mesh = self._fusion_meshes(object_nodes)
            visited_nodes = visited_nodes.union(object_nodes)


            if len(node_fusion_mesh.vertices) == 0:
                visited_nodes.add(node)
                for child in new_object_notes.difference(visited_nodes):
                    to_visit_new_node.add((child, body_parent))
                continue

            body = self._trimesh_to_body(node_fusion_mesh, node)
            world_elements[node] = body


            # Connect to body_parent
            if body_parent in connection:
                connection[body_parent].append(node)
            connection[node] = []

            # Children of this node get this node as body_parent
            for child in new_object_notes.difference(visited_nodes):
                to_visit_new_node.add((child, node))

            visited_nodes.add(node)

        return self._build_world_from_elements(world_elements, connection, world)

    def _apply(self, world: World) -> World:
        """Load GLTF/GLB file and create world objects."""
        try:
            self.scene = trimesh.load(self.file_path)
            if self.scene is None:
                raise ValueError("Failed to load scene from file")
        except Exception as e:
            raise ValueError(f"Failed to load file: {e}")

        # Handle case where trimesh loads a single mesh instead of a Scene
        if isinstance(self.scene, trimesh.Trimesh):
            mesh = self.scene
            self.scene = trimesh.Scene()
            self.scene.add_geometry(mesh, node_name="root", geom_name="root_geom")

        if len(self.scene.geometry) == 0:
            root = self._get_root_node()
            empty_body = Body(
                name=PrefixedName(root),
                collision=ShapeCollection([]),
                visual=ShapeCollection([])
            )
            world.add_kinematic_structure_entity(empty_body)
            return world

        return self._create_world_objects(world)


