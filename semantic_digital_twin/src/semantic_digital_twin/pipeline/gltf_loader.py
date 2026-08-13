import logging
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Set, Optional
import re

import numpy as np
import trimesh

from .pipeline import Step
from ..exceptions import RootNodeNotFoundError
from ..world import World
from ..world_description.world_entity import Body
from ..world_description.connections import FixedConnection
from ..world_description.shape_collection import ShapeCollection
from ..world_description.geometry import Mesh, Scale
from ..spatial_types import HomogeneousTransformationMatrix
from ..datastructures.prefixed_name import PrefixedName


@dataclass
class NodeProcessingResult:
    """Result of processing a single node in the scene graph."""

    body: Body
    """The Body object created from the node's geometry."""

    visited_nodes: Set[str]
    """Set of node names that were visited during processing."""

    children_to_visit: Set[str]
    """Set of child node names that need to be processed next."""


@dataclass
class GLTFLoader(Step):
    """
    Load GLTF/GLB files into a World.

    Parses GLTF/GLB files and creates Body objects with FixedConnection
    relationships matching the scene hierarchy. Supports FreeCAD exports
    by fusing similarly named meshes.

    :raises ValueError: If the file cannot be loaded or parsed.
    """

    file_path: str
    """Path to the GLTF/GLB file."""

    max_grouping_iterations: int = 10000
    """Maximum iterations for grouping similar meshes."""

    scene: Optional[trimesh.Scene] = field(default=None, init=False)
    """The loaded trimesh Scene."""

    def _get_root_node(self) -> str:
        """
        Identify the primary root node of the scene graph.

        :raises RootNodeNotFoundError: If the scene has no root or multiple candidates.
        """
        base_frame = self.scene.graph.base_frame
        root_children = self.scene.graph.transforms.children.get(base_frame, [])
        if not root_children:
            raise RootNodeNotFoundError(candidates=[])

        if len(root_children) > 1:
            raise RootNodeNotFoundError(candidates=list(root_children))
        return root_children[0]

    def _get_relative_transform(
        self, parent_node: str, child_node: str
    ) -> HomogeneousTransformationMatrix:
        """Determine the transformation from a parent node to its child."""
        world_T_parent, _ = self.scene.graph.get(parent_node)
        world_T_child, _ = self.scene.graph.get(child_node)

        # Compute relative transform: parent_T_child = parent_T_world @ world_T_child
        parent_T_world = np.linalg.inv(world_T_parent)
        parent_T_child = parent_T_world @ world_T_child

        return HomogeneousTransformationMatrix(parent_T_child)

    def _trimesh_to_body(self, mesh: trimesh.Trimesh, name: str) -> Body:
        """Create a Body representation from a trimesh object."""
        # Create TriangleMesh geometry from trimesh
        triangle_mesh = Mesh.from_trimesh(
            mesh=mesh,
            origin=HomogeneousTransformationMatrix.from_xyz_rpy(),  # Identity transform
            scale=Scale(1.0, 1.0, 1.0),  # No scaling
        )

        # Create ShapeCollection for collision and visual
        shape_collection = ShapeCollection([triangle_mesh])

        # Create Body
        body = Body(
            name=PrefixedName(name),
            collision=shape_collection,
            visual=shape_collection,  # Use same for both collision and visual
        )

        return body

    def _extract_base_name(self, node_name: str) -> str:
        """Determine the base name of a node by removing suffixes."""
        match = re.match(r"^([^_]+)", node_name)
        return match.group(1) if match else node_name

    def _collect_matching_children(
        self, node: str, base_name: str, object_nodes: Set[str]
    ) -> Tuple[Set[str], Set[str]]:
        """Find child nodes sharing the same base name."""
        matching = set()
        non_matching = set()
        for child in self.scene.graph.transforms.children.get(node, []):
            if child in object_nodes:
                continue
            if self._extract_base_name(child) == base_name:
                matching.add(child)
            else:
                non_matching.add(child)
        return matching, non_matching

    def _grouping_similar_meshes(self, base_node: str) -> Tuple[Set[str], Set[str]]:
        """Identify related mesh nodes based on their naming patterns."""
        base_name = self._extract_base_name(base_node)
        object_nodes = {base_node}
        new_object_nodes = set()
        to_search = [base_node]

        for _ in range(self.max_grouping_iterations):
            if not to_search:
                break
            node = to_search.pop()
            matching, non_matching = self._collect_matching_children(
                node, base_name, object_nodes
            )
            object_nodes.update(matching)
            to_search.extend(matching)
            new_object_nodes.update(non_matching)
        else:
            logging.warning(
                "Hit max iterations (%d) in _grouping_similar_meshes for node '%s'",
                self.max_grouping_iterations,
                base_node,
            )

        return object_nodes, new_object_nodes

    def _fusion_meshes(self, object_nodes: Set[str]) -> trimesh.Trimesh:
        """Combine several geometry nodes into one unified mesh."""
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
            return trimesh.util.concatenate(meshes)  # type: ignore[return-value]
        return trimesh.Trimesh()  # Empty mesh if no geometry found

    def _add_child_connection(
        self,
        world: World,
        parent_node: str,
        child_node: str,
        world_elements: Dict[str, Body],
    ) -> None:
        """Establish a fixed relationship between two bodies in the world."""
        if child_node not in world_elements or parent_node not in world_elements:
            return
        parent_body = world_elements[parent_node]
        child_body = world_elements[child_node]
        world.add_kinematic_structure_entity(child_body)
        relative_transform = self._get_relative_transform(parent_node, child_node)
        conn = FixedConnection(
            parent=parent_body,
            child=child_body,
            parent_T_connection_expression=relative_transform,
            name=PrefixedName(f"{parent_node}_{child_node}"),
        )
        world.add_connection(conn)

    def _build_world_from_elements(
        self,
        world_elements: Dict[str, Body],
        connection: Dict[str, List[str]],
        world: World,
    ) -> World:
        """
        Construct the world hierarchy from parsed bodies and connections.

        :raises RootNodeNotFoundError: If the scene root is missing from elements.
        """
        object_root = self._get_root_node()
        if object_root not in world_elements:
            raise RootNodeNotFoundError(candidates=list(world_elements.keys()))

        object_root_body = world_elements[object_root]
        world.add_kinematic_structure_entity(object_root_body)

        # Connect root to world root if exists
        if world.root is not None and world.root != object_root_body:
            root_transform, _ = self.scene.graph.get(object_root)
            conn = FixedConnection(
                parent=world.root,
                child=object_root_body,
                parent_T_connection_expression=HomogeneousTransformationMatrix(
                    root_transform
                ),
                name=PrefixedName(f"object_root_{object_root}"),
            )
            world.add_connection(conn)

        # Add all child connections via BFS
        to_add_nodes = [object_root]
        while to_add_nodes:
            node = to_add_nodes.pop()
            for child in connection.get(node, []):
                to_add_nodes.append(child)
                self._add_child_connection(world, node, child, world_elements)

        return world

    def _create_empty_body(self, name: str) -> Body:
        """Generate a body instance without any geometry."""
        return Body(name=PrefixedName(name))

    def _process_node(
        self, node: str, body_parent: str, visited_nodes: Set[str]
    ) -> Tuple[Optional[NodeProcessingResult], Set[Tuple[str, str]]]:
        """Analyze a scene node and convert it to a body if it contains geometry."""
        object_nodes, remaining_children = self._grouping_similar_meshes(node)
        mesh = self._fusion_meshes(object_nodes)

        children_raw = self.scene.graph.transforms.children.get(node, [])
        non_matching = {
            c for c in children_raw if c not in object_nodes and c not in visited_nodes
        }

        if len(mesh.vertices) > 0:
            base_name = self._extract_base_name(node)
            body = self._trimesh_to_body(mesh, base_name)
            result = NodeProcessingResult(
                body=body,
                visited_nodes=object_nodes,
                children_to_visit=remaining_children.difference(visited_nodes),
            )
            children_to_visit = {(c, node) for c in result.children_to_visit} | {
                (c, body_parent) for c in non_matching
            }
            return result, children_to_visit

        # No geometry - pass children through to current body_parent
        children_to_visit = {
            (c, body_parent) for c in children_raw if c not in visited_nodes
        }
        return None, children_to_visit

    def _process_root_node(
        self, root: str
    ) -> Tuple[Body, Set[str], Set[Tuple[str, str]]]:
        """Initial processing of the scene's root node."""
        result, children_to_visit = self._process_node(root, root, set())

        if result is not None and len(result.body.visual.shapes) > 0:
            return result.body, result.visited_nodes, children_to_visit

        # Root has no geometry - create empty body
        body = self._create_empty_body(root)
        children_raw = self.scene.graph.transforms.children.get(root, [])
        return body, {root}, {(child, root) for child in children_raw}

    def _add_body_to_world_elements(
        self,
        world_elements: Dict[str, Body],
        base_name_to_node: Dict[str, str],
        node: str,
        body: Body,
    ) -> None:
        """Integrate a body into the element collection, merging if necessary."""
        base_name = str(body.name)

        existing_node = base_name_to_node.get(base_name)

        if existing_node is not None:
            # Merge meshes from new body into existing body
            existing_body = world_elements[existing_node]
            merged_shapes = list(existing_body.visual.shapes) + list(body.visual.shapes)
            merged_collision = list(existing_body.collision.shapes) + list(
                body.collision.shapes
            )
            existing_body.visual = ShapeCollection(merged_shapes)
            existing_body.collision = ShapeCollection(merged_collision)
            # Map the new node to the existing body for connection tracking
            world_elements[node] = existing_body
        else:
            base_name_to_node[base_name] = node
            world_elements[node] = body

    def _create_world_objects(self, world: World) -> World:
        """Traverse the scene graph and populate the world with objects."""
        root = self._get_root_node()
        world_elements: Dict[str, Body] = {}
        base_name_to_node: Dict[str, str] = {}
        connection: Dict[str, List[str]] = {}
        visited_nodes: Set[str] = set()

        # Process root
        root_body, root_visited, to_visit_new_node = self._process_root_node(root)
        world_elements[root] = root_body
        base_name_to_node[str(root_body.name)] = root
        visited_nodes = visited_nodes.union(root_visited)

        while to_visit_new_node:
            node, body_parent = to_visit_new_node.pop()

            if node in visited_nodes:
                continue

            result, children_to_visit = self._process_node(
                node, body_parent, visited_nodes
            )

            if result is not None:
                # Node created a body
                self._add_body_to_world_elements(
                    world_elements, base_name_to_node, node, result.body
                )
                visited_nodes.update(result.visited_nodes | {node})
                connection.setdefault(body_parent, []).append(node)
                connection[node] = []
                to_visit_new_node.update(children_to_visit)
            else:
                # No geometry - just pass through to children
                visited_nodes.add(node)
                to_visit_new_node.update(children_to_visit)

        return self._build_world_from_elements(world_elements, connection, world)

    def _apply(self, world: World) -> World:
        """
        Execute the loading process from file to world.

        :raises ValueError: If the file is inaccessible.
        """
        try:
            self.scene = trimesh.load(self.file_path)  # type: ignore[assignment]
        except Exception as e:
            raise ValueError(
                f"Failed to load GLTF/GLB file '{self.file_path}': {type(e).__name__}: {e}"
            ) from e

        # Handle case where trimesh loads a single mesh instead of a Scene
        if isinstance(self.scene, trimesh.Trimesh):
            mesh = self.scene
            self.scene = trimesh.Scene()
            self.scene.add_geometry(mesh, node_name="root", geom_name="root_geom")

        if len(self.scene.geometry) == 0:
            root = self._get_root_node()
            world.add_kinematic_structure_entity(self._create_empty_body(root))
            return world

        return self._create_world_objects(world)
