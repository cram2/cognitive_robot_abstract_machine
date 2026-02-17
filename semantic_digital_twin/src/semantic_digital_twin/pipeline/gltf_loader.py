from typing import List, Tuple, Dict

import trimesh
from .pipeline import Step
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.world_entity import Body
from semantic_digital_twin.world_description.connections import FixedConnection
from semantic_digital_twin.world_description.shape_collection import ShapeCollection
from semantic_digital_twin.world_description.geometry import TriangleMesh, Scale
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
import re

class GLTFLoader(Step):

    file_path: str #where i get it??
    scene: trimesh.Scene = None

    def __init__(self, file_path: str):
        self.file_path = file_path

    def _get_root_node(self) -> str:
        base_frame = self.scene.graph.base_frame
        root_children = self.scene.graph.transforms.children.get(base_frame, [])
        if len(root_children) > 1 or len(root_children) == 0:
            raise ValueError("More than one root node found in the scene, or no root node found.")
        return root_children[0]

    def _get_relative_transform(self, parent_node: str, child_node: str) -> HomogeneousTransformationMatrix:
        """Get the relative transform from parent to child node."""
        parent_transform, _ = self.scene.graph.get(parent_node)
        child_transform, _ = self.scene.graph.get(child_node)

        # Compute relative transform: parent_inv @ child
        import numpy as np
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

    def _grouping_similar_meshes(self, base_node) -> Tuple[set, set]:
        base_name_pattern = re.compile(r"^(.*?)(_[A-Za-z0-9]+)?$")#could be diffrent systems but freeCAD export always this
        #biggest problem if same names connect it will fuse then like Bolt_Simple
        base_name, _ = base_name_pattern.match(base_node).groups()
        object_nodes = set()
        object_nodes.add(base_node)
        new_object_notes = set()
        to_search = [base_node]
        while to_search:
            node = to_search.pop()
            children = self.scene.graph.transforms.children.get(node, [])
            for child in children:
                if child in object_nodes:
                    continue
                elif base_name_pattern.match(str(child)):
                    object_nodes.add(child)
                    to_search.append(child)
                else:
                    new_object_notes.add(child)
        return object_nodes, new_object_notes

    def _fusion_meshes(self, object_nodes) -> trimesh.Trimesh:
        meshes: List[trimesh.Trimesh] = []
        for node in object_nodes:
            transform, geometry_name = self.scene.graph.get(node)
            mesh = self.scene.geometry.get(geometry_name).copy()
            if mesh is None:
                continue #should not happen but just in case
            mesh.apply_transform(transform)
            meshes.append(mesh)
        if meshes:
            return trimesh.util.concatenate(meshes)
        return trimesh.Trimesh() # should not happen but just in case

    def _build_world_from_elements(self, world_elements: Dict[str, Body], connection: Dict[str, List[str]],
                                   world: World) -> World:

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
        print(len(world_elements), len(connection))
        return world

    def _create_world_objects(self, world) -> World:
        root = self._get_root_node()
        world_elements = {}
        connection = {} # will be filled greedy first to note no more against cycle if directed
        visited_nodes = set()
        to_visit_new_node = set()
        # Root is special case, no geometry cant be ingnored.
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
            root_body = self._trimesh_to_body(node_fusion_mesh, root)
            world_elements[root] = root_body
            connection[root] = []

        while to_visit_new_node:
            node, body_parent = to_visit_new_node.pop()
            if node in visited_nodes:
                continue

            _, geometry_name = self.scene.graph.get(node)

            if geometry_name is None:
                # Non-geometry node, just track connections
                new_nodes = self.scene.graph.transforms.children.get(node, [])
                for child in new_nodes:
                    if child not in visited_nodes:
                        to_visit_new_node.add((child, body_parent))
                visited_nodes.add(node)
                continue

            # Geometry node: create body
            object_nodes, new_object_notes = self._grouping_similar_meshes(node)
            node_fusion_mesh = self._fusion_meshes(object_nodes)
            body = self._trimesh_to_body(node_fusion_mesh, node)
            world_elements[node] = body

            # Connect to body_parent
            if body_parent in connection:
                connection[body_parent].append(node)
            connection[node] = []

            # Children of this node get this node as body_parent
            for child in new_object_notes.difference(visited_nodes):
                to_visit_new_node.add((child, node))

            visited_nodes = visited_nodes.union(object_nodes)
            visited_nodes.add(node)

        return self._build_world_from_elements(world_elements, connection, world)

    #Wolrd is empty complet overwrite???
    def _apply(self, world: World) -> World:
        self.scene = trimesh.load(self.file_path)
        if self.scene is None:
            raise ValueError("Failed to load scene from file.")
        return self._create_world_objects(world)


