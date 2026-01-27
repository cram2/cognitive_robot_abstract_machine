from typing import List, Tuple

import trimesh
from pipeline import Step
from semantic_digital_twin.world import World
import re

class GLTFLoader(Step):

    file_path: str #where i get it??
    scene: trimesh.Scene = None

    def _get_root_node(self) -> str:
        base_frame = self.scene.graph.base_frame
        root_children = self.scene.graph.transforms.children.get(base_frame, [])
        if len(root_children) > 1 or len(root_children) == 0:
            raise ValueError("More than one root node found in the scene, or no root node found.")
        return root_children[0]

    def _grouping_similar_meshes(self, base_node) -> Tuple[List, List]:
        base_name_pattern = re.compile(r"^(.*?)(_\d+)?$")#could be diffrent systems but freeCAD export always this
        #biggest problem if same names connect it will fuse then like Bolt_Simple
        base_name, _ = base_name_pattern.match(str(base_node)).groups()
        object_nodes = set(base_node)
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
        return list(object_nodes), list(new_object_notes)

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

    def _build_world_from_elements(self, world_elements, connection, world) -> World:
        object_root = self._get_root_node()
        if world.root is None:
            # Set object_root as the root
            world.add_kinematic_structure_entity(None, world_elements[object_root], [])
            world.root = object_root
        else:
            # Add object_root as a child of the existing root
            world.add_kinematic_structure_entity(world.root, world_elements[object_root], [])
            connection[world.root] = [object_root]
        to_add_nodes = [object_root]
        while to_add_nodes:
            node = to_add_nodes.pop()
            children = connection.get(node, [])
            for child in children:
                mesh = world_elements.get(child)
                if mesh is None:
                    continue
                world.add_kinematic_structure_entity(node, mesh, [])
                to_add_nodes.append(child)
        return world

    def _create_world_objects(self, world) -> World:
        root = self._get_root_node()
        world_elements = {}
        connection = {} # will be filled greedy first to note no more against cycle if directed
        visited_nodes = set()
        to_visit_new_object = set()
        to_visit_new_object.add(root)
        while to_visit_new_object:
            node = to_visit_new_object.pop()
            _, geometry_name = self.scene.graph.get(node)
            if geometry_name is None:
                new_nodes = self.scene.graph.transforms.children.get(node, [])
                # will add meshless elements that will be ignored later on
                connection[node] = [n for n in new_nodes if n not in visited_nodes]
                to_visit_new_object.update()
                visited_nodes.add(node)
                continue
            object_nodes, new_object_notes = self._grouping_similar_meshes(node)
            node_fusion_mesh = self._fusion_meshes(object_nodes)
            world_elements[node] = node_fusion_mesh
            truly_new_nodes = [n for n in new_object_notes if n not in visited_nodes]
            to_visit_new_object.update(truly_new_nodes)
            visited_nodes.add(node)
            connection[node] = truly_new_nodes
        return self._build_world_from_elements(world_elements, connection, world)

    #Wolrd is empty complet overwrite???
    def _apply(self, world: World) -> World:
        self.scene = trimesh.load(self.file_path)
        if self.scene is None:
            raise ValueError("Failed to load scene from file.")
        return self._create_world_objects(world)

