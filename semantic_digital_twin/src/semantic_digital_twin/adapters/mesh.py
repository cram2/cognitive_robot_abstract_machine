import os
from dataclasses import dataclass

from ..world_description.geometry import FileMesh
from ..datastructures.prefixed_name import PrefixedName
from ..spatial_types import HomogeneousTransformationMatrix
from ..world import World
from ..world_description.shape_collection import ShapeCollection
from ..world_description.world_entity import Body


@dataclass
class MeshParser:
    """
    Adapter for mesh files.
    """

    file_path: str
    """
    The path to the mesh file.
    """

    def parse(self) -> World:
        """
        Parse the mesh file to a body and return a world containing that body.

        :return: A World object containing the parsed body.
        """
        file_name = os.path.basename(self.file_path)

        mesh_shape = FileMesh(
            origin=HomogeneousTransformationMatrix(), filename=self.file_path
        )
        body = Body(
            name=PrefixedName(file_name),
            collision=ShapeCollection([mesh_shape]),
            visual=ShapeCollection([mesh_shape]),
        )

        world = World()
        with world.modify_world():
            world.add_kinematic_structure_entity(body)

        return world


@dataclass
class STLParser(MeshParser):
    pass


@dataclass
class OBJParser(MeshParser):
    pass


@dataclass
class DAEParser(MeshParser):
    pass


@dataclass
class PLYParser(MeshParser):
    pass


@dataclass
class OFFParser(MeshParser):
    pass


@dataclass
class GLBParser(MeshParser):
    pass


@dataclass
class XYZParser(MeshParser):
    pass
