from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import fbxloader
import numpy as np
import trimesh
from fbxloader import Object3D, Mesh as FBXMesh, Scene

from semantic_digital_twin.adapters.exceptions import (
    UnexpectedArrayShape,
    UnsupportedFBXAxis,
)
from semantic_digital_twin.adapters.mesh import MeshParser
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from krrood_spatial_math.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import FixedConnection
from semantic_digital_twin.world_description.geometry import Mesh
from semantic_digital_twin.world_description.shape_collection import ShapeCollection
from semantic_digital_twin.world_description.world_entity import Body


@dataclass
class CoordinateAxis(Enum):
    """
    Enum for coordinate axes with direction.

    The value is a tuple of (axis_index, sign), where axis_index is 0 for X, 1 for Y,
    and 2 for Z, and sign is 1 for positive direction and -1 for negative direction.
    """

    POSITIVE_X = (0, 1)
    NEGATIVE_X = (0, -1)
    POSITIVE_Y = (1, 1)
    NEGATIVE_Y = (1, -1)
    POSITIVE_Z = (2, 1)
    NEGATIVE_Z = (2, -1)

    @classmethod
    def from_fbx(cls, axis_value: int, sign: int) -> CoordinateAxis:
        """
        Map FBX axis index + sign into a CoordinateAxis enum.

        :raises UnsupportedFBXAxis: If the index and sign name no axis.
        """
        for member in cls:
            if member.value[0] == axis_value and member.value[1] == sign:
                return member
        raise UnsupportedFBXAxis(axis_index=axis_value, sign=sign)

    def to_vector(self) -> np.ndarray:
        """
        Convert the CoordinateAxis to a 3D unit vector.
        """
        idx, sgn = self.value
        v = np.zeros(3)
        v[idx] = float(sgn)
        return v


@dataclass
class FBXGlobalSettings:
    """
    Class to handle FBX global settings, particularly the coordinate system.

    This class extracts the up, front, and coordinate axes from the FBX file and
    provides a method to get the transformation matrix from FBX to Semantic Digital Twin
    coordinate system.
    """

    fbx_loader: fbxloader.FBXLoader
    """
    The FBXLoader instance containing the loaded FBX file.
    """

    up_axis: CoordinateAxis = field(init=False)
    """
    The up axis of the FBX file.
    """

    front_axis: CoordinateAxis = field(init=False)
    """
    The front axis of the FBX file.
    """

    coord_axis: CoordinateAxis = field(init=False)
    """
    The final, third axis of the FBX file, called the coordinate axis.
    """

    def __post_init__(self):
        fbx = self.fbx_loader

        self.up_axis = CoordinateAxis.from_fbx(
            fbx.fbxtree["GlobalSettings"]["UpAxis"]["value"],
            fbx.fbxtree["GlobalSettings"]["UpAxisSign"]["value"],
        )

        self.front_axis = CoordinateAxis.from_fbx(
            fbx.fbxtree["GlobalSettings"]["FrontAxis"]["value"],
            fbx.fbxtree["GlobalSettings"]["FrontAxisSign"]["value"],
        )

        self.coord_axis = CoordinateAxis.from_fbx(
            fbx.fbxtree["GlobalSettings"]["CoordAxis"]["value"],
            fbx.fbxtree["GlobalSettings"]["CoordAxisSign"]["value"],
        )

    def get_semantic_digital_twin_T_fbx(self) -> np.ndarray:
        """
        Get the transformation matrix from FBX to Semantic Digital Twin coordinate
        system.
        """
        sX = self.front_axis.to_vector()
        sY = self.coord_axis.to_vector()
        sZ = self.up_axis.to_vector()
        S = np.column_stack((sX, sY, sZ))

        R = S.T

        M = np.eye(4)
        M[:3, :3] = R
        return M


@dataclass
class FBXParser(MeshParser):
    """
    Adapter for FBX files.
    """

    @staticmethod
    def transform_vertices(
        fbx_vertices: np.ndarray, semantic_digital_twin_T_fbx: np.ndarray
    ) -> np.ndarray:
        """
        Transform vertices from FBX coordinate system to Semantic Digital Twin
        coordinate system.

        :raises UnexpectedArrayShape: If the vertices are not an (N, 3) array or the
            transformation is not a 4x4 matrix.
        """
        if fbx_vertices.ndim != 2 or fbx_vertices.shape[1] != 3:
            raise UnexpectedArrayShape(
                array_name="fbx_vertices",
                expected_shape="(N, 3)",
                actual_shape=list(fbx_vertices.shape),
            )
        if semantic_digital_twin_T_fbx.shape != (4, 4):
            raise UnexpectedArrayShape(
                array_name="semantic_digital_twin_T_fbx",
                expected_shape="(4, 4)",
                actual_shape=list(semantic_digital_twin_T_fbx.shape),
            )

        fbx_vertices_h = np.hstack(
            [
                fbx_vertices,
                np.ones((fbx_vertices.shape[0], 1), dtype=fbx_vertices.dtype),
            ]
        )

        semantic_digital_twin_vertices = fbx_vertices_h @ np.linalg.inv(
            semantic_digital_twin_T_fbx
        )
        return semantic_digital_twin_vertices[:, :3]

    def create_world(self) -> World:
        """
        :return: A world holding one body for every object in the FBX file.
        """
        fbx = fbxloader.FBXLoader(self.file_path)

        global_settings = FBXGlobalSettings(fbx)
        semantic_digital_twin_T_fbx = global_settings.get_semantic_digital_twin_T_fbx()

        world = World()

        with world.modify_world():
            for obj_id, obj in fbx.objects.items():
                # Create a body for each object in the FBX file
                if type(obj) is Object3D:
                    name = fbx.fbxtree["Objects"]["Model"][obj_id]["attrName"].split(
                        "\x00"
                    )[0]
                    meshes = []
                    for o in obj.children:
                        if isinstance(o, FBXMesh):
                            transformed_vertices = (
                                self.transform_vertices(
                                    o.vertices, semantic_digital_twin_T_fbx
                                )
                                / 100
                            )

                            t_mesh = Mesh.from_trimesh(
                                origin=HomogeneousTransformationMatrix(),
                                mesh=trimesh.Trimesh(
                                    vertices=transformed_vertices, faces=o.faces
                                ),
                            )

                            meshes.append(t_mesh)
                    body = Body(
                        name=PrefixedName(name),
                        collision=ShapeCollection(meshes),
                        visual=ShapeCollection(meshes),
                    )
                    world.add_body(body)

            for obj in fbx.objects.values():
                if type(obj) is Object3D:
                    name = fbx.fbxtree["Objects"]["Model"][obj.id]["attrName"].split(
                        "\x00"
                    )[0]
                    parent_name = (
                        fbx.fbxtree["Objects"]["Model"][obj.parent.id][
                            "attrName"
                        ].split("\x00")[0]
                        if type(obj.parent) is not Scene
                        else None
                    )
                    if not parent_name:
                        continue

                    obj_body = world.get_body_by_name(name)
                    parent_body = world.get_body_by_name(parent_name)

                    parent_T_child = HomogeneousTransformationMatrix(obj.matrix)

                    connection = FixedConnection(
                        parent=parent_body,
                        child=obj_body,
                        parent_T_connection_expression=parent_T_child,
                    )
                    world.add_connection(connection)

        return world
