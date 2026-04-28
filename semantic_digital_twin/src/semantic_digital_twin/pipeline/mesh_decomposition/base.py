from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, TYPE_CHECKING

from pygments.styles import vs

from semantic_digital_twin.pipeline.pipeline import Step
from semantic_digital_twin.world_description.geometry import Shape, Mesh
from semantic_digital_twin.world_description.shape_collection import ShapeCollection
from semantic_digital_twin.world_description.world_entity import Body

if TYPE_CHECKING:
    from semantic_digital_twin.world import World


@dataclass
class MeshDecomposer(Step, ABC):
    """
    MeshDecomposer is an abstract base class for decomposing complex 3D meshes into simpler convex components.
    It provides methods to apply the decomposition to meshes, shapes, bodies, and entire worlds.
    Subclasses should implement the `apply_to_mesh` method to define the specific decomposition algorithm.
    3D meshes are represented using the `trimesh` library, and the decomposed parts are returned as a list of `TriangleMesh` objects.
    """

    @abstractmethod
    def apply_to_mesh(self, mesh: Mesh) -> List[Shape]:
        """
        Apply the mesh decomposition to a given mesh.
        Returns a list of TriangleMesh objects representing the decomposed convex parts.
        """
        ...

    def apply_to_shape(self, shape: Shape) -> List[Mesh]:
        """
        Apply the mesh decomposition to a given shape.
        If the shape is a Mesh, it will be decomposed into multiple TriangleMesh objects.
        Otherwise, the shape will be returned as is in a list.
        """
        new_geometry = []
        if isinstance(shape, Mesh):
            new_geometry = self.apply_to_mesh(shape)
        else:
            new_geometry.append(shape)

        return new_geometry

    def apply_to_body(self, body: Body) -> Body:
        """
        Apply the mesh decomposition to all shapes in a given body.
        The body's collision shapes will be replaced with the decomposed shapes.
        Returns the modified body.
        """
        new_geometry = []
        for shape in body.visual:
            decomposed_shapes = self.apply_to_shape(shape)
            new_geometry.extend(decomposed_shapes)

        body.collision = ShapeCollection(new_geometry)
        return body

    def _apply(self, world: World) -> World:
        """
        Apply the mesh decomposition to all bodies in a given world.
        Each body's collision shapes will be replaced with the decomposed shapes.
        Returns the modified world.
        """
        for body in world.bodies:
            self.apply_to_body(body)

        return world
