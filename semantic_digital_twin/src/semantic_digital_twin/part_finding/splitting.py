"""
Splitting a part, such as a handle, off the shape of an object into a body of its own,
annotated as that part of the object's annotation.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import cached_property

import numpy as np
import trimesh
from numpy.typing import NDArray
from trimesh.visual.texture import TextureVisuals
from typing_extensions import TYPE_CHECKING, Callable, TypeVar

from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.exceptions import CollisionDiffersFromVisualError
from semantic_digital_twin.part_finding.relationships import PartRelationshipInspector
from semantic_digital_twin.pipeline.mesh_decomposition.base import MeshDecomposer
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.world_description.connections import FixedConnection
from semantic_digital_twin.world_description.geometry import Mesh, Shape
from semantic_digital_twin.world_description.inertial_properties import Inertial
from semantic_digital_twin.world_description.shape_collection import ShapeCollection
from semantic_digital_twin.world_description.world_entity import Body

if TYPE_CHECKING:
    from semantic_digital_twin.semantic_annotations.mixins import (
        HasRootBody,
        PartWholeRelationship,
    )

T = TypeVar("T")

# %% splitting a mesh


@dataclass
class MeshSplit:
    """
    A mesh cut into a part and the rest of it.
    """

    mesh: trimesh.Trimesh
    """
    The mesh that is cut.
    """

    belongs_to_part_mask: NDArray[np.bool_]
    """
    For each face of :attr:`mesh`, whether it belongs to the part.
    """

    @classmethod
    def from_mesh(
        cls, mesh: trimesh.Trimesh, belongs_to_part_mask: NDArray[np.bool_]
    ) -> MeshSplit | None:
        """
        :param mesh: The mesh to split.
        :param belongs_to_part_mask: For each face of ``mesh``, whether it belongs to
            the part.
        :return: The faces of the part and the remaining faces; ``None`` if either
            would be empty.
        """
        if not belongs_to_part_mask.any() or belongs_to_part_mask.all():
            return None
        return cls(mesh=mesh, belongs_to_part_mask=belongs_to_part_mask)

    @property
    def part_faces(self) -> NDArray[np.int64]:
        """
        :return: The indices of the faces of :attr:`mesh` that belong to the part.
        """
        return np.flatnonzero(self.belongs_to_part_mask)

    @property
    def rest_faces(self) -> NDArray[np.int64]:
        """
        :return: The indices of the remaining faces of :attr:`mesh`.
        """
        return np.flatnonzero(~self.belongs_to_part_mask)

    @cached_property
    def part(self) -> trimesh.Trimesh:
        """
        :return: The faces of the part, as a mesh.
        """
        return self.mesh.submesh([self.part_faces], append=True)

    @cached_property
    def rest(self) -> trimesh.Trimesh:
        """
        :return: The remaining faces, as a mesh.
        """
        return self.mesh.submesh([self.rest_faces], append=True)


class PartFinder(ABC):
    """
    Finds the piece of an object's mesh that is one of its parts.
    """

    @abstractmethod
    def split(self, mesh: trimesh.Trimesh) -> MeshSplit | None:
        """
        :param mesh: The object's mesh, in its root body's frame.
        :return: ``mesh`` split into the part and the rest; ``None`` if the mesh has
            no such part.
        """


# %% splitting a part off a body


@dataclass
class PartSplitter:
    """
    Splits a part off the shape of a body into a new body fixed to it, optionally
    annotating the new body as a part of the body's annotation.

    The body keeps its inertial properties, which describe the whole object; each split
    is one block of the world's modification history.

    The visual shapes are cut, and the pieces collide as they look. A body whose
    collision shapes differ from its visual shapes, such as a convex decomposition, is
    only split with a :attr:`decomposer`; otherwise split before decomposing.
    """

    part_type: type[HasRootBody]
    """
    The type of annotation the part is given, which also names the part's body.
    """

    finder: PartFinder
    """
    Finds the part in a body's shape.
    """

    decomposer: MeshDecomposer | None = None
    """
    Gives the part and the rest new collision shapes from their visual shapes, for
    bodies whose collision shapes differ from their visual shapes.
    """

    def part_of(self, whole: PartWholeRelationship) -> HasRootBody | None:
        """
        The part of one annotation, split off its shape the first time it is asked for.

        :param whole: The annotation whose part is wanted.
        :return: The first part the annotation already has in the relationship field for
            :attr:`part_type`, if any; otherwise the part split off its shape, ``None``
            if there is none to split off.
        """
        relationship = PartRelationshipInspector.for_part(type(whole), self.part_type)
        existing = relationship.parts_of(whole)
        if existing:
            return existing[0]
        return self.split(whole)

    def split(
        self, whole: PartWholeRelationship, field_name: str = ""
    ) -> HasRootBody | None:
        """
        Split the part off the shape of one annotation's root body and add it to the
        annotation, unless the relationship field holds a single part and is set
        already.

        :param whole: The annotation whose root body's shape is split.
        :param field_name: The part-whole relationship field of the annotation the part
            is added to; found by the part's type if not given.
        :return: The part, added to the annotation and its world; ``None`` if the field
            is set already, the finder finds none or the root body has no shape.
        """
        relationship = PartRelationshipInspector.for_part(
            type(whole), self.part_type, field_name
        )
        if relationship.is_filled_in(whole) and not relationship.holds_many:
            return None
        cut = self._cut(whole.root)
        if cut is None:
            return None
        world = whole.root._world
        with world.modify_world():
            part = self.part_type(root=cut.apply())
            world.add_semantic_annotation(part)
            whole.add(part, field_name=relationship.name)
        return part

    def split_body(self, body: Body) -> Body | None:
        """
        Split the part off the shape of a body, without annotating either of them.

        :param body: The body whose shape is split; it keeps the rest of its shape.
        :return: The new body of the part, fixed to ``body`` in its world; ``None`` if
            the finder finds none or the body has no shape.
        """
        cut = self._cut(body)
        if cut is None:
            return None
        with body._world.modify_world():
            return cut.apply()

    def _cut(self, body: Body) -> CutBody | None:
        """
        :param body: The body whose shape is split.
        :return: The part's body and the rest of the shape, not yet in the world;
            ``None`` if the finder finds no part or the body has no shape.
        :raises CollisionDiffersFromVisualError: If the body's collision shapes differ
            from its visual shapes and there is no :attr:`decomposer`.
        """
        shapes = body.visual or body.collision
        if not shapes:
            return None
        has_own_collision = self._has_own_collision(body)
        if has_own_collision and self.decomposer is None:
            raise CollisionDiffersFromVisualError(body.name)
        colored = ColoredShape(shapes=shapes)
        split = self.finder.split(colored.mesh)
        if split is None:
            return None
        part = Body(name=self._part_name(body), inertial=Inertial.negligible())
        part.visual = colored.piece(split.part_faces, part)
        rest = colored.piece(split.rest_faces, body)
        if has_own_collision:
            part.collision = self._decomposed(part.visual, part)
            rest_collision = self._decomposed(rest, body)
        elif body.collision:
            part.collision = part.visual
            rest_collision = rest
        else:
            rest_collision = ShapeCollection()
        return CutBody(body=body, part=part, visual=rest, collision=rest_collision)

    @staticmethod
    def _has_own_collision(body: Body) -> bool:
        """
        :param body: A body.
        :return: Whether it has both visual and collision shapes, and they differ.
        """
        if not body.visual or not body.collision:
            return False
        if [id(shape) for shape in body.visual] == [
            id(shape) for shape in body.collision
        ]:
            return False
        visual, collision = body.visual.combined_mesh, body.collision.combined_mesh
        return not (
            np.array_equal(visual.vertices, collision.vertices)
            and np.array_equal(visual.faces, collision.faces)
        )

    def _decomposed(self, shapes: ShapeCollection, body: Body) -> ShapeCollection:
        """
        :param shapes: The visual shapes of a piece.
        :param body: The body the piece belongs to.
        :return: The collision shapes :attr:`decomposer` makes of them.
        """
        return ShapeCollection(
            [
                convex
                for shape in shapes
                for convex in self.decomposer.apply_to_shape(shape)
            ],
            reference_frame=body,
        )

    def _part_name(self, body: Body) -> PrefixedName:
        """
        :param body: The body the part is split off.
        :return: The part's type, under the body's name.
        """
        return PrefixedName(self.part_type.__name__.lower(), str(body.name))


@dataclass
class CutBody:
    """
    A body's shape cut into the part's body and the rest, to be applied to the world.
    """

    body: Body
    """
    The body whose shape is cut.
    """

    part: Body
    """
    The part's body, not yet in the world.
    """

    visual: ShapeCollection
    """
    What the body looks like without the part.
    """

    collision: ShapeCollection
    """
    How the body collides without the part.
    """

    def apply(self) -> Body:
        """
        Give the body the rest of its shape and fix the part's body to it.

        Must be called while the world is being modified.

        :return: The part's body.
        """
        self.body.update_shapes(visual=self.visual, collision=self.collision)
        self.body._world.add_connection(
            FixedConnection(parent=self.body, child=self.part)
        )
        return self.part


@dataclass
class ColoredShape:
    """
    The shapes of a body as one mesh whose faces keep the colors of the shapes, so that
    the pieces cut from it can be colored like it.
    """

    shapes: ShapeCollection
    """
    The shapes of the body.
    """

    @cached_property
    def mesh(self) -> trimesh.Trimesh:
        """
        :return: The shapes as one mesh in the body's frame, each face colored as its
            shape looks there.
        """
        meshes = []
        for shape in self.shapes:
            mesh = shape.mesh.copy()
            if isinstance(mesh.visual, TextureVisuals):
                mesh.visual = mesh.visual.to_color()
            if mesh.visual.kind is None:
                mesh.visual.face_colors = trimesh.visual.color.to_rgba(
                    shape.color.to_rgba()
                )
            mesh.apply_transform(shape.origin.to_np())
            meshes.append(mesh)
        return trimesh.util.concatenate(meshes)

    def piece(self, faces: NDArray[np.int64], body: Body) -> ShapeCollection:
        """
        :param faces: The indices of the faces of :attr:`mesh` that make up the piece.
        :param body: The body the piece belongs to, whose frame is the frame of
            :attr:`mesh`.
        :return: The piece as the shapes of ``body``, colored like its faces, and with
            the surface finish and simulator properties that all of :attr:`shapes`
            share.
        """
        mesh = self.mesh.submesh([faces], append=True)
        mesh.visual = trimesh.visual.ColorVisuals(
            mesh, face_colors=self.mesh.visual.face_colors[faces]
        )
        shape = Mesh.from_trimesh(
            mesh=mesh,
            origin=HomogeneousTransformationMatrix(reference_frame=body),
            finish=self.shared(lambda shape: shape.finish),
        )
        shape.simulator_additional_properties = list(
            self.shared(lambda shape: shape.simulator_additional_properties) or []
        )
        return ShapeCollection([shape], reference_frame=body)

    def shared(self, attribute: Callable[[Shape], T]) -> T | None:
        """
        :param attribute: Reads an attribute of a shape.
        :return: The attribute's value, if all :attr:`shapes` have the same; otherwise
            ``None``.
        """
        values = [attribute(shape) for shape in self.shapes]
        return values[0] if all(value == values[0] for value in values) else None
