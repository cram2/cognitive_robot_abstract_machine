"""
Semantic annotations for the objects that appear in the ``icub_montessori_no_hands``
episode: the shape-sorting board, its shape-holes, and the loose shapes that are dropped
through them.
"""

from __future__ import annotations

import math
from abc import abstractmethod
from dataclasses import dataclass, field
from enum import StrEnum

from typing_extensions import Dict, Optional, Type

from experiments.montessori.exceptions import NoMatchingHoleError
from krrood.ormatic.utils import classproperty
from semantic_digital_twin.semantic_annotations.mixins import (
    HasApertures,
    HasCaseAsRootBody,
    HasDrawers,
    HasRootBody,
    HasRootKinematicStructureEntity,
)
from semantic_digital_twin.semantic_annotations.part_whole import (
    IsPartWholeRelationship,
)
from semantic_digital_twin.semantic_annotations.semantic_annotations import Aperture
from semantic_digital_twin.spatial_types import Vector3
from semantic_digital_twin.spatial_types.spatial_types import Point3, Pose


class MontessoriShapeCategory(StrEnum):
    """
    The geometric shape of a loose Montessori piece and of the single hole in the
    ShapeSortingBoard it is meant to be dropped through.
    """

    CUBE = "cube"
    CYLINDER = "cylinder"
    DISK = "disk"
    SPHERE = "sphere"
    TRIANGULAR_PRISM = "triangular_prism"
    RECTANGULAR_PRISM = "rectangular_prism"


@dataclass(eq=False)
class MontessoriShape(HasRootBody):
    """
    A loose Montessori piece that a child (or robot) drops through the matching
    :class:`ShapeSortingHole` of a :class:`ShapeSortingBoard`.

    Concrete per :class:`MontessoriShapeCategory` (e.g. :class:`CubeShape`), each fixing
    its own :attr:`shape_category` and knowing the :class:`Pose` it needs relative to a
    hole to actually drop through it (see :meth:`insertion_pose_relative_to_hole`).
    """

    @classproperty
    @abstractmethod
    def shape_category(self) -> MontessoriShapeCategory:
        """
        The geometric shape of this piece, matched against
        :attr:`ShapeSortingHole.shape_category` to decide which hole it fits through.
        """
        ...

    def insertion_pose_relative_to_hole(
        self, hole: ShapeSortingHole, horizontal_offset: Point3, hover_height: float
    ) -> Pose:
        """
        The :class:`Pose`, expressed in ``hole``'s own frame, at which this shape must
        be released to drop through it.

        Every shape's loose body and its matching hole are built from the same local
        cross-section (see
        :func:`~experiments.montessori.world._footprint_shape_mesh`) and spawned
        without any relative rotation, so hovering directly above the hole's own
        origin, unrotated, already lines any shape up with its hole regardless of
        category; a shape category whose fit depends on some other relative
        orientation would override this.

        :param hole: The hole this shape is being inserted through.
        :param horizontal_offset: Offset from the hole's own origin, in its local
            ``(x, y)`` plane, to release the shape at.
        :param hover_height: Height above the hole's own origin, along its local
            z-axis, to release the shape at.
        """
        return Pose.from_xyz_rpy(
            horizontal_offset.x,
            horizontal_offset.y,
            hover_height,
            reference_frame=hole.root,
        )

    @property
    def cross_section_size(self) -> float:
        """
        The larger of this shape's own local ``(x, y)`` footprint extents: the
        practical diameter or width a hole must be at least as large as for this shape
        to actually pass through it (see :meth:`fits_through`).
        """
        bounds = self.root.collision.combined_mesh.bounds
        return float(max(bounds[1][0] - bounds[0][0], bounds[1][1] - bounds[0][1]))

    def fits_through(self, hole: ShapeSortingHole) -> bool:
        """
        Whether this shape can actually pass through ``hole``.

        Matching :attr:`shape_category` is necessary but not sufficient once more
        than one hole shares a category (e.g. the board's two circular holes are both
        :attr:`MontessoriShapeCategory.CYLINDER`, but sized differently), so this also
        requires this shape to be no larger than ``hole``.

        :param hole: The hole to check this shape against.
        """
        return (
            hole.shape_category == self.shape_category
            and self.cross_section_size <= hole.cross_section_size
        )


@dataclass(eq=False)
class CubeShape(MontessoriShape):
    """
    A loose cube-shaped Montessori piece.
    """

    @classproperty
    def shape_category(self) -> MontessoriShapeCategory:
        return MontessoriShapeCategory.CUBE


@dataclass(eq=False)
class CylinderShape(MontessoriShape):
    """
    A loose cylinder-shaped Montessori piece.
    """

    @classproperty
    def shape_category(self) -> MontessoriShapeCategory:
        return MontessoriShapeCategory.CYLINDER


@dataclass(eq=False)
class DiskShape(MontessoriShape):
    """
    A loose disk-shaped Montessori piece: a flat coin whose matching hole is a narrow
    slot rather than a coin-shaped opening (see
    :func:`~experiments.montessori.hole_geometry._classify_hole_shape`), so unlike
    every other shape it must be tipped onto its edge to fit through, not just hover
    above it flat.
    """

    @classproperty
    def shape_category(self) -> MontessoriShapeCategory:
        return MontessoriShapeCategory.DISK

    def insertion_pose_relative_to_hole(
        self, hole: ShapeSortingHole, horizontal_offset: Point3, hover_height: float
    ) -> Pose:
        """
        Overrides :meth:`MontessoriShape.insertion_pose_relative_to_hole`: a disk
        lying flat presents its full diameter to the hole's narrow slot and cannot
        pass through it, so this rotates the disk a quarter turn about the hole's
        local y-axis, presenting its thin edge (matching the slot's narrow width)
        instead of its flat face.
        """
        return Pose.from_xyz_rpy(
            horizontal_offset.x,
            horizontal_offset.y,
            hover_height,
            pitch=math.pi / 2,
            reference_frame=hole.root,
        )


@dataclass(eq=False)
class SphereShape(MontessoriShape):
    """
    A loose sphere-shaped Montessori piece; the board has no hole shaped to accept it
    (see :meth:`ShapeSortingBoard.hole_for`).
    """

    @classproperty
    def shape_category(self) -> MontessoriShapeCategory:
        return MontessoriShapeCategory.SPHERE


@dataclass(eq=False)
class TriangularPrismShape(MontessoriShape):
    """
    A loose triangular-prism-shaped Montessori piece.
    """

    @classproperty
    def shape_category(self) -> MontessoriShapeCategory:
        return MontessoriShapeCategory.TRIANGULAR_PRISM


@dataclass(eq=False)
class RectangularPrismShape(MontessoriShape):
    """
    A loose rectangular-prism-shaped Montessori piece.
    """

    @classproperty
    def shape_category(self) -> MontessoriShapeCategory:
        return MontessoriShapeCategory.RECTANGULAR_PRISM


MONTESSORI_SHAPE_CLASSES: Dict[MontessoriShapeCategory, Type[MontessoriShape]] = {
    MontessoriShapeCategory.CUBE: CubeShape,
    MontessoriShapeCategory.CYLINDER: CylinderShape,
    MontessoriShapeCategory.DISK: DiskShape,
    MontessoriShapeCategory.SPHERE: SphereShape,
    MontessoriShapeCategory.TRIANGULAR_PRISM: TriangularPrismShape,
    MontessoriShapeCategory.RECTANGULAR_PRISM: RectangularPrismShape,
}
"""
The concrete :class:`MontessoriShape` subclass for a given
:class:`MontessoriShapeCategory`, for building a loose shape from its detected or
authored category (see
:meth:`~experiments.montessori.world.MontessoriWorld._build_shapes`).
"""


@dataclass(eq=False)
class ShapeSortingHole(Aperture):
    """
    A hole cut into the lid of a :class:`ShapeSortingBoard`, shaped after a single
    :class:`MontessoriShapeCategory`; only a :class:`MontessoriShape` of the matching
    category passes through it.

    Unlike a generic :class:`Aperture`, mounting this onto its board does not cut an
    axis-aligned bounding box out of the board's geometry: the board's mesh already has
    this hole's true (possibly non-rectangular) shape cut into it when it is built, from
    the same mesh this hole's shape was detected from.
    """

    shape_category: Optional[MontessoriShapeCategory] = field(
        kw_only=True, default=None
    )
    """
    The geometric shape of this hole, matched against
    :attr:`MontessoriShape.shape_category` to decide which pieces fit through it.
    """

    @property
    def cross_section_size(self) -> float:
        """
        The larger of this hole's own local ``(x, y)`` footprint extents; see
        :meth:`MontessoriShape.fits_through`.
        """
        bounds = self.root.area.combined_mesh.bounds
        return float(max(bounds[1][0] - bounds[0][0], bounds[1][1] - bounds[0][1]))

    def _mount_strategy(
        self,
        main_has_root_body_annotation: HasRootBody,
        relationship: IsPartWholeRelationship,
    ) -> None:
        """
        Mount as a kinematic child of the board without cutting its geometry, since the
        board's mesh already has this hole's true shape cut into it.

        :param main_has_root_body_annotation: The board this hole is being added to.
        :param relationship: The metadata of the part-whole relationship field being
            mounted into.
        """
        HasRootKinematicStructureEntity._mount_strategy(
            self, main_has_root_body_annotation, relationship
        )


@dataclass(eq=False)
class ShapeSortingBoard(HasCaseAsRootBody, HasDrawers, HasApertures):
    """
    The Montessori shape-sorting board: a wooden case whose lid has one
    :class:`ShapeSortingHole` per :class:`MontessoriShapeCategory` and whose body houses
    drawers for storing the shapes when they are not on the board.
    """

    @classproperty
    def hole_direction(self) -> Vector3:
        return Vector3.Z()

    def hole_for(self, montessori_shape: MontessoriShape) -> ShapeSortingHole:
        """
        Find this board's :class:`ShapeSortingHole` that ``montessori_shape`` actually
        fits through (see :meth:`MontessoriShape.fits_through`).

        More than one hole can share a category (e.g. the board's two circular holes
        are both :attr:`MontessoriShapeCategory.CYLINDER`, but sized differently). A
        hole whose name matches ``montessori_shape``'s own name (stripped of its
        ``"_shape"`` suffix, the naming convention :meth:`~experiments.montessori.world.MontessoriWorld._build_shapes`
        pairs a shape and the hole it was built for with) is preferred: relying on
        size alone breaks once a shape sized after one hole (see
        :func:`~experiments.montessori.world._footprint_shape_mesh`) is also small
        enough to fit through a different, same-category hole, which silently resolves
        it to the wrong one instead of the hole it was actually built for. Falls back
        to the smallest fitting hole (for shapes with no such name pairing, e.g. in
        tests) when no name matches.

        :raises NoMatchingHoleError: If this board has no hole ``montessori_shape``
            fits through.
        """
        fitting_holes = [
            hole
            for hole in self.apertures
            if isinstance(hole, ShapeSortingHole) and montessori_shape.fits_through(hole)
        ]
        if not fitting_holes:
            raise NoMatchingHoleError(montessori_shape, self)
        shape_key = montessori_shape.name.name.removesuffix("_shape")
        for hole in fitting_holes:
            if hole.name.name == shape_key:
                return hole
        return min(fitting_holes, key=lambda hole: hole.cross_section_size)
