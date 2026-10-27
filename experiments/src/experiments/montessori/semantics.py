"""
Semantic annotations for the objects that appear in the ``icub_montessori_no_hands``
episode: the shape-sorting board, its shape-holes, and the loose shapes that are dropped
through them.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from typing_extensions import Optional, Type

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


@dataclass(eq=False)
class MontessoriShape(HasRootBody):
    """
    A loose Montessori piece that a child (or robot) drops through the matching
    :class:`ShapeSortingHole` of a :class:`ShapeSortingBoard`.

    Each concrete subclass (e.g. :class:`CubeShape`) is one geometric shape, matched
    against :attr:`ShapeSortingHole.shape_category`, and knows the :class:`Pose` it
    needs relative to a hole to actually drop through it (see
    :meth:`insertion_pose_relative_to_hole`).
    """

    def insertion_pose_relative_to_hole(
        self, hole: ShapeSortingHole, horizontal_offset: Point3, hover_height: float
    ) -> Pose:
        """
        The :class:`Pose`, expressed in ``hole``'s own frame, at which this shape must
        be released to drop through it.

        The shape hovers above the hole, turned about the hole's vertical axis so that
        its own footprint follows the hole's; a subclass whose fit depends on some other
        relative orientation overrides this. A shape with no geometry of its own has no
        footprint to line up and is released unturned.

        :param hole: The hole this shape is being inserted through.
        :param horizontal_offset: Offset from the hole's own origin, in its local ``(x,
            y)`` plane, to release the shape at.
        :param hover_height: Height above the hole's own origin, along its local z-axis,
            to release the shape at.
        """
        return Pose.from_xyz_rpy(
            horizontal_offset.x,
            horizontal_offset.y,
            hover_height,
            yaw=self._turn_onto(hole),
            reference_frame=hole.root,
        )

    def _turn_onto(self, hole: ShapeSortingHole) -> float:
        """
        The turn about the vertical axis, in radians, at which this shape's footprint
        follows the hole's.

        :param hole: The hole to line this shape up with.
        """
        if not self.root.collision.shapes:
            return 0.0
        own = self.root.collision.convex_footprint_in_frame(self.root)
        return own.turn_onto(hole.root.area.convex_footprint_in_frame(hole.root))

    @property
    def cross_section_size(self) -> float:
        """
        The larger of this shape's own local ``(x, y)`` footprint extents: the practical
        diameter or width a hole must be at least as large as for this shape to actually
        pass through it (see :meth:`fits_through`).
        """
        bounds = self.root.collision.combined_mesh.bounds
        return float(max(bounds[1][0] - bounds[0][0], bounds[1][1] - bounds[0][1]))

    def fits_through(self, hole: ShapeSortingHole) -> bool:
        """
        Whether this shape can actually pass through ``hole``.

        Being of the hole's :attr:`~ShapeSortingHole.shape_category` is necessary but
        not sufficient once more than one hole is cut for the same class, sized
        differently, so this also requires this shape to be no larger than ``hole``.

        :param hole: The hole to check this shape against.
        """
        return (
            type(self) is hole.shape_category
            and self.cross_section_size <= hole.cross_section_size
        )


@dataclass(eq=False)
class CubeShape(MontessoriShape):
    """
    A loose cube-shaped Montessori piece.
    """


@dataclass(eq=False)
class CylinderShape(MontessoriShape):
    """
    A loose cylinder-shaped Montessori piece.
    """


@dataclass(eq=False)
class DiskShape(MontessoriShape):
    """
    A loose disk-shaped Montessori piece: a flat coin whose matching hole is a narrow
    slot rather than a coin-shaped opening (see
    :class:`~experiments.montessori.hole_geometry.HoleShapeClassifier`), so unlike every
    other shape it must be tipped onto its edge to fit through, not just hover above it
    flat.
    """

    def insertion_pose_relative_to_hole(
        self, hole: ShapeSortingHole, horizontal_offset: Point3, hover_height: float
    ) -> Pose:
        """
        Overrides :meth:`MontessoriShape.insertion_pose_relative_to_hole`: a disk lying
        flat presents its full diameter to the hole's narrow slot and cannot pass
        through it, so this rotates the disk a quarter turn about the hole's local
        y-axis, presenting its thin edge (matching the slot's narrow width) instead of
        its flat face.
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
    A loose sphere-shaped Montessori piece.
    """


@dataclass(eq=False)
class TriangularPrismShape(MontessoriShape):
    """
    A loose triangular-prism-shaped Montessori piece.
    """


@dataclass(eq=False)
class RectangularPrismShape(MontessoriShape):
    """
    A loose rectangular-prism-shaped Montessori piece.
    """


@dataclass(eq=False)
class ShapeSortingHole(Aperture):
    """
    A hole cut into the lid of a :class:`ShapeSortingBoard`, shaped after a single
    :class:`MontessoriShape` subclass; only a piece of that subclass passes through it.

    Unlike a generic :class:`Aperture`, mounting this onto its board does not cut an
    axis-aligned bounding box out of the board's geometry: the board's mesh already has
    this hole's true (possibly non-rectangular) shape cut into it when it is built, from
    the same mesh this hole's shape was detected from.
    """

    shape_category: Optional[Type[MontessoriShape]] = field(kw_only=True, default=None)
    """
    The kind of piece this hole is cut for; only pieces of exactly this class fit
    through it.
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
    :class:`ShapeSortingHole` per :class:`MontessoriShape` subclass and whose body
    houses drawers for storing the shapes when they are not on the board.
    """

    @classproperty
    def hole_direction(self) -> Vector3:
        return Vector3.Z()

    def hole_for(self, montessori_shape: MontessoriShape) -> ShapeSortingHole:
        """
        Find the smallest :class:`ShapeSortingHole` of this board that
        ``montessori_shape`` fits through (see :meth:`MontessoriShape.fits_through`).

        :raises NoMatchingHoleError: If this board has no hole ``montessori_shape`` fits
            through.
        """
        fitting_holes = [
            hole
            for hole in self.apertures
            if isinstance(hole, ShapeSortingHole)
            and montessori_shape.fits_through(hole)
        ]
        if not fitting_holes:
            raise NoMatchingHoleError(montessori_shape, self)
        return min(fitting_holes, key=lambda hole: hole.cross_section_size)
