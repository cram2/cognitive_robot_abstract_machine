"""
The semantics of a shape-sorting toy: its loose pieces, each a kind of piece, the holes
cut through its board's lid, and the board.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from krrood.ormatic.utils import classproperty
from krrood_spatial_math.spatial_types import Point3, Pose, RotationMatrix, Vector3
from semantic_digital_twin.exceptions import NoGraspGeometry
from semantic_digital_twin.grasping.grasp_candidates import GraspCandidate
from semantic_digital_twin.semantic_annotations.mixins import (
    HasApertures,
    HasCaseAsRootBody,
    HasDrawers,
    HasGraspCandidates,
)
from semantic_digital_twin.semantic_annotations.part_whole import (
    IsPartWholeRelationship,
)
from semantic_digital_twin.semantic_annotations.semantic_annotations import Aperture
from semantic_digital_twin.shape_sorting_toy.exceptions import NoMatchingHoleError
from semantic_digital_twin.world_description.geometry import Color, Polygon2D

# %% the pieces


@dataclass(eq=False, kw_only=True)
class ShapeSortingPiece(HasGraspCandidates):
    """
    A loose piece of a shape-sorting toy, which drops through the hole of its
    own outline.

    Each subclass is one kind of piece, matched against
    :attr:`ShapeSortingHole.piece_kind`.
    """

    grasp_height: float
    """
    How far above its bottom face the piece is gripped.

    Gripped low, the piece hangs as little as possible below the fingers when it is let
    go of over its hole; how low depends on how far the closing fingers reach past the
    point they are brought to.
    """

    def default_grasp_candidates(self) -> list[GraspCandidate]:
        """
        :return: The grasp from above, on the piece's vertical axis at
            :attr:`grasp_height`, closing along the piece's own x.
        :raises NoGraspGeometry: If the root body has no collision geometry to find the
            bottom face in.
        """
        if not self.root.has_collision():
            raise NoGraspGeometry(self)
        bottom_face = (
            self.root.collision.as_bounding_box_collection_in_frame(self.root)
            .bounding_box()
            .min_z
        )
        return [
            GraspCandidate(
                self,
                Pose(
                    position=Point3(
                        0.0,
                        0.0,
                        bottom_face + self.grasp_height,
                        reference_frame=self.root,
                    ),
                    orientation=RotationMatrix.from_vectors(
                        x=Vector3.NEGATIVE_Z(), y=Vector3.X()
                    ).quaternion,
                    reference_frame=self.root,
                ),
            )
        ]

    @property
    def footprint(self) -> Polygon2D:
        """
        The outline the piece presents resting on its own flat face, in its own frame.
        """
        return self.root.collision.convex_footprint_in_frame(reference_frame=self.root)

    @property
    def height(self) -> float:
        """
        How far the piece's top face stands above the surface it rests on.
        """
        return (
            self.root.collision.as_bounding_box_collection_in_frame(self.root)
            .bounding_box()
            .height
        )

    @property
    def color(self) -> Color:
        """
        The colour the piece wears: the colour of its one visual shape, since a piece
        is made of a single material.
        """
        [shape] = self.root.visual.shapes
        return shape.color

    @property
    def cross_section_size(self) -> float:
        """
        The larger of the piece's own x and y extents: how wide a hole must be at least
        for the piece to pass through it.
        """
        return max(self.footprint.bounding_box.dimensions)

    def fits_through(self, *, hole: ShapeSortingHole) -> bool:
        """
        Whether this piece passes through a hole: the hole is cut for this kind of piece
        and is no smaller than it.

        :param hole: The hole to check this piece against.
        """
        return (
            type(self) is hole.piece_kind
            and self.cross_section_size <= hole.cross_section_size
        )

    def insertion_pose_relative_to_hole(
        self, *, hole: ShapeSortingHole, horizontal_offset: Point3, hover_height: float
    ) -> Pose:
        """
        The pose, in the hole's own frame, at which this piece is let go of to drop
        through it: hovering above the hole, turned about the hole's vertical axis so
        that its footprint follows the hole's. A piece without geometry is not turned.

        :param hole: The hole this piece is inserted through.
        :param horizontal_offset: Offset from the hole's origin in its x-y plane.
        :param hover_height: Height above the hole's origin along its z-axis.
        """
        return Pose.from_xyz_rpy(
            horizontal_offset.x,
            horizontal_offset.y,
            hover_height,
            yaw=self.turn_onto(hole=hole),
            reference_frame=hole.root,
        )

    def turn_onto(self, *, hole: ShapeSortingHole) -> float:
        """
        The turn about the vertical axis, in radians, at which this piece's footprint
        follows the hole's; zero for a piece without geometry.

        :param hole: The hole to line this piece up with.
        """
        if not self.root.collision.shapes:
            return 0.0
        return self.footprint.turn_onto(other=hole.footprint)


@dataclass(eq=False, kw_only=True)
class CubePiece(ShapeSortingPiece):
    """
    A cube-shaped piece of a shape-sorting toy.
    """


@dataclass(eq=False, kw_only=True)
class CylinderPiece(ShapeSortingPiece):
    """
    A cylinder-shaped piece of a shape-sorting toy.
    """


@dataclass(eq=False, kw_only=True)
class DiskPiece(ShapeSortingPiece):
    """
    A flat, coin-shaped piece of a shape-sorting toy, whose hole is a narrow slot it
    passes through only standing on its edge.
    """

    def insertion_pose_relative_to_hole(
        self, *, hole: ShapeSortingHole, horizontal_offset: Point3, hover_height: float
    ) -> Pose:
        """
        Unlike :meth:`ShapeSortingPiece.insertion_pose_relative_to_hole`, the disk is
        tipped a quarter turn about the hole's y-axis, presenting its thin edge to the
        slot instead of its flat face.
        """
        return Pose.from_xyz_rpy(
            horizontal_offset.x,
            horizontal_offset.y,
            hover_height,
            pitch=math.pi / 2,
            reference_frame=hole.root,
        )


@dataclass(eq=False, kw_only=True)
class SpherePiece(ShapeSortingPiece):
    """
    A sphere-shaped piece of a shape-sorting toy.
    """


@dataclass(eq=False, kw_only=True)
class TriangularPrismPiece(ShapeSortingPiece):
    """
    A piece of a shape-sorting toy shaped as an upright prism of triangular
    cross-section.
    """


@dataclass(eq=False, kw_only=True)
class RectangularPrismPiece(ShapeSortingPiece):
    """
    A piece of a shape-sorting toy shaped as an upright prism of rectangular, not
    square, cross-section.
    """


# %% the holes and the board


@dataclass(eq=False, kw_only=True)
class ShapeSortingHole(Aperture):
    """
    A hole cut through the lid of a :class:`ShapeSortingBoard`, shaped after one kind of
    :class:`ShapeSortingPiece`.
    """

    piece_kind: type[ShapeSortingPiece] | None = None
    """
    The kind of piece the hole is cut for; only pieces of exactly this class pass
    through it.
    """

    @property
    def footprint(self) -> Polygon2D:
        """
        The outline of the hole, in its own frame.
        """
        return self.root.area.convex_footprint_in_frame(reference_frame=self.root)

    @property
    def cross_section_size(self) -> float:
        """
        The larger of the hole's own x and y extents; see
        :meth:`ShapeSortingPiece.fits_through`.
        """
        return max(self.footprint.bounding_box.dimensions)


@dataclass(eq=False, kw_only=True)
class ShapeSortingBoard(HasCaseAsRootBody, HasDrawers, HasApertures):
    """
    A shape-sorting board: a case whose lid has a :class:`ShapeSortingHole` for each
    kind of :class:`ShapeSortingPiece`, and whose drawers store the pieces.
    """

    apertures: list[Aperture] = field(
        default_factory=list,
        hash=False,
        metadata=IsPartWholeRelationship(
            removes_part_geometry_from_whole=False
        ).as_dict(),
    )
    """
    The holes through the lid. The board's geometry already has every hole cut through
    it, so mounting a hole leaves the board's material as it is.
    """

    @classproperty
    def _hole_direction_axis(cls) -> Vector3:
        return Vector3.Z()

    def hole_for(self, *, piece: ShapeSortingPiece) -> ShapeSortingHole:
        """
        The smallest hole of this board that a piece fits through.

        :param piece: The piece to find a hole for.
        :raises NoMatchingHoleError: If the piece fits through none of the holes.
        """
        fitting_holes = [
            hole
            for hole in self.apertures
            if isinstance(hole, ShapeSortingHole) and piece.fits_through(hole=hole)
        ]
        if not fitting_holes:
            raise NoMatchingHoleError(piece=piece, board=self)
        return min(fitting_holes, key=lambda hole: hole.cross_section_size)
