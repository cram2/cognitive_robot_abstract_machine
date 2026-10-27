"""
The loose Montessori pieces this lab's physical sets contain.

Each set is a world of its own holding one body per piece, built from the measurements
taken off the real pieces rather than derived from the board's own holes: a piece is cut
smaller than the hole it drops through, so the hole's footprint is the wrong size to
recognise a piece by or to build one from.

What a piece is recognised by -- the outline it presents resting on its own flat face,
how tall it then stands and the colour it wears -- is read off its body rather than
stored beside it.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np
from typing_extensions import Dict, Optional, Tuple, Type

from experiments.montessori.semantics import (
    CubeShape,
    CylinderShape,
    MontessoriShape,
    RectangularPrismShape,
    TriangularPrismShape,
)
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import FixedConnection
from semantic_digital_twin.world_description.geometry import (
    Box,
    Color,
    Cylinder,
    Hue,
    Mesh,
    Polygon2D,
    Scale,
    Shape,
)
from semantic_digital_twin.world_description.shape_collection import ShapeCollection
from semantic_digital_twin.world_description.world_entity import Body

# %% the colours they were measured to be


class PieceHue(Hue, Enum):
    """
    The hues the pieces of this lab's sets wear, measured off the rectified camera
    image.
    """

    CYAN = 172.0
    """
    Hue of the full-size set's pale blue pieces.
    """

    YELLOW = 42.0
    """
    Hue of the full-size set's yellow pieces.

    The bare table has no colour to speak of, so nothing on it competes for this; the
    board's own lid does share it, but the lid is searched on its own plane and excluded
    from the loose pieces by its outline.
    """

    SMALLER_SET_BLUE = 196.0
    """
    Hue of the smaller set's blue pieces: a sky blue where the full-size set's is a pale
    cyan.
    """

    SMALLER_SET_YELLOW = 52.0
    """
    Hue of the smaller set's yellow pieces.

    Within :attr:`PieceSet.hue_tolerance` of the board's own wooden lid, so a piece of
    this set on the lid is told from it by its outline alone.
    """


# %% the sets they make up


@dataclass(frozen=True)
class PieceSet:
    """
    The loose pieces of one physical set, and how closely a measured colour must match
    one of theirs to be taken for it.
    """

    world: World
    """
    The world holding one body per piece, each standing on its own flat face at its own
    origin and annotated with the kind of piece it is.
    """

    hue_tolerance: float = 8.0
    """
    How far a measured colour may sit from a piece's own and still be taken for it, in
    degrees round the colour circle.

    Measured on this table, every piece read within 4 degrees of its own recorded colour
    while the two things standing on the table that are not pieces read 12 and 14
    degrees away, so this sits midway between and turns them away. It is the one number
    here that a real change of lighting would have to be re-measured for.
    """

    @classmethod
    def full_size(cls) -> PieceSet:
        """
        The set the captures of August 2026 hold.

        The disk and the sphere are left out because the lab's sets have neither.
        """
        return cls.printed(scale=1.0, blue=PieceHue.CYAN, yellow=PieceHue.YELLOW)

    @classmethod
    def smaller(cls) -> PieceSet:
        """
        The set on the table since September 2026: the full-size set printed at four
        fifths of its size, its cube measuring 24 millimetres where the full-size one
        measures 30, in plastics of its own.
        """
        return cls.printed(
            scale=0.8,
            blue=PieceHue.SMALLER_SET_BLUE,
            yellow=PieceHue.SMALLER_SET_YELLOW,
        )

    @classmethod
    def printed(cls, scale: float, blue: Hue, yellow: Hue) -> PieceSet:
        """
        The full-size set's kinds of piece printed at a size and in two colours, with
        every dimension measured off the full-size pieces, in metres.

        :param scale: How large the pieces are against the full-size set's.
        :param blue: The colour the cube and the cylinder are printed in.
        :param yellow: The colour the two prisms are printed in.
        """
        triangular_prism = Mesh.from_trimesh(
            mesh=Polygon2D.equilateral_triangle(side=0.037 * scale).extrude(
                0.03 * scale
            )
        )
        triangular_prism.color = Color.from_hsv(yellow)
        return cls.of_shapes(
            {
                CubeShape: Box(
                    scale=Scale(0.03 * scale, 0.03 * scale, 0.03 * scale),
                    color=Color.from_hsv(blue),
                ),
                CylinderShape: Cylinder(
                    width=0.028 * scale,
                    height=0.03 * scale,
                    color=Color.from_hsv(blue),
                ),
                RectangularPrismShape: Box(
                    scale=Scale(0.02 * scale, 0.04 * scale, 0.03 * scale),
                    color=Color.from_hsv(yellow),
                ),
                TriangularPrismShape: triangular_prism,
            }
        )

    @classmethod
    def of_shapes(cls, shapes: Dict[Type[MontessoriShape], Shape]) -> PieceSet:
        """
        A set holding one piece of each given kind, shaped and coloured as given.

        :param shapes: The shape each kind of piece is, centred on its own origin with
            its flat face down.
        """
        world = World()
        root = Body(name=PrefixedName("piece_set"))
        with world.modify_world():
            world.add_body(root)
            for kind, shape in shapes.items():
                name = PrefixedName(kind.__name__)
                body = Body.from_shape_collection(name, ShapeCollection([shape]))
                world.add_connection(FixedConnection(parent=root, child=body))
                world.add_semantic_annotation(kind(name=name, root=body))
        return cls(world=world)

    @property
    def pieces(self) -> Tuple[MontessoriShape, ...]:
        """
        Every piece in the set.
        """
        return tuple(self.world.get_semantic_annotations_by_type(MontessoriShape))

    def piece_for(self, category: Type[MontessoriShape]) -> Optional[MontessoriShape]:
        """
        The piece of a given kind, or None when this set has none of that kind.

        :param category: The kind of piece to look up.
        """
        for piece in self.pieces:
            if type(piece) is category:
                return piece
        return None

    @staticmethod
    def outline_of(piece: MontessoriShape) -> Polygon2D:
        """
        The outline a piece presents resting on its own flat face, about its own centre.

        :param piece: A piece of a set.
        """
        return piece.root.collision.convex_footprint_in_frame(piece.root)

    @staticmethod
    def height_of(piece: MontessoriShape) -> float:
        """
        How far a piece's top face stands above the surface it rests on, in metres.

        :param piece: A piece of a set.
        """
        return (
            piece.root.collision.as_bounding_box_collection_in_frame(piece.root)
            .bounding_box()
            .height
        )

    @staticmethod
    def color_of(piece: MontessoriShape) -> Color:
        """
        The colour a piece wears.

        A piece is printed in a single plastic, so this is the colour of its one visual
        shape.

        :param piece: A piece of a set.
        """
        [shape] = piece.root.visual.shapes
        return shape.color

    @classmethod
    def hue_of(cls, piece: MontessoriShape) -> Hue:
        """
        Where the colour a piece wears lies on the colour circle.

        :param piece: A piece of a set.
        """
        return cls.color_of(piece).hue

    @property
    def hues(self) -> Tuple[Hue, ...]:
        """
        Every colour a piece in this set wears, in order round the colour circle.

        What a piece stands on is whatever the table happens to be covered with, so it
        is these that say a pixel belongs to a piece rather than anything about the
        surface under it.
        """
        return tuple(
            sorted(
                {self.hue_of(piece) for piece in self.pieces},
                key=lambda hue: hue.degrees,
            )
        )

    def could_be(self, piece: MontessoriShape, hue: Optional[Hue]) -> bool:
        """
        Whether a piece's own colour is close enough to a measured one to be it.

        :param piece: The piece to consider.
        :param hue: The colour measured, or None where there was none to read.
        """
        if hue is None:
            return True
        return bool(self.hue_of(piece).distance(hue.degrees) <= self.hue_tolerance)

    def wears_a_piece_colour(self, degrees: np.ndarray) -> np.ndarray:
        """
        Which of the given hues lie close enough to one a piece in this set wears.

        :param degrees: Hues in degrees round the colour circle, of any shape.
        :return: A boolean array of the same shape.
        """
        apart = np.stack([worn.distance(degrees) for worn in self.hues])
        return apart.min(axis=0) <= self.hue_tolerance
