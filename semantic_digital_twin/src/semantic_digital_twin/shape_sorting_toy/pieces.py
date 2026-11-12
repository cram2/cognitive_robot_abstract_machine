"""
The loose pieces of the lab's physical shape-sorting sets.

Each set is a world of its own holding one body per piece, built from the measurements
taken off the real pieces rather than derived from the board's holes: a piece is cut
smaller than the hole it drops through.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.semantic_annotations.semantic_annotations import (
    CubePiece,
    CylinderPiece,
    RectangularPrismPiece,
    ShapeSortingPiece,
    TriangularPrismPiece,
)
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import FixedConnection
from semantic_digital_twin.world_description.geometry import (
    Box,
    Color,
    Cylinder,
    Mesh,
    Polygon2D,
    Scale,
    Shape,
)
from semantic_digital_twin.world_description.shape_collection import ShapeCollection
from semantic_digital_twin.world_description.world_entity import Body

# %% the colours they were measured to be


class PieceColor(Enum):
    """
    The colours the pieces of the lab's sets wear, measured off the rectified camera
    image.
    """

    CYAN = Color(hue=172 / 360, saturation=1.0, value=1.0)
    """
    The full-size set's pale blue, 172 degrees round the hue wheel.
    """

    YELLOW = Color(hue=42 / 360, saturation=1.0, value=1.0)
    """
    The full-size set's yellow, 42 degrees round the hue wheel. The board's own lid
    shares it.
    """

    SMALLER_SET_BLUE = Color(hue=196 / 360, saturation=1.0, value=1.0)
    """
    The smaller set's sky blue, 196 degrees round the hue wheel.
    """

    SMALLER_SET_YELLOW = Color(hue=52 / 360, saturation=1.0, value=1.0)
    """
    The smaller set's yellow, 52 degrees round the hue wheel.
    """


# %% the sets they make up


@dataclass(frozen=True, kw_only=True)
class PieceSet:
    """
    The loose pieces of one physical set.
    """

    world: World
    """
    The world holding one body per piece, each standing on its own flat face at its own
    origin and annotated with the kind of piece it is.
    """

    @classmethod
    def full_size(cls, *, grasp_height: float) -> PieceSet:
        """
        The set the captures of August 2026 hold. It has no disk and no sphere.

        :param grasp_height: How far above its bottom face each piece is gripped.
        """
        return cls.printed(
            scale=1.0,
            blue=PieceColor.CYAN.value,
            yellow=PieceColor.YELLOW.value,
            grasp_height=grasp_height,
        )

    @classmethod
    def smaller(cls, *, grasp_height: float) -> PieceSet:
        """
        The set on the table since September 2026: the full-size set printed at four
        fifths of its size, in plastics of its own.

        :param grasp_height: How far above its bottom face each piece is gripped.
        """
        return cls.printed(
            scale=0.8,
            blue=PieceColor.SMALLER_SET_BLUE.value,
            yellow=PieceColor.SMALLER_SET_YELLOW.value,
            grasp_height=grasp_height,
        )

    @classmethod
    def printed(
        cls, *, scale: float, blue: Color, yellow: Color, grasp_height: float
    ) -> PieceSet:
        """
        The full-size set's kinds of piece printed at a size and in two colours, every
        dimension measured off the full-size pieces, in metres.

        :param scale: How large the pieces are against the full-size set's.
        :param blue: The colour the cube and the cylinder are printed in.
        :param yellow: The colour the two prisms are printed in.
        :param grasp_height: How far above its bottom face each piece is gripped.
        """
        triangular_prism = Mesh.from_trimesh(
            mesh=Polygon2D.equilateral_triangle(side=0.037 * scale).extrude(
                thickness=0.03 * scale
            )
        )
        triangular_prism.color = yellow
        return cls.from_shapes(
            shapes={
                CubePiece: Box(
                    scale=Scale(0.03 * scale, 0.03 * scale, 0.03 * scale), color=blue
                ),
                CylinderPiece: Cylinder(
                    width=0.028 * scale, height=0.03 * scale, color=blue
                ),
                RectangularPrismPiece: Box(
                    scale=Scale(0.02 * scale, 0.04 * scale, 0.03 * scale),
                    color=yellow,
                ),
                TriangularPrismPiece: triangular_prism,
            },
            grasp_height=grasp_height,
        )

    @classmethod
    def from_shapes(
        cls,
        *,
        shapes: dict[type[ShapeSortingPiece], Shape],
        grasp_height: float,
    ) -> PieceSet:
        """
        A set holding one piece of each given kind, shaped and coloured as given.

        :param shapes: The shape each kind of piece is, centred on its own origin with
            its flat face down.
        :param grasp_height: How far above its bottom face each piece is gripped.
        """
        world = World()
        root = Body(name=PrefixedName("piece_set"))
        with world.modify_world():
            world.add_body(root)
            for kind, shape in shapes.items():
                name = PrefixedName(kind.__name__)
                body = Body.from_shape_collection(name, ShapeCollection([shape]))
                world.add_connection(FixedConnection(parent=root, child=body))
                world.add_semantic_annotation(
                    kind(name=name, root=body, grasp_height=grasp_height)
                )
        return cls(world=world)

    @property
    def pieces(self) -> tuple[ShapeSortingPiece, ...]:
        """
        Every piece in the set.
        """
        return tuple(self.world.get_semantic_annotations_by_type(ShapeSortingPiece))

    def find_piece(self, *, kind: type[ShapeSortingPiece]) -> ShapeSortingPiece | None:
        """
        :param kind: The kind of piece to look up.
        :return: The piece of that kind, or ``None`` when this set has none.
        """
        return next((piece for piece in self.pieces if type(piece) is kind), None)

    @property
    def colors(self) -> tuple[Color, ...]:
        """
        Every colour a piece in this set wears, in order round the hue wheel.
        """
        return tuple(
            sorted({piece.color for piece in self.pieces}, key=lambda color: color.hue)
        )
