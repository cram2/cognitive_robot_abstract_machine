"""
The loose Montessori pieces this lab's physical set actually contains.

Every measurement here was taken off the pieces themselves rather than derived from the
board's own holes: a piece is cut smaller than the hole it drops through, so the hole's
footprint is the wrong size to recognise a piece by or to build one from.

Each piece is described by the outline it presents while resting on its own flat face,
the colour it was measured to be, and how far it can be turned about its standing axis
before it looks as it did. That last one is what makes a piece's orientation a *minimal*
turn rather than an absolute one: a square is unchanged by a quarter turn, so there is
no sense in reporting that it was turned by more.
"""

from __future__ import annotations

import colorsys
import math
from dataclasses import dataclass

import numpy as np
from typing_extensions import Dict, Optional, Tuple

from experiments.montessori.semantics import MontessoriShapeCategory
from semantic_digital_twin.world_description.geometry import Color

# %% measured dimensions

CUBE_EDGE = 0.03
"""
Edge length, in metres, of this scene's physical cube piece.
"""

CYLINDER_DIAMETER = 0.028
"""
Diameter, in metres, of this scene's one physical cylindrical piece.
"""

CYLINDER_HEIGHT = 0.03
"""
Height, in metres, of this scene's physical cylindrical piece (see
:const:`CYLINDER_DIAMETER`).
"""

RECTANGULAR_PRISM_WIDTH = 0.02
"""
Width, in metres, of this scene's physical rectangular-prism piece.
"""

RECTANGULAR_PRISM_LENGTH = 0.04
"""
Length, in metres, of this scene's physical rectangular-prism piece (see
:const:`RECTANGULAR_PRISM_WIDTH`).
"""

RECTANGULAR_PRISM_HEIGHT = 0.03
"""
Height, in metres, of this scene's physical rectangular-prism piece (see
:const:`RECTANGULAR_PRISM_WIDTH`).
"""

TRIANGULAR_PRISM_SIDE = 0.037
"""
Side length, in metres, of this scene's physical triangular-prism piece's equilateral
cross-section.
"""

TRIANGULAR_PRISM_HEIGHT = 0.03
"""
Height, in metres, of this scene's physical triangular-prism piece (see
:const:`TRIANGULAR_PRISM_SIDE`).
"""

# %% the outlines they present

_CIRCLE_CORNERS = 64
"""
How many corners a circular outline is drawn with, which at this set's sizes puts every
corner within a tenth of a millimetre of the true circle.
"""


def equilateral_triangle_boundary(side: float) -> np.ndarray:
    """
    Vertices of an equilateral triangle centered on its own centroid, apex pointing
    along local +y.

    :param side: Length of each of the triangle's three sides.
    """
    circumradius = side / math.sqrt(3)
    inradius = side / (2 * math.sqrt(3))
    return np.array(
        [
            [0.0, circumradius],
            [-side / 2, -inradius],
            [side / 2, -inradius],
        ]
    )


def rectangle_boundary(width: float, length: float) -> np.ndarray:
    """
    Corners of a rectangle centered on its own middle, its length along local +y.

    :param width: The rectangle's shorter side.
    :param length: The rectangle's longer side.
    """
    return np.array(
        [
            [-width / 2, -length / 2],
            [width / 2, -length / 2],
            [width / 2, length / 2],
            [-width / 2, length / 2],
        ]
    )


def points_along(outline: np.ndarray, spacing: float) -> np.ndarray:
    """
    Spread points evenly along a closed outline, corners included.

    :param outline: The outline's corners, as ``(n, 2)`` ``(x, y)`` points in metres.
    :param spacing: How far apart, in metres, to place the points.
    :return: The points, as ``(m, 2)`` ``(x, y)`` points in metres.
    """
    corners = np.vstack([outline, outline[:1]])
    walked = []
    for start, end in zip(corners[:-1], corners[1:]):
        steps = max(1, int(round(float(np.linalg.norm(end - start)) / spacing)))
        walked.append(start + np.outer(np.arange(steps) / steps, end - start))
    return np.vstack(walked)


def circle_boundary(diameter: float) -> np.ndarray:
    """
    Corners of a many-sided polygon standing in for a circle centered on its own middle.

    :param diameter: The circle's diameter.
    """
    angles = np.linspace(0.0, 2 * math.pi, _CIRCLE_CORNERS, endpoint=False)
    return np.stack([np.cos(angles), np.sin(angles)], axis=1) * diameter / 2


# %% the colours they were measured to be

HUE_RANGE = 180
"""
Number of hues OpenCV fits into a byte, which its hue channel wraps around at.
"""


def hue_distance(one: int, other: int) -> int:
    """
    How far apart two hues lie on the colour circle.

    :param one: A hue as OpenCV reports it.
    :param other: The hue to compare it against.
    """
    apart = abs(int(one) - int(other))
    return min(apart, HUE_RANGE - apart)


HUE_TOLERANCE = 4
"""
How far a measured colour may sit from a piece's own and still be taken for it.

Measured on this table, every piece read within 2 of its own recorded colour while the
two things standing on the table that are not pieces read 6 and 7 away, so this sits
midway between and turns them away. It is the one number here that a real change of
lighting would have to be re-measured for.
"""

CYAN_HUE = 86
"""
Hue of the pale blue pieces in this set, measured off the rectified camera image.
"""

YELLOW_HUE = 21
"""
Hue of the yellow pieces in this set, measured off the rectified camera image.

The bare table has no colour to speak of, so nothing on it competes for this; the
board's own lid does share it, but the lid is searched on its own plane and excluded
from the loose pieces by its outline.
"""


# %% one kind of piece


@dataclass(frozen=True, eq=False)
class KnownPiece:
    """
    One kind of loose piece this set contains, as measured off the piece itself.
    """

    category: MontessoriShapeCategory
    """
    The geometric shape it is, and so the hole it belongs in.
    """

    outline: np.ndarray
    """
    The outline it presents while resting on its own flat face, as ``(n, 2)`` ``(x, y)``
    points in metres about its own centre, at zero turn.
    """

    height: float
    """
    How far its top face stands above the surface it rests on, in metres.
    """

    hue: int
    """
    The colour it was measured to be, as OpenCV reports hue.
    """

    rotation_period: Optional[float]
    """
    The smallest turn about its standing axis, in radians, that leaves it looking as it
    did, or None when every turn does.

    An orientation is only ever reported within half of this either way, since a larger
    turn is indistinguishable from a smaller one.
    """

    @property
    def color(self) -> Color:
        """
        The colour to draw this piece in.

        Only :attr:`hue` was measured off the piece, so this is that hue at full
        saturation and brightness -- the pure form of the colour it wears, rather than
        the shade any one photograph of it happened to catch.
        """
        red, green, blue = colorsys.hsv_to_rgb(self.hue / HUE_RANGE, 1.0, 1.0)
        return Color(red, green, blue)

    @property
    def radius(self) -> float:
        """
        How far its outline reaches from its own centre, in metres.
        """
        return float(np.abs(self.outline).max())

    def turned_outline(self, angle: float) -> np.ndarray:
        """
        Its outline turned about its own centre.

        :param angle: How far to turn it, in radians about the world frame's z-axis.
        :return: The turned outline, as ``(n, 2)`` ``(x, y)`` points in metres.
        """
        cosine, sine = math.cos(angle), math.sin(angle)
        return self.outline @ np.array([[cosine, sine], [-sine, cosine]])

    def smallest_equivalent_turn(self, angle: float) -> float:
        """
        The smallest turn that leaves this piece looking the way the given one does.

        :param angle: A turn about the world frame's z-axis, in radians.
        :return: The same turn brought within half a :attr:`rotation_period` of zero, or
            zero for a piece no turn changes.
        """
        if self.rotation_period is None:
            return 0.0
        half = self.rotation_period / 2
        return (angle + half) % self.rotation_period - half


KNOWN_PIECES: Tuple[KnownPiece, ...] = (
    KnownPiece(
        category=MontessoriShapeCategory.CUBE,
        outline=rectangle_boundary(CUBE_EDGE, CUBE_EDGE),
        height=CUBE_EDGE,
        hue=CYAN_HUE,
        rotation_period=math.pi / 2,
    ),
    KnownPiece(
        category=MontessoriShapeCategory.CYLINDER,
        outline=circle_boundary(CYLINDER_DIAMETER),
        height=CYLINDER_HEIGHT,
        hue=CYAN_HUE,
        rotation_period=None,
    ),
    KnownPiece(
        category=MontessoriShapeCategory.RECTANGULAR_PRISM,
        outline=rectangle_boundary(RECTANGULAR_PRISM_WIDTH, RECTANGULAR_PRISM_LENGTH),
        height=RECTANGULAR_PRISM_HEIGHT,
        hue=YELLOW_HUE,
        rotation_period=math.pi,
    ),
    KnownPiece(
        category=MontessoriShapeCategory.TRIANGULAR_PRISM,
        outline=equilateral_triangle_boundary(TRIANGULAR_PRISM_SIDE),
        height=TRIANGULAR_PRISM_HEIGHT,
        hue=YELLOW_HUE,
        rotation_period=2 * math.pi / 3,
    ),
)
"""
Every kind of loose piece this set contains.

The disk and the sphere are left out because this physical set has neither.
"""

KNOWN_PIECE_BY_CATEGORY: Dict[MontessoriShapeCategory, KnownPiece] = {
    piece.category: piece for piece in KNOWN_PIECES
}
"""
:data:`KNOWN_PIECES` keyed by the shape each one is.
"""

PIECE_HUES: Tuple[int, ...] = tuple(sorted({piece.hue for piece in KNOWN_PIECES}))
"""
Every colour a loose piece in this set wears.

What a piece stands on is whatever the table happens to be covered with, so it is these
that say a pixel belongs to a piece rather than anything about the surface under it.
"""
