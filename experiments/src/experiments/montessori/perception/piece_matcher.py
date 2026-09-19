"""
Recognise a loose piece by laying the pieces this set is known to contain over the edges
the camera saw, and keeping the one that follows them best.

Rather than measuring proportions of an outline and deciding from thresholds what shape
they suggest (which is how the holes in the board's lid are still read, see
:class:`~experiments.montessori.perception.footprint.CrossSectionClassifier`), each known
piece is placed and turned until its own outline lies along the edges in the picture. The
set is small and every piece in it has been measured, so the question is not *what shape
is this* but *which of these four is it, where, and how is it turned* -- a question with
a far narrower answer, and one whose answer carries how well it fitted.

Fitting to edges rather than to a segmented colour is what lets a piece be recognised on
a mirror-finish table (see
:mod:`~experiments.montessori.perception.edges`): the colour of a piece runs on into its
own reflection, so a coloured region says roughly where a piece is but neither how large
it is nor which one it is. The fit answers all of that at once, and refuses an outline no
known piece follows instead of reporting whichever threshold it happened to fall between.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from typing_extensions import List, Optional, Sequence, Tuple

from experiments.montessori.perception.edges import EdgeDistances
from experiments.montessori.pieces import (
    HUE_TOLERANCE,
    KNOWN_PIECES,
    KnownPiece,
    hue_distance,
    points_along,
)

# %% what a fit came to


@dataclass(frozen=True, eq=False)
class MatchedPiece:
    """
    Which known piece the edges were recognised as, and where it stands.
    """

    piece: KnownPiece
    """
    The piece the edges were recognised as.
    """

    center: Tuple[float, float]
    """
    The world-frame ``(x, y)`` its own centre was fitted to, in metres.
    """

    yaw: float
    """
    How far it is turned about the world frame's z-axis, in radians.

    Always the smallest turn that leaves the piece looking as it does, so a square is
    reported within an eighth of a turn of straight and a circle at zero.
    """

    outline_agreement: float
    """
    How much of this piece's own outline lay along an edge the camera saw.

    One is a perfect fit. A low value says no placement of this piece follows what is
    actually in the picture.
    """


# %% fitting them


@dataclass(frozen=True)
class PieceMatcher:
    """
    Recognises the piece the edges around one spot on the table belong to.

    The search runs twice: once coarsely and forgivingly, to find where the piece lies
    even when the spot it was seeded from is centimetres off, and once finely around
    that placement, to settle its position and its turn.
    """

    candidates: Tuple[KnownPiece, ...] = KNOWN_PIECES
    """
    The pieces that may be found on the table.
    """

    hue_tolerance: int = HUE_TOLERANCE
    """
    How far a measured colour may sit from a piece's own before that piece is ruled out.
    """

    minimum_agreement: float = 0.62
    """
    How much of a piece's outline must lie along a seen edge before it is reported at
    all.

    On the real table a correctly recognised piece reaches between 0.63 and 0.86, while
    the best any *other* piece of the same colour reaches on the same spot is 0.62; this
    sits at that boundary, so a piece is refused rather than reported as its neighbour.
    """

    search_radius: float = 0.024
    """
    How far, in metres, a piece may be found from the spot it was seeded at.

    A piece seen together with its own reflection is seeded from the middle of the two,
    which on this table is up to fifteen millimetres off.
    """

    coarse_step: float = 0.003
    """
    How far apart, in metres, the placements of the first search stand.
    """

    step: float = 0.001
    """
    How far apart, in metres, the placements of the second search stand, matching the
    rectified image's own resolution.
    """

    coarse_angle_step: float = math.radians(6.0)
    """
    How finely a piece is turned in the first search, in radians.
    """

    angle_step: float = math.radians(2.0)
    """
    How finely a piece is turned in the second search, in radians.

    Two degrees moves the far corner of the largest piece here by under a millimetre, so
    a finer sweep would be answering below the resolution the edges were found at.
    """

    coarse_reach: float = 0.008
    """
    How far, in metres, an edge may lie from an outline and still count in the first
    search.

    Wide on purpose: the first search only has to find which placement to look around,
    and a reach narrower than the step it walks in would step over the answer.
    """

    reach: float = 0.003
    """
    How far, in metres, an edge may lie from an outline and still count in the second
    search.
    """

    outline_spacing: float = 0.002
    """
    How far apart, in metres, the points an outline is compared to the picture at stand.
    """

    def match(
        self,
        edges: EdgeDistances,
        seed: Tuple[float, float],
        hue: Optional[int],
    ) -> Optional[MatchedPiece]:
        """
        Recognise the piece standing around one spot.

        :param edges: The edges seen in the plane the piece's top face stands on.
        :param seed: The world-frame ``(x, y)`` to search around.
        :param hue: The colour that spot was measured to be, or None where it had no
            colour to read.
        :return: The best fit, or None if no known piece follows the edges well enough.
        """
        candidates = [piece for piece in self.candidates if self._could_be(piece, hue)]
        if not candidates:
            return None
        best = max(
            (self._fit(piece, edges, seed) for piece in candidates),
            key=lambda fit: fit.outline_agreement,
        )
        if best.outline_agreement < self.minimum_agreement:
            return None
        return best

    def _could_be(self, piece: KnownPiece, hue: Optional[int]) -> bool:
        """
        Whether a piece's own colour is close enough to a measured one to be it.

        :param piece: The piece to consider.
        :param hue: The colour measured, or None where there was none to read.
        """
        if hue is None:
            return True
        return hue_distance(hue, piece.hue) <= self.hue_tolerance

    def _fit(
        self, piece: KnownPiece, edges: EdgeDistances, seed: Tuple[float, float]
    ) -> MatchedPiece:
        """
        Place and turn one piece until its outline follows the edges as closely as it
        can.

        :param piece: The piece to lay over the edges.
        :param edges: The edges seen in the plane its top face stands on.
        :param seed: The world-frame ``(x, y)`` to search around.
        :return: The best placement this piece reaches.
        """
        coarse = self._sweep(
            piece,
            edges,
            seed,
            self._turns_of(piece, self.coarse_angle_step),
            self.search_radius,
            self.coarse_step,
            self.coarse_reach,
        )
        return self._sweep(
            piece,
            edges,
            coarse.center,
            self._turns_around(piece, coarse.yaw),
            self.coarse_step,
            self.step,
            self.reach,
        )

    def _sweep(
        self,
        piece: KnownPiece,
        edges: EdgeDistances,
        center: Tuple[float, float],
        angles: Sequence[float],
        radius: float,
        step: float,
        reach: float,
    ) -> MatchedPiece:
        """
        Try one piece at every placement of a grid of positions and turns.

        :param piece: The piece to place.
        :param edges: The edges seen in the plane its top face stands on.
        :param center: The world-frame ``(x, y)`` the grid is centred on.
        :param angles: The turns to try, in radians.
        :param radius: How far, in metres, the grid reaches from its centre.
        :param step: How far apart, in metres, the grid's positions stand.
        :param reach: How far an edge may lie from the outline and still count.
        :return: The best placement.
        """
        walk = np.arange(-radius, radius + step / 2, step)
        positions = np.stack(np.meshgrid(walk, walk, indexing="ij"), axis=-1).reshape(
            -1, 2
        ) + np.asarray(center, dtype=float)
        return max(
            (
                self._best_position(piece, edges, positions, angle, reach)
                for angle in angles
            ),
            key=lambda fit: fit.outline_agreement,
        )

    def _best_position(
        self,
        piece: KnownPiece,
        edges: EdgeDistances,
        positions: np.ndarray,
        angle: float,
        reach: float,
    ) -> MatchedPiece:
        """
        Where a piece turned to one angle follows the edges best.

        :param piece: The piece to place.
        :param edges: The edges seen in the plane its top face stands on.
        :param positions: The world-frame ``(n, 2)`` positions to try it at.
        :param angle: The turn to try it at, in radians.
        :param reach: How far an edge may lie from the outline and still count.
        :return: The best of those positions.
        """
        outline = points_along(piece.turned_outline(angle), self.outline_spacing)
        agreements = edges.agreement(outline[None, :, :] + positions[:, None, :], reach)
        best = int(np.argmax(agreements))
        return MatchedPiece(
            piece=piece,
            center=(float(positions[best][0]), float(positions[best][1])),
            yaw=piece.smallest_equivalent_turn(angle),
            outline_agreement=float(agreements[best]),
        )

    @staticmethod
    def _turns_of(piece: KnownPiece, step: float) -> List[float]:
        """
        The turns a piece is worth trying, which span one of its rotation periods about
        zero and so hold every orientation it can be told apart in.

        :param piece: The piece to turn.
        :param step: How finely to turn it, in radians.
        """
        if piece.rotation_period is None:
            return [0.0]
        steps = int(piece.rotation_period / 2 / step)
        return list(np.arange(-steps, steps + 1) * step)

    def _turns_around(self, piece: KnownPiece, angle: float) -> List[float]:
        """
        The turns worth trying either side of one the coarse search settled on, which
        reach a full coarse step in both directions.

        :param piece: The piece to turn.
        :param angle: The turn to search around, in radians.
        """
        if piece.rotation_period is None:
            return [0.0]
        steps = int(round(self.coarse_angle_step / self.angle_step))
        return list(angle + np.arange(-steps, steps + 1) * self.angle_step)
