"""
Telling the pieces of a shape-sorting set apart by the colour a camera measures on them.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from semantic_digital_twin.semantic_annotations.semantic_annotations import (
    ShapeSortingPiece,
)
from semantic_digital_twin.shape_sorting_toy.pieces import PieceSet
from semantic_digital_twin.world_description.geometry import Color

# %% matching measured colours against a set's pieces


@dataclass(frozen=True, kw_only=True)
class PieceColorMatcher:
    """
    How closely a measured colour must match the colour a piece of a set wears to be
    taken for it.
    """

    piece_set: PieceSet
    """
    The set whose pieces measured colours are matched against.
    """

    hue_tolerance: float = 8 / 360
    """
    How far, in turns of the hue wheel, a measured hue may sit from a piece's own and
    still be taken for it: 8 degrees, midway between the 4 degrees every piece read
    within on this table and the 12 degrees the nearest non-piece read at.
    """

    def could_be(self, *, piece: ShapeSortingPiece, measured: Color | None) -> bool:
        """
        :param piece: The piece to consider.
        :param measured: The colour measured, or ``None`` where there was none to read.
        :return: Whether the piece's own colour is close enough to the measured one for
            it to be the piece.
        """
        if measured is None:
            return True
        return bool(piece.color.hue_distance(hues=measured.hue) <= self.hue_tolerance)

    def is_piece_color(self, *, hues: npt.ArrayLike) -> npt.NDArray[np.bool_]:
        """
        :param hues: Positions around the hue wheel, from zero to one, of any shape.
        :return: Whether each hue lies close enough to one a piece of the set wears.
        """
        apart = np.stack(
            [worn.hue_distance(hues=hues) for worn in self.piece_set.colors]
        )
        return apart.min(axis=0) <= self.hue_tolerance
