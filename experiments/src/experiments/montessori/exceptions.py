"""
The ways the Montessori scene's semantics can fail.
"""

from __future__ import annotations

from dataclasses import dataclass

from typing_extensions import TYPE_CHECKING

from krrood.exceptions import DataclassException

if TYPE_CHECKING:
    from experiments.montessori.semantics import MontessoriShape, ShapeSortingBoard


@dataclass
class NoMatchingHoleError(DataclassException):
    """
    Raised when a :class:`~experiments.montessori.semantics.ShapeSortingBoard` has no
    :class:`~experiments.montessori.semantics.ShapeSortingHole` whose category matches a
    given :class:`~experiments.montessori.semantics.MontessoriShape`.
    """

    montessori_shape: MontessoriShape
    """
    The shape that has no matching hole.
    """

    board: ShapeSortingBoard
    """
    The board that has no hole matching :attr:`montessori_shape`.
    """

    def error_message(self) -> str:
        return (
            f"{self.board.name} has no hole matching {self.montessori_shape.name}'s "
            f"category {self.montessori_shape.shape_category}."
        )

    def suggest_correction(self) -> str:
        return ""
