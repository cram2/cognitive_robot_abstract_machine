"""
The ways reading a shape-sorting toy's board, or sorting a piece into it, can fail.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing_extensions import TYPE_CHECKING

from krrood.exceptions import DataclassException
from semantic_digital_twin.exceptions import UsageError

if TYPE_CHECKING:
    from semantic_digital_twin.shape_sorting_toy.semantic_annotations import (
        ShapeSortingBoard,
        ShapeSortingPiece,
    )

# %% reading the board's mesh


@dataclass(kw_only=True)
class NoPerforatedPart(DataclassException):
    """
    Raised when no part of a board's mesh has holes cut through it, so the mesh has no
    lid to read holes off.
    """

    mesh_filename: str
    """
    The mesh that has no part with holes through it.
    """

    def error_message(self) -> str:
        return f"No watertight part of the mesh '{self.mesh_filename}' has holes through it."

    def suggest_correction(self) -> str:
        return "Load the mesh of a board whose lid has its holes cut through it."


# %% sorting a piece


@dataclass(kw_only=True)
class NoMatchingHoleError(UsageError):
    """
    Raised when a shape-sorting board has no hole that a piece fits through.
    """

    piece: ShapeSortingPiece
    """
    The piece that fits through none of the board's holes.
    """

    board: ShapeSortingBoard
    """
    The board whose holes the piece fits through none of.
    """

    def error_message(self) -> str:
        return (
            f"{self.board.name.name} has no hole that {self.piece.name.name}, a "
            f"{type(self.piece).__name__}, fits through."
        )

    def suggest_correction(self) -> str:
        return (
            "add a hole cut for this kind of piece, at least as wide as the piece, to "
            "the board."
        )
