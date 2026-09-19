"""
What each capture taken off the real camera shows.

Read off the picture by eye, so a detection result is measured against the scene rather
than against an earlier run of the same code.
"""

from __future__ import annotations

from dataclasses import dataclass

from typing_extensions import Dict, Tuple

from experiments.montessori.semantics import MontessoriShapeCategory


@dataclass(frozen=True)
class CaptureTruth:
    """
    What one capture really holds, as a reader of the picture can see it.
    """

    pieces_on_table: Tuple[MontessoriShapeCategory, ...]
    """
    Every loose piece resting on the bare table, one entry per physical piece.
    """

    pieces_on_lid: Tuple[MontessoriShapeCategory, ...]
    """
    Every piece resting on, or standing in a hole of, the board's lid.
    """

    @property
    def pieces(self) -> Tuple[MontessoriShapeCategory, ...]:
        """
        Every piece in the scene, wherever it rests.
        """
        return self.pieces_on_table + self.pieces_on_lid


CAPTURE_TRUTHS: Dict[str, CaptureTruth] = {
    "objects_on_montessori": CaptureTruth(
        pieces_on_table=(
            MontessoriShapeCategory.CYLINDER,
            MontessoriShapeCategory.TRIANGULAR_PRISM,
        ),
        pieces_on_lid=(
            MontessoriShapeCategory.CUBE,
            MontessoriShapeCategory.RECTANGULAR_PRISM,
        ),
    ),
    "stuck_cube_in_hole": CaptureTruth(
        pieces_on_table=(
            MontessoriShapeCategory.CYLINDER,
            MontessoriShapeCategory.RECTANGULAR_PRISM,
            MontessoriShapeCategory.TRIANGULAR_PRISM,
        ),
        pieces_on_lid=(MontessoriShapeCategory.CUBE,),
    ),
    "disoriented_cube_on_hole": CaptureTruth(
        pieces_on_table=(
            MontessoriShapeCategory.CYLINDER,
            MontessoriShapeCategory.RECTANGULAR_PRISM,
            MontessoriShapeCategory.TRIANGULAR_PRISM,
        ),
        pieces_on_lid=(MontessoriShapeCategory.CUBE,),
    ),
    "displaced_cube_from_hole": CaptureTruth(
        pieces_on_table=(
            MontessoriShapeCategory.CYLINDER,
            MontessoriShapeCategory.RECTANGULAR_PRISM,
            MontessoriShapeCategory.TRIANGULAR_PRISM,
        ),
        pieces_on_lid=(MontessoriShapeCategory.CUBE,),
    ),
    "non_inserted_objects": CaptureTruth(
        pieces_on_table=(),
        pieces_on_lid=(
            MontessoriShapeCategory.CUBE,
            MontessoriShapeCategory.CYLINDER,
            MontessoriShapeCategory.RECTANGULAR_PRISM,
        ),
    ),
    "tracy_pickup_demo": CaptureTruth(
        pieces_on_table=(
            MontessoriShapeCategory.RECTANGULAR_PRISM,
            MontessoriShapeCategory.TRIANGULAR_PRISM,
        ),
        pieces_on_lid=(
            MontessoriShapeCategory.CUBE,
            MontessoriShapeCategory.CYLINDER,
        ),
    ),
}
"""
What each shipped capture shows, keyed by the capture's own name.

Every one of them holds the shape-sorting board, so only the loose pieces differ.
"""
