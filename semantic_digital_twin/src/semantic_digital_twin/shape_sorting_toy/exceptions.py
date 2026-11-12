"""
The ways reading a shape-sorting toy's board can fail.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from krrood.exceptions import DataclassException

# %% reading the board's mesh


@dataclass(kw_only=True)
class NoPerforatedPart(DataclassException):
    """
    Raised when no part of a board's mesh has holes cut through it, so the mesh has no
    lid to read holes off.
    """

    mesh_path: Path
    """
    The mesh that has no part with holes through it.
    """

    def error_message(self) -> str:
        return (
            f"No watertight part of the mesh '{self.mesh_path}' has holes through it."
        )

    def suggest_correction(self) -> str:
        return "Load the mesh of a board whose lid has its holes cut through it."
