"""
What goes wrong when a live session is asked for a moment it does not have.
"""

from __future__ import annotations

from dataclasses import dataclass

from krrood.exceptions import DataclassException

# %% snapshots and cells


@dataclass
class UnknownSnapshotError(DataclassException):
    """
    Raised when a session is asked to restore a snapshot under a name it has none under.
    """

    snapshot_name: str
    """
    The name asked for.
    """

    known_names: list[str]
    """
    The names the session has snapshots under.
    """

    def error_message(self) -> str:
        return f"The session has no snapshot named {self.snapshot_name!r}."

    def suggest_correction(self) -> str:
        return f"Restore one of {self.known_names}, or take the snapshot first."


@dataclass
class NoPerformedCellError(DataclassException):
    """
    Raised when a session is asked to go back to a cell it has not performed.
    """

    cell_index: int
    """
    The position in the session's history asked for.
    """

    performed_cells: int
    """
    How many cells the session has performed.
    """

    def error_message(self) -> str:
        return (
            f"The session has performed {self.performed_cells} cells, so there is no "
            f"cell {self.cell_index} to go back to."
        )

    def suggest_correction(self) -> str:
        return "Pick a cell from the session's history."
