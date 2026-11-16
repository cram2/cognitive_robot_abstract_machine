"""
The cells a live session performs: pieces of code, each one step a developer asked for.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from typing_extensions import Self

# %% cells


class CellMarker(StrEnum):
    """
    The text that marks the cells of a cell script.
    """

    START = "# %% cell "
    """
    Begins a line that starts a cell; the rest of the line is the cell's title.
    """


@dataclass(kw_only=True)
class Cell:
    """
    One step of a live session, as code.
    """

    title: str
    """
    What the step does, in the developer's words.
    """

    source: str
    """
    The code of the step.
    """


@dataclass(kw_only=True)
class CellScript:
    """
    The cells of a script, in the order they are written in.
    """

    cells: list[Cell]
    """
    The cells, first to last.
    """

    @classmethod
    def from_file(cls, *, path: Path) -> Self:
        """
        :param path: A script whose cells each begin with a line starting with
            :attr:`CellMarker.START`; text before the first such line is no cell.
        :return: The cells of the script.
        """
        [_, *marked_cells] = path.read_text().split(CellMarker.START)
        cells = []
        for marked_cell in marked_cells:
            title, _, source = marked_cell.partition("\n")
            cells.append(Cell(title=title, source=source))
        return cls(cells=cells)
