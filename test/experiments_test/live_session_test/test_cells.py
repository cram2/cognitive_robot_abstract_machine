"""
Tests for splitting a cell script into the cells a live session performs.
"""

from __future__ import annotations

from pathlib import Path

from experiments.live_session.cells import CellMarker, CellScript

# %% splitting a script


def test_a_cell_script_is_split_into_its_cells_in_order(swing_the_hinge_script: Path):
    script = CellScript.from_file(path=swing_the_hinge_script)

    assert [cell.title for cell in script.cells] == [
        "swing the hinge to half a radian",
        "swing the hinge back",
    ]


def test_the_cells_of_a_script_hold_all_of_its_text(swing_the_hinge_script: Path):
    script = CellScript.from_file(path=swing_the_hinge_script)

    rejoined = "".join(
        f"{CellMarker.START}{cell.title}\n{cell.source}" for cell in script.cells
    )
    assert rejoined == swing_the_hinge_script.read_text()
