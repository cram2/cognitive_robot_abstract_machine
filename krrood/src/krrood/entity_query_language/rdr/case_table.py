"""
Render a concrete case object as a readable terminal table for the interactive expert.

The case's ``(attribute, value)`` pairs are laid out across a configurable number of
*columns* (pairs per row, default :data:`DEFAULT_COLUMNS`) using the full terminal width,
keeping the table short rather than one tall pair-per-row column. Any value that overflows
its column budget is clipped with an ellipsis.

Kept deliberately small and dependency-light (just ``tabulate``); the legacy
``ripple_down_rules`` rendering is not imported because its package ``__init__`` spins up a
QApplication and heavy imports.
"""

from __future__ import annotations

import dataclasses
import shutil

from dataclasses import dataclass

from typing_extensions import Any, List, Optional, Tuple

from tabulate import tabulate

#: Default number of ``(attribute, value)`` pairs laid out side by side per row.
DEFAULT_COLUMNS = 2

#: Ellipsis appended to a clipped cell.
_ELLIPSIS = "..."

#: Floor for a value column so a clip never collapses to just the ellipsis.
_MIN_VALUE_WIDTH = 8

#: Longest attribute name rendered before clipping.
_MAX_KEY_WIDTH = 30

#: Per-pair horizontal padding ``simple_grid`` adds around the key and value cells.
_PAIR_BORDER = 7


def _case_items(case: Any) -> List[Tuple[str, Any]]:
    """:return: ``(public_attribute_name, value)`` pairs describing the case."""
    if dataclasses.is_dataclass(case) and not isinstance(case, type):
        return [
            (f.name, getattr(case, f.name))
            for f in dataclasses.fields(case)
            if not f.name.startswith("_")
        ]
    if hasattr(case, "__dict__"):
        return [(k, v) for k, v in vars(case).items() if not k.startswith("_")]
    return [("value", case)]


def _format_value(value: Any) -> str:
    text = "" if value is None else str(value)
    if text in ("True", "False"):
        text = text.lower()
    return text


def _clip(text: str, width: int) -> str:
    """:return: ``text`` truncated to ``width`` characters, ending in an ellipsis if cut."""
    if len(text) <= width:
        return text
    if width <= len(_ELLIPSIS):
        return text[:width]
    return text[: width - len(_ELLIPSIS)] + _ELLIPSIS


def _terminal_width() -> int:
    return shutil.get_terminal_size((80, 20)).columns


@dataclass
class CaseTableRenderer:
    """Renders a case as a width-bounded, multi-column ``(attribute, value)`` table."""

    columns: int = DEFAULT_COLUMNS
    """Number of ``(attribute, value)`` pairs placed side by side on each row."""
    max_width: Optional[int] = None
    """Total table width budget; defaults to the current terminal width."""

    def render(self, case: Any) -> str:
        """
        :return: A multi-column table of the case's public attributes sized to
            :attr:`max_width`. Falls back to ``repr(case)`` when the case has no inspectable
            attributes.
        """
        items = [(name, _format_value(value)) for name, value in _case_items(case)]
        if not items:
            return repr(case)
        columns = max(1, self.columns)
        width = self.max_width or _terminal_width()
        value_width = self._value_width(items, width, columns)
        return tabulate(self._rows(items, columns, value_width), tablefmt="simple_grid")

    @staticmethod
    def _value_width(items: List[Tuple[str, str]], width: int, columns: int) -> int:
        key_width = min(_MAX_KEY_WIDTH, max(len(name) for name, _ in items))
        return max(_MIN_VALUE_WIDTH, width // columns - key_width - _PAIR_BORDER)

    @staticmethod
    def _rows(
        items: List[Tuple[str, str]], columns: int, value_width: int
    ) -> List[List[str]]:
        cells = [
            (_clip(name, _MAX_KEY_WIDTH), _clip(value, value_width))
            for name, value in items
        ]
        rows: List[List[str]] = []
        for start in range(0, len(cells), columns):
            chunk = cells[start : start + columns]
            row: List[str] = [field for pair in chunk for field in pair]
            row += [""] * ((columns - len(chunk)) * 2)
            rows.append(row)
        return rows


def render_case_table(
    case: Any, columns: int = DEFAULT_COLUMNS, max_width: Optional[int] = None
) -> str:
    """Convenience wrapper around :class:`CaseTableRenderer`."""
    return CaseTableRenderer(columns=columns, max_width=max_width).render(case)
