"""
Render a concrete case object as a readable terminal table for the interactive expert.

The case's ``(attribute, value)`` pairs are laid out across as many columns as the
terminal is wide enough for: the number of pairs per row is derived from the available
width and :data:`DEFAULT_MIN_COLUMN_WIDTH` (the smallest a pair column may be), and the
value columns are then stretched so the table spans the full width rather than hugging
its content. Values that overflow their budget are wrapped across multiple lines so the
full content is visible rather than truncated.

Kept deliberately small and dependency-light (just ``tabulate``);
"""

from __future__ import annotations

import dataclasses
import shutil
import textwrap

from dataclasses import dataclass

from typing_extensions import Any, List, Optional, Tuple

import tabulate as _tabulate

from colorama import Fore, Style

#: Smallest total width (key + value + borders) a single pair column may occupy. The
#: number of pairs laid side by side is ``terminal_width // min_column_width``.
DEFAULT_MIN_COLUMN_WIDTH = 24

#: Ellipsis appended to a clipped cell.
_ELLIPSIS = "..."

#: Longest attribute name rendered before clipping.
_MAX_KEY_WIDTH = 30

#: Per-pair horizontal padding ``simple_grid`` adds around the key and value cells.
_PAIR_BORDER = 7

#: Width assumed when the terminal size cannot be detected.
_FALLBACK_WIDTH = 100


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
    return shutil.get_terminal_size((_FALLBACK_WIDTH, 24)).columns


@dataclass
class CaseTableRenderer:
    """Renders a case as a full-width, multi-column ``(attribute, value)`` table."""

    min_column_width: int = DEFAULT_MIN_COLUMN_WIDTH
    """Smallest total width a pair column may take; sets how many pairs fit per row."""
    max_width: Optional[int] = None
    """Total table width budget; defaults to the current terminal width."""
    use_color: bool = True
    """Render field names in a distinct accent so they stand apart from their values."""

    def render(self, case: Any) -> str:
        """
        :return: A multi-column table of the case's public attributes that spans the full
            available width. Falls back to ``repr(case)`` when the case has no inspectable
            attributes.
        """
        items = [(name, _format_value(value)) for name, value in _case_items(case)]
        if not items:
            return repr(case)
        width = self.max_width or _terminal_width()
        key_width = min(_MAX_KEY_WIDTH, max(len(name) for name, _ in items))
        pairs_per_row = self._pairs_per_row(len(items), width, key_width)
        value_width = self._value_width(width, pairs_per_row, key_width)
        rows = self._rows(items, pairs_per_row, key_width, value_width)
        return self._tabulate_full_width(rows)

    def _style_key(self, text: str) -> str:
        """Wrap a (already width-padded) field name in the accent style, if colour is on."""
        if not self.use_color:
            return text
        return f"{Style.BRIGHT}{Fore.CYAN}{text}{Style.RESET_ALL}"

    @staticmethod
    def _tabulate_full_width(rows: List[List[str]]) -> str:
        """Tabulate without trimming the padding that stretches columns to full width."""
        preserve = _tabulate.PRESERVE_WHITESPACE
        _tabulate.PRESERVE_WHITESPACE = True
        try:
            return _tabulate.tabulate(rows, tablefmt="simple_grid")
        finally:
            _tabulate.PRESERVE_WHITESPACE = preserve

    def _pairs_per_row(self, item_count: int, width: int, key_width: int) -> int:
        budget = max(self.min_column_width, key_width + _PAIR_BORDER + 1)
        return max(1, min(item_count, width // budget))

    @staticmethod
    def _value_width(width: int, pairs_per_row: int, key_width: int) -> int:
        """Stretch the value column so ``pairs_per_row`` pairs span the full width."""
        pair_width = width // pairs_per_row
        return max(1, pair_width - key_width - _PAIR_BORDER)

    def _rows(
        self,
        items: List[Tuple[str, str]],
        pairs_per_row: int,
        key_width: int,
        value_width: int,
    ) -> List[List[str]]:
        cells = [
            (
                self._style_key(_clip(name, key_width).ljust(key_width)),
                textwrap.fill(value, width=value_width) if value else "",
            )
            for name, value in items
        ]
        empty_pair = ["".ljust(key_width), ""]
        rows: List[List[str]] = []
        for start in range(0, len(cells), pairs_per_row):
            chunk = cells[start : start + pairs_per_row]
            row: List[str] = [field for pair in chunk for field in pair]
            for _ in range(pairs_per_row - len(chunk)):
                row += empty_pair
            rows.append(row)
        return rows


def render_case_table(
    case: Any,
    min_column_width: int = DEFAULT_MIN_COLUMN_WIDTH,
    max_width: Optional[int] = None,
    use_color: bool = True,
) -> str:
    """Convenience wrapper around :class:`CaseTableRenderer`."""
    return CaseTableRenderer(
        min_column_width=min_column_width, max_width=max_width, use_color=use_color
    ).render(case)
