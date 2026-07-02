from __future__ import annotations

import html
import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import StrEnum
from typing_extensions import ClassVar, Optional

from krrood.entity_query_language.verbalization.fragments.roles import SemanticRole

_TOOLTIP_ATTR = "title"

_log = logging.getLogger(__name__)


#: Hex colour string (or ``None`` for no colour) for each semantic role, matching the
#: query-graph palette.  Colour is a *formatter* concern — the fragment tree carries only the
#: role tag.
ROLE_COLORS: dict[SemanticRole, Optional[str]] = {
    SemanticRole.KEYWORD: "#eded18",  # ConclusionSelector yellow
    SemanticRole.VARIABLE: "cornflowerblue",
    SemanticRole.AGGREGATION: "#F54927",  # Aggregator red-orange
    SemanticRole.OPERATOR: "#ff7f0e",  # Comparator orange
    SemanticRole.LOGICAL: "#2ca02c",  # LogicalOperator green
    SemanticRole.LITERAL: "#949292",  # Literal gray
    SemanticRole.ATTRIBUTE: "#8FC7B8",  # MappedVariable teal
    SemanticRole.PLAIN: None,
}


class BulletStyle(StrEnum):
    """Bullet character used for hierarchical list items."""

    DASH = "-"
    """A hyphen bullet."""
    DOT = "•"
    """A round bullet."""
    ASTERISK = "*"
    """An asterisk bullet."""


class IndentSize(StrEnum):
    """Indentation string used per nesting level in hierarchical rendering."""

    TWO_SPACES = "  "
    """Two-space indent (the default)."""
    FOUR_SPACES = "    "
    """Four-space indent."""
    TAB = "\t"
    """Hard tab character."""


def detect_osc8_support() -> bool:
    """:return: ``True`` when the current terminal is known to support OSC 8 hyperlinks."""
    if os.environ.get("VTE_VERSION"):  # GNOME Terminal, Tilix, …
        return True
    term_prog = os.environ.get("TERM_PROGRAM", "")
    if term_prog in {"vscode", "WezTerm", "iTerm.app"}:
        return True
    if os.environ.get("TERM") == "xterm-kitty":
        return True
    return False


@dataclass
class Formatter(ABC):
    """
    Single source of truth for all format-specific characters and colour markup.

    Concrete subclasses determine how colours, spaces, newlines, and hyperlinks are encoded in
    the output string:

    * ``PlainFormatter`` — no colour, ASCII space/newline.
    * ``ANSIFormatter`` — 24-bit ANSI escape sequences with optional OSC 8 links.
    * ``HTMLFormatter`` — ``<span style="color:...">`` tags and ``<a href="...">`` links.
    """

    @abstractmethod
    def colorize(self, text: str, role: SemanticRole) -> str:
        """
        Wrap *text* in format-specific colour markup for *role*.

        :param text: Plain display text to colourize.
        :param role: Semantic role determining the colour.
        :return: Coloured string (or *text* unchanged when no colour is defined for *role*).
        """
        ...

    @property
    @abstractmethod
    def space(self) -> str:
        """Inline word separator character(s) (e.g. ``" "`` or ``"&nbsp;"``)."""
        ...

    @property
    @abstractmethod
    def newline(self) -> str:
        """Line break character(s) (e.g. ``"\\n"`` or ``"<br>"``)."""
        ...

    def wrap_link(self, text: str, url: str, tooltip: Optional[str] = None) -> str:
        """
        Wrap already-rendered *text* with a hyperlink to *url*.

        The base implementation is a no-op (hyperlinks are not supported for this format).

        :param text: Already-colourized display text.
        :param url: Destination URL.
        :param tooltip: Optional single-line docstring summary shown on hover (HTML-escaped).
        :return: *text* unchanged (base); linked string (subclasses).
        """
        return text


@dataclass
class PlainFormatter(Formatter):
    """
    No colour markup; standard ASCII space (``" "``) and newline (``"\\n"``).
    """

    def colorize(self, text: str, role: SemanticRole) -> str:
        """Return *text* unchanged (no colour markup in plain mode)."""
        return text

    @property
    def space(self) -> str:
        return " "

    @property
    def newline(self) -> str:
        return "\n"


@dataclass
class ANSIFormatter(Formatter):
    """
    True-color ANSI escape sequences (24-bit, ``\\033[38;2;R;G;Bm``).

    Compatible with VS Code terminal, GNOME Terminal, iTerm2, Windows Terminal,
    and any terminal supporting the ISO-8613-3 direct-color extension.

    OSC 8 hyperlinks are enabled automatically when the terminal is detected as
    capable (``VTE_VERSION``, ``TERM_PROGRAM`` in ``{vscode, WezTerm, iTerm.app}``,
    or ``TERM=xterm-kitty``).  On unsupported terminals links fall back to plain
    coloured text with no link markup.
    """

    _RESET: ClassVar[str] = "\033[0m"
    _NAMED: ClassVar[dict[str, tuple[int, int, int]]] = {
        "cornflowerblue": (100, 149, 237),
    }

    _hyperlinks_enabled: bool = field(default_factory=detect_osc8_support, init=False)

    def colorize(self, text: str, role: SemanticRole) -> str:
        color = ROLE_COLORS.get(role)
        if color is None:
            return text
        r, g, b = self._hex_to_rgb(color)
        return f"\033[38;2;{r};{g};{b}m{text}{self._RESET}"

    def wrap_link(self, text: str, url: str, tooltip: Optional[str] = None) -> str:
        if not self._hyperlinks_enabled:
            return text
        # OSC 8 format: ESC ] 8 ; ; URL ST  text  ESC ] 8 ; ; ST
        return f"\033]8;;{url}\033\\{text}\033]8;;\033\\"

    @property
    def space(self) -> str:
        return " "

    @property
    def newline(self) -> str:
        return "\n"

    def _hex_to_rgb(self, color: str) -> tuple[int, int, int]:
        """Convert a hex colour string (e.g. ``\"#ff7f0e\"``) or named colour to an ``(R, G, B)`` tuple."""
        if color.startswith("#"):
            h = color.lstrip("#")
            return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
        return self._NAMED.get(color.lower(), (255, 255, 255))


@dataclass
class HTMLFormatter(Formatter):
    """
    HTML output with ``<span style="color: …">`` colour tags, ``&nbsp;`` spaces,
    and ``<br>`` newlines.

    Suitable for Jupyter notebooks, GitLab Markdown, and any renderer that
    passes through inline HTML.  Hyperlinks use standard ``<a href="…">`` anchors.
    """

    def colorize(self, text: str, role: SemanticRole) -> str:
        color = ROLE_COLORS.get(role)
        if color is None:
            return text
        return f'<span style="color:{color}">{text}</span>'

    def wrap_link(self, text: str, url: str, tooltip: Optional[str] = None) -> str:
        tooltip_attribute = (
            f' {_TOOLTIP_ATTR}="{html.escape(tooltip, quote=True)}"' if tooltip else ""
        )
        return f'<a target="_blank" rel="noopener" href="{url}"{tooltip_attribute}>{text}</a>'

    @property
    def space(self) -> str:
        return "&nbsp;"

    @property
    def newline(self) -> str:
        return "<br>"
