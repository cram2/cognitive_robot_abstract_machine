from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, StrEnum

from typing_extensions import Dict

from semantic_digital_twin.world_description.geometry import Color
from giskardpy.motion_statechart.data_types import (
    ObservationStateValues,
    TransitionKind,
)

# %% fonts


class Font(StrEnum):
    """
    The font families the drawing writes its text in.
    """

    SANS_SERIF = "sans-serif"
    """
    The proportional family used for the names of nodes and goals.
    """

    MONOSPACE = "monospace"
    """
    The fixed width family used for text taken verbatim from the statechart, such as
    conditions and state badges.
    """


# %% how a node is drawn


class NodeDrawingStyle(Enum):
    """
    The graphviz style and shape each kind of node is drawn with.
    """

    style: str
    """
    The graphviz style of a node of this kind.
    """

    shape: str
    """
    The graphviz shape of a node of this kind.
    """

    MONITOR = "filled, rounded", "rectangle"
    TASK = "filled, diagonals", "rectangle"
    GOAL = "filled", "none"

    def __init__(self, style: str, shape: str) -> None:
        """
        :param style: The graphviz style of a node of this kind.
        :param shape: The graphviz shape of a node of this kind.
        """
        self.style = style
        self.shape = shape


class BorderStyle(StrEnum):
    """
    The graphviz styles of the extra borders drawn around a node.
    """

    ROUNDED = "rounded"
    DASHED_ROUNDED = "dashed, rounded"


# %% sizes


@dataclass(frozen=True)
class DrawingMetrics:
    """
    The measurements the drawing is laid out with.
    """

    line_width: float = 4
    """
    The thickness in points of every drawn border and edge.
    """

    font_size: int = 15
    """
    The size in points of the text in nodes and goals.
    """

    rank_separation: float = 1
    """
    The distance in inches between two consecutive rows of nodes.
    """

    node_separation: float = 1
    """
    The distance in inches between two nodes of the same row.
    """

    arrow_size: float = 1
    """
    The size of the arrow head of a transition, relative to the graphviz default.
    """

    compact_separation_factor: float = 0.5
    """
    The multiple of both separations the drawing keeps in compact mode.
    """

    compact_bottom_padding_factor: float = 2.5
    """
    The multiple of :attr:`line_width` a node reserves below its badges in compact mode,
    where no conditions follow them.
    """


DRAWING_METRICS = DrawingMetrics()
"""
The measurements every drawing is laid out with.
"""


# %% conditions

DISABLED_CONDITION_COLOR = Color.from_hex("#9CA3AF")
"""
The color a condition that cannot become true any more is written in.
"""


# %% how a trinary value is drawn


@dataclass(frozen=True)
class ObservationDrawingStyle:
    """
    How one trinary value is drawn as a line or as text.

    ..note:: These are the badge colors of :class:`ObservationStateValues` darkened for
        drawing on the page. A badge color fills a cell behind black text and has to stay
        light, which makes it far too pale for a stroke.
    """

    color: Color
    """
    The color a line or a term carrying this value is drawn in.
    """

    line_width: float
    """
    The width of a line carrying this value.
    """


OBSERVATION_DRAWING_STYLES: Dict[ObservationStateValues, ObservationDrawingStyle] = {
    ObservationStateValues.TRUE: ObservationDrawingStyle(
        color=Color.from_hex("#2E7D32"), line_width=DRAWING_METRICS.line_width * 1.25
    ),
    ObservationStateValues.FALSE: ObservationDrawingStyle(
        color=Color.from_hex("#C62828"), line_width=DRAWING_METRICS.line_width * 1.25
    ),
    ObservationStateValues.UNKNOWN: ObservationDrawingStyle(
        color=Color.from_hex("#000000"), line_width=DRAWING_METRICS.line_width * 0.5
    ),
}
"""
The drawing style of every trinary value.

Only true and false are colors; a value with no answer yet is plain black and drawn
thin, so it recedes instead of presenting itself as a third value color.
"""

# %% how far apart a dependency pushes its endpoints


MINIMUM_RANK_DISTANCES: Dict[TransitionKind, int] = {
    TransitionKind.START: 1,
    TransitionKind.PAUSE: 0,
    TransitionKind.END: 1,
    TransitionKind.RESET: 0,
}
"""
How many rows apart a dependency of each transition kind draws its endpoints at least.

A node read by a pause or a reset condition may sit beside the node reading it; the
other two kinds are drawn a row above it.
"""
