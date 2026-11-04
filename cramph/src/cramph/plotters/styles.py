from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, StrEnum


from semantic_digital_twin.world_description.geometry import Color
from cramph.data_types import ObservationStateValues, TransitionKind

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
    COMPOSITE = "filled", "none"

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

    side_by_side_rank_distance: int = 0
    """
    How many rows apart a dependency of a pause or a reset condition draws its endpoints
    at least, so the node read may sit beside the node reading it.
    """

    stacked_rank_distance: int = 1
    """
    How many rows apart a dependency of every other transition kind draws its endpoints
    at least, so the node read is drawn a row above the node reading it.
    """

    def minimum_rank_distance(self, transition_kind: TransitionKind) -> int:
        """
        :param transition_kind: The kind of the condition a dependency feeds.
        :return: How many rows apart that dependency draws its endpoints at least.
        """
        if transition_kind in (TransitionKind.PAUSE, TransitionKind.RESET):
            return self.side_by_side_rank_distance
        return self.stacked_rank_distance


# %% how a trinary value is drawn


class ObservationDrawingStyle(Enum):
    """
    How one trinary value is drawn as a line or as text.

    Only true and false are colors; a value with no answer yet is plain black and drawn
    thin, so it recedes instead of presenting itself as a third value color.

    ..note:: These are the badge colors of :class:`ObservationStateValues` darkened for
        drawing on the page. A badge color fills a cell behind black text and has to stay
        light, which makes it far too pale for a stroke.
    """

    color: Color
    """
    The color a line or a term carrying this value is drawn in.
    """

    line_width_factor: float
    """
    The multiple of :attr:`DrawingMetrics.line_width` a line carrying this value is
    drawn with.
    """

    TRUE = "#2E7D32", 1.25
    FALSE = "#C62828", 1.25
    UNKNOWN = "#000000", 0.5

    def __init__(self, color_hex: str, line_width_factor: float) -> None:
        """
        :param color_hex: The color a line or a term carrying this value is drawn in.
        :param line_width_factor: The multiple of :attr:`DrawingMetrics.line_width` a
            line carrying this value is drawn with.
        """
        self.color = Color.from_hex(color_hex)
        self.line_width_factor = line_width_factor

    @classmethod
    def of_observation(
        cls, observation: ObservationStateValues
    ) -> ObservationDrawingStyle:
        """
        :param observation: The trinary value to draw.
        :return: The style that value is drawn in.
        """
        return cls[observation.name]

    def line_width(self, metrics: DrawingMetrics) -> float:
        """
        :param metrics: The measurements the drawing is laid out with.
        :return: The width of a line carrying this value.
        """
        return metrics.line_width * self.line_width_factor
