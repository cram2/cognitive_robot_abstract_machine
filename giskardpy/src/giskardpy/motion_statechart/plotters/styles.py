from __future__ import annotations

from dataclasses import dataclass

from typing_extensions import (
    Dict,
    TYPE_CHECKING,
)

from semantic_digital_twin.world_description.geometry import Color
from giskardpy.motion_statechart.data_types import (
    ObservationStateValues,
    TransitionKind,
)

if TYPE_CHECKING:
    pass

NotStartedColor = "#9F9F9F"
MyBLUE = "#0000DD"
MyGREEN = "#006600"
MyRED = "#993000"
MyGRAY = "#E0E0E0"

ChatGPTGreen = "#28A745"
ChatGPTOrange = "#E6AC00"
ChatGPTRed = "#DC3545"
ChatGPTBlue = "#007BFF"
ChatGPTGray = "#8F959E"

FONT = "sans-serif"
LineWidth = 4
NodeSep = 1
RankSep = 1
ArrowSize = 1
Fontsize = 15
GoalNodeStyle = "filled"
GoalNodeShape = "none"
GoalClusterStyle = "filled"
MonitorStyle = "filled, rounded"
MonitorShape = "rectangle"
TaskStyle = "filled, diagonals"
TaskShape = "rectangle"
ConditionFont = "monospace"
DisabledConditionColor = Color.from_hex("#9CA3AF")


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
        color=Color.from_hex("#2E7D32"), line_width=LineWidth * 1.25
    ),
    ObservationStateValues.FALSE: ObservationDrawingStyle(
        color=Color.from_hex("#C62828"), line_width=LineWidth * 1.25
    ),
    ObservationStateValues.UNKNOWN: ObservationDrawingStyle(
        color=Color.from_hex("#000000"), line_width=LineWidth * 0.5
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
