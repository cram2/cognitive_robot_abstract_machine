from __future__ import annotations

from dataclasses import dataclass

from typing_extensions import (
    Dict,
    TYPE_CHECKING,
)

from giskardpy.motion_statechart.data_types import (
    LifeCycleValues,
    ObservationStateValues,
)

if TYPE_CHECKING:
    pass

NotStartedColor = "#9F9F9F"
MyBLUE = "#0000DD"
MyGREEN = "#006600"
MyORANGE = "#996900"
MyRED = "#993000"
MyGRAY = "#E0E0E0"

ChatGPTGreen = "#28A745"
ChatGPTOrange = "#E6AC00"
ChatGPTRed = "#DC3545"
ChatGPTBlue = "#007BFF"
ChatGPTGray = "#8F959E"


@dataclass
class ConditionColors:
    StartCondColor = ChatGPTGreen
    PauseCondColor = ChatGPTOrange
    EndCondColor = ChatGPTRed
    ResetCondColor = ChatGPTGray


MonitorTrueGreen = "#B6E5A0"
MonitorFalseRed = "#FF5024"
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

ResetSymbol = "⟲"

ObservationStateToColor: Dict[ObservationStateValues, str] = {
    ObservationStateValues.UNKNOWN: ConditionColors.ResetCondColor,
    ObservationStateValues.TRUE: MonitorTrueGreen,
    ObservationStateValues.FALSE: MonitorFalseRed,
}

ObservationStateToSymbol: Dict[ObservationStateValues, str] = {
    ObservationStateValues.UNKNOWN: "?",
    ObservationStateValues.TRUE: "True",
    ObservationStateValues.FALSE: "False",
}

ObservationStateToEdgeStyle: Dict[ObservationStateValues, Dict[str, str]] = {
    ObservationStateValues.UNKNOWN: {
        "penwidth": (LineWidth * 1.5) / 2,
        # 'label': '<<FONT FACE="monospace"><B>?</B></FONT>>',
        "fontsize": Fontsize * 1.333,
    },
    ObservationStateValues.TRUE: {"penwidth": LineWidth * 1.5},
    ObservationStateValues.FALSE: {"style": "dashed", "penwidth": LineWidth * 1.5},
}

LifeCycleStateToColor: Dict[LifeCycleValues, str] = {
    LifeCycleValues.NOT_STARTED: ConditionColors.ResetCondColor,
    LifeCycleValues.RUNNING: ConditionColors.StartCondColor,
    LifeCycleValues.PAUSED: ConditionColors.PauseCondColor,
    LifeCycleValues.SUCCEEDED: MonitorTrueGreen,
    LifeCycleValues.FAILED: MonitorFalseRed,
    LifeCycleValues.INTERRUPTED: MyORANGE,
}

LifeCycleStateToSymbol: Dict[LifeCycleValues, str] = {
    # LifeCycleState.not_started: '○',
    LifeCycleValues.NOT_STARTED: "—",
    LifeCycleValues.RUNNING: "▶",
    # LifeCycleState.paused: '⏸',
    LifeCycleValues.PAUSED: "<B>||</B>",
    LifeCycleValues.SUCCEEDED: "✔",
    LifeCycleValues.FAILED: "✖",
    LifeCycleValues.INTERRUPTED: "■",
}
