from __future__ import annotations

from dataclasses import dataclass, field

from typing_extensions import (
    Any,
    Callable,
    List,
    TYPE_CHECKING,
    Self,
)

from krrood.patterns.field_metadata import JSONMetadata
from giskardpy.motion_statechart.plotters.styles import (
    BorderStyle,
    NodeDrawingStyle,
)

if TYPE_CHECKING:
    pass


@dataclass
class NodePlotSpec:
    visible: bool = True
    collapse_children: bool = False
    """
    Whether the descendants of this node are omitted from the drawing.

    Only has an effect on nodes that own children, i.e. goals.
    """

    style: str = NodeDrawingStyle.MONITOR.style
    shape: str = NodeDrawingStyle.MONITOR.shape
    extra_border_styles: List[str] = field(default_factory=list)

    @classmethod
    def create_monitor_style(cls) -> Self:
        return cls(
            visible=True,
            style=NodeDrawingStyle.MONITOR.style,
            shape=NodeDrawingStyle.MONITOR.shape,
            extra_border_styles=[],
        )

    @classmethod
    def create_task_style(cls) -> Self:
        return cls(
            visible=True,
            style=NodeDrawingStyle.TASK.style,
            shape=NodeDrawingStyle.TASK.shape,
            extra_border_styles=[],
        )

    @classmethod
    def create_goal_style(cls) -> Self:
        return cls(
            visible=True,
            style=NodeDrawingStyle.GOAL.style,
            shape=NodeDrawingStyle.GOAL.shape,
            extra_border_styles=[],
        )

    @classmethod
    def create_collapsed_goal_style(cls) -> Self:
        """
        :return: A goal style whose descendants are left out of the drawing.
        """
        goal_style = cls.create_goal_style()
        goal_style.collapse_children = True
        return goal_style

    @classmethod
    def create_end_style(cls):
        return cls(
            visible=True,
            style=NodeDrawingStyle.MONITOR.style,
            shape=NodeDrawingStyle.MONITOR.shape,
            extra_border_styles=[BorderStyle.ROUNDED],
        )

    @classmethod
    def create_cancel_style(cls):
        return cls(
            visible=True,
            style=NodeDrawingStyle.MONITOR.style,
            shape=NodeDrawingStyle.MONITOR.shape,
            extra_border_styles=[BorderStyle.DASHED_ROUNDED],
        )


def plot_specification_field(default_factory: Callable[[], NodePlotSpec]) -> Any:
    """
    Declares the plot spec field of a node, styled by `default_factory`.

    Plot specs are not constructor arguments, so they need to be marked as serializable
    explicitly to survive a JSON round trip.
    """
    return field(
        default_factory=default_factory,
        kw_only=True,
        init=False,
        metadata=JSONMetadata(serialize=True).as_dict(),
    )
