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

    Only has an effect on nodes that own children, i.e. composite statechart nodes.
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
    def create_composite_statechart_node_style(cls) -> Self:
        return cls(
            visible=True,
            style=NodeDrawingStyle.COMPOSITE.style,
            shape=NodeDrawingStyle.COMPOSITE.shape,
            extra_border_styles=[],
        )

    @classmethod
    def create_collapsed_composite_statechart_node_style(cls) -> Self:
        """
        :return: A composite statechart node style whose descendants are left out of the drawing.
        """
        composite_style = cls.create_composite_statechart_node_style()
        composite_style.collapse_children = True
        return composite_style

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
