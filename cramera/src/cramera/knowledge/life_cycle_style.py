"""
Native lifecycle presentation in the graph viewer's JSON shape.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import StrEnum

from typing_extensions import Any

from giskardpy.motion_statechart.data_types import LifeCycleValues
from giskardpy.motion_statechart.plotters.styles import DRAWING_METRICS


# %% graph payload vocabulary
class StatusPresentationField(StrEnum):
    """
    Graph payload keys carrying the lifecycle palette and its legend order.
    """

    STYLES = "statusStyles"
    """
    Lifecycle styles indexed by native state names.
    """

    ORDER = "statusOrder"
    """
    Native state names in lifecycle declaration order.
    """


# %% graph ring measurements
@dataclass(frozen=True)
class LifeCycleDrawingMetrics:
    """
    Lifecycle ring measurements relative to the native renderer's line width.
    """

    active_width_factor: float = 4
    """
    Width of an active or failed ring relative to the native drawing line width.
    """

    default_width_factor: float = 3
    """
    Width of a completed or suspended ring relative to the native drawing line width.
    """

    pending_width_factor: float = 1.75
    """
    Width of a not-started ring relative to the native drawing line width.
    """

    pending_dashes: tuple[int, int] = (4, 5)
    """
    Short dashes identifying a node that has not started.
    """

    suspended_dashes: tuple[int, int] = (9, 7)
    """
    Longer dashes identifying a paused or interrupted node.
    """


# %% lifecycle presentation
@dataclass(frozen=True)
class LifeCycleStyle:
    """
    A native lifecycle's label and color, with its graph ring measurements.
    """

    label: str
    """
    Human-readable lifecycle name, shared by rings, legends and step pills.
    """

    color: str
    """
    Native lifecycle color as a CSS hex color.
    """

    width: float
    """
    Graph ring width in vis-network's drawing units.
    """

    dashes: tuple[int, int] | None
    """
    Alternating dash and gap lengths, or None for a solid ring.
    """

    show_label: bool
    """
    Whether the graph node appends the lifecycle label beneath its own name.
    """

    @classmethod
    def of_life_cycle(
        cls, state: LifeCycleValues, metrics: LifeCycleDrawingMetrics
    ) -> LifeCycleStyle:
        """
        Derive the presentation of one native lifecycle value.

        :param state: The lifecycle value to present.
        :param metrics: The graph ring measurements.
        :return: The lifecycle's labels, color and ring style.
        """
        width_factor = metrics.default_width_factor
        dashes = None
        if state is LifeCycleValues.NOT_STARTED:
            width_factor = metrics.pending_width_factor
            dashes = metrics.pending_dashes
        elif state in (LifeCycleValues.RUNNING, LifeCycleValues.FAILED):
            width_factor = metrics.active_width_factor
        elif state in (LifeCycleValues.PAUSED, LifeCycleValues.INTERRUPTED):
            dashes = metrics.suspended_dashes
        return cls(
            label=state.name.lower().replace("_", " "),
            color=state.color.to_hex(),
            width=DRAWING_METRICS.line_width * width_factor,
            dashes=dashes,
            show_label=state is not LifeCycleValues.NOT_STARTED,
        )

    @classmethod
    def presentation(cls) -> dict[str, Any]:
        """
        Serialize the native palette and legend order.

        :return: JSON-ready style and legend entries for a graph payload.
        """
        metrics = LifeCycleDrawingMetrics()
        styles = {
            state.name: asdict(cls.of_life_cycle(state, metrics))
            for state in LifeCycleValues
        }
        return {
            StatusPresentationField.STYLES: styles,
            StatusPresentationField.ORDER: [state.name for state in LifeCycleValues],
        }
