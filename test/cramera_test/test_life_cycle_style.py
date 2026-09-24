"""
The browser lifecycle palette follows the native enum and renderer configuration.
"""

import json

import pytest

from cramera.knowledge.life_cycle_style import (
    LifeCycleDrawingMetrics,
    LifeCycleStyle,
    StatusPresentationField,
)
from cramera.knowledge.views.chart import ChartViewPayload
from giskardpy.motion_statechart.data_types import LifeCycleValues
from giskardpy.motion_statechart.plotters.styles import DRAWING_METRICS
from semantic_digital_twin.world_description.geometry import Color


# %% native source
@pytest.mark.parametrize("state", LifeCycleValues)
def test_a_lifecycle_keeps_its_native_name_and_color(state: LifeCycleValues) -> None:
    """
    Every native value supplies its own browser label and color.

    :param state: The native lifecycle value to present.
    """
    style = LifeCycleStyle.of_life_cycle(state, LifeCycleDrawingMetrics())

    assert style.label == state.name.lower().replace("_", " ")
    assert style.color == state.color.to_hex()
    assert style.show_label is (state is not LifeCycleValues.NOT_STARTED)


def test_the_palette_contains_only_native_lifecycles() -> None:
    """
    Publish exactly the native lifecycle values in the palette and legend.
    """
    presentation = json.loads(json.dumps(LifeCycleStyle.presentation()))

    assert set(presentation[StatusPresentationField.STYLES]) == {
        state.name for state in LifeCycleValues
    }
    assert presentation[StatusPresentationField.ORDER] == [
        state.name for state in LifeCycleValues
    ]


def test_the_palette_reads_the_current_native_color(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    A graph response reflects a changed native palette without a second definition.

    :param monkeypatch: The active monkeypatch fixture.
    """
    color = Color(0.2, 0.4, 0.6)
    monkeypatch.setattr(LifeCycleValues.RUNNING, "color", color)

    payload = ChartViewPayload().to_payload()

    assert (
        payload[StatusPresentationField.STYLES][LifeCycleValues.RUNNING.name]["color"]
        == color.to_hex()
    )


# %% graph ring legibility
def test_ring_weights_keep_active_nodes_visible() -> None:
    """
    Active and failed rings are heavier than pending rings at the native line width.
    """
    metrics = LifeCycleDrawingMetrics()
    pending = LifeCycleStyle.of_life_cycle(LifeCycleValues.NOT_STARTED, metrics)
    active = LifeCycleStyle.of_life_cycle(LifeCycleValues.RUNNING, metrics)
    failed = LifeCycleStyle.of_life_cycle(LifeCycleValues.FAILED, metrics)
    succeeded = LifeCycleStyle.of_life_cycle(LifeCycleValues.SUCCEEDED, metrics)

    assert (
        active.width
        == failed.width
        == DRAWING_METRICS.line_width * metrics.active_width_factor
    )
    assert pending.width == DRAWING_METRICS.line_width * metrics.pending_width_factor
    assert succeeded.width == DRAWING_METRICS.line_width * metrics.default_width_factor
    assert active.width > succeeded.width > pending.width
    assert pending.dashes == metrics.pending_dashes
    assert active.dashes is None


@pytest.mark.parametrize("state", [LifeCycleValues.PAUSED, LifeCycleValues.INTERRUPTED])
def test_a_suspended_ring_is_distinct_from_a_pending_ring(
    state: LifeCycleValues,
) -> None:
    """
    Suspended execution has a heavier ring with longer dashes than pending execution.

    :param state: The suspended lifecycle to present.
    """
    metrics = LifeCycleDrawingMetrics()
    pending = LifeCycleStyle.of_life_cycle(LifeCycleValues.NOT_STARTED, metrics)
    suspended = LifeCycleStyle.of_life_cycle(state, metrics)

    assert suspended.dashes == metrics.suspended_dashes
    assert suspended.dashes != pending.dashes
    assert suspended.width > pending.width
