"""
Tests for the live event dashboard: the page it serves, the events it hands a browser
at once and as they come, starting and stopping it, and refusing to run without flask.
"""

from __future__ import annotations

import importlib
import importlib.util
import json
import urllib.request

import pytest

from segmind.exceptions import DashboardNeedsFlask, OptionalDependency

pytest.importorskip(OptionalDependency.FLASK)

import segmind.dashboard
from segmind.dashboard.server import (
    DashboardAddress,
    DashboardRoute,
    LiveEventDashboard,
)
from segmind.datastructures.events import GraspEvent, PickUpEvent
from segmind.event_feed import EventFeed, EventRow, FeedField
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.world_description.world_entity import Body

ANY_FREE_PORT = 0
"""
The port that asks the operating system for any free one.
"""


def _grasp_and_pick_up() -> list:
    shape, tool = Body(name=PrefixedName("cube")), Body(name=PrefixedName("tool"))
    return [
        GraspEvent(tracked_object=shape, with_object=tool),
        PickUpEvent(tracked_object=shape),
    ]


def _dashboard_over(events: list) -> LiveEventDashboard:
    feed = EventFeed()
    feed.receive(events)
    return LiveEventDashboard(feed=feed, address=DashboardAddress(port=ANY_FREE_PORT))


# %% what it serves


def test_the_events_route_hands_over_every_event_so_far():
    events = _grasp_and_pick_up()
    dashboard = _dashboard_over(events)

    response = dashboard.app.test_client().get(DashboardRoute.EVENTS)

    assert response.get_json() == [EventRow.of(event).to_json() for event in events]


def test_the_stream_starts_with_the_events_so_far():
    events = _grasp_and_pick_up()
    dashboard = _dashboard_over(events)

    response = dashboard.app.test_client().get(
        DashboardRoute.EVENT_STREAM, buffered=False
    )
    first_message = next(iter(response.response)).decode()
    response.close()

    assert first_message == f"data: {json.dumps(EventRow.of(events[0]).to_json())}\n\n"


def test_the_page_reads_every_field_an_event_is_shown_by():
    dashboard = _dashboard_over([])

    page = dashboard.app.test_client().get(DashboardRoute.PAGE).get_data(as_text=True)

    assert all(field.value in page for field in FeedField)
    assert DashboardRoute.EVENT_STREAM.value in page


# %% running it


def test_a_started_dashboard_answers_until_stopped():
    events = _grasp_and_pick_up()
    dashboard = _dashboard_over(events)

    dashboard.start()
    url = f"http://{dashboard.address.host}:{dashboard.port}{DashboardRoute.EVENTS}"
    with urllib.request.urlopen(url) as response:
        served = json.loads(response.read())
    dashboard.stop()

    assert served == [EventRow.of(event).to_json() for event in events]


def test_a_dashboard_watching_a_monitor_listens_to_it():
    monitor_listeners = []

    class HasListeners:
        listeners = monitor_listeners

    dashboard = LiveEventDashboard.watching(
        HasListeners(), DashboardAddress(port=ANY_FREE_PORT)
    )

    assert monitor_listeners == [dashboard.feed]


# %% without flask


def test_the_dashboard_refuses_to_load_without_flask(monkeypatch):
    real_find_spec = importlib.util.find_spec
    monkeypatch.setattr(
        importlib.util,
        "find_spec",
        lambda name, *args: (
            None if name == OptionalDependency.FLASK else real_find_spec(name, *args)
        ),
    )

    with pytest.raises(DashboardNeedsFlask):
        importlib.reload(segmind.dashboard)

    monkeypatch.undo()
    importlib.reload(segmind.dashboard)
