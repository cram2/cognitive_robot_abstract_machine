"""
Tests for the event feed a dashboard reads: it keeps every event it is told about, a
subscription takes each of them once and in order starting with those received before
it, and an event is reduced to the fields the page shows.
"""

from __future__ import annotations

import numpy as np
import pytest

from segmind.datastructures.events import GraspEvent, PickUpEvent, SupportEvent
from segmind.detectors.spatial_relation_detector_nodes import SupportDetector
from segmind.event_feed import EventFeed, EventRow, FeedField
from segmind.live_segmenter import LiveSegmenter
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.spatial_types.spatial_types import (
    HomogeneousTransformationMatrix,
)
from semantic_digital_twin.world_description.world_entity import Body

TICKS_TO_WATCH = 3
"""
How many ticks a test lets a watched run go on for.
"""


def _bodies() -> tuple[Body, Body]:
    """
    A tracked body and a tool frame body, standing in no world.
    """
    return Body(name=PrefixedName("cube")), Body(name=PrefixedName("tool"))


# %% keeping what it is told


def test_the_snapshot_holds_every_event_received_in_order():
    feed = EventFeed()
    shape, tool = _bodies()
    grasp = GraspEvent(tracked_object=shape, with_object=tool)
    pick_up = PickUpEvent(tracked_object=shape)

    feed.receive([grasp])
    feed.receive([pick_up])

    assert feed.snapshot() == [grasp, pick_up]


# %% reading it as events come


def test_a_late_subscription_takes_the_history_then_live_events():
    feed = EventFeed()
    shape, _ = _bodies()
    before = PickUpEvent(tracked_object=shape)
    feed.receive([before])

    subscription = feed.subscribe()
    after = PickUpEvent(tracked_object=shape)
    feed.receive([after])

    assert subscription.take() == [before, after]


def test_a_subscription_takes_each_event_once():
    feed = EventFeed()
    shape, _ = _bodies()
    feed.receive([PickUpEvent(tracked_object=shape)])
    subscription = feed.subscribe()
    subscription.take()

    assert subscription.take() == []


def test_two_subscriptions_each_take_every_event():
    feed = EventFeed()
    shape, _ = _bodies()
    first, second = feed.subscribe(), feed.subscribe()
    event = PickUpEvent(tracked_object=shape)

    feed.receive([event])

    assert first.take() == [event]
    assert second.take() == [event]


# %% what the page shows of an event


def test_an_event_is_shown_by_what_it_tracks_what_it_is_with_and_when():
    shape, tool = _bodies()
    grasp = GraspEvent(tracked_object=shape, with_object=tool)

    assert EventRow.of(grasp).to_json() == {
        FeedField.TRACKED_OBJECT: str(shape.name),
        FeedField.EVENT_TYPE: GraspEvent.__name__,
        FeedField.WITH_OBJECT: str(tool.name),
        FeedField.TIMESTAMP: grasp.timestamp.isoformat(),
    }


def test_an_event_with_nothing_to_be_with_shows_no_other_object():
    shape, _ = _bodies()

    row = EventRow.of(PickUpEvent(tracked_object=shape))

    assert row.to_json()[FeedField.WITH_OBJECT] is None


# %% fed by a monitor


@pytest.fixture
def milk_resting_on_the_table(_simple_apartment_setup):
    """
    The apartment with its milk standing where it rests, put back where it was afterwards.
    """
    world = _simple_apartment_setup
    milk = world.get_body_by_name("milk.stl")
    milk.parent_connection.origin = HomogeneousTransformationMatrix.from_xyz_rpy(
        -1.7, 0, 0.93, reference_frame=world.root
    )
    yield world, milk
    milk.parent_connection.origin = HomogeneousTransformationMatrix.from_xyz_rpy(
        -1.7, 0, 1.07, yaw=np.pi, reference_frame=world.root
    )


def test_a_feed_listening_to_a_watched_run_holds_what_that_run_detected(
    milk_resting_on_the_table,
):
    world, milk = milk_resting_on_the_table
    feed = EventFeed()
    segmenter = LiveSegmenter(
        world=world,
        detectors=[SupportDetector(tracked_object=milk)],
        listeners=[feed],
    )

    for _ in range(TICKS_TO_WATCH):
        segmenter.tick()

    assert feed.snapshot() == segmenter.event_logger.get_events()
    assert any(isinstance(event, SupportEvent) for event in feed.snapshot())
