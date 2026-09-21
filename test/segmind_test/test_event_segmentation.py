"""
Tests for a run stating what it wants watched, what it wants detected and whether it
shows the events while it goes on.
"""

from __future__ import annotations

import threading
from collections import Counter

from segmind.datastructures.events import TranslationEvent
from segmind.detector_selection import DetectorSelection
from segmind.detectors.coarse_event_detector_nodes import PickUpDetector
from segmind.event_segmentation import EventSegmentation
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix

from .test_detectors.test_detection_without_casadi import (  # noqa: F401 (fixture)
    RESTING_ON_THE_TABLE,
    milk_in_the_apartment,
)

TICK_TIMEOUT = 10.0
"""
Seconds a test waits for the watching thread to have detected something.
"""

MOVED_ALONG_X = 0.2
"""
How far a test moves the milk while the run is watched.
"""


def _stand_the_milk_on_the_table(milk) -> None:
    """
    Put the milk where it rests on the table, so a move of it is a move from rest.
    """
    rest_x, rest_y, rest_z = RESTING_ON_THE_TABLE
    milk.parent_connection.origin = HomogeneousTransformationMatrix.from_xyz_rpy(
        rest_x, rest_y, rest_z, reference_frame=milk.parent_connection.parent
    )


def test_a_run_watches_the_bodies_it_names(milk_in_the_apartment):
    world, milk, box = milk_in_the_apartment

    event_segmentation = EventSegmentation.watching_bodies_named(
        world, (milk.name.name, box.name.name)
    )

    assert event_segmentation.bodies == [milk, box]


def test_a_run_is_given_every_detector_what_it_asks_for_is_read_from(
    milk_in_the_apartment,
):
    """
    A run says what it wants detected; the detectors that is concluded from come with
    it, and the kinds ticked are named once each.
    """
    world, milk, _ = milk_in_the_apartment

    event_segmentation = EventSegmentation(
        world=world, bodies=[milk], detectors=[PickUpDetector]
    )

    assert Counter(event_segmentation.detector_names) == Counter(
        detector_type.__name__
        for detector_type in DetectorSelection.of(PickUpDetector).detector_types
    )


def test_what_happens_while_a_run_is_watched_is_detected(milk_in_the_apartment):
    world, milk, _ = milk_in_the_apartment
    _stand_the_milk_on_the_table(milk)
    event_segmentation = EventSegmentation(world=world, bodies=[milk])
    translated = threading.Event()
    event_segmentation.segmenter.event_logger.add_callback(
        TranslationEvent, lambda event: translated.set()
    )

    with event_segmentation:
        rest_x, rest_y, rest_z = RESTING_ON_THE_TABLE
        milk.parent_connection.origin = HomogeneousTransformationMatrix.from_xyz_rpy(
            rest_x + MOVED_ALONG_X,
            rest_y,
            rest_z,
            reference_frame=milk.parent_connection.parent,
        )
        seen = translated.wait(TICK_TIMEOUT)

    assert seen
    [translation] = [
        event
        for event in event_segmentation.segmenter.event_logger.get_events()
        if isinstance(event, TranslationEvent)
    ]
    assert translation.tracked_object is milk


def test_a_run_showing_its_events_serves_them_only_while_it_goes_on(
    milk_in_the_apartment,
):
    world, milk, _ = milk_in_the_apartment
    event_segmentation = EventSegmentation(
        world=world, bodies=[milk], show_live_events=True
    )

    with event_segmentation:
        dashboard = event_segmentation.dashboard
        assert dashboard.feed in event_segmentation.segmenter.listeners
        assert dashboard.port > 0

    assert event_segmentation.dashboard is None


def test_a_run_that_shows_nothing_serves_nothing(milk_in_the_apartment):
    world, milk, _ = milk_in_the_apartment

    with EventSegmentation(world=world, bodies=[milk]) as event_segmentation:
        assert event_segmentation.dashboard is None
