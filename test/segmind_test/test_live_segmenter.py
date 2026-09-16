"""
Tests for watching a world with SegMind on a thread of its own while something else, such
as a plan, changes the world.
"""

from __future__ import annotations

import json
import threading
import time
from collections import Counter

from krrood.adapters.json_serializer import from_json, to_json
from segmind.datastructures.events import (
    StopTranslationEvent,
    SupportEvent,
    TranslationEvent,
)
from segmind.detectors.atomic_event_detectors_nodes import (
    StopTranslationDetector,
    TranslationDetector,
)
from segmind.live_segmenter import (
    EVENT_COMBINING_DETECTOR_TYPES,
    OBJECT_DETECTOR_TYPES,
    EventRecord,
    LiveSegmenter,
    SegmindEnvironmentVariable,
)
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix

from .dataset.detector_counting_its_ticks import DetectorCountingItsTicks
from .dataset.detector_taking_its_time import DetectorTakingItsTime
from .test_detectors.test_detection_without_casadi import (  # noqa: F401 (fixture)
    RESTING_ON_THE_TABLE,
    milk_in_the_apartment,
)

TICK_TIMEOUT = 10.0
"""
Seconds a test waits for the watching thread to have ticked or detected something.
"""

QUIET_PERIOD = 0.3
"""
Seconds a test holds the world still to check that the watching thread does not tick.
"""

MOVED_ALONG_X = 0.2
"""
How far a test moves the milk while it is watched.
"""

CHANGES_WHILE_WATCHED = 20
"""
How many changes a test makes to the world while it is watched.
"""


# %% watching on a thread of its own


def test_detectors_are_ticked_on_a_thread_of_their_own(milk_in_the_apartment):
    world, _, _ = milk_in_the_apartment
    detector = DetectorCountingItsTicks()

    with LiveSegmenter(world=world, detectors=[detector]):
        ticked = detector.ticked.wait(TICK_TIMEOUT)

    assert ticked
    assert threading.get_ident() not in detector.tick_threads


def test_a_body_moved_while_watched_is_seen_translating(milk_in_the_apartment):
    world, milk, _ = milk_in_the_apartment
    rest_x, rest_y, rest_z = RESTING_ON_THE_TABLE
    milk.parent_connection.origin = HomogeneousTransformationMatrix.from_xyz_rpy(
        rest_x, rest_y, rest_z, reference_frame=milk.parent_connection.parent
    )
    translated = threading.Event()
    segmenter = LiveSegmenter(
        world=world, detectors=[TranslationDetector(tracked_object=milk)]
    )
    segmenter.event_logger.add_callback(
        TranslationEvent, lambda event: translated.set()
    )

    with segmenter:
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
        for event in segmenter.event_logger.get_events()
        if isinstance(event, TranslationEvent)
    ]
    assert translation.tracked_object is milk


def test_detectors_are_not_ticked_while_the_world_is_being_modified(
    milk_in_the_apartment,
):
    """
    A plan changing the world holds it for the whole change, so the watching thread
    never reads a world half changed.
    """
    world, _, _ = milk_in_the_apartment
    detector = DetectorCountingItsTicks()

    with LiveSegmenter(world=world, detectors=[detector]):
        detector.ticked.wait(TICK_TIMEOUT)
        with world.modify_world():
            ticks_when_the_change_began = detector.ticks
            time.sleep(QUIET_PERIOD)
            ticks_when_the_change_ended = detector.ticks

    assert ticks_when_the_change_ended == ticks_when_the_change_began


def test_changing_the_world_is_not_held_up_by_watching_it(milk_in_the_apartment):
    """
    Whatever changes the world, such as a plan, gets the world between two ticks, however
    slow a tick is.
    """
    world, milk, _ = milk_in_the_apartment
    rest_x, rest_y, rest_z = RESTING_ON_THE_TABLE
    detector = DetectorTakingItsTime()

    with LiveSegmenter(world=world, detectors=[detector]):
        started = time.monotonic()
        for step in range(CHANGES_WHILE_WATCHED):
            milk.parent_connection.origin = (
                HomogeneousTransformationMatrix.from_xyz_rpy(
                    rest_x + 0.01 * step,
                    rest_y,
                    rest_z,
                    reference_frame=milk.parent_connection.parent,
                )
            )
        seconds_taken = time.monotonic() - started

    assert seconds_taken < CHANGES_WHILE_WATCHED * (
        detector.seconds_per_tick + TICK_TIMEOUT / CHANGES_WHILE_WATCHED
    )


def test_a_body_that_stopped_as_watching_ends_is_seen_at_rest(milk_in_the_apartment):
    """
    Detecting that something came to rest takes ticks of a still world, so a run that
    ends right after its last motion is still seen ending at rest.
    """
    world, milk, _ = milk_in_the_apartment
    rest_x, rest_y, rest_z = RESTING_ON_THE_TABLE
    milk.parent_connection.origin = HomogeneousTransformationMatrix.from_xyz_rpy(
        rest_x, rest_y, rest_z, reference_frame=milk.parent_connection.parent
    )
    translated = threading.Event()
    segmenter = LiveSegmenter(
        world=world,
        detectors=[
            TranslationDetector(tracked_object=milk),
            StopTranslationDetector(tracked_object=milk),
        ],
    )
    segmenter.event_logger.add_callback(
        TranslationEvent, lambda event: translated.set()
    )

    with segmenter:
        milk.parent_connection.origin = HomogeneousTransformationMatrix.from_xyz_rpy(
            rest_x + MOVED_ALONG_X,
            rest_y,
            rest_z,
            reference_frame=milk.parent_connection.parent,
        )
        translated.wait(TICK_TIMEOUT)

    assert StopTranslationEvent in {
        type(event) for event in segmenter.event_logger.get_events()
    }


def test_stopping_ends_the_watching_thread(milk_in_the_apartment):
    world, _, _ = milk_in_the_apartment
    detector = DetectorCountingItsTicks()
    segmenter = LiveSegmenter(world=world, detectors=[detector])

    with segmenter:
        detector.ticked.wait(TICK_TIMEOUT)

    assert not segmenter.is_alive()


# %% watching the objects a plan handles


def _milk_resting_on_the_table(milk) -> None:
    rest_x, rest_y, rest_z = RESTING_ON_THE_TABLE
    milk.parent_connection.origin = HomogeneousTransformationMatrix.from_xyz_rpy(
        rest_x, rest_y, rest_z, reference_frame=milk.parent_connection.parent
    )


def test_watching_bodies_ticks_every_object_detector_for_each_body(
    milk_in_the_apartment,
):
    world, milk, box = milk_in_the_apartment

    segmenter = LiveSegmenter.watching(world, [milk, box])

    tracked = Counter(
        (type(detector), detector.tracked_object)
        for detector in segmenter.detectors
        if type(detector) in OBJECT_DETECTOR_TYPES
    )
    assert tracked == Counter(
        {
            (detector_type, body): 1
            for detector_type in OBJECT_DETECTOR_TYPES
            for body in (milk, box)
        }
    )


def test_watching_bodies_combines_their_events_once_for_all_of_them(
    milk_in_the_apartment,
):
    world, milk, box = milk_in_the_apartment

    segmenter = LiveSegmenter.watching(world, [milk, box])

    assert Counter(
        type(detector)
        for detector in segmenter.detectors
        if type(detector) in EVENT_COMBINING_DETECTOR_TYPES
    ) == Counter(EVENT_COMBINING_DETECTOR_TYPES)


# %% handing detected events over


def test_an_event_record_names_the_event_and_its_bodies(milk_in_the_apartment):
    _, milk, box = milk_in_the_apartment

    record = EventRecord.of(SupportEvent(tracked_object=milk, with_object=box))

    assert record == EventRecord(
        event_type=SupportEvent.__name__,
        tracked_object=milk.name.name,
        with_object=box.name.name,
    )


def test_an_event_record_survives_json():
    record = EventRecord(
        event_type=TranslationEvent.__name__,
        tracked_object="milk.stl",
        with_object=None,
    )

    assert from_json(json.loads(json.dumps(to_json(record)))) == record


def test_written_events_read_back_as_their_records(milk_in_the_apartment, tmp_path):
    world, milk, _ = milk_in_the_apartment
    _milk_resting_on_the_table(milk)
    segmenter = LiveSegmenter.watching(world, [milk])
    segmenter.tick()
    events_file = tmp_path / "events.json"

    segmenter.write_event_records(events_file)

    records = EventRecord.read_all(events_file)
    assert records == [
        EventRecord.of(event) for event in segmenter.event_logger.get_events()
    ]
    assert SupportEvent.__name__ in {record.event_type for record in records}


def test_events_are_written_where_the_environment_asks(
    milk_in_the_apartment, tmp_path, monkeypatch
):
    world, milk, _ = milk_in_the_apartment
    _milk_resting_on_the_table(milk)
    segmenter = LiveSegmenter.watching(world, [milk])
    segmenter.tick()
    events_file = tmp_path / "requested.json"
    monkeypatch.setenv(SegmindEnvironmentVariable.EVENTS_FILE, str(events_file))

    segmenter.write_event_records_where_requested()

    assert EventRecord.read_all(events_file) == [
        EventRecord.of(event) for event in segmenter.event_logger.get_events()
    ]
