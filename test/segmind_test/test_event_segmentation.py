"""
Tests for a run stating what it wants watched and what it wants detected, and for
SegMind watching the world on a thread of its own while something else, such as a plan,
changes it.
"""

from __future__ import annotations

import threading
import time
from collections import Counter
from dataclasses import dataclass, field

import pytest
from giskardpy.motion_statechart.context import MotionStatechartContext
from krrood.exceptions import DataclassException
from typing_extensions import List

from segmind.datastructures.events import (
    DetectionEvent,
    StopTranslationEvent,
    TranslationEvent,
)
from segmind.detector_selection import DetectorSelection
from segmind.detectors.atomic_event_detectors_nodes import TranslationDetector
from segmind.detectors.base import (
    AbstractDetector,
    EventCombiningDetector,
    SegmindContext,
)
from segmind.detectors.coarse_event_detector_nodes import PickUpDetector
from segmind.event_segmentation import Segmind
from segmind.exceptions import NoSemanticAnnotationToWatch
from semantic_digital_twin.semantic_annotations.semantic_annotations import Food, Milk
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.world_description.world_entity import Body

from .conftest import RESTING_ON_THE_TABLE

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


def _stand_the_milk_on_the_table(milk: Body) -> None:
    """
    Put the milk where it rests on the table, so a move of it is a move from rest.
    """
    rest_x, rest_y, rest_z = RESTING_ON_THE_TABLE
    milk.parent_connection.origin = HomogeneousTransformationMatrix.from_xyz_rpy(
        rest_x, rest_y, rest_z, reference_frame=milk.parent_connection.parent
    )


# %% what a run asks to be watched


def test_a_run_watches_the_bodies_of_every_annotation_of_a_type_it_names(
    milk_annotated_in_the_apartment,
):
    world, milk, _ = milk_annotated_in_the_apartment

    segmentation = Segmind.create_for_semantic_annotation_types(world, [Milk])

    assert segmentation.watched_bodies == milk.bodies


def test_a_body_annotated_by_two_of_the_types_named_is_watched_once(
    milk_annotated_in_the_apartment,
):
    world, milk, _ = milk_annotated_in_the_apartment

    segmentation = Segmind.create_for_semantic_annotation_types(world, [Milk, Food])

    assert segmentation.watched_bodies == milk.bodies


def test_a_run_cannot_watch_a_type_the_world_holds_no_annotation_of(
    milk_in_the_apartment,
):
    world, _, _ = milk_in_the_apartment

    with pytest.raises(NoSemanticAnnotationToWatch) as raised:
        Segmind.create_for_semantic_annotation_types(world, [Milk])

    assert raised.value.semantic_annotation_type is Milk


def test_a_type_the_world_holds_no_annotation_of_is_reported_like_any_cram_error(
    milk_in_the_apartment,
):
    world, _, _ = milk_in_the_apartment

    with pytest.raises(DataclassException):
        Segmind.create_for_semantic_annotation_types(world, [Milk])


def test_a_run_is_given_every_detector_what_it_asks_for_is_read_from(
    milk_in_the_apartment,
):
    """
    A run says what it wants detected; the detectors that is concluded from come with
    it, and the kinds ticked are named once each.
    """
    world, milk, _ = milk_in_the_apartment

    segmentation = Segmind.create_for_bodies(
        world, [milk], detector_types=[PickUpDetector]
    )

    assert Counter(segmentation.detector_names) == Counter(
        detector_type.__name__
        for detector_type in DetectorSelection.of(PickUpDetector).detector_types
    )


def test_what_happens_while_a_run_is_watched_is_detected(milk_in_the_apartment):
    world, milk, _ = milk_in_the_apartment
    _stand_the_milk_on_the_table(milk)
    segmentation = Segmind.create_for_bodies(world, [milk])
    translated = threading.Event()
    segmentation.event_logger.add_callback(
        TranslationEvent, lambda event: translated.set()
    )

    with segmentation:
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
        for event in segmentation.event_logger.get_events()
        if isinstance(event, TranslationEvent)
    ]
    assert translation.tracked_object is milk


# %% detectors that detect nothing, to watch the watching itself


@dataclass(eq=False, repr=False)
class DetectorCountingItsTicks(AbstractDetector):
    """
    Detects nothing, and records how often and on which thread it was ticked.
    """

    ticks: int = field(default=0, init=False)
    """
    How often this detector was ticked.
    """

    tick_threads: List[int] = field(default_factory=list, init=False)
    """
    The identifier of the thread each tick ran on, in order.
    """

    ticked: threading.Event = field(default_factory=threading.Event, init=False)
    """
    Set once this detector has been ticked.
    """

    def update_context_and_events(
        self,
        context: MotionStatechartContext,
        segmind_context: SegmindContext,
        tracked_objects: List[Body],
    ) -> List[DetectionEvent]:
        self.ticks += 1
        self.tick_threads.append(threading.get_ident())
        self.ticked.set()
        return []


@dataclass(eq=False, repr=False)
class DetectorTakingItsTime(AbstractDetector):
    """
    Detects nothing, and spends a fixed time on every tick.
    """

    seconds_per_tick: float = 0.02
    """
    How long each tick takes.
    """

    def update_context_and_events(
        self,
        context: MotionStatechartContext,
        segmind_context: SegmindContext,
        tracked_objects: List[Body],
    ) -> List[DetectionEvent]:
        time.sleep(self.seconds_per_tick)
        return []


# %% watching on a thread of its own


def test_detectors_are_ticked_on_a_thread_of_their_own(milk_in_the_apartment):
    world, _, _ = milk_in_the_apartment
    detector = DetectorCountingItsTicks()

    with Segmind(world=world, detectors=[detector]):
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
    segmentation = Segmind(
        world=world, detectors=[TranslationDetector(tracked_object=milk)]
    )
    segmentation.event_logger.add_callback(
        TranslationEvent, lambda event: translated.set()
    )

    with segmentation:
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
        for event in segmentation.event_logger.get_events()
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

    with Segmind(world=world, detectors=[detector]):
        detector.ticked.wait(TICK_TIMEOUT)
        with world.modify_world():
            ticks_when_the_change_began = detector.ticks
            time.sleep(QUIET_PERIOD)
            ticks_when_the_change_ended = detector.ticks

    assert ticks_when_the_change_ended == ticks_when_the_change_began


def test_changing_the_world_is_not_held_up_by_watching_it(milk_in_the_apartment):
    """
    Whatever changes the world, such as a plan, gets the world between two ticks,
    however slow a tick is.
    """
    world, milk, _ = milk_in_the_apartment
    rest_x, rest_y, rest_z = RESTING_ON_THE_TABLE
    detector = DetectorTakingItsTime()

    with Segmind(world=world, detectors=[detector]):
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
    segmentation = Segmind(
        world=world,
        detectors=[TranslationDetector(tracked_object=milk)],
    )
    segmentation.event_logger.add_callback(
        TranslationEvent, lambda event: translated.set()
    )

    with segmentation:
        milk.parent_connection.origin = HomogeneousTransformationMatrix.from_xyz_rpy(
            rest_x + MOVED_ALONG_X,
            rest_y,
            rest_z,
            reference_frame=milk.parent_connection.parent,
        )
        translated.wait(TICK_TIMEOUT)

    assert StopTranslationEvent in {
        type(event) for event in segmentation.event_logger.get_events()
    }


def test_stopping_ends_the_watching_thread(milk_in_the_apartment):
    world, _, _ = milk_in_the_apartment
    detector = DetectorCountingItsTicks()
    segmentation = Segmind(world=world, detectors=[detector])

    with segmentation:
        detector.ticked.wait(TICK_TIMEOUT)

    assert not segmentation.is_watching


# %% watching the objects a plan handles


def test_watching_bodies_ticks_every_object_detector_for_each_body(
    milk_in_the_apartment,
):
    world, milk, box = milk_in_the_apartment
    object_detector_types = [
        detector_type
        for detector_type in DetectorSelection.of_every_kind().detector_types
        if not issubclass(detector_type, EventCombiningDetector)
    ]

    segmentation = Segmind.create_for_bodies(world, [milk, box])

    tracked = Counter(
        (type(detector), detector.tracked_object)
        for detector in segmentation.detectors
        if type(detector) in object_detector_types
    )
    assert tracked == Counter(
        {
            (detector_type, body): 1
            for detector_type in object_detector_types
            for body in (milk, box)
        }
    )


def test_watching_bodies_combines_their_events_once_for_all_of_them(
    milk_in_the_apartment,
):
    world, milk, box = milk_in_the_apartment
    event_combining_detector_types = [
        detector_type
        for detector_type in DetectorSelection.of_every_kind().detector_types
        if issubclass(detector_type, EventCombiningDetector)
    ]

    segmentation = Segmind.create_for_bodies(world, [milk, box])

    assert Counter(
        type(detector)
        for detector in segmentation.detectors
        if type(detector) in event_combining_detector_types
    ) == Counter(event_combining_detector_types)


def test_watching_for_what_is_asked_watches_each_body_with_what_that_is_read_from(
    milk_in_the_apartment,
):
    """
    Asking for pick-ups is enough: each body is watched by every kind of detector a
    pick-up is read from, and the pick-ups themselves are concluded once for all of
    them.
    """
    world, milk, box = milk_in_the_apartment

    segmentation = Segmind.create_for_bodies(
        world, [milk, box], detector_types=[PickUpDetector]
    )

    chosen = DetectorSelection.of(PickUpDetector).detector_types
    assert Counter(
        (type(detector), detector.tracked_object) for detector in segmentation.detectors
    ) == Counter(
        {
            **{
                (detector_type, body): 1
                for detector_type in chosen
                if detector_type is not PickUpDetector
                for body in (milk, box)
            },
            (PickUpDetector, None): 1,
        }
    )
