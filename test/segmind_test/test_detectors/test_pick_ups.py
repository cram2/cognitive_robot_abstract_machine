"""
Tests for concluding pick-ups from the translations and losses of support already
detected: one pick-up for each time an object is lifted.
"""

from __future__ import annotations

from datetime import datetime, timedelta

from typing_extensions import List

from segmind.datastructures.events import (
    DetectionEvent,
    LossOfSupportEvent,
    PickUpEvent,
    TranslationEvent,
)
from segmind.detectors.base import SegmindContext
from segmind.detectors.coarse_event_detector_nodes import PickUpDetector

from .test_detection_without_casadi import milk_in_the_apartment  # noqa: F401

LIFTS_APART = timedelta(minutes=5)
"""
How far apart in time a test lifts the same object twice; farther than any interaction
is allowed to span.
"""


def _logged(events: List[DetectionEvent]) -> SegmindContext:
    """
    :return: A context whose logger holds ``events``.
    """
    segmind_context = SegmindContext()
    for event in events:
        segmind_context.logger.log_event(event, segmind_context.tracker_registry)
    return segmind_context


def _translation_of(body, at: datetime) -> TranslationEvent:
    pose = body.numeric_global_pose
    return TranslationEvent(
        tracked_object=body, start_pose=pose, current_pose=pose, timestamp=at
    )


def _pick_ups_concluded_from(segmind_context: SegmindContext) -> List[DetectionEvent]:
    return [
        event
        for event in PickUpDetector().update_context_and_events(
            None, segmind_context, []
        )
        if isinstance(event, PickUpEvent)
    ]


def test_losing_several_supports_in_one_lift_is_one_pick_up(milk_in_the_apartment):
    world, milk, box = milk_in_the_apartment
    lifted_at = datetime.now()
    segmind_context = _logged(
        [
            _translation_of(milk, lifted_at),
            LossOfSupportEvent(
                tracked_object=milk, with_object=box, timestamp=lifted_at
            ),
            LossOfSupportEvent(
                tracked_object=milk,
                with_object=world.get_body_by_name("box_2"),
                timestamp=lifted_at,
            ),
        ]
    )

    assert len(_pick_ups_concluded_from(segmind_context)) == 1


def test_each_lift_of_the_same_object_is_a_pick_up_of_its_own(milk_in_the_apartment):
    _, milk, box = milk_in_the_apartment
    first_lift = datetime.now()
    second_lift = first_lift + LIFTS_APART
    segmind_context = _logged(
        [
            _translation_of(milk, first_lift),
            LossOfSupportEvent(
                tracked_object=milk, with_object=box, timestamp=first_lift
            ),
            _translation_of(milk, second_lift),
            LossOfSupportEvent(
                tracked_object=milk, with_object=box, timestamp=second_lift
            ),
        ]
    )

    assert len(_pick_ups_concluded_from(segmind_context)) == 2


def test_one_loss_of_support_is_the_pick_up_of_one_translation(milk_in_the_apartment):
    """
    An object carried in starts and stops is translated many times after it was lifted,
    but it lost its support only once.
    """
    _, milk, box = milk_in_the_apartment
    lifted_at = datetime.now()
    segmind_context = _logged(
        [
            _translation_of(milk, lifted_at),
            LossOfSupportEvent(
                tracked_object=milk, with_object=box, timestamp=lifted_at
            ),
            _translation_of(milk, lifted_at + timedelta(seconds=1)),
        ]
    )

    assert len(_pick_ups_concluded_from(segmind_context)) == 1
