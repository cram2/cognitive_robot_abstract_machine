"""
Tests for detectors that report a relation being established and being lost: one
detector of a kind reports both, so asking for the kind is enough.
"""

from __future__ import annotations

import numpy as np

from giskardpy.motion_statechart.context import MotionStatechartContext
from segmind.datastructures.events import (
    ContactEvent,
    ContainmentEvent,
    LossOfContactEvent,
    LossOfContainmentEvent,
    LossOfSupportEvent,
    RotationEvent,
    StopRotationEvent,
    StopTranslationEvent,
    SupportEvent,
    TranslationEvent,
)
from segmind.detectors.atomic_event_detectors_nodes import (
    ContactDetector,
    RotationDetector,
    TranslationDetector,
)
from segmind.detectors.base import AbstractDetector, SegmindContext
from segmind.detectors.spatial_relation_detector_nodes import (
    ContainmentDetector,
    SupportDetector,
)
from segmind.episode_segmenter import EpisodeSegmenterExecutor
from segmind.statecharts.segmind_statechart import SegmindStatechart
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.world_entity import Body

from ..conftest import RESTING_ON_THE_TABLE, WHERE_THE_MILK_STOOD

MOVING_TICKS = 5
"""
How many ticks a test keeps a body moving, which is more than a motion detector's window.
"""


def _ticking(world: World, detector: AbstractDetector):
    """
    :return: An executor ticking only ``detector``, and the context it logs to.
    """
    executor = EpisodeSegmenterExecutor(context=MotionStatechartContext(world=world))
    segmind_context = executor.context.require_extension(SegmindContext)
    executor.compile(SegmindStatechart().build_statechart([detector]))
    executor.tick()
    return executor, segmind_context


def _events_of(segmind_context: SegmindContext, event_type):
    return [
        event
        for event in segmind_context.logger.get_events()
        if isinstance(event, event_type)
    ]


def _place(body: Body, x: float, y: float, z: float, **orientation) -> None:
    body.parent_connection.origin = HomogeneousTransformationMatrix.from_xyz_rpy(
        x, y, z, reference_frame=body.parent_connection.parent, **orientation
    )


def _put_back(milk: Body) -> None:
    x, y, z = WHERE_THE_MILK_STOOD
    _place(milk, x, y, z, yaw=np.pi)


def test_the_contact_detector_reports_gaining_and_losing_a_contact(
    milk_in_the_apartment,
):
    world, milk, box = milk_in_the_apartment
    executor, segmind_context = _ticking(world, ContactDetector())

    _place(milk, box.global_pose.x, box.global_pose.y, box.global_pose.z)
    executor.tick()
    assert len(_events_of(segmind_context, ContactEvent)) == 1
    assert _events_of(segmind_context, LossOfContactEvent) == []

    _place(milk, 0, 0, 1)
    executor.tick()

    [lost] = _events_of(segmind_context, LossOfContactEvent)
    assert lost.tracked_object is milk
    _put_back(milk)


def test_the_support_detector_reports_gaining_and_losing_a_support(
    milk_in_the_apartment,
):
    world, milk, _ = milk_in_the_apartment
    executor, segmind_context = _ticking(world, SupportDetector())

    _place(milk, *RESTING_ON_THE_TABLE)
    executor.tick()
    assert len(_events_of(segmind_context, SupportEvent)) == 1
    assert _events_of(segmind_context, LossOfSupportEvent) == []

    _place(milk, 0, 0, 1)
    executor.tick()

    [lost] = _events_of(segmind_context, LossOfSupportEvent)
    assert lost.tracked_object is milk
    _put_back(milk)


def test_the_containment_detector_reports_gaining_and_losing_a_containment(
    milk_in_the_apartment,
):
    world, milk, box = milk_in_the_apartment
    executor, segmind_context = _ticking(world, ContainmentDetector())

    _place(milk, box.global_pose.x, box.global_pose.y, box.global_pose.z)
    executor.tick()
    assert len(_events_of(segmind_context, ContainmentEvent)) == 1
    assert _events_of(segmind_context, LossOfContainmentEvent) == []

    _place(milk, 0, 0, 1)
    executor.tick()

    [lost] = _events_of(segmind_context, LossOfContainmentEvent)
    assert lost.tracked_object is milk
    _put_back(milk)


def test_the_translation_detector_reports_starting_and_stopping(
    milk_in_the_apartment,
):
    world, milk, _ = milk_in_the_apartment
    executor, segmind_context = _ticking(world, TranslationDetector())

    for step in range(MOVING_TICKS):
        _place(milk, 1 + step * 0.1, -3, 0.25)
        executor.tick()
    assert len(_events_of(segmind_context, TranslationEvent)) == 1
    assert _events_of(segmind_context, StopTranslationEvent) == []

    for _ in range(MOVING_TICKS):
        executor.tick()

    [stopped] = _events_of(segmind_context, StopTranslationEvent)
    assert stopped.tracked_object is milk
    _put_back(milk)


def test_the_rotation_detector_reports_starting_and_stopping(milk_in_the_apartment):
    world, milk, _ = milk_in_the_apartment
    executor, segmind_context = _ticking(world, RotationDetector())

    for step in range(MOVING_TICKS):
        _place(milk, 0, 0, 0, roll=step * 0.1)
        executor.tick()
    assert len(_events_of(segmind_context, RotationEvent)) == 1
    assert _events_of(segmind_context, StopRotationEvent) == []

    for _ in range(MOVING_TICKS):
        executor.tick()

    [stopped] = _events_of(segmind_context, StopRotationEvent)
    assert stopped.tracked_object is milk
    _put_back(milk)
