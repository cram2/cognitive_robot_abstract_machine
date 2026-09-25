"""
Tests for detectors each watching one of several bodies: a detector judges only the body it
watches, so one body's relations are never reported lost by the detector of another.
"""

from __future__ import annotations

from giskardpy.motion_statechart.context import MotionStatechartContext
from typing_extensions import List, Type

from segmind.datastructures.events import (
    ContactEvent,
    ContainmentEvent,
    DetectionEvent,
    LossOfContactEvent,
    LossOfContainmentEvent,
    LossOfSupportEvent,
    SupportEvent,
)
from segmind.detectors.atomic_event_detectors_nodes import (
    ContactDetector,
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

from ..conftest import RESTING_ON_THE_TABLE

TICKS_WITHOUT_ANYTHING_MOVING = 4
"""
How many ticks a test runs while nothing in the world moves.
"""


def _ticked_while_nothing_moves(
    world: World, detector_types: List[Type[AbstractDetector]], watched: List[Body]
) -> SegmindContext:
    """
    Tick every detector type for every watched body a few times, with nothing moving.

    :return: The context holding the events detected.
    """
    executor = EpisodeSegmenterExecutor(context=MotionStatechartContext(world=world))
    executor.compile(
        SegmindStatechart().build_statechart(
            [
                detector_type(tracked_object=body)
                for body in watched
                for detector_type in detector_types
            ]
        )
    )
    for _ in range(TICKS_WITHOUT_ANYTHING_MOVING):
        executor.tick()
    return executor.context.require_extension(SegmindContext)


def _events_of(
    segmind_context: SegmindContext, event_type: Type[DetectionEvent], body: Body
) -> List[DetectionEvent]:
    return [
        event
        for event in segmind_context.logger.get_events()
        if type(event) is event_type and event.tracked_object is body
    ]


def _place(body: Body, x: float, y: float, z: float) -> None:
    body.parent_connection.origin = HomogeneousTransformationMatrix.from_xyz_rpy(
        x, y, z, reference_frame=body.parent_connection.parent
    )


# %% losses are judged by the detector watching the body


def test_a_contact_that_lasts_is_not_reported_lost_by_another_bodys_detector(
    milk_in_the_apartment,
):
    world, milk, box = milk_in_the_apartment
    box_x, box_y, box_z = box.global_pose.to_position().to_np()[:3]
    _place(milk, box_x, box_y, box_z)

    segmind_context = _ticked_while_nothing_moves(world, [ContactDetector], [milk, box])

    assert len(_events_of(segmind_context, ContactEvent, milk)) == 1
    assert _events_of(segmind_context, LossOfContactEvent, milk) == []


def test_a_support_that_lasts_is_not_reported_lost_by_another_bodys_detector(
    milk_in_the_apartment,
):
    world, milk, box = milk_in_the_apartment
    _place(milk, *RESTING_ON_THE_TABLE)

    segmind_context = _ticked_while_nothing_moves(world, [SupportDetector], [milk, box])

    assert len(_events_of(segmind_context, SupportEvent, milk)) == 1
    assert _events_of(segmind_context, LossOfSupportEvent, milk) == []


def test_a_containment_that_lasts_is_not_reported_lost_by_another_bodys_detector(
    milk_in_the_apartment,
):
    world, milk, box = milk_in_the_apartment
    box_x, box_y, box_z = box.global_pose.to_position().to_np()[:3]
    _place(milk, box_x, box_y, box_z)

    segmind_context = _ticked_while_nothing_moves(
        world, [ContainmentDetector], [milk, box]
    )

    assert len(_events_of(segmind_context, ContainmentEvent, milk)) == 1
    assert _events_of(segmind_context, LossOfContainmentEvent, milk) == []
