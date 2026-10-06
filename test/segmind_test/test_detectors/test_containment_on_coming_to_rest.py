"""
Tests for containment being looked for only once an object comes to rest: an object put
into something is set down in it, and one merely carried through something is not in it.
"""

from __future__ import annotations

from giskardpy.motion_statechart.context import MotionStatechartContext
from typing_extensions import List, Tuple, Type

from segmind.datastructures.events import (
    ContainmentEvent,
    DetectionEvent,
    LossOfContainmentEvent,
)
from segmind.detectors.base import SegmindContext
from segmind.detectors.spatial_relation_detector_nodes import (
    ContainmentDetector,
    SupportDetector,
)
from segmind.episode_segmenter import EpisodeSegmenterExecutor
from segmind.statecharts.segmind_statechart import SegmindStatechart
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.world_entity import Body

from ..trays import add_tray, hold_up_in, lift_out_of, set_down_in, world_with_a_box


def _ticking(
    world: World, box: Body
) -> Tuple[EpisodeSegmenterExecutor, SegmindContext]:
    """
    :return: An executor ticking the support and the containment detector for ``box``,
        and the context they record into.
    """
    executor = EpisodeSegmenterExecutor(context=MotionStatechartContext(world=world))
    executor.compile(
        SegmindStatechart().build_statechart(
            [
                SupportDetector(tracked_object=box),
                ContainmentDetector(tracked_object=box),
            ]
        )
    )
    return executor, executor.context.require_extension(SegmindContext)


def _containers_in(
    segmind_context: SegmindContext, event_type: Type[DetectionEvent]
) -> List[Body]:
    """
    :return: What each event of ``event_type`` names the box as being in.
    """
    return [
        event.with_object
        for event in segmind_context.logger.get_events()
        if isinstance(event, event_type)
    ]


def test_an_object_set_down_in_a_container_is_contained_in_it():
    world, box = world_with_a_box()
    tray = add_tray(world, "tray")
    set_down_in(box, tray)
    executor, segmind_context = _ticking(world, box)

    executor.tick()

    assert _containers_in(segmind_context, ContainmentEvent) == [tray]


def test_an_object_held_up_inside_a_container_is_not_contained_in_it():
    """
    Something carried through a container's walls has not been put into it.
    """
    world, box = world_with_a_box()
    tray = add_tray(world, "tray")
    hold_up_in(box, tray)
    executor, segmind_context = _ticking(world, box)

    executor.tick()

    assert _containers_in(segmind_context, ContainmentEvent) == []


def test_an_object_lifted_out_of_its_container_is_no_longer_contained_in_it():
    """
    Leaving a container is seen while the object is still carried, not only once it
    is set down again.
    """
    world, box = world_with_a_box()
    tray = add_tray(world, "tray")
    set_down_in(box, tray)
    executor, segmind_context = _ticking(world, box)
    executor.tick()

    lift_out_of(box, tray)
    executor.tick()

    assert _containers_in(segmind_context, LossOfContainmentEvent) == [tray]
