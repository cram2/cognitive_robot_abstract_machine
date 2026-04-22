import pytest

from giskardpy.motion_statechart.context import MotionStatechartContext
from segmind.datastructures.events import (
    ContactEvent,
    LossOfContactEvent,
    SupportEvent,
    LossOfSupportEvent,
    LossOfContainmentEvent,
    ContainmentEvent,
    InsertionEvent,
    TranslationEvent,
    StopTranslationEvent,
    PickUpEvent,
    PlacingEvent,
    RotationEvent,
    StopRotationEvent,
)
from segmind.detectors.atomic_event_detectors_nodes import RotationDetector, StopRotationDetector
from segmind.detectors.base import SegmindContext
from segmind.episode_segmenter import EpisodeSegmenterExecutor
from segmind.statecharts.segmind_statechart import SegmindStatechart
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from test.segmind_test import setup_contact_world, setup_support_world


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_executor(world):
    context = MotionStatechartContext(world=world)
    segmind_executor = EpisodeSegmenterExecutor(context=context)
    sc = SegmindStatechart().build_statechart()
    segmind_executor.compile(sc)
    segmind_context = segmind_executor.context.require_extension(SegmindContext)
    return segmind_executor, segmind_context, sc


def events_of(segmind_context, event_type):
    return [e for e in segmind_context.logger.get_events() if isinstance(e, event_type)]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def contact_setup():
    executor, segmind_context, sc = _build_executor(setup_contact_world())
    cylinder = executor.context.world.get_body_by_name("cylinder_body")
    return executor, segmind_context, sc, cylinder


@pytest.fixture
def support_setup():
    executor, segmind_context, sc = _build_executor(setup_support_world())
    world = executor.context.world
    cylinder = world.get_body_by_name("cylinder_body")
    table = world.get_body_by_name("table_body")
    cabinet = world.get_body_by_name("cabinet")
    hole = world.get_body_by_name("hole_body")
    return executor, segmind_context, sc, cylinder, table, cabinet, hole


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_contact_detector(contact_setup):
    executor, segmind_context, _, cylinder = contact_setup

    assert len(events_of(segmind_context, ContactEvent)) == 0

    cylinder.parent_connection.origin = HomogeneousTransformationMatrix.from_xyz_rpy(y=-0.4)
    executor.tick()
    assert len(events_of(segmind_context, ContactEvent)) == 2

    executor.tick()
    assert len(events_of(segmind_context, ContactEvent)) == 2

    cylinder.parent_connection.origin = HomogeneousTransformationMatrix.from_xyz_rpy(z=2)
    executor.tick()
    assert len(events_of(segmind_context, LossOfContactEvent)) == 2

    executor.tick()
    assert len(events_of(segmind_context, LossOfContactEvent)) == 2

    cylinder.parent_connection.origin = HomogeneousTransformationMatrix.from_xyz_rpy(y=-0.4)
    executor.tick()
    assert len(events_of(segmind_context, ContactEvent)) == 4

    cylinder.parent_connection.origin = HomogeneousTransformationMatrix.from_xyz_rpy(z=2)
    executor.tick()
    assert len(events_of(segmind_context, LossOfContactEvent)) == 4


def test_support_detector(support_setup):
    executor, segmind_context, _, cylinder, table, cabinet, _ = support_setup

    assert len(events_of(segmind_context, SupportEvent)) == 0
    assert len(events_of(segmind_context, LossOfSupportEvent)) == 0

    executor.tick()

    cylinder.parent_connection.origin = HomogeneousTransformationMatrix.from_xyz_rpy(
        x=table.global_pose.x,
        y=table.global_pose.y,
        z=table.global_pose.z + 0.2,
    )
    executor.tick()
    executor.tick()
    assert len(events_of(segmind_context, SupportEvent)) == 1

    cylinder.parent_connection.origin = HomogeneousTransformationMatrix.from_xyz_rpy(
        x=cabinet.global_pose.x,
        y=cabinet.global_pose.y,
        z=cabinet.global_pose.z,
    )
    executor.tick()
    executor.tick()
    assert len(events_of(segmind_context, LossOfSupportEvent)) == 1


def test_containment_detector(support_setup):
    executor, segmind_context, _, cylinder, table, cabinet, _ = support_setup

    assert len(events_of(segmind_context, ContactEvent)) == 0
    assert len(events_of(segmind_context, ContainmentEvent)) == 0

    cylinder.parent_connection.origin = HomogeneousTransformationMatrix.from_xyz_rpy(
        x=cabinet.global_pose.x,
        y=cabinet.global_pose.y,
        z=cabinet.global_pose.z,
    )
    executor.tick()
    assert len(events_of(segmind_context, ContainmentEvent)) == 1

    cylinder.parent_connection.origin = HomogeneousTransformationMatrix.from_xyz_rpy(
        x=table.global_pose.x,
        y=table.global_pose.y,
        z=table.global_pose.z + 0.2,
    )
    executor.tick()
    assert len(events_of(segmind_context, LossOfContainmentEvent)) == 1


def test_insertion_detector(support_setup):
    executor, segmind_context, _, cylinder, _, cabinet, hole = support_setup

    assert len(segmind_context.holes) == 1
    assert len(events_of(segmind_context, InsertionEvent)) == 0

    cylinder.parent_connection.origin = HomogeneousTransformationMatrix.from_xyz_rpy(
        x=hole.global_pose.x,
        y=hole.global_pose.y - 0.03,
        z=hole.global_pose.z,
    )
    executor.tick()

    cylinder.parent_connection.origin = HomogeneousTransformationMatrix.from_xyz_rpy(
        x=cabinet.global_pose.x,
        y=cabinet.global_pose.y,
        z=cabinet.global_pose.z,
    )
    executor.tick()

    contact_events_with_holes = [
        e for e in events_of(segmind_context, ContactEvent)
        if e.with_object in segmind_context.holes
    ]

    assert len(events_of(segmind_context, ContainmentEvent)) == 1
    assert len(contact_events_with_holes) == 1
    assert len(events_of(segmind_context, InsertionEvent)) == 1


def test_pickup(support_setup):
    executor, segmind_context, _, cylinder, table, _, _ = support_setup

    cylinder.parent_connection.origin = HomogeneousTransformationMatrix.from_xyz_rpy(
        x=table.global_pose.x,
        y=table.global_pose.y,
        z=table.global_pose.z + 0.2,
    )
    executor.tick()
    executor.tick()
    assert len(events_of(segmind_context, SupportEvent)) == 1

    for i in range(5):
        cylinder.parent_connection.origin = HomogeneousTransformationMatrix.from_xyz_rpy(
            x=table.global_pose.x,
            y=table.global_pose.y,
            z=table.global_pose.z + 0.3 + i * 0.1,
        )
        executor.tick()

    assert len(events_of(segmind_context, TranslationEvent)) >= 1
    assert len(events_of(segmind_context, LossOfSupportEvent)) == 1
    assert len(events_of(segmind_context, PickUpEvent)) == 1


def test_placing(support_setup):
    executor, segmind_context, _, cylinder, table, _, _ = support_setup

    for i in range(5):
        cylinder.parent_connection.origin = HomogeneousTransformationMatrix.from_xyz_rpy(
            x=table.global_pose.x,
            y=table.global_pose.y,
            z=table.global_pose.z + 0.5 - i * 0.05,
        )
        executor.tick()

    assert len(events_of(segmind_context, TranslationEvent)) >= 1

    cylinder.parent_connection.origin = HomogeneousTransformationMatrix.from_xyz_rpy(
        x=table.global_pose.x,
        y=table.global_pose.y,
        z=table.global_pose.z + 0.2,
    )
    for _ in range(5):
        executor.tick()

    assert len(events_of(segmind_context, SupportEvent)) == 1
    assert len(events_of(segmind_context, StopTranslationEvent)) == 1
    assert len(events_of(segmind_context, PlacingEvent)) == 1


def test_translation(contact_setup):
    executor, segmind_context, _, cylinder = contact_setup

    assert len(events_of(segmind_context, TranslationEvent)) == 0

    for i in range(5):
        cylinder.parent_connection.origin = HomogeneousTransformationMatrix.from_xyz_rpy(
            x=1 + i * 0.1, y=-3, z=0.25
        )
        executor.tick()

    assert len(events_of(segmind_context, TranslationEvent)) == 1


def test_stop_translation(contact_setup):
    executor, segmind_context, _, cylinder = contact_setup

    for i in range(5):
        cylinder.parent_connection.origin = HomogeneousTransformationMatrix.from_xyz_rpy(
            x=1 + i * 0.1, y=-3, z=0.25
        )
        executor.tick()

    assert len(events_of(segmind_context, TranslationEvent)) == 1

    for _ in range(5):
        executor.tick()

    assert len(events_of(segmind_context, StopTranslationEvent)) == 1


@pytest.mark.skip(reason="Buggy")
def test_rotation():
    world = setup_contact_world()
    context = MotionStatechartContext(world=world)
    segmind_executor = EpisodeSegmenterExecutor(context=context)
    statechart = SegmindStatechart()
    sc = statechart.build_statechart()
    sc.add_node(RotationDetector())
    segmind_executor.compile(sc)
    segmind_context = segmind_executor.context.require_extension(SegmindContext)

    cylinder = world.get_body_by_name("cylinder_body")
    assert len([i for i in segmind_context.logger.get_events() if isinstance(i, RotationEvent)]) == 0

    for i in range(5):
        cylinder.parent_connection.origin = (
            HomogeneousTransformationMatrix.from_xyz_rpy(x=1, y=-3, z=0.25, roll=i*0.1)
        )
        segmind_executor.tick()

    assert len([i for i in segmind_context.logger.get_events() if isinstance(i, RotationEvent)]) >= 1

@pytest.mark.skip(reason="Buggy")
def test_stop_rotation():
    world = setup_contact_world()
    context = MotionStatechartContext(world=world)
    segmind_executor = EpisodeSegmenterExecutor(context=context)
    statechart = SegmindStatechart()
    sc = statechart.build_statechart()
    sc.add_node(RotationDetector())
    sc.add_node(StopRotationDetector())
    segmind_executor.compile(sc)
    segmind_context = segmind_executor.context.require_extension(SegmindContext)



    cylinder = world.get_body_by_name("cylinder_body")

    assert len([i for i in segmind_context.logger.get_events() if isinstance(i, RotationEvent)]) == 0

    for i in range(5):
        cylinder.parent_connection.origin = (
            HomogeneousTransformationMatrix.from_xyz_rpy(x=1, y=-3, z=0.25, roll=i*0.1)
        )
        segmind_executor.tick()
    assert len([i for i in segmind_context.logger.get_events() if isinstance(i, RotationEvent)]) >= 1

    for _ in range(5):
        segmind_executor.tick()
    assert len([i for i in segmind_context.logger.get_events() if isinstance(i, StopRotationEvent)]) >= 1

