import numpy as np
import pytest
import rclpy

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
from segmind.detectors.atomic_event_detectors_nodes import RotationDetector, StopRotationDetector, ContactDetector, \
    LossOfContactDetector, TranslationDetector, StopTranslationDetector
from segmind.detectors.base import SegmindContext
from segmind.detectors.coarse_event_detector_nodes import PickUpDetector, PlacingDetector
from segmind.detectors.spatial_relation_detector_nodes import SupportDetector, LossOfSupportDetector, \
    ContainmentDetector, LossOfContainmentDetector, InsertionDetector
from segmind.episode_segmenter import EpisodeSegmenterExecutor
from segmind.statecharts.segmind_statechart import SegmindStatechart
from semantic_digital_twin.adapters.ros.visualization.viz_marker import VizMarkerPublisher
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.world_description.connections import FixedConnection
from semantic_digital_twin.world_description.geometry import Box, Scale
from semantic_digital_twin.world_description.shape_collection import ShapeCollection
from semantic_digital_twin.world_description.world_entity import Body


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_executor(world):
    context = MotionStatechartContext(world=world)
    milk = world.get_body_by_name("milk.stl")
    box1 = world.get_body_by_name("box")
    box2 = world.get_body_by_name("box_2")
    segmind_executor = EpisodeSegmenterExecutor(context=context)
    segmind_context = segmind_executor.context.require_extension(SegmindContext)
    return segmind_executor, segmind_context, milk, box1, box2


def events_of(segmind_context, event_type):
    return [e for e in segmind_context.logger.get_events() if isinstance(e, event_type)]

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_contact_detector(simple_apartment_setup):
    segmind_executor, segmind_context, milk, box1, box2 = _build_executor(simple_apartment_setup)
    statechart = SegmindStatechart().build_statechart([ContactDetector(),LossOfContactDetector()])
    segmind_executor.compile(statechart)
    segmind_executor.tick()
    assert len(events_of(segmind_context, ContactEvent)) == 1

    milk.parent_connection.origin = HomogeneousTransformationMatrix.from_xyz_rpy(z=1)
    segmind_executor.tick()
    assert len(events_of(segmind_context, LossOfContactEvent)) == 1

    milk.parent_connection.origin = HomogeneousTransformationMatrix.from_xyz_rpy(box1.global_pose.x, box1.global_pose.y, box1.global_pose.z)
    segmind_executor.tick()
    assert len(events_of(segmind_context, ContactEvent)) == 2
    assert len(events_of(segmind_context, LossOfContactEvent)) == 1

    milk.parent_connection.origin = HomogeneousTransformationMatrix.from_xyz_rpy(box2.global_pose.x, box2.global_pose.y, box2.global_pose.z)
    segmind_executor.tick()
    assert len(events_of(segmind_context, ContactEvent)) == 3
    assert len(events_of(segmind_context, LossOfContactEvent)) == 2

    milk.parent_connection.origin = HomogeneousTransformationMatrix.from_xyz_rpy(z=1)
    segmind_executor.tick()
    assert len(events_of(segmind_context, LossOfContactEvent)) == 3
    milk.parent_connection.origin = HomogeneousTransformationMatrix.from_xyz_rpy(-1.7, 0, 1.07, yaw=np.pi)

def test_support_detector(simple_apartment_setup):
    segmind_executor, segmind_context, milk, box1, box2 = _build_executor(simple_apartment_setup)
    statechart = SegmindStatechart().build_statechart([SupportDetector(), LossOfSupportDetector()])
    segmind_executor.compile(statechart)
    segmind_executor.tick()
    assert len(events_of(segmind_context, SupportEvent)) == 1

    milk.parent_connection.origin = HomogeneousTransformationMatrix.from_xyz_rpy(z=1)
    segmind_executor.tick()
    assert len(events_of(segmind_context, LossOfSupportEvent)) == 1

    milk.parent_connection.origin = HomogeneousTransformationMatrix.from_xyz_rpy(box1.global_pose.x, box1.global_pose.y,
                                                                                 box1.global_pose.z + 0.575)
    segmind_executor.tick()
    assert len(events_of(segmind_context, SupportEvent)) == 2
    assert len(events_of(segmind_context, LossOfSupportEvent)) == 1

    milk.parent_connection.origin = HomogeneousTransformationMatrix.from_xyz_rpy(box2.global_pose.x, box2.global_pose.y,
                                                                                 box2.global_pose.z + 0.575)
    segmind_executor.tick()
    assert len(events_of(segmind_context, SupportEvent)) == 3
    assert len(events_of(segmind_context, LossOfSupportEvent)) == 2

    milk.parent_connection.origin = HomogeneousTransformationMatrix.from_xyz_rpy(z=1)
    segmind_executor.tick()
    assert len(events_of(segmind_context, LossOfSupportEvent)) == 3
    milk.parent_connection.origin = HomogeneousTransformationMatrix.from_xyz_rpy(-1.7, 0, 1.07, yaw=np.pi)

def test_containment_detector(simple_apartment_setup):
    segmind_executor, segmind_context, milk, box1, box2 = _build_executor(simple_apartment_setup)
    statechart = SegmindStatechart().build_statechart([ContainmentDetector(), LossOfContainmentDetector()])
    segmind_executor.compile(statechart)
    segmind_executor.tick()

    assert len(events_of(segmind_context, ContainmentEvent)) == 0

    milk.parent_connection.origin = HomogeneousTransformationMatrix.from_xyz_rpy(box1.global_pose.x, box1.global_pose.y,
                                                                                 box1.global_pose.z)
    segmind_executor.tick()
    assert len(events_of(segmind_context, ContainmentEvent)) == 1

    milk.parent_connection.origin = HomogeneousTransformationMatrix.from_xyz_rpy(box2.global_pose.x, box2.global_pose.y,
                                                                                 box2.global_pose.z)
    segmind_executor.tick()
    assert len(events_of(segmind_context, ContainmentEvent)) == 2
    assert len(events_of(segmind_context, LossOfContainmentEvent)) == 1

    milk.parent_connection.origin = HomogeneousTransformationMatrix.from_xyz_rpy(z=1)
    segmind_executor.tick()
    assert len(events_of(segmind_context, LossOfContainmentEvent)) == 2
    milk.parent_connection.origin = HomogeneousTransformationMatrix.from_xyz_rpy(-1.7, 0, 1.07, yaw=np.pi)

def test_pickup(simple_apartment_setup):
    segmind_executor, segmind_context, milk, box1, box2 = _build_executor(simple_apartment_setup)
    statechart = SegmindStatechart().build_statechart([PickUpDetector(), SupportDetector(), TranslationDetector(), LossOfSupportDetector()])
    segmind_executor.compile(statechart)
    segmind_executor.tick()

    assert len(events_of(segmind_context, SupportEvent)) == 1

    for i in range(5):
        milk.parent_connection.origin = HomogeneousTransformationMatrix.from_xyz_rpy(
            x=box2.global_pose.x,
            y=box2.global_pose.y,
            z=box2.global_pose.z + 0.57 + i * 0.1,
        )
        segmind_executor.tick()

    assert len(events_of(segmind_context, TranslationEvent)) >= 1
    assert len(events_of(segmind_context, LossOfSupportEvent)) == 1
    assert len(events_of(segmind_context, PickUpEvent)) == 1
    milk.parent_connection.origin = HomogeneousTransformationMatrix.from_xyz_rpy(-1.7, 0, 1.07, yaw=np.pi)

def test_placing(simple_apartment_setup):
    segmind_executor, segmind_context, milk, box1, box2 = _build_executor(simple_apartment_setup)
    statechart = SegmindStatechart().build_statechart(
        [SupportDetector(), TranslationDetector(), StopTranslationDetector(), PlacingDetector()])
    segmind_executor.compile(statechart)
    segmind_executor.tick()

    for i in range(5):
        milk.parent_connection.origin = HomogeneousTransformationMatrix.from_xyz_rpy(
            x=box2.global_pose.x,
            y=box2.global_pose.y,
            z=box2.global_pose.z + 0.97 - i * 0.1,
        )
        segmind_executor.tick()

    assert len(events_of(segmind_context, TranslationEvent)) >= 1

    milk.parent_connection.origin = HomogeneousTransformationMatrix.from_xyz_rpy(
        x=box2.global_pose.x,
        y=box2.global_pose.y,
        z=box2.global_pose.z + 0.57,
    )
    for _ in range(5):
        segmind_executor.tick()

    assert len(events_of(segmind_context, SupportEvent)) == 1
    assert len(events_of(segmind_context, StopTranslationEvent)) == 1
    assert len(events_of(segmind_context, PlacingEvent)) == 1
    milk.parent_connection.origin = HomogeneousTransformationMatrix.from_xyz_rpy(-1.7, 0, 1.07, yaw=np.pi)

def test_translation(simple_apartment_setup):
    segmind_executor, segmind_context, milk, box1, box2 = _build_executor(simple_apartment_setup)
    statechart = SegmindStatechart().build_statechart(
        [TranslationDetector()])
    segmind_executor.compile(statechart)
    segmind_executor.tick()

    assert len(events_of(segmind_context, TranslationEvent)) == 0

    for i in range(5):
        milk.parent_connection.origin = HomogeneousTransformationMatrix.from_xyz_rpy(
            x=1 + i * 0.1, y=-3, z=0.25
        )
        segmind_executor.tick()

    assert len(events_of(segmind_context, TranslationEvent)) == 1
    milk.parent_connection.origin = HomogeneousTransformationMatrix.from_xyz_rpy(-1.7, 0, 1.07, yaw=np.pi)


def test_stop_translation(simple_apartment_setup):
    segmind_executor, segmind_context, milk, box1, box2 = _build_executor(simple_apartment_setup)
    statechart = SegmindStatechart().build_statechart(
        [SupportDetector(), TranslationDetector(), StopTranslationDetector(), PlacingDetector()])
    segmind_executor.compile(statechart)
    segmind_executor.tick()

    for i in range(5):
        milk.parent_connection.origin = HomogeneousTransformationMatrix.from_xyz_rpy(
            x=1 + i * 0.1, y=-3, z=0.25
        )
        segmind_executor.tick()

    assert len(events_of(segmind_context, TranslationEvent)) == 1

    for _ in range(5):
        segmind_executor.tick()

    assert len(events_of(segmind_context, StopTranslationEvent)) == 1
    milk.parent_connection.origin = HomogeneousTransformationMatrix.from_xyz_rpy(-1.7, 0, 1.07, yaw=np.pi)


def test_insertion(simple_apartment_setup):
    segmind_executor, segmind_context, milk, box1, box2 = _build_executor(simple_apartment_setup)
    statechart = SegmindStatechart().build_statechart(
        [ContactDetector(), InsertionDetector(), LossOfContactDetector(), ContainmentDetector()])

    with segmind_executor.context.world.modify_world():
        hole = Body(
            name=PrefixedName("box_hole"),
            collision=ShapeCollection([Box(scale=Scale(1, 1, 1))]),
            visual=ShapeCollection([Box(scale=Scale(1, 1, 1))]),
        )
        hole_connection = FixedConnection(
            parent=segmind_executor.context.world.root,
            child=hole,
            parent_T_connection_expression=HomogeneousTransformationMatrix.from_xyz_rpy(
                2, 2, 2, reference_frame=segmind_executor.context.world.root
            ),
        )
        segmind_executor.context.world.add_connection(hole_connection)
    segmind_executor.compile(statechart)
    segmind_executor.tick()

    assert len(events_of(segmind_context, InsertionEvent)) == 0
    milk.parent_connection.origin = HomogeneousTransformationMatrix.from_xyz_rpy(hole.global_pose.x, hole.global_pose.y,
                                                                                 hole.global_pose.z)

    segmind_executor.tick()

    milk.parent_connection.origin = HomogeneousTransformationMatrix.from_xyz_rpy(box2.global_pose.x,box2.global_pose.y,box2.global_pose.z)
    segmind_executor.tick()

    assert len(events_of(segmind_context, InsertionEvent)) == 1
    milk.parent_connection.origin = HomogeneousTransformationMatrix.from_xyz_rpy(-1.7, 0, 1.07, yaw=np.pi)


def test_rotation(simple_apartment_setup):
    segmind_executor, segmind_context, milk, box1, box2 = _build_executor(simple_apartment_setup)
    statechart = SegmindStatechart().build_statechart(
        [RotationDetector()])
    segmind_executor.compile(statechart)
    segmind_executor.tick()


    assert len([i for i in segmind_context.logger.get_events() if isinstance(i, RotationEvent)]) == 0

    for i in range(5):
        milk.parent_connection.origin = (
            HomogeneousTransformationMatrix.from_xyz_rpy(roll=i*0.1)
        )
        segmind_executor.tick()

    assert len([i for i in segmind_context.logger.get_events() if isinstance(i, RotationEvent)]) >= 1


def test_stop_rotation(simple_apartment_setup):
    segmind_executor, segmind_context, milk, box1, box2 = _build_executor(simple_apartment_setup)
    statechart = SegmindStatechart().build_statechart(
        [RotationDetector(), StopRotationDetector()])
    segmind_executor.compile(statechart)
    segmind_executor.tick()

    assert len([i for i in segmind_context.logger.get_events() if isinstance(i, RotationEvent)]) == 0

    for i in range(5):
        milk.parent_connection.origin = (
            HomogeneousTransformationMatrix.from_xyz_rpy(roll=i*0.1)
        )
        segmind_executor.tick()
    assert len([i for i in segmind_context.logger.get_events() if isinstance(i, RotationEvent)]) >= 1

    for _ in range(5):
        segmind_executor.tick()
    assert len([i for i in segmind_context.logger.get_events() if isinstance(i, StopRotationEvent)]) >= 1

