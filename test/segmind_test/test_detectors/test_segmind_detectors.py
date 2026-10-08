import numpy as np
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
from segmind.detectors.atomic_event_detectors_nodes import RotationDetector, ContactDetector, \
    TranslationDetector
from segmind.detectors.base import SegmindContext
from segmind.detectors.coarse_event_detector_nodes import PickUpDetector, PlacingDetector
from segmind.detectors.spatial_relation_detector_nodes import SupportDetector, \
    ContainmentDetector, InsertionDetector
from segmind.episode_segmenter import EpisodeSegmenterExecutor
from segmind.statecharts.segmind_statechart import SegmindStatechart
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.reasoning.predicates import InContactWith, SupportedBy
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.world_description.connections import FixedConnection
from semantic_digital_twin.world_description.geometry import Box, Scale
from semantic_digital_twin.world_description.shape_collection import ShapeCollection
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.world_entity import Body

from ..conftest import (
    HELD_UP_IN_THE_TRAY,
    LIFTED_OUT_OF_THE_TRAY,
    SET_DOWN_IN_THE_HOLE,
    SET_DOWN_IN_THE_TRAY,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_executor(world: World):
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

def test_contact_detector(_simple_apartment_setup):
    segmind_executor, segmind_context, milk, box1, box2 = _build_executor(_simple_apartment_setup)
    statechart = SegmindStatechart().build_statechart([ContactDetector()])
    segmind_executor.compile(statechart)
    segmind_executor.tick()

    assert len(events_of(segmind_context, ContactEvent)) == 0

    milk.parent_connection.origin = HomogeneousTransformationMatrix.from_xyz_rpy(z=1, reference_frame=milk.parent_connection.parent)
    segmind_executor.tick()
    assert len(events_of(segmind_context, LossOfContactEvent)) == 0

    milk.parent_connection.origin = HomogeneousTransformationMatrix.from_xyz_rpy(box1.global_pose.x, box1.global_pose.y, box1.global_pose.z, reference_frame=milk.parent_connection.parent)
    segmind_executor.tick()
    assert len(events_of(segmind_context, ContactEvent)) == 1
    assert len(events_of(segmind_context, LossOfContactEvent)) == 0

    milk.parent_connection.origin = HomogeneousTransformationMatrix.from_xyz_rpy(box2.global_pose.x, box2.global_pose.y, box2.global_pose.z, reference_frame=milk.parent_connection.parent)
    segmind_executor.tick()
    assert len(events_of(segmind_context, ContactEvent)) == 2
    assert len(events_of(segmind_context, LossOfContactEvent)) == 1

    milk.parent_connection.origin = HomogeneousTransformationMatrix.from_xyz_rpy(z=1, reference_frame=milk.parent_connection.parent)
    segmind_executor.tick()
    assert len(events_of(segmind_context, LossOfContactEvent)) == 2
    milk.parent_connection.origin = HomogeneousTransformationMatrix.from_xyz_rpy(-1.7, 0, 1.07, yaw=np.pi, reference_frame=milk.parent_connection.parent)

def test_support_detector(_simple_apartment_setup):
    segmind_executor, segmind_context, milk, box1, box2 = _build_executor(_simple_apartment_setup)
    statechart = SegmindStatechart().build_statechart([SupportDetector()])
    milk.parent_connection.origin = HomogeneousTransformationMatrix.from_xyz_rpy(-1.7, 0, 0.93, reference_frame=milk.parent_connection.parent)
    segmind_executor.compile(statechart)
    segmind_executor.tick()
    assert len(events_of(segmind_context, SupportEvent)) == 1

    milk.parent_connection.origin = HomogeneousTransformationMatrix.from_xyz_rpy(z=1, reference_frame=milk.parent_connection.parent)
    segmind_executor.tick()
    assert len(events_of(segmind_context, LossOfSupportEvent)) == 1

    milk.parent_connection.origin = HomogeneousTransformationMatrix.from_xyz_rpy(box1.global_pose.x, box1.global_pose.y,
                                                                                 box1.global_pose.z + 0.56, reference_frame=milk.parent_connection.parent)
    segmind_executor.tick()
    assert len(events_of(segmind_context, SupportEvent)) == 2
    assert len(events_of(segmind_context, LossOfSupportEvent)) == 1

    milk.parent_connection.origin = HomogeneousTransformationMatrix.from_xyz_rpy(box2.global_pose.x, box2.global_pose.y,
                                                                                 box2.global_pose.z + 0.56, reference_frame=milk.parent_connection.parent)
    segmind_executor.tick()
    assert len(events_of(segmind_context, SupportEvent)) == 3
    assert len(events_of(segmind_context, LossOfSupportEvent)) == 2

    milk.parent_connection.origin = HomogeneousTransformationMatrix.from_xyz_rpy(z=1, reference_frame=milk.parent_connection.parent)
    segmind_executor.tick()
    assert len(events_of(segmind_context, LossOfSupportEvent)) == 3
    milk.parent_connection.origin = HomogeneousTransformationMatrix.from_xyz_rpy(-1.7, 0, 1.07, yaw=np.pi, reference_frame=milk.parent_connection.parent)

def test_containment_detector(box_and_trays):
    world, box, _, _ = box_and_trays
    segmind_executor = EpisodeSegmenterExecutor(context=MotionStatechartContext(world=world))
    segmind_context = segmind_executor.context.require_extension(SegmindContext)
    statechart = SegmindStatechart().build_statechart([SupportDetector(), ContainmentDetector()])
    segmind_executor.compile(statechart)
    segmind_executor.tick()

    assert len(events_of(segmind_context, ContainmentEvent)) == 0

    box.parent_connection.origin = HomogeneousTransformationMatrix.from_xyz_rpy(*SET_DOWN_IN_THE_TRAY, reference_frame=box.parent_connection.parent)
    segmind_executor.tick()
    assert len(events_of(segmind_context, ContainmentEvent)) == 1

    box.parent_connection.origin = HomogeneousTransformationMatrix.from_xyz_rpy(*SET_DOWN_IN_THE_HOLE, reference_frame=box.parent_connection.parent)
    segmind_executor.tick()
    assert len(events_of(segmind_context, ContainmentEvent)) == 2
    assert len(events_of(segmind_context, LossOfContainmentEvent)) == 1

    box.parent_connection.origin = HomogeneousTransformationMatrix.from_xyz_rpy(*LIFTED_OUT_OF_THE_TRAY, reference_frame=box.parent_connection.parent)
    segmind_executor.tick()
    assert len(events_of(segmind_context, LossOfContainmentEvent)) == 2


def test_containment_detector_ignores_an_object_held_up_inside_a_container(box_and_trays):
    """
    Something carried through a container's walls has not been put into it.
    """
    world, box, _, _ = box_and_trays
    segmind_executor = EpisodeSegmenterExecutor(context=MotionStatechartContext(world=world))
    segmind_context = segmind_executor.context.require_extension(SegmindContext)
    statechart = SegmindStatechart().build_statechart([SupportDetector(), ContainmentDetector()])
    segmind_executor.compile(statechart)

    box.parent_connection.origin = HomogeneousTransformationMatrix.from_xyz_rpy(*HELD_UP_IN_THE_TRAY, reference_frame=box.parent_connection.parent)
    segmind_executor.tick()

    assert len(events_of(segmind_context, ContainmentEvent)) == 0

def test_containment_is_not_looked_for_without_the_supports_it_is_read_from(box_and_trays):
    """
    Containment is looked for once an object comes to rest, which only a support
    detector ticked alongside can tell.
    """
    world, box, _, _ = box_and_trays
    segmind_executor = EpisodeSegmenterExecutor(context=MotionStatechartContext(world=world))
    segmind_context = segmind_executor.context.require_extension(SegmindContext)
    segmind_executor.compile(SegmindStatechart().build_statechart([ContainmentDetector()]))

    box.parent_connection.origin = HomogeneousTransformationMatrix.from_xyz_rpy(*SET_DOWN_IN_THE_TRAY, reference_frame=box.parent_connection.parent)
    segmind_executor.tick()

    assert len(events_of(segmind_context, ContainmentEvent)) == 0


def test_asking_which_bodies_came_to_rest_twice_gives_the_same_answer(box_and_trays):
    _, box, tray, _ = box_and_trays
    detector = ContainmentDetector()
    segmind_context = SegmindContext()
    segmind_context.latest_support[box] = {tray}

    first_answer = detector.bodies_come_to_rest(segmind_context, [box])
    second_answer = detector.bodies_come_to_rest(segmind_context, [box])

    assert first_answer == second_answer == [box]


def test_pickup(_simple_apartment_setup):
    segmind_executor, segmind_context, milk, box1, box2 = _build_executor(_simple_apartment_setup)
    statechart = SegmindStatechart().build_statechart([PickUpDetector(), SupportDetector(), TranslationDetector()])
    milk.parent_connection.origin = HomogeneousTransformationMatrix.from_xyz_rpy(-1.7, 0, 0.93, reference_frame=milk.parent_connection.parent)

    segmind_executor.compile(statechart)
    segmind_executor.tick()

    assert len(events_of(segmind_context, SupportEvent)) == 1

    for i in range(5):
        milk.parent_connection.origin = HomogeneousTransformationMatrix.from_xyz_rpy(
            x=box2.global_pose.x,
            y=box2.global_pose.y,
            z=box2.global_pose.z + 0.56 + i * 0.1, reference_frame=milk.parent_connection.parent)
        segmind_executor.tick()

    assert len(events_of(segmind_context, TranslationEvent)) >= 1
    assert len(events_of(segmind_context, LossOfSupportEvent)) == 1
    assert len(events_of(segmind_context, PickUpEvent)) == 1
    milk.parent_connection.origin = HomogeneousTransformationMatrix.from_xyz_rpy(-1.7, 0, 1.07, yaw=np.pi, reference_frame=milk.parent_connection.parent)

def test_placing(_simple_apartment_setup):
    segmind_executor, segmind_context, milk, box1, box2 = _build_executor(_simple_apartment_setup)
    statechart = SegmindStatechart().build_statechart(
        [SupportDetector(), TranslationDetector(), PlacingDetector()])
    segmind_executor.compile(statechart)
    segmind_executor.tick()

    for i in range(5):
        milk.parent_connection.origin = HomogeneousTransformationMatrix.from_xyz_rpy(
            x=box2.global_pose.x,
            y=box2.global_pose.y,
            z=box2.global_pose.z + 0.97 - i * 0.1, reference_frame=milk.parent_connection.parent)
        segmind_executor.tick()

    assert len(events_of(segmind_context, TranslationEvent)) >= 1

    milk.parent_connection.origin = HomogeneousTransformationMatrix.from_xyz_rpy(
        x=box2.global_pose.x,
        y=box2.global_pose.y,
        z=box2.global_pose.z + 0.56, reference_frame=milk.parent_connection.parent)
    for _ in range(5):
        segmind_executor.tick()

    assert len(events_of(segmind_context, SupportEvent)) == 1
    assert len(events_of(segmind_context, StopTranslationEvent)) == 1
    assert len(events_of(segmind_context, PlacingEvent)) == 1
    milk.parent_connection.origin = HomogeneousTransformationMatrix.from_xyz_rpy(-1.7, 0, 1.07, yaw=np.pi, reference_frame=milk.parent_connection.parent)

def test_pickup_then_place_back_on_same_surface(_simple_apartment_setup):
    segmind_executor, segmind_context, milk, box1, box2 = _build_executor(_simple_apartment_setup)
    statechart = SegmindStatechart().build_statechart(
        [PickUpDetector(), PlacingDetector(), SupportDetector(),
         TranslationDetector()])
    milk.parent_connection.origin = HomogeneousTransformationMatrix.from_xyz_rpy(
        x=box2.global_pose.x, y=box2.global_pose.y, z=box2.global_pose.z + 0.56, reference_frame=milk.parent_connection.parent)

    segmind_executor.compile(statechart)
    segmind_executor.tick()

    assert len(events_of(segmind_context, SupportEvent)) == 1

    # Pick the milk up off box2.
    for i in range(5):
        milk.parent_connection.origin = HomogeneousTransformationMatrix.from_xyz_rpy(
            x=box2.global_pose.x,
            y=box2.global_pose.y,
            z=box2.global_pose.z + 0.56 + i * 0.1, reference_frame=milk.parent_connection.parent)
        segmind_executor.tick()

    assert len(events_of(segmind_context, LossOfSupportEvent)) == 1
    assert len(events_of(segmind_context, PickUpEvent)) == 1

    for _ in range(5):
        segmind_executor.tick()

    # Place the milk back down onto the very same surface (box2) it was picked up from.
    for i in range(5):
        milk.parent_connection.origin = HomogeneousTransformationMatrix.from_xyz_rpy(
            x=box2.global_pose.x,
            y=box2.global_pose.y,
            z=box2.global_pose.z + 0.97 - i * 0.1, reference_frame=milk.parent_connection.parent)
        segmind_executor.tick()

    milk.parent_connection.origin = HomogeneousTransformationMatrix.from_xyz_rpy(
        x=box2.global_pose.x,
        y=box2.global_pose.y,
        z=box2.global_pose.z + 0.56, reference_frame=milk.parent_connection.parent)
    for _ in range(5):
        segmind_executor.tick()

    assert len(events_of(segmind_context, SupportEvent)) == 2
    assert len(events_of(segmind_context, PlacingEvent)) == 1
    milk.parent_connection.origin = HomogeneousTransformationMatrix.from_xyz_rpy(-1.7, 0, 1.07, yaw=np.pi, reference_frame=milk.parent_connection.parent)


def test_translation(_simple_apartment_setup):
    segmind_executor, segmind_context, milk, box1, box2 = _build_executor(_simple_apartment_setup)
    statechart = SegmindStatechart().build_statechart(
        [TranslationDetector()])
    segmind_executor.compile(statechart)
    segmind_executor.tick()

    assert len(events_of(segmind_context, TranslationEvent)) == 0

    for i in range(5):
        milk.parent_connection.origin = HomogeneousTransformationMatrix.from_xyz_rpy(
            x=1 + i * 0.1, y=-3, z=0.25, reference_frame=milk.parent_connection.parent)
        segmind_executor.tick()

    assert len(events_of(segmind_context, TranslationEvent)) == 1
    milk.parent_connection.origin = HomogeneousTransformationMatrix.from_xyz_rpy(-1.7, 0, 1.07, yaw=np.pi, reference_frame=milk.parent_connection.parent)


def test_a_motion_event_states_its_poses_in_the_world_frame(_simple_apartment_setup):
    world = _simple_apartment_setup
    segmind_executor, segmind_context, milk, box1, box2 = _build_executor(world)
    segmind_executor.compile(SegmindStatechart().build_statechart([TranslationDetector()]))
    segmind_executor.tick()

    for i in range(5):
        milk.parent_connection.origin = HomogeneousTransformationMatrix.from_xyz_rpy(
            x=1 + i * 0.1, y=-3, z=0.25, reference_frame=milk.parent_connection.parent)
        segmind_executor.tick()

    [translation] = events_of(segmind_context, TranslationEvent)
    assert translation.world_T_start_pose.reference_frame is world.root
    assert translation.world_T_current_pose.reference_frame is world.root
    milk.parent_connection.origin = HomogeneousTransformationMatrix.from_xyz_rpy(-1.7, 0, 1.07, yaw=np.pi, reference_frame=milk.parent_connection.parent)


def test_a_contact_event_states_the_poses_of_both_objects_in_the_world_frame(
    _simple_apartment_setup,
):
    world = _simple_apartment_setup
    segmind_executor, segmind_context, milk, box1, box2 = _build_executor(world)
    segmind_executor.compile(SegmindStatechart().build_statechart([ContactDetector()]))
    segmind_executor.tick()
    milk.parent_connection.origin = HomogeneousTransformationMatrix.from_xyz_rpy(
        box1.global_pose.x, box1.global_pose.y, box1.global_pose.z, reference_frame=milk.parent_connection.parent)
    segmind_executor.tick()

    [contact_event] = events_of(segmind_context, ContactEvent)
    assert contact_event.world_T_tracked_object.reference_frame is world.root
    assert contact_event.world_T_with_object.reference_frame is world.root
    milk.parent_connection.origin = HomogeneousTransformationMatrix.from_xyz_rpy(-1.7, 0, 1.07, yaw=np.pi, reference_frame=milk.parent_connection.parent)


def test_stop_translation(_simple_apartment_setup):
    segmind_executor, segmind_context, milk, box1, box2 = _build_executor(_simple_apartment_setup)
    statechart = SegmindStatechart().build_statechart(
        [SupportDetector(), TranslationDetector(), PlacingDetector()])
    segmind_executor.compile(statechart)
    segmind_executor.tick()

    for i in range(5):
        milk.parent_connection.origin = HomogeneousTransformationMatrix.from_xyz_rpy(
            x=1 + i * 0.1, y=-3, z=0.25, reference_frame=milk.parent_connection.parent)
        segmind_executor.tick()

    assert len(events_of(segmind_context, TranslationEvent)) == 1

    for _ in range(5):
        segmind_executor.tick()

    assert len(events_of(segmind_context, StopTranslationEvent)) == 1
    milk.parent_connection.origin = HomogeneousTransformationMatrix.from_xyz_rpy(-1.7, 0, 1.07, yaw=np.pi, reference_frame=milk.parent_connection.parent)


def test_insertion(box_and_trays):
    world, box, _, _ = box_and_trays
    segmind_executor = EpisodeSegmenterExecutor(context=MotionStatechartContext(world=world))
    segmind_context = segmind_executor.context.require_extension(SegmindContext)
    statechart = SegmindStatechart().build_statechart(
        [ContactDetector(), SupportDetector(), InsertionDetector(), ContainmentDetector()])
    segmind_executor.compile(statechart)
    segmind_executor.tick()

    assert len(events_of(segmind_context, InsertionEvent)) == 0

    box.parent_connection.origin = HomogeneousTransformationMatrix.from_xyz_rpy(*SET_DOWN_IN_THE_HOLE, reference_frame=box.parent_connection.parent)
    segmind_executor.tick()

    box.parent_connection.origin = HomogeneousTransformationMatrix.from_xyz_rpy(*LIFTED_OUT_OF_THE_TRAY, reference_frame=box.parent_connection.parent)
    segmind_executor.tick()

    assert len(events_of(segmind_context, InsertionEvent)) == 1


def test_rotation(_simple_apartment_setup):
    segmind_executor, segmind_context, milk, box1, box2 = _build_executor(_simple_apartment_setup)
    statechart = SegmindStatechart().build_statechart(
        [RotationDetector()])
    segmind_executor.compile(statechart)
    segmind_executor.tick()


    assert len([i for i in segmind_context.logger.get_events() if isinstance(i, RotationEvent)]) == 0

    for i in range(5):
        milk.parent_connection.origin = (
            HomogeneousTransformationMatrix.from_xyz_rpy(
                roll=i * 0.1, reference_frame=milk.parent_connection.parent
            )
        )
        segmind_executor.tick()

    assert len([i for i in segmind_context.logger.get_events() if isinstance(i, RotationEvent)]) >= 1


def test_stop_rotation(_simple_apartment_setup):
    segmind_executor, segmind_context, milk, box1, box2 = _build_executor(_simple_apartment_setup)
    statechart = SegmindStatechart().build_statechart(
        [RotationDetector()])
    segmind_executor.compile(statechart)
    segmind_executor.tick()

    assert len([i for i in segmind_context.logger.get_events() if isinstance(i, RotationEvent)]) == 0

    for i in range(5):
        milk.parent_connection.origin = (
            HomogeneousTransformationMatrix.from_xyz_rpy(
                roll=i * 0.1, reference_frame=milk.parent_connection.parent
            )
        )
        segmind_executor.tick()
    assert len([i for i in segmind_context.logger.get_events() if isinstance(i, RotationEvent)]) >= 1

    for _ in range(5):
        segmind_executor.tick()
    assert len([i for i in segmind_context.logger.get_events() if isinstance(i, StopRotationEvent)]) >= 1


def test_slow_motion_with_all_motion_detectors(_simple_apartment_setup):
    """
    Runs every motion detector in one statechart on an object that drifts slowly.

    Each step stays below distance_threshold and rotation_threshold; only the displacement
    accumulated across the whole window exceeds them. Detecting this therefore requires every
    detector to hold a pose window that spans window_size ticks. While the window lived on the
    shared context, all four detectors appended to it on every tick, so it only ever spanned a
    single tick and drift this slow was never reported.
    """
    segmind_executor, segmind_context, milk, box1, box2 = _build_executor(_simple_apartment_setup)
    statechart = SegmindStatechart().build_statechart(
        [TranslationDetector(), RotationDetector()])
    segmind_executor.compile(statechart)

    # Move the object to its start pose and let the pose windows settle, so that the events
    # triggered by that jump are not mistaken for the slow drift below.
    milk.parent_connection.origin = HomogeneousTransformationMatrix.from_xyz_rpy(x=1, y=-3, z=0.25, reference_frame=milk.parent_connection.parent)
    for _ in range(8):
        segmind_executor.tick()

    translations = len(events_of(segmind_context, TranslationEvent))
    rotations = len(events_of(segmind_context, RotationEvent))

    # Per tick this is 0.002m and 0.04rad, both below their detection thresholds. Across the
    # window it accumulates to more than 0.005m and 0.1rad, so it has to be reported.
    for i in range(1, 9):
        milk.parent_connection.origin = HomogeneousTransformationMatrix.from_xyz_rpy(
            x=1 + i * 0.002, y=-3, z=0.25, roll=i * 0.04, reference_frame=milk.parent_connection.parent)
        segmind_executor.tick()

    assert len(events_of(segmind_context, TranslationEvent)) > translations
    assert len(events_of(segmind_context, RotationEvent)) > rotations

    for _ in range(8):
        segmind_executor.tick()

    assert len(events_of(segmind_context, StopTranslationEvent)) >= 1
    assert len(events_of(segmind_context, StopRotationEvent)) >= 1

    milk.parent_connection.origin = HomogeneousTransformationMatrix.from_xyz_rpy(-1.7, 0, 1.07, yaw=np.pi, reference_frame=milk.parent_connection.parent)


# %% relations between tracked bodies


def _unit_box(name: str) -> Body:
    """
    A body whose collision is a one-metre cube centred on its own frame.
    """
    body = Body(name=PrefixedName(name))
    collision = Box(
        scale=Scale(1.0, 1.0, 1.0),
        origin=HomogeneousTransformationMatrix.from_xyz_rpy(reference_frame=body),
    )
    body.collision = ShapeCollection([collision], reference_frame=body)
    return body


@pytest.fixture
def box_resting_on_another_beside_a_distant_one():
    """
    A box resting on top of another, and a third box far from both.

    :return: The world, the box on top, and the box it rests on.
    """
    world = World()
    bottom = _unit_box("bottom")
    top = _unit_box("top")
    distant = _unit_box("distant")
    with world.modify_world():
        world.add_connection(
            FixedConnection(
                parent=bottom,
                child=top,
                parent_T_connection_expression=HomogeneousTransformationMatrix.from_xyz_rpy(
                    z=1.0, reference_frame=bottom
                ),
            )
        )
        world.add_connection(
            FixedConnection(
                parent=bottom,
                child=distant,
                parent_T_connection_expression=HomogeneousTransformationMatrix.from_xyz_rpy(
                    x=5.0, reference_frame=bottom
                ),
            )
        )
    return world, top, bottom


@pytest.mark.parametrize("relation", [SupportedBy, InContactWith])
def test_a_relation_relates_only_the_bodies_it_holds_between(
    box_resting_on_another_beside_a_distant_one, relation
):
    """
    A detector is handed the relation itself, and relates a tracked body only to the
    bodies that relation holds for.
    """
    world, top, bottom = box_resting_on_another_beside_a_distant_one
    context = MotionStatechartContext(world=world)

    related = SupportDetector().get_relation(context, [top], relation)

    assert related == {top: {bottom}}
