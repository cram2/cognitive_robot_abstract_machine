from segmind.datastructures.events import (
    ContactEvent,
    LossOfContactEvent,
    SupportEvent,
    LossOfSupportEvent, LossOfContainmentEvent, ContainmentEvent, InsertionEvent,
    TranslationEvent, StopTranslationEvent, PickUpEvent, PlacingEvent,
)
from segmind.detectors.base import SegmindContext
from segmind.episode_segmenter import EpisodeSegmenterExecutor
from segmind.event_logger import EventLogger
from segmind.statecharts.segmind_statechart import SegmindStatechart
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from test import setup_contact_world, setup_support_world




def test_contact_detector():
    world = setup_contact_world()
    logger = EventLogger()
    context = SegmindContext(
        world=world,
        logger=logger,
    )
    statechart = SegmindStatechart()
    sc = statechart.build_statechart(context)

    cylinder = world.get_body_by_name("cylinder_body")

    segmind_executor = EpisodeSegmenterExecutor(context=context)
    segmind_executor.compile(sc)

    assert (
        len(
            [
                i
                for i in context.logger.get_events()
                if isinstance(i, ContactEvent)
            ]
        )
        == 0
    )


    cylinder.parent_connection.origin = (
        HomogeneousTransformationMatrix.from_xyz_rpy(y=-0.4)
    )
    segmind_executor.tick()

    assert len([i for i in logger.get_events() if isinstance(i, ContactEvent)]) == 2


    segmind_executor.tick()

    assert len([i for i in logger.get_events() if isinstance(i, ContactEvent)]) == 2

    cylinder.parent_connection.origin = (
        HomogeneousTransformationMatrix.from_xyz_rpy(z=2)
    )

    segmind_executor.tick()
    assert (
        len([i for i in logger.get_events() if isinstance(i, LossOfContactEvent)])
        == 2
    )

    segmind_executor.tick()
    assert (
        len([i for i in logger.get_events() if isinstance(i, LossOfContactEvent)])
        == 2
    )

    cylinder.parent_connection.origin = (
        HomogeneousTransformationMatrix.from_xyz_rpy(y=-0.4)
    )
    segmind_executor.tick()

    assert len([i for i in logger.get_events() if isinstance(i, ContactEvent)]) == 4

    cylinder.parent_connection.origin = (
        HomogeneousTransformationMatrix.from_xyz_rpy(z=2)
    )

    segmind_executor.tick()
    assert (
        len([i for i in logger.get_events() if isinstance(i, LossOfContactEvent)])
        == 4
    )


def test_support_detector():
    world = setup_support_world()
    logger = EventLogger()
    context = SegmindContext(
        world=world,
        logger=logger,
    )
    statechart = SegmindStatechart()
    sc = statechart.build_statechart(context)

    cylinder = world.get_body_by_name("cylinder_body")
    table = world.get_body_by_name("table_body")
    cabinet = world.get_body_by_name("cabinet")

    segmind_executor = EpisodeSegmenterExecutor(context=context)
    segmind_executor.compile(sc)

    assert len([i for i in logger.get_events() if isinstance(i, SupportEvent)]) == 0
    assert (
        len([i for i in logger.get_events() if isinstance(i, LossOfSupportEvent)])
        == 0
    )

    segmind_executor.tick()

    cylinder.parent_connection.origin = (
        HomogeneousTransformationMatrix.from_xyz_rpy(
            x=table.global_pose.x,
            y=table.global_pose.y,
            z=table.global_pose.z + 0.2,
        )
    )
    segmind_executor.tick()

    segmind_executor.tick()
    assert len([i for i in logger.get_events() if isinstance(i, SupportEvent)]) == 1

    cylinder.parent_connection.origin = (
        HomogeneousTransformationMatrix.from_xyz_rpy(
            x=cabinet.global_pose.x,
            y=cabinet.global_pose.y,
            z=cabinet.global_pose.z,
        )
    )
    segmind_executor.tick()
    segmind_executor.tick()

    assert (len([i for i in logger.get_events() if isinstance(i, LossOfSupportEvent)]) == 1)


def test_containment_detector():
    world = setup_support_world()
    logger = EventLogger()
    context = SegmindContext(
        world=world,
        logger=logger,
    )
    statechart = SegmindStatechart()
    sc = statechart.build_statechart(context)

    cylinder = world.get_body_by_name("cylinder_body")
    cabinet = world.get_body_by_name("cabinet")
    table = world.get_body_by_name("table_body")

    segmind_executor = EpisodeSegmenterExecutor(context=context)
    segmind_executor.compile(sc)

    assert len([i for i in logger.get_events() if isinstance(i, ContactEvent)]) == 0
    assert len([i for i in logger.get_events() if isinstance(i, ContainmentEvent)]) == 0


    cylinder.parent_connection.origin = (
        HomogeneousTransformationMatrix.from_xyz_rpy(
            x=cabinet.global_pose.x,
            y=cabinet.global_pose.y,
            z=cabinet.global_pose.z,
        )
    )

    segmind_executor.tick()

    assert len([i for i in logger.get_events() if isinstance(i, ContainmentEvent)]) == 1

    cylinder.parent_connection.origin = (
        HomogeneousTransformationMatrix.from_xyz_rpy(
            x=table.global_pose.x,
            y=table.global_pose.y,
            z=table.global_pose.z + 0.2,
        )
    )
    segmind_executor.tick()


    assert len([i for i in logger.get_events() if isinstance(i, LossOfContainmentEvent)]) == 1


def test_insertion_detector():
    world = setup_support_world()
    logger = EventLogger()
    context = SegmindContext(
        world=world,
        logger=logger,
    )
    statechart = SegmindStatechart()
    sc = statechart.build_statechart(context)

    hole = world.get_body_by_name("hole_body")
    cylinder = world.get_body_by_name("cylinder_body")
    cabinet = world.get_body_by_name("cabinet")

    segmind_executor = EpisodeSegmenterExecutor(context=context)
    segmind_executor.compile(sc)


    assert len(context.holes) == 1
    assert len([i for i in logger.get_events() if  isinstance(i, InsertionEvent)]) == 0
    cylinder.parent_connection.origin = (
        HomogeneousTransformationMatrix.from_xyz_rpy(
            x=hole.global_pose.x,
            y=hole.global_pose.y - 0.03,
            z=hole.global_pose.z,
        )
    )
    segmind_executor.tick()


    cylinder.parent_connection.origin = (
        HomogeneousTransformationMatrix.from_xyz_rpy(
            x=cabinet.global_pose.x,
            y=cabinet.global_pose.y,
            z=cabinet.global_pose.z,
        )
    )

    segmind_executor.tick()

    contact_events = [i for i in context.logger.get_events() if isinstance(i, ContactEvent)]
    contact_events_with_holes = [i for i in contact_events if i.with_object in context.holes]

    assert len([i for i in logger.get_events() if isinstance(i, ContainmentEvent)]) == 1
    assert len(contact_events_with_holes) == 1
    assert len([i for i in logger.get_events() if isinstance(i, InsertionEvent)]) == 1


def test_pickup():
    world = setup_support_world()
    logger = EventLogger()
    context = SegmindContext(
        world=world,
        logger=logger,
    )
    statechart = SegmindStatechart()
    sc = statechart.build_statechart(context)

    cylinder = world.get_body_by_name("cylinder_body")
    table = world.get_body_by_name("table_body")

    segmind_executor = EpisodeSegmenterExecutor(context=context)
    segmind_executor.compile(sc)

    cylinder.parent_connection.origin = (
        HomogeneousTransformationMatrix.from_xyz_rpy(
            x=table.global_pose.x,
            y=table.global_pose.y,
            z=table.global_pose.z + 0.2,
        )
    )
    segmind_executor.tick()
    segmind_executor.tick()
    assert len([i for i in logger.get_events() if isinstance(i, SupportEvent)]) == 1

    for i in range(5):
        cylinder.parent_connection.origin = (
            HomogeneousTransformationMatrix.from_xyz_rpy(
                x=table.global_pose.x,
                y=table.global_pose.y,
                z=table.global_pose.z + 0.3 + i * 0.1,
            )
        )
        segmind_executor.tick()

    assert len([i for i in logger.get_events() if isinstance(i, TranslationEvent)]) >= 1
    assert len([i for i in logger.get_events() if isinstance(i, LossOfSupportEvent)]) == 1
    assert len([i for i in logger.get_events() if isinstance(i, PickUpEvent)]) == 1


def test_placing():
    world = setup_support_world()
    logger = EventLogger()
    context = SegmindContext(
        world=world,
        logger=logger,
    )
    statechart = SegmindStatechart()
    sc = statechart.build_statechart(context)

    cylinder = world.get_body_by_name("cylinder_body")
    table = world.get_body_by_name("table_body")

    segmind_executor = EpisodeSegmenterExecutor(context=context)
    segmind_executor.compile(sc)

    for i in range(5):
        cylinder.parent_connection.origin = (
            HomogeneousTransformationMatrix.from_xyz_rpy(
                x=table.global_pose.x,
                y=table.global_pose.y,
                z=table.global_pose.z + 0.5 - i * 0.05,
            )
        )
        segmind_executor.tick()

    assert len([i for i in logger.get_events() if isinstance(i, TranslationEvent)]) >= 1

    cylinder.parent_connection.origin = (
        HomogeneousTransformationMatrix.from_xyz_rpy(
            x=table.global_pose.x,
            y=table.global_pose.y,
            z=table.global_pose.z + 0.2,
        )
    )
    for _ in range(5):
        segmind_executor.tick()

    assert len([i for i in logger.get_events() if isinstance(i, SupportEvent)]) == 1
    assert len([i for i in logger.get_events() if isinstance(i, StopTranslationEvent)]) == 1
    assert len([i for i in logger.get_events() if isinstance(i, PlacingEvent)]) == 1


def test_translation():
    world = setup_contact_world()
    logger = EventLogger()
    context = SegmindContext(
        world=world,
        logger=logger,
    )
    statechart = SegmindStatechart()
    sc = statechart.build_statechart(context)

    cylinder = world.get_body_by_name("cylinder_body")

    segmind_executor = EpisodeSegmenterExecutor(context=context)
    segmind_executor.compile(sc)

    assert len([i for i in logger.get_events() if isinstance(i, TranslationEvent)]) == 0

    for i in range(5):
        cylinder.parent_connection.origin = (
            HomogeneousTransformationMatrix.from_xyz_rpy(x=1 + i * 0.1, y=-3, z=0.25)
        )
        segmind_executor.tick()

    assert len([i for i in logger.get_events() if isinstance(i, TranslationEvent)]) == 1


def test_stop_translation():
    world = setup_contact_world()
    logger = EventLogger()
    context = SegmindContext(
        world=world,
        logger=logger,
    )
    statechart = SegmindStatechart()
    sc = statechart.build_statechart(context)

    cylinder = world.get_body_by_name("cylinder_body")

    segmind_executor = EpisodeSegmenterExecutor(context=context)
    segmind_executor.compile(sc)

    for i in range(5):
        cylinder.parent_connection.origin = (
            HomogeneousTransformationMatrix.from_xyz_rpy(x=1 + i * 0.1, y=-3, z=0.25)
        )
        segmind_executor.tick()

    assert len([i for i in logger.get_events() if isinstance(i, TranslationEvent)]) == 1

    for _ in range(5):
        segmind_executor.tick()

    assert len([i for i in logger.get_events() if isinstance(i, StopTranslationEvent)]) == 1


def test_rotation():
    world = setup_contact_world()
    logger = EventLogger()
    context = SegmindContext(
        world=world,
        logger=logger,
    )

    from segmind.detectors.atomic_event_detectors_nodes import RotationDetector
    statechart = SegmindStatechart()
    sc = statechart.build_statechart(context)
    rotation_detector = RotationDetector(name="rotation_detector", context=context)
    sc.add_node(rotation_detector)

    cylinder = world.get_body_by_name("cylinder_body")

    segmind_executor = EpisodeSegmenterExecutor(context=context)
    segmind_executor.compile(sc)

    from segmind.datastructures.events import RotationEvent
    assert len([i for i in logger.get_events() if isinstance(i, RotationEvent)]) == 0

    for i in range(5):
        cylinder.parent_connection.origin = (
            HomogeneousTransformationMatrix.from_xyz_rpy(x=1, y=-3, z=0.25, roll=i*0.1)
        )
        segmind_executor.tick()

    assert len([i for i in logger.get_events() if isinstance(i, RotationEvent)]) >= 1


def test_stop_rotation():
    world = setup_contact_world()
    logger = EventLogger()
    context = SegmindContext(
        world=world,
        logger=logger,
    )

    from segmind.detectors.atomic_event_detectors_nodes import RotationDetector, StopRotationDetector
    statechart = SegmindStatechart()
    sc = statechart.build_statechart(context)
    sc.add_node(RotationDetector(name="rotation_detector", context=context))
    sc.add_node(StopRotationDetector(name="stop_rotation_detector", context=context))

    cylinder = world.get_body_by_name("cylinder_body")

    segmind_executor = EpisodeSegmenterExecutor(context=context)
    segmind_executor.compile(sc)

    from segmind.datastructures.events import RotationEvent, StopRotationEvent

    assert len([i for i in logger.get_events() if isinstance(i, RotationEvent)]) == 0

    for i in range(5):
        cylinder.parent_connection.origin = (
            HomogeneousTransformationMatrix.from_xyz_rpy(x=1, y=-3, z=0.25, roll=i*0.1)
        )
        segmind_executor.tick()
    assert len([i for i in logger.get_events() if isinstance(i, RotationEvent)]) >= 1

    for _ in range(5):
        segmind_executor.tick()
    assert len([i for i in logger.get_events() if isinstance(i, StopRotationEvent)]) >= 1




