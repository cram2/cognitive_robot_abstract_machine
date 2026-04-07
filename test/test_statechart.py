import rclpy
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
from semantic_digital_twin.adapters.ros.visualization.viz_marker import (
    VizMarkerPublisher,
)
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from test import setup_contact_world, setup_support_world


class TestMotionStatechart:


    def test_contact_detector(self):

        world = setup_contact_world()
        self.visualize(world)
        logger = EventLogger()
        self.context = SegmindContext(
            world=world,
            logger=logger,
        )
        self.statechart = SegmindStatechart()
        sc = self.statechart.build_statechart(self.context)

        cylinder = world.get_body_by_name("cylinder_body")

        self.segmind_executor = EpisodeSegmenterExecutor(context=self.context)
        self.segmind_executor.compile(sc)

        assert (
            len(
                [
                    i
                    for i in self.context.logger.get_events()
                    if isinstance(i, ContactEvent)
                ]
            )
            == 0
        )


        cylinder.parent_connection.origin = (
            HomogeneousTransformationMatrix.from_xyz_rpy(y=-0.4)
        )
        self.segmind_executor.tick()

        assert len([i for i in logger.get_events() if isinstance(i, ContactEvent)]) == 2


        self.segmind_executor.tick()

        assert len([i for i in logger.get_events() if isinstance(i, ContactEvent)]) == 2

        cylinder.parent_connection.origin = (
            HomogeneousTransformationMatrix.from_xyz_rpy(z=2)
        )

        self.segmind_executor.tick()
        assert (
            len([i for i in logger.get_events() if isinstance(i, LossOfContactEvent)])
            == 2
        )

        self.segmind_executor.tick()
        assert (
            len([i for i in logger.get_events() if isinstance(i, LossOfContactEvent)])
            == 2
        )

        cylinder.parent_connection.origin = (
            HomogeneousTransformationMatrix.from_xyz_rpy(y=-0.4)
        )
        self.segmind_executor.tick()

        assert len([i for i in logger.get_events() if isinstance(i, ContactEvent)]) == 4

        cylinder.parent_connection.origin = (
            HomogeneousTransformationMatrix.from_xyz_rpy(z=2)
        )

        self.segmind_executor.tick()
        assert (
            len([i for i in logger.get_events() if isinstance(i, LossOfContactEvent)])
            == 4
        )

        # rclpy.shutdown()

    def test_support_detector(self):
        world = setup_support_world()
        self.visualize(world)
        logger = EventLogger()
        self.context = SegmindContext(
            world=world,
            logger=logger,
        )
        self.statechart = SegmindStatechart()
        sc = self.statechart.build_statechart(self.context)

        cylinder = world.get_body_by_name("cylinder_body")
        table = world.get_body_by_name("table_body")
        cabinet = world.get_body_by_name("cabinet")

        self.segmind_executor = EpisodeSegmenterExecutor(context=self.context)
        self.segmind_executor.compile(sc)

        assert len([i for i in logger.get_events() if isinstance(i, SupportEvent)]) == 0
        assert (
            len([i for i in logger.get_events() if isinstance(i, LossOfSupportEvent)])
            == 0
        )

        self.segmind_executor.tick()

        cylinder.parent_connection.origin = (
            HomogeneousTransformationMatrix.from_xyz_rpy(
                x=table.global_pose.x,
                y=table.global_pose.y,
                z=table.global_pose.z + 0.2,
            )
        )
        self.segmind_executor.tick()

        self.segmind_executor.tick()
        assert len([i for i in logger.get_events() if isinstance(i, SupportEvent)]) == 1

        cylinder.parent_connection.origin = (
            HomogeneousTransformationMatrix.from_xyz_rpy(
                x=cabinet.global_pose.x,
                y=cabinet.global_pose.y,
                z=cabinet.global_pose.z,
            )
        )
        self.segmind_executor.tick()
        self.segmind_executor.tick()

        assert (len([i for i in logger.get_events() if isinstance(i, LossOfSupportEvent)]) == 1)

        # rclpy.shutdown()

    def test_containment_detector(self):
        world = setup_support_world()
        self.visualize(world)
        logger = EventLogger()
        self.context = SegmindContext(
            world=world,
            logger=logger,
        )
        self.statechart = SegmindStatechart()
        sc = self.statechart.build_statechart(self.context)

        cylinder = world.get_body_by_name("cylinder_body")
        cabinet = world.get_body_by_name("cabinet")
        table = world.get_body_by_name("table_body")

        self.segmind_executor = EpisodeSegmenterExecutor(context=self.context)
        self.segmind_executor.compile(sc)

        assert len([i for i in logger.get_events() if isinstance(i, ContactEvent)]) == 0
        assert len([i for i in logger.get_events() if isinstance(i, ContainmentEvent)]) == 0


        cylinder.parent_connection.origin = (
            HomogeneousTransformationMatrix.from_xyz_rpy(
                x=cabinet.global_pose.x,
                y=cabinet.global_pose.y,
                z=cabinet.global_pose.z,
            )
        )

        self.segmind_executor.tick()

        assert len([i for i in logger.get_events() if isinstance(i, ContainmentEvent)]) == 1

        cylinder.parent_connection.origin = (
            HomogeneousTransformationMatrix.from_xyz_rpy(
                x=table.global_pose.x,
                y=table.global_pose.y,
                z=table.global_pose.z + 0.2,
            )
        )
        self.segmind_executor.tick()


        assert len([i for i in logger.get_events() if isinstance(i, LossOfContainmentEvent)]) == 1


        # rclpy.shutdown()

    def test_insertion_detector(self):
        world = setup_support_world()
        self.visualize(world)
        logger = EventLogger()
        self.context = SegmindContext(
            world=world,
            logger=logger,
        )
        self.statechart = SegmindStatechart()
        sc = self.statechart.build_statechart(self.context)

        hole = world.get_body_by_name("hole_body")
        cylinder = world.get_body_by_name("cylinder_body")
        cabinet = world.get_body_by_name("cabinet")

        self.segmind_executor = EpisodeSegmenterExecutor(context=self.context)
        self.segmind_executor.compile(sc)


        assert len(self.context.holes) == 1
        assert len([i for i in logger.get_events() if  isinstance(i, InsertionEvent)]) == 0
        cylinder.parent_connection.origin = (
            HomogeneousTransformationMatrix.from_xyz_rpy(
                x=hole.global_pose.x,
                y=hole.global_pose.y - 0.03,
                z=hole.global_pose.z,
            )
        )
        self.segmind_executor.tick()


        cylinder.parent_connection.origin = (
            HomogeneousTransformationMatrix.from_xyz_rpy(
                x=cabinet.global_pose.x,
                y=cabinet.global_pose.y,
                z=cabinet.global_pose.z,
            )
        )

        self.segmind_executor.tick()

        contact_events = [i for i in self.context.logger.get_events() if isinstance(i, ContactEvent)]
        contact_events_with_holes = [i for i in contact_events if i.with_object in self.context.holes]

        assert len([i for i in logger.get_events() if isinstance(i, ContainmentEvent)]) == 1
        assert len(contact_events_with_holes) == 1
        assert len([i for i in logger.get_events() if isinstance(i, InsertionEvent)]) == 1

    def test_pickup(self):
        world = setup_support_world()
        self.visualize(world)
        logger = EventLogger()
        self.context = SegmindContext(
            world=world,
            logger=logger,
        )
        self.statechart = SegmindStatechart()
        sc = self.statechart.build_statechart(self.context)

        cylinder = world.get_body_by_name("cylinder_body")
        table = world.get_body_by_name("table_body")

        self.segmind_executor = EpisodeSegmenterExecutor(context=self.context)
        self.segmind_executor.compile(sc)

        # Initial state: supported by table
        cylinder.parent_connection.origin = (
            HomogeneousTransformationMatrix.from_xyz_rpy(
                x=table.global_pose.x,
                y=table.global_pose.y,
                z=table.global_pose.z + 0.2,
            )
        )
        self.segmind_executor.tick()
        self.segmind_executor.tick()
        assert len([i for i in logger.get_events() if isinstance(i, SupportEvent)]) == 1

        # Move cylinder up to trigger Translation and LossOfSupport
        # We need a few ticks for TranslationDetector window
        for i in range(5):
            cylinder.parent_connection.origin = (
                HomogeneousTransformationMatrix.from_xyz_rpy(
                    x=table.global_pose.x,
                    y=table.global_pose.y,
                    z=table.global_pose.z + 0.3 + i * 0.1,
                )
            )
            self.segmind_executor.tick()

        assert len([i for i in logger.get_events() if isinstance(i, TranslationEvent)]) >= 1
        assert len([i for i in logger.get_events() if isinstance(i, LossOfSupportEvent)]) == 1
        assert len([i for i in logger.get_events() if isinstance(i, PickUpEvent)]) == 1

        # # rclpy.shutdown()

    def test_placing(self):
        world = setup_support_world()
        self.visualize(world)
        logger = EventLogger()
        self.context = SegmindContext(
            world=world,
            logger=logger,
        )
        self.statechart = SegmindStatechart()
        sc = self.statechart.build_statechart(self.context)

        cylinder = world.get_body_by_name("cylinder_body")
        table = world.get_body_by_name("table_body")

        self.segmind_executor = EpisodeSegmenterExecutor(context=self.context)
        self.segmind_executor.compile(sc)

        # Start moving
        for i in range(5):
            cylinder.parent_connection.origin = (
                HomogeneousTransformationMatrix.from_xyz_rpy(
                    x=table.global_pose.x,
                    y=table.global_pose.y,
                    z=table.global_pose.z + 0.5 - i * 0.05,
                )
            )
            self.segmind_executor.tick()

        assert len([i for i in logger.get_events() if isinstance(i, TranslationEvent)]) >= 1

        # Place on table (SupportEvent + StopTranslationEvent)
        cylinder.parent_connection.origin = (
            HomogeneousTransformationMatrix.from_xyz_rpy(
                x=table.global_pose.x,
                y=table.global_pose.y,
                z=table.global_pose.z + 0.2,
            )
        )
        # Tick multiple times to ensure StopTranslation is detected (window-based)
        for _ in range(5):
            self.segmind_executor.tick()

        assert len([i for i in logger.get_events() if isinstance(i, SupportEvent)]) == 1
        assert len([i for i in logger.get_events() if isinstance(i, StopTranslationEvent)]) == 1
        assert len([i for i in logger.get_events() if isinstance(i, PlacingEvent)]) == 1

        # rclpy.shutdown()

    def test_translation(self):
        world = setup_contact_world()
        self.visualize(world)
        logger = EventLogger()
        self.context = SegmindContext(
            world=world,
            logger=logger,
        )
        self.statechart = SegmindStatechart()
        sc = self.statechart.build_statechart(self.context)

        cylinder = world.get_body_by_name("cylinder_body")

        self.segmind_executor = EpisodeSegmenterExecutor(context=self.context)
        self.segmind_executor.compile(sc)

        assert len([i for i in logger.get_events() if isinstance(i, TranslationEvent)]) == 0

        # Move cylinder
        for i in range(5):
            cylinder.parent_connection.origin = (
                HomogeneousTransformationMatrix.from_xyz_rpy(x=1 + i * 0.1, y=-3, z=0.25)
            )
            self.segmind_executor.tick()

        assert len([i for i in logger.get_events() if isinstance(i, TranslationEvent)]) == 1

        # rclpy.shutdown()

    def test_stop_translation(self):
        world = setup_contact_world()
        self.visualize(world)
        logger = EventLogger()
        self.context = SegmindContext(
            world=world,
            logger=logger,
        )
        self.statechart = SegmindStatechart()
        sc = self.statechart.build_statechart(self.context)

        cylinder = world.get_body_by_name("cylinder_body")

        self.segmind_executor = EpisodeSegmenterExecutor(context=self.context)
        self.segmind_executor.compile(sc)

        # Move cylinder
        for i in range(5):
            cylinder.parent_connection.origin = (
                HomogeneousTransformationMatrix.from_xyz_rpy(x=1 + i * 0.1, y=-3, z=0.25)
            )
            self.segmind_executor.tick()

        assert len([i for i in logger.get_events() if isinstance(i, TranslationEvent)]) == 1

        # Stop moving
        for _ in range(5):
            self.segmind_executor.tick()

        assert len([i for i in logger.get_events() if isinstance(i, StopTranslationEvent)]) == 1

        # rclpy.shutdown()

    def test_rotation(self):
        world = setup_contact_world()
        self.visualize(world)
        logger = EventLogger()
        self.context = SegmindContext(
            world=world,
            logger=logger,
        )
        
        from segmind.detectors.atomic_event_detectors_nodes import RotationDetector
        self.statechart = SegmindStatechart()
        sc = self.statechart.build_statechart(self.context)
        rotation_detector = RotationDetector(name="rotation_detector", context=self.context)
        sc.add_node(rotation_detector)

        cylinder = world.get_body_by_name("cylinder_body")

        self.segmind_executor = EpisodeSegmenterExecutor(context=self.context)
        self.segmind_executor.compile(sc)

        from segmind.datastructures.events import RotationEvent
        assert len([i for i in logger.get_events() if isinstance(i, RotationEvent)]) == 0

        # Rotate cylinder
        for i in range(5):
            cylinder.parent_connection.origin = (
                HomogeneousTransformationMatrix.from_xyz_rpy(x=1, y=-3, z=0.25, roll=i*0.1)
            )
            self.segmind_executor.tick()

        assert len([i for i in logger.get_events() if isinstance(i, RotationEvent)]) >= 1

        # rclpy.shutdown()

    def test_stop_rotation(self):
        world = setup_contact_world()
        self.visualize(world)
        logger = EventLogger()
        self.context = SegmindContext(
            world=world,
            logger=logger,
        )
        
        from segmind.detectors.atomic_event_detectors_nodes import RotationDetector, StopRotationDetector
        self.statechart = SegmindStatechart()
        sc = self.statechart.build_statechart(self.context)
        sc.add_node(RotationDetector(name="rotation_detector", context=self.context))
        sc.add_node(StopRotationDetector(name="stop_rotation_detector", context=self.context))

        cylinder = world.get_body_by_name("cylinder_body")

        self.segmind_executor = EpisodeSegmenterExecutor(context=self.context)
        self.segmind_executor.compile(sc)

        from segmind.datastructures.events import RotationEvent, StopRotationEvent
        
        assert len([i for i in logger.get_events() if isinstance(i, RotationEvent)]) == 0

        # Rotate
        for i in range(5):
            cylinder.parent_connection.origin = (
                HomogeneousTransformationMatrix.from_xyz_rpy(x=1, y=-3, z=0.25, roll=i*0.1)
            )
            self.segmind_executor.tick()
        assert len([i for i in logger.get_events() if isinstance(i, RotationEvent)]) >= 1

        # Stop rotating
        for _ in range(5):
            self.segmind_executor.tick()
        assert len([i for i in logger.get_events() if isinstance(i, StopRotationEvent)]) >= 1

        # rclpy.shutdown()



    def visualize(self, world):
        if not rclpy.ok():
            rclpy.init()
        node = rclpy.create_node("test_node")
        viz = VizMarkerPublisher(_world=world, node=node)
        viz.with_tf_publisher()
