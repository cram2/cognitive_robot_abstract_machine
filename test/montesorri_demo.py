import datetime
import os
import time
from dataclasses import dataclass
from pathlib import Path
from unittest import TestCase

import rclpy

from segmind.datastructures.events import (
    SupportEvent,
    ContactEvent,
    ContainmentEvent,
    TranslationEvent,
    StopTranslationEvent,
    PlacingEvent, InsertionEvent, PickUpEvent,
)
from segmind.detectors.atomic_event_detectors import DetectorStateChart
from segmind.detectors.atomic_event_detectors_nodes import (
    SegmindContext,
    ContactDetector,
    LossOfContactDetector,
    TranslationDetector,
    StopTranslationDetector,
)
from segmind.detectors.coarse_event_detector_nodes import PlacingDetector, PickUpDetector
from segmind.detectors.spatial_relation_detector_nodes import (
    SupportDetector,
    LossOfSupportDetector,
    ContainmentDetector, InsertionDetector,
)
from segmind.episode_segmenter import EpisodeSegmenterExecutor
from segmind.event_logger import EventLogger
from segmind.players.csv_player import CSVEpisodePlayer
from segmind.players.data_player import DataPlayer
from semantic_digital_twin.adapters.ros.visualization.viz_marker import (
    VizMarkerPublisher,
)
from semantic_digital_twin.adapters.urdf import URDFParser
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.spatial_types import Vector3
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import FixedConnection
from semantic_digital_twin.world_description.world_entity import Body


class TestMultiverseEpisodeSegmenter(TestCase):

    @classmethod
    def setUpClass(cls):
        cls.world = World()
        root = Body(name=PrefixedName(name="root", prefix="world"))
        with cls.world.modify_world():
            cls.world.add_kinematic_structure_entity(root)
        rclpy.init()
        cls.node = rclpy.create_node("test_node")
        cls.viz_marker_publisher = VizMarkerPublisher(node=cls.node, _world=cls.world)

        cls.viz_marker_publisher.with_tf_publisher()
        cls.logger = EventLogger()
        cls.sc = DetectorStateChart()
        cls.context = SegmindContext(world=cls.world, logger=cls.logger)
        cls.file_player = CSVEpisodePlayer(
            file_path="/home/sorin/dev/workspace/Segmind/resources/multiverse_episodes/icub_montessori_no_hands/data.csv",
            world=cls.world,
            time_between_frames=datetime.timedelta(milliseconds=4),
            position_shift=Vector3(0, 0, -0.05),
        )
        cls.episode_executor = EpisodeSegmenterExecutor(
            context=cls.context, player=cls.file_player
        )
        cls.episode_executor.spawn_scene(
            models_dir="/home/sorin/dev/workspace/Segmind/resources/multiverse_episodes/icub_montessori_no_hands/models/"
        )

    def test_replay_episode(self):
        sc = DetectorStateChart()
        logger = EventLogger()


        self.context = self.episode_executor.context

        contact_detector = ContactDetector(
            name="contact_detector", context=self.context
        )
        loss_of_contact_detector = LossOfContactDetector(
            name="loss_of_contact_detector",
            context=self.context,
        )
        support_detector = SupportDetector(
            name="support_detector",
            context=self.context,
        )
        loss_of_support_detector = LossOfSupportDetector(
            name="los_detector",
            context=self.context,
        )
        containment_detector = ContainmentDetector(
            name="containment_detector",
            context=self.context,
        )
        translation_detector = TranslationDetector(
            name="translation_detector", context=self.context
        )

        stop_translation_detector = StopTranslationDetector(
            name="stop_translation_detector", context=self.context
        )

        placing_detector = PlacingDetector(
            name="placing_detector", context=self.context
        )

        insertion_detector = InsertionDetector(
            name="insertion_detector", context=self.context
        )

        pickup_detector = PickUpDetector(
            name="pickup_detector", context=self.context
        )

        sc.add_nodes(
            [
                contact_detector,
                loss_of_contact_detector,
                support_detector,
                loss_of_support_detector,
                translation_detector,
                containment_detector,
                stop_translation_detector,
                placing_detector,
                insertion_detector,
                pickup_detector
            ]
        )

        support_detector.start_condition = contact_detector.observation_variable
        loss_of_support_detector.start_condition = (
            loss_of_contact_detector.observation_variable
        )
        containment_detector.start_condition = support_detector.observation_variable
        placing_detector.start_condition = support_detector.observation_variable
        insertion_detector.start_condition = containment_detector.observation_variable
        pickup_detector.start_condition = loss_of_support_detector.observation_variable

        self.episode_executor.compile(sc)

        print(f"Number of holes: {len(self.context.holes)}")
        assert len(self.context.holes) > 0

        time.sleep(5)
        while self.episode_executor.player.is_alive():
            time.sleep(0.01)
            self.episode_executor.tick()

        translation_events = [
            i for i in self.logger.get_events() if isinstance(i, TranslationEvent)
        ]
        stop_translation_events = [
            i for i in self.logger.get_events() if isinstance(i, StopTranslationEvent)
        ]
        placing_events = [i for i in self.logger.get_events() if isinstance(i, PlacingEvent)]

        support_events = [i for i in self.logger.get_events() if isinstance(i, SupportEvent)]

        insertion_events = [i for i in self.logger.get_events() if isinstance(i, InsertionEvent)]

        contact_events = [i for i in self.logger.get_events() if isinstance(i, ContactEvent)]

        containment_events = [i for i in self.logger.get_events() if isinstance(i, ContainmentEvent)]

        pickup_events = [i for i in self.logger.get_events() if isinstance(i, PickUpEvent)]

        print(f"Number of pickup events: {len(pickup_events)}")
        print(f"Number of support events: {len(support_events)}")
        print(f"Number of translation events: {len(translation_events)}")
        print(f"Number of stop translation events: {len(stop_translation_events)}")
        print(f"Number of placing events: {len(placing_events)}")
        print(f"Number of insertion events: {len(insertion_events)}")
        print(f"Number of contact events: {len(containment_events)}")

        for e in translation_events:
            print(f"Translation Event: {e}")

        for e in stop_translation_events:
            print(f"Stop Translation Event: {e}")

        for e in placing_events:
            print(f"Placing Event: {e}")

        for e in support_events:
            print(f"Support Event: {e}")

        assert len([i for i in logger.get_events() if isinstance(i, SupportEvent)]) > 0
        assert len([i for i in logger.get_events() if isinstance(i, ContactEvent)]) > 0
        for e in insertion_events:
            print(f"Insertion Event: {e}")

        for e in contact_events:
            print(f"Contact Event: {e}")

        for e in containment_events:
            print(f"Containment Event: {e}")

        for e in pickup_events:
            print(f"Pickup Event: {e}")

        assert len([i for i in self.logger.get_events() if isinstance(i, SupportEvent)]) > 0
        assert len([i for i in self.logger.get_events() if isinstance(i, ContactEvent)]) > 0
        assert (
            len([i for i in self.logger.get_events() if isinstance(i, ContainmentEvent)]) > 0
        )
        assert (
            len(translation_events) == len(stop_translation_events)
            and len(translation_events) > 0
        )

        self.episode_executor.statechart.draw("/home/sorin/dev/Segmind/plots/" + "sony.pdf")
