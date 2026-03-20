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
    PlacingEvent,
)
from segmind.detectors.atomic_event_detectors import DetectorStateChart
from segmind.detectors.atomic_event_detectors_nodes import (
    SegmindContext,
    ContactDetector,
    LossOfContactDetector,
    TranslationDetector,
    StopTranslationDetector,
)
from segmind.detectors.coarse_event_detector_nodes import PlacingDetector
from segmind.detectors.spatial_relation_detector_nodes import (
    SupportDetector,
    LossOfSupportDetector,
    ContainmentDetector,
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
        cls.context = SegmindContext(world=cls.world)
        cls.file_player = CSVEpisodePlayer(
            file_path="/home/sorin/dev/workspace/Segmind/resources/multiverse_episodes/icub_montessori_no_hands/data.csv",
            world=cls.world,
            time_between_frames=datetime.timedelta(milliseconds=4),
            position_shift=Vector3(0, 0, 0),
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

        self.context = SegmindContext(world=self.world, logger=logger)

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
            ]
        )

        support_detector.start_condition = contact_detector.observation_variable
        loss_of_support_detector.start_condition = (
            loss_of_contact_detector.observation_variable
        )
        containment_detector.start_condition = support_detector.observation_variable
        placing_detector.start_condition = support_detector.observation_variable
        self.episode_executor.compile(sc)
        time.sleep(5)
        while self.episode_executor.player.is_alive():
            time.sleep(0.1)
            self.episode_executor.tick()

        translation_events = [
            i for i in logger.get_events() if isinstance(i, TranslationEvent)
        ]
        stop_translation_events = [
            i for i in logger.get_events() if isinstance(i, StopTranslationEvent)
        ]
        placing_events = [i for i in logger.get_events() if isinstance(i, PlacingEvent)]

        support_events = [i for i in logger.get_events() if isinstance(i, SupportEvent)]

        print(f"Number of support events: {len(support_events)}")
        print(f"Number of translation events: {len(translation_events)}")
        print(f"Number of stop translation events: {len(stop_translation_events)}")
        print(f"Number of placing events: {len(placing_events)}")
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
        assert (
            len([i for i in logger.get_events() if isinstance(i, ContainmentEvent)]) > 0
        )
        assert (
            len(translation_events) == len(stop_translation_events)
            and len(translation_events) > 0
        )
        print(f"Number of holes: {len(self.episode_executor.context.holes)}")
        assert len(self.episode_executor.context.holes) > 0
