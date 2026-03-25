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
    InsertionEvent,
    PickUpEvent,
    LossOfContactEvent, LossOfSupportEvent,
)
from segmind.detectors.atomic_event_detectors import DetectorStateChart
from segmind.detectors.atomic_event_detectors_nodes import (
    SegmindContext,
    ContactDetector,
    LossOfContactDetector,
    TranslationDetector,
    StopTranslationDetector,
)
from segmind.detectors.coarse_event_detector_nodes import (
    PlacingDetector,
    PickUpDetector,
)
from segmind.detectors.spatial_relation_detector_nodes import (
    SupportDetector,
    LossOfSupportDetector,
    ContainmentDetector,
    InsertionDetector,
)
from segmind.episode_segmenter import EpisodeSegmenterExecutor
from segmind.event_logger import EventLogger
from segmind.players.csv_player import CSVEpisodePlayer
from semantic_digital_twin.adapters.ros.visualization.viz_marker import (
    VizMarkerPublisher,
)
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.spatial_types import Vector3
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.world_entity import Body

from segmind.statecharts.segmind_statechart import SegmindStatechart


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
            position_shift=Vector3(0, 0, 0),
        )
        cls.episode_executor = EpisodeSegmenterExecutor(
            context=cls.context, player=cls.file_player
        )
        cls.episode_executor.spawn_scene(
            models_dir="/home/sorin/dev/workspace/Segmind/resources/multiverse_episodes/icub_montessori_no_hands/models/"
        )

    def test_replay_episode(self):
        statechart = SegmindStatechart()
        sc = statechart.build_statechart(self.context)

        self.episode_executor.compile(sc)

        assert len(self.context.holes) > 0

        time.sleep(5)

        self.episode_executor.tick_until_end(10000000)

        translation_events = [
            i for i in self.logger.get_events() if isinstance(i, TranslationEvent)
        ]
        stop_translation_events = [
            i for i in self.logger.get_events() if isinstance(i, StopTranslationEvent)
        ]
        placing_events = [
            i for i in self.logger.get_events() if isinstance(i, PlacingEvent)
        ]

        support_events = [
            i for i in self.logger.get_events() if isinstance(i, SupportEvent)
        ]

        insertion_events = [
            i for i in self.logger.get_events() if isinstance(i, InsertionEvent)
        ]

        contact_events = [
            i for i in self.logger.get_events() if isinstance(i, ContactEvent)
        ]

        containment_events = [
            i for i in self.logger.get_events() if isinstance(i, ContainmentEvent)
        ]

        pickup_events = [
            i for i in self.logger.get_events() if isinstance(i, PickUpEvent)
        ]

        loss_of_support_events = [
            i for i in self.logger.get_events() if isinstance(i, LossOfSupportEvent)
        ]

        loss_of_contact_events = [
            i for i in self.logger.get_events() if isinstance(i, LossOfContactEvent)
        ]

        assert len(self.context.holes) > 0
        assert len(contact_events) > 0
        assert len(loss_of_contact_events) > 0

        # ToDo: Fix bug in placing detector and insertion
        assert len(support_events) >= len(placing_events) > 0
        assert len(translation_events) >= len(stop_translation_events) > 0
        assert len(containment_events) >= len(insertion_events) > 0
        assert len(loss_of_support_events) >= len(pickup_events) >= 0