import datetime
import os
import time
from dataclasses import dataclass
from pathlib import Path
from unittest import TestCase

import rclpy

from segmind.detectors.atomic_event_detectors import DetectorStateChart
from segmind.detectors.atomic_event_detectors_nodes import (
    SegmindContext,
    ContactDetector,
    LossOfContactDetector,
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
        # self.file_player.start()
        sc = DetectorStateChart()
        logger = EventLogger()

        self.context = SegmindContext(
            world=self.world, logger=logger, latest_contact_bodies={}
        )

        contact_detector = ContactDetector(
            name="contact_detector", context=self.context
        )
        loss_of_contact_detector = LossOfContactDetector(
            name="loss_of_contact_detector",
            context=self.context,
        )

        sc.add_nodes([contact_detector, loss_of_contact_detector])
        self.episode_executor.compile(sc)
        time.sleep(5)
        for i in range(100):
            self.episode_executor.tick()

        # self.file_player.start()
        # self.episode_executor.player.start()
        # So the context actually acts as the subcriber.

        # self.episode_executor.tick_until_end()
