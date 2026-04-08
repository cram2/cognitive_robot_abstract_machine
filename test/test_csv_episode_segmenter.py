import datetime
import os
from os.path import dirname
from unittest import TestCase
from segmind import logger, set_logger_level, LogLevel
import rclpy

from segmind.detectors.base import SegmindContext
from segmind.episode_segmenter import EpisodeSegmenterExecutor
from segmind.event_logger import EventLogger
from semantic_digital_twin.adapters.ros.visualization.viz_marker import (
    VizMarkerPublisher,
)
from semantic_digital_twin.adapters.urdf import URDFParser
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.spatial_types import Vector3

from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import (
    FixedConnection,

)
from segmind.players.csv_player import CSVEpisodePlayer
from semantic_digital_twin.world_description.world_entity import (
    Body,
)


try:
    from pycram.worlds.multiverse2 import Multiverse
except ImportError:
    Multiverse = None

set_logger_level(LogLevel.DEBUG)


class TestMultiverseEpisodeSegmenter(TestCase):
    world: World
    file_player: CSVEpisodePlayer
    episode_segmenter: EpisodeSegmenterExecutor
    viz_marker_publisher: VizMarkerPublisher

    @classmethod
    def setUpClass(cls):
        multiverse_episodes_dir = (
            f"{dirname(__file__)}/../resources/multiverse_episodes"
        )
        selected_episode = "icub_montessori_no_hands"
        episode_dir = os.path.join(multiverse_episodes_dir, selected_episode)
        csv_file = os.path.join(episode_dir, f"data.csv")
        models_dir = os.path.join(episode_dir, "models/")
        cls.world = World()
        cls.logger = EventLogger()
        cls.context = SegmindContext(world=cls.world, logger=cls.logger)
        root = Body(name=PrefixedName(name="root", prefix="world"))
        with cls.world.modify_world():
            cls.world.add_kinematic_structure_entity(root)
        rclpy.init()
        cls.node = rclpy.create_node("test_node")
        cls.viz_marker_publisher = VizMarkerPublisher(node=cls.node, _world=cls.world)
        cls.viz_marker_publisher.with_tf_publisher()
        cls.file_player = CSVEpisodePlayer(
            file_path=csv_file,
            world=cls.world,
            time_between_frames=datetime.timedelta(milliseconds=4),
            position_shift=Vector3(0, 0, 0),
        )
        cls.episode_segmenter = EpisodeSegmenterExecutor(player=cls.file_player, context=cls.context, ignored_objects=["iCub"])
        cls.episode_segmenter.spawn_scene(
            models_dir=models_dir
        )

    def test_csv_replay(self):
        self.episode_segmenter.start()
        assert self.episode_segmenter.player.is_alive()


