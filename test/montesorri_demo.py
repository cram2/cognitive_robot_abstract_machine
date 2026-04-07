import datetime
from unittest import TestCase
import rclpy
from segmind.detectors.base import DetectorStateChart, SegmindContext
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
            file_path="/home/sorin/dev/Segmind/resources/multiverse_episodes/icub_montessori_no_hands/data.csv",
            world=cls.world,
            time_between_frames=datetime.timedelta(milliseconds=4),
            position_shift=Vector3(0, 0, 0),
        )
        cls.episode_executor = EpisodeSegmenterExecutor(
            context=cls.context, player=cls.file_player, ignored_objects=["iCub"]
        )
        cls.episode_executor.spawn_scene(
            models_dir="/home/sorin/dev/Segmind/resources/multiverse_episodes/icub_montessori_no_hands/models/"
        )

    def test_replay_episode(self):
        statechart = SegmindStatechart()
        sc = statechart.build_statechart(self.context)

        self.episode_executor.compile(sc)

