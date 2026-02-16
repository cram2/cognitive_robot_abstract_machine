import datetime
import os
import shutil
from datetime import timedelta
from pathlib import Path
from unittest import TestCase
from os.path import dirname
import pytest


import pycram.ros
from pycram.datastructures.pose import PoseStamped
from semantic_digital_twin.adapters.ros.visualization.viz_marker import VizMarkerPublisher
from semantic_digital_twin.world import World
from segmind.players.csv_player import CSVEpisodePlayer
from segmind.episode_segmenter import NoAgentEpisodeSegmenter
from segmind.detectors.coarse_event_detectors import GeneralPickUpDetector
from segmind.detectors.spatial_relation_detector import InsertionDetector


try:
    from segmind.players.multiverse_player import MultiversePlayer
except ImportError:
    MultiversePlayer = None

try:
    from pycram.worlds.multiverse2 import Multiverse
except ImportError:
    Multiverse = None


@pytest.mark.skipif(MultiversePlayer is None, reason="MultiversePlayer not available")
class TestMultiverseEpisodeSegmenter(TestCase):
    world: World
    player: MultiversePlayer
    episode_segmenter: NoAgentEpisodeSegmenter
    viz_marker_publisher: VizMarkerPublisher

    @classmethod
    def setUpClass(cls):
        rdm = RobotDescriptionManager()
        rdm.load_description("iCub")

        cls.world: BulletWorld = BulletWorld(WorldMode.GUI)
        pycram.ros.set_logger_level(pycram.datastructures.enums.LoggerLevel.ERROR)
        cls.viz_marker_publisher = VizMarkerPublisher()

        episode_name = "icub_montessori_no_hands"
        cls.spawn_objects(f"{dirname(__file__)}/../resources/multiverse_episodes/{episode_name}/models")

        cls.player = MultiversePlayer(world=cls.world,
                                      time_between_frames=datetime.timedelta(milliseconds=4))

        cls.episode_segmenter = NoAgentEpisodeSegmenter(cls.player, annotate_events=True,
                                                        plot_timeline=True,
                                                        plot_save_path=f'{dirname(__file__)}/test_results/multiverse_episode',
                                                        detectors_to_start=[GeneralPickUpDetector],
                                                        initial_detectors=[InsertionDetector])

    @classmethod
    def spawn_objects(cls, models_dir):
        cls.copy_model_files_to_world_data_dir(models_dir)
        directory = Path(models_dir)
        urdf_files = [f.name for f in directory.glob('*.urdf')]
        for file in urdf_files:
            obj_name = Path(file).stem
            pose = PoseStamped()
            if obj_name == "iCub":
                obj_type = Robot
                pose = PoseStamped(Pose(Vector3(-0.8, 0, 0.55)))
            elif obj_name == "scene":
                obj_type = Location
            else:
                obj_type = PhysicalObject
            obj = Object(obj_name, obj_type, path=file, pose=pose)

    @classmethod
    def copy_model_files_to_world_data_dir(cls, models_dir):
        """
        Copy the model files to the world data directory.
        """
        # Copy the entire folder and its contents
        shutil.copytree(models_dir, cls.world.conf.cache_dir + "/objects", dirs_exist_ok=True)

    @classmethod
    def tearDownClass(cls):
        cls.viz_marker_publisher._stop_publishing()
        cls.world.exit()
        cls.episode_segmenter.join()

    def test_multiverse_replay(self):
        self.episode_segmenter.start()
