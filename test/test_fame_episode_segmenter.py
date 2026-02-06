import datetime
from pathlib import Path
from unittest import TestCase
from os.path import dirname

import pycram.ros

from pycram.datastructures.enums import WorldMode
from semantic_digital_twin.adapters.ros.visualization.viz_marker import VizMarkerPublisher
from semantic_digital_twin.semantic_annotations.semantic_annotations import Bowl, Cup
from semantic_digital_twin.world import World

from segmind.players.json_player import JSONPlayer
from segmind.episode_segmenter import NoAgentEpisodeSegmenter
from segmind.players.json_player import JSONPlayer
from segmind.detectors.coarse_event_detectors import GeneralPickUpDetector
from pycram.datastructures.enums import WorldMode


Multiverse = None
try:
    from pycram.worlds.multiverse import Multiverse
except ImportError:
    pass


class TestFileEpisodeSegmenter(TestCase):
    world: World
    file_player: JSONPlayer
    episode_segmenter: NoAgentEpisodeSegmenter
    viz_marker_publisher: VizMarkerPublisher

    @classmethod
    def setUpClass(cls):
        json_file = f"{dirname(__file__)}/../resources/fame_episodes/alessandro_with_ycp_objects_in_max_room_2/refined_poses.json"
        # json_file = "../resources/fame_episodes/alessandro_sliding_bueno/refined_poses.json"
        # simulator = BulletWorld if Multiverse is None else Multiverse
        simulator = BulletWorld
        annotate_events = True if simulator == BulletWorld else False
        cls.world = simulator(WorldMode.DIRECT)
        pycram.ros.set_logger_level(pycram.datastructures.enums.LoggerLevel.DEBUG)
        cls.viz_marker_publisher = VizMarkerPublisher()
        obj_id_to_name = {1: "chips", 3: "bowl", 4: "cup", 6: "bueno"}
        obj_id_to_type = {1: Container, 3: Bowl, 4: Cup, 6: Container}
        cls.file_player = JSONPlayer(json_file, world=cls.world,
                                            time_between_frames=datetime.timedelta(milliseconds=50),
                                            objects_to_ignore=[5],
                                            obj_id_to_name=obj_id_to_name,
                                            obj_id_to_type=obj_id_to_type)
        cls.episode_segmenter = NoAgentEpisodeSegmenter(cls.file_player, annotate_events=annotate_events,
                                                        plot_timeline=True,
                                                        plot_save_path=f'{dirname(__file__)}/test_results/{Path(dirname(json_file)).stem}',
                                                        detectors_to_start=[GeneralPickUpDetector])

    @classmethod
    def tearDownClass(cls):
        cls.viz_marker_publisher._stop_publishing()
        cls.world.exit()
        cls.episode_segmenter.join()

    def test_replay_episode(self):
        self.episode_segmenter.start()
