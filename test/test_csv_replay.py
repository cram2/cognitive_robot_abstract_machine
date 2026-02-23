import datetime
import os
from pathlib import Path

import rclpy

from cognitive_robot_abstract_machine.test.krrood_test.dataset.semantic_world_like_classes import FixedConnection
from segmind import logger, set_logger_level, LogLevel
from segmind.episode_segmenter import NoAgentEpisodeSegmenter, EpisodeSegmenter
from segmind.players.csv_player import CSVEpisodePlayer
from semantic_digital_twin.adapters.ros.visualization.viz_marker import VizMarkerPublisher
from semantic_digital_twin.adapters.urdf import URDFParser
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.spatial_types import Vector3
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.world_entity import Body

set_logger_level(LogLevel.DEBUG)
class TestCSVReplay:
    world: World
    file_player: CSVEpisodePlayer
    viz_marker_publisher: VizMarkerPublisher


    def test_csv_replay(self):
        csv_file = "/home/sorin/dev/workspace/Segmind/resources/multiverse_episodes/icub_montessori_no_hands/data.csv"
        self.world = World()
        self.visualize(self.world)
        root = Body(name=PrefixedName(name="root", prefix="world"))
        models_dir = os.path.join(
            "/home/sorin/dev/workspace/Segmind/resources/multiverse_episodes/icub_montessori_no_hands",
            "models/")
        directory = Path(models_dir)
        urdf_files = [f.name for f in directory.glob("montessori*.urdf")]
        with self.world.modify_world():
            self.world.add_kinematic_structure_entity(root)

        for file in urdf_files:
            file_path = models_dir + file
            try:
                obj = URDFParser.from_file(file_path).parse()
                self.world.merge_world(
                    obj)
            except Exception as e:
                logger.debug(f"Error: {e}"),
                logger.debug(f"Could not spawn object with file_path {file_path} from file {file}. Skipping.")
                continue
        self.file_player = CSVEpisodePlayer(csv_file, world=self.world,
                                            time_between_frames=datetime.timedelta(milliseconds=4),
                                            position_shift=Vector3(0,0,-0.05)
        )
        self.episode_segmenter = NoAgentEpisodeSegmenter(episode_player=self.file_player)
        self.episode_segmenter.start()

    def visualize(self, world):
        logger.debug("Starting Visualization")
        rclpy.init()
        self.node = rclpy.create_node("test_node")
        self.world = world
        logger.debug("Node created")
        self.viz_marker_publisher = VizMarkerPublisher(world=self.world, node=self.node)
        self.viz_marker_publisher.with_tf_publisher()

