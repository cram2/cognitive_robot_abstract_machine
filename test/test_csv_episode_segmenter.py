import datetime
import os
import shutil
import threading
from os.path import dirname
from pathlib import Path
from unittest import TestCase

import rclpy

import pycram.ros
from cognitive_robot_abstract_machine.test.krrood_test.dataset.semantic_world_like_classes import FixedConnection
from pycram.datastructures.enums import WorldMode
from pycram.datastructures.pose import PoseStamped
from pycram.testing import SemanticWorldTestCase, setup_world
from semantic_digital_twin.adapters.ros.visualization.viz_marker import VizMarkerPublisher
from semantic_digital_twin.adapters.urdf import URDFParser
from semantic_digital_twin.orm.model import HomogeneousTransformationMatrixMapping
from semantic_digital_twin.spatial_types import Vector3
from semantic_digital_twin.spatial_types.spatial_types import Pose, HomogeneousTransformationMatrix
from semantic_digital_twin.world import World
from sqlalchemy import create_engine
from sqlalchemy.orm.session import Session

from segmind.datastructures.events import ContainmentEvent
from segmind.detectors.coarse_event_detectors import GeneralPickUpDetector, PlacingDetector
from segmind.detectors.spatial_relation_detector import InsertionDetector, SupportDetector, ContainmentDetector
from segmind.episode_segmenter import NoAgentEpisodeSegmenter
from segmind.players.csv_player import CSVEpisodePlayer
from semantic_digital_twin.world_description.world_entity import Agent, Region, Body, Connection

# from segmind.orm.ormatic_interface import *
try:
    from pycram.worlds.multiverse2 import Multiverse
except ImportError:
    Multiverse = None


class TestMultiverseEpisodeSegmenter(TestCase):
    world: World
    file_player: CSVEpisodePlayer
    episode_segmenter: NoAgentEpisodeSegmenter
    viz_marker_publisher: VizMarkerPublisher

    @classmethod
    def setUpClass(cls):
        multiverse_episodes_dir = f"{dirname(__file__)}/../resources/multiverse_episodes"
        selected_episode = "icub_montessori_no_hands"
        episode_dir = os.path.join(multiverse_episodes_dir, selected_episode)
        csv_file = os.path.join(episode_dir, f"data.csv")
        models_dir = os.path.join(episode_dir, "models")
        cls.world = setup_world()

        cls.spawn_objects(models_dir)
        rclpy.init()
        cls.node = rclpy.create_node("test_node")
        cls.viz_marker_publisher = VizMarkerPublisher(world=cls.world, node=cls.node)
        cls.file_player = CSVEpisodePlayer(csv_file, world=cls.world,
                                           time_between_frames=datetime.timedelta(milliseconds=4),
                                           position_shift=Vector3(0, 0, -0.05))
        cls.episode_segmenter = NoAgentEpisodeSegmenter(cls.file_player, annotate_events=True,
                                                        plot_timeline=True,
                                                        plot_save_path=f'{dirname(__file__)}/test_results/{Path(dirname(csv_file)).stem}',
                                                        detectors_to_start=[GeneralPickUpDetector, PlacingDetector],
                                                        initial_detectors=[InsertionDetector, SupportDetector,
                                                                           ContainmentDetector])

    @classmethod
    def spawn_objects(cls, models_dir):
        cls.copy_model_files_to_world_data_dir(models_dir)
        directory = Path(models_dir)
        urdf_files = [f.name for f in directory.glob('*.urdf')]
        for file in urdf_files:
            file_path = "/home/sorin/dev/workspace/Segmind/resources/multiverse_episodes/icub_montessori_no_hands/models/" + file
            obj_name = Path(file).stem
            pose = PoseStamped()
            if obj_name == "iCub":
                file = "iCub.urdf"
                obj_type = Agent
                pose = [-0.8, 0, 0.55]
                #PoseStamped(Pose(Vector3(-0.8, 0, 0.55)))
            elif obj_name == "scene":
                obj_type = Region
            else:
                obj_type = Body
            try:
                obj_world = URDFParser.from_file(file_path).parse()
                new_obj_connection = HomogeneousTransformationMatrix.from_xyz_rpy(
                x=pose[0], y=pose[1], z=pose[2]
                )
                with cls.world.modify_world():
                    cls.world.merge_world(obj_world,
                                          root_connection=Connection(
                                              parent=cls.world.root,
                                              child=obj_world.root,
                                              parent_T_connection_expression=new_obj_connection
                                          ))

                    #obj = Object(obj_name, obj_type, path=file, pose=pose)
            except Exception as e:
                #import pdb
                #pdb.set_trace()
                print(f"Error: {e}"),
                print(f"Could not spawn object {obj_name} from file {file}. Skipping.")
                continue

    @classmethod
    def copy_model_files_to_world_data_dir(cls, models_dir):
        """
        Copy the model files to the world data directory.
        """
        # Copy the entire folder and its contents
       # shutil.copytree(models_dir, cls.world.conf.cache_dir + "/objects", dirs_exist_ok=True)

    @classmethod
    def tearDownClass(cls):
        cls.viz_marker_publisher._stop_publishing()
        print("Viz marker publisher has been stopped, exiting the world...")
        # cls.world.exit()
        print("World has been exited.")

    def tearDown(self):
        self.episode_segmenter.reset()
        self.file_player.reset()
        print("File player and episode segmenter have been reset.")

    def test_containment_detector(self):
        """
        Test the ContainmentDetector by checking if the iCub is contained within the scene.
        """
        self.episode_segmenter.reset()
        self.episode_segmenter.detectors_to_start = [PlacingDetector]
        self.episode_segmenter.initial_detectors = [ContainmentDetector, SupportDetector]
        self.episode_segmenter.start()
        self.assertTrue(any([isinstance(e, ContainmentEvent) for e in self.episode_segmenter.logger.get_events()]))

    def test_csv_replay(self):
        # engine = create_engine('sqlite:///:memory:')
        # session = Session(engine)
        # mapper_registry.metadata.create_all(engine)
        #
        self.episode_segmenter.start()
        # session.add_all(self.episode_segmenter.logger.get_events())
        # session.commit()
