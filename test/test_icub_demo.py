import datetime
import os
import shutil
import threading
import time
from os.path import dirname
from pathlib import Path

import pytest
from pycram.tf_transformations import quaternion_from_euler
from typing_extensions import Tuple

from segmind.datastructures.events import AbstractAgentObjectInteractionEvent
from segmind.detectors.coarse_event_detectors import GeneralPickUpDetector, PlacingDetector
from segmind.detectors.spatial_relation_detector import InsertionDetector, SupportDetector, ContainmentDetector
from segmind.episode_segmenter import NoAgentEpisodeSegmenter

try:
    from segmind.players.multiverse_player import MultiversePlayer
except ImportError:
    MultiversePlayer = None
    
import pycram
from pycram.datastructures.enums import WorldMode, Arms, Grasp
from pycram.datastructures.grasp import GraspDescription
from pycram.datastructures.pose import PoseStamped,
from pycram.designator import ObjectDesignatorDescription

from pycram.external_interfaces import giskard
from pycram.failures import ObjectNotGraspedError
from pycram.language import SequentialPlan
from pycram.process_module import simulated_robot, real_robot



@pytest.fixture(scope="module")
def set_up_demo_fixture(episode_name: str = "icub_montessori_no_hands"):
    rdm = RobotDescriptionManager()
    rdm.load_description("iCub")

    world: BulletWorld = BulletWorld(WorldMode.DIRECT)
    # viz_marker_publisher = VizMarkerPublisher()
    pycram.ros.set_logger_level(pycram.datastructures.enums.LoggerLevel.ERROR)

    multiverse_episodes_dir = f"{dirname(__file__)}/../resources/multiverse_episodes"
    episode_dir = os.path.join(multiverse_episodes_dir, episode_name)
    models_dir = os.path.join(episode_dir, "models")

    spawn_objects(models_dir)

    csv_file = os.path.join(episode_dir, f"data.csv")
    multiverse_player = MultiversePlayer(world=world,
                                         time_between_frames=datetime.timedelta(milliseconds=4),
                                         stop_after_ready=False)
    multiverse_player.start()
    episode_segmenter = NoAgentEpisodeSegmenter(multiverse_player, annotate_events=True,
                                                    plot_timeline=True,
                                                    plot_save_path=f'{dirname(__file__)}/test_results/multiverse_episode',
                                                    detectors_to_start=[GeneralPickUpDetector, PlacingDetector],
                                                    # initial_detectors=[SupportDetector, ContainmentDetector])
                                                    initial_detectors=[InsertionDetector, SupportDetector, ContainmentDetector])
    # episode_segmenter.start()

    # while not multiverse_player.ready:
    #     time.sleep(0.1)

    yield episode_segmenter
    # viz_marker_publisher._stop_publishing()


def spawn_objects(models_dir):
    copy_model_files_to_world_data_dir(models_dir)
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


def copy_model_files_to_world_data_dir(models_dir):
    """
    Copy the model files to the world data directory.
    """
    # Copy the entire folder and its contents
    shutil.copytree(models_dir, World.current_world.conf.cache_dir + "/objects", dirs_exist_ok=True)


def test_icub_demo(set_up_demo_fixture):
    episode_segmenter = set_up_demo_fixture
    # Create a thread
    thread = threading.Thread(target=episode_segmenter.start)
    # Start the thread
    thread.start()
    time.sleep(10000)
    # input("Press Enter to continue...")

    all_events = episode_segmenter.logger.get_events()
    actionable_events = [event for event in all_events if isinstance(event, AbstractAgentObjectInteractionEvent)]
    for actionable_event in actionable_events:
        print(next(actionable_event.action_description.__iter__()))

    thread.join()


def test_icub_pick_up_and_insert(set_up_demo_fixture):
    # set_up_demo_fixture.episode_player.stop()
    time.sleep(2)
    # set_up_demo_fixture.episode_player.sync_robot_only = True
    obj_name = "montessori_object_5"
    obj = World.current_world.get_object_by_root_link_name(obj_name)
    obj_pose = obj.pose
    obj_pose = World.current_world.get_object_by_name("scene").links["circular_hole_1"].pose
    # obj_pose.position.z += 0.04
    # obj_pose.orientation = Quaternion(*quaternion_from_euler(0, 1.57/2, 0))
    # obj.set_pose(obj_pose)
    arm, grasp = get_arm_and_grasp_description_for_object(obj)

    scene_obj = World.current_world.get_object_by_name("scene")
    square_hole_pose = scene_obj.get_link_pose("circular_hole_1")
    object_description = ObjectDesignatorDescription(names=[obj_name])
    with real_robot:
        plan = SequentialPlan(ParkArmsActionDescription(Arms.BOTH),
                              PickUpActionDescription(object_description,arm=arm,
                                                      grasp_description=grasp),
                              PlaceActionDescription(object_description, square_hole_pose, arm, insert=True)
                              )
        plan.perform()

    # plan.plot()


def get_arm_and_grasp_description_for_object(obj: Object) -> Tuple[Arms, GraspDescription]:
    obj_pose = obj.pose
    left_arm_pose = World.current_world.robot.get_link_pose("l_gripper_tool_frame")
    right_arm_pose = World.current_world.robot.get_link_pose("r_gripper_tool_frame")
    obj_distance_from_left_arm = left_arm_pose.position.euclidean_distance(obj_pose.position)
    obj_distance_from_right_arm = right_arm_pose.position.euclidean_distance(obj_pose.position)
    if obj_distance_from_left_arm < obj_distance_from_right_arm:
        arm = Arms.LEFT
        grasp = GraspDescription(Grasp.LEFT, Grasp.TOP)
    else:
        arm = Arms.RIGHT
        grasp = GraspDescription(Grasp.RIGHT, Grasp.TOP)
    return arm, grasp
