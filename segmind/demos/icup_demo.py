import datetime
import os
import shutil
import threading
import time
from os.path import dirname
from pathlib import Path
from typing import Tuple, List, Optional

import pycram
from pycram.datastructures.enums import WorldMode, Arms
from pycram.datastructures.grasp import GraspDescription
from pycram.datastructures.partial_designator import PartialDesignator
from pycram.datastructures.pose import PoseStamped, Pose, Vector3
from pycram.datastructures.world import World
from pycram.datastructures.world_entity import PhysicalBody
from pycram.datastructures.dataclasses import AxisAlignedBoundingBox
from pycram.designator import ActionDescription
from pycram.designators.action_designator import PickUpActionDescription, PlaceActionDescription, \
    ParkArmsActionDescription, PlaceAction
from pycram.language import SequentialPlan
from pycram.process_module import real_robot
from pycram.robot_description import RobotDescriptionManager, RobotDescription
from pycram.ros import logerr, loginfo, logdebug
from pycram.world_concepts.world_object import Object, Link
from pycram.worlds.bullet_world import BulletWorld
from pycrap.ontologies import Robot, Location, PhysicalObject
from typing_extensions import Dict, Set

from segmind.datastructures.events import AbstractAgentObjectInteractionEvent, PlacingEvent, PickUpEvent, InsertionEvent
from segmind.datastructures.object_tracker import ObjectTrackerFactory, ObjectTracker
from segmind.detectors.coarse_event_detectors import GeneralPickUpDetector, select_transportable_objects, \
    PlacingDetector
from segmind.detectors.spatial_relation_detector import InsertionDetector, SupportDetector, ContainmentDetector
from segmind.episode_segmenter import NoAgentEpisodeSegmenter
from segmind.players.multiverse_player import MultiversePlayer
from segmind.utils import get_arm_and_grasp_description_for_object, text_to_speech
# from PyQt6.QtWidgets import QApplication
# from ripple_down_rules.user_interface.gui import RDRCaseViewer
import numpy as np
import sys

objects_dir = World.conf.cache_dir + "/objects"

obj_name_map = {"montessori_object_1": "Disk",
                "montessori_object_4": "Sphere",
                "montessori_object_6": "Cylinder",
                "montessori_object_3": "Cube",
                "montessori_object_5": "Cuboid",
                "montessori_object_2": "Triangle", }

obj_hole_map = {"montessori_object_1": "disk_hole",
                "montessori_object_4": "circular_hole_2",
                "montessori_object_6": "circular_hole_1",
                "montessori_object_3": "square_hole",
                "montessori_object_5": "rectangular_hole",
                "montessori_object_2": "triangle_hole", }

use_gui = False
# app: Optional[QApplication] = None
# viewer: Optional[RDRCaseViewer] = None
# if RDRCaseViewer is not None and QApplication is not None and use_gui:
#     app = QApplication(sys.argv)
#     viewer = RDRCaseViewer()

def spawn_objects(models_dir: str):
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


def copy_model_files_to_world_data_dir(models_dir: str):
    """
    Copy the model files to the world data directory.
    """
    # Copy the entire folder and its contents
    shutil.copytree(models_dir, objects_dir, dirs_exist_ok=True)


rdm = RobotDescriptionManager()
rdm.load_description("iCub")

world: BulletWorld = BulletWorld(WorldMode.GUI)

# viz_marker_publisher = VizMarkerPublisher()
pycram.ros.set_logger_level(pycram.datastructures.enums.LoggerLevel.ERROR)

multiverse_episodes_dir = f"{dirname(__file__)}/../resources/multiverse_episodes"
episode_name = "icub_montessori_no_hands"
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
                                            initial_detectors=[InsertionDetector, SupportDetector, ContainmentDetector])

# Create a thread
thread = threading.Thread(target=episode_segmenter.start)
# Start the thread
thread.start()

time.sleep(5)

match_shapes: bool = False
pickable_objects: Set[PhysicalBody] = set()
latest_event: Optional[AbstractAgentObjectInteractionEvent] = None
all_inserted_objects: Set[PhysicalBody] = set()
excluded_objects: List[Object] = [World.current_world.get_object_by_name("montessori_object_1"),
World.current_world.get_object_by_name("montessori_object_4")]
while True:
    pickable_objects = set(select_transportable_objects(World.current_world.objects, not_contained=True))
    pickable_objects = {pobj for pobj in pickable_objects if pobj not in excluded_objects}
    pickable_objects = {obj for obj in pickable_objects if obj not in all_inserted_objects}
    if not match_shapes or len(pickable_objects) == 0:
        user_input = input("Continue? (y/n) ")
        if user_input == "n":
            break

    # input("Press Enter to continue...")
    if match_shapes and len(pickable_objects) > 0:
        actionable_events = []
        for obj in pickable_objects:
            actionable_events.append(PickUpEvent(obj))
            scene_obj = World.current_world.get_object_by_name("scene")
            through_hole = scene_obj.links[obj_hole_map[obj.name]]
            actionable_events.append(InsertionEvent(obj, [through_hole], through_hole))
    else:
        all_events = episode_segmenter.logger.get_events()
        actionable_events = [event for event in all_events if isinstance(event, AbstractAgentObjectInteractionEvent) and not isinstance(event, PlacingEvent)]
        actionable_events = sorted(actionable_events, key=lambda event: event.timestamp)
        if latest_event is not None:
            actionable_events = list(filter(lambda e: e.timestamp > latest_event.timestamp, actionable_events))
        elif len(actionable_events) > 0:
            latest_event = actionable_events[-1]
        pickable_objects = set(select_transportable_objects(World.current_world.objects, not_contained=True))
        pickable_objects = {pobj for pobj in pickable_objects if pobj not in excluded_objects}
        all_inserted_objects = {event.tracked_object for event in actionable_events if
                                isinstance(event, InsertionEvent)}
        pickable_objects = {obj for obj in pickable_objects if obj not in all_inserted_objects}
    action_descriptions: List[Tuple[Optional[AbstractAgentObjectInteractionEvent],
                                    PartialDesignator[ActionDescription]]] = []
    object_pick_up_actions: Dict[Object, PickUpActionDescription] = {}
    object_picked_arm_and_grasp: Dict[Object, Tuple[Arms, GraspDescription, PoseStamped]] = {}
    mapped_objects: Dict[Object, Object] = {}
    loginfo(str(actionable_events))
    objects_to_insert = []

    for i, actionable_event in enumerate(actionable_events):
        action_descriptions.append((actionable_event, actionable_event.action_description))
        pickable_objects = set(select_transportable_objects(World.current_world.objects, not_contained=True))
        pickable_objects = {pobj for pobj in pickable_objects if pobj not in excluded_objects}
        pickable_objects = {obj for obj in pickable_objects if obj not in all_inserted_objects}
        pickable_objects = {obj for obj in pickable_objects if obj not in mapped_objects.values()}


        if isinstance(actionable_event, PickUpEvent):
            if len(action_descriptions) > 1:
                if isinstance(action_descriptions[-2][0], PickUpEvent):
                    action_descriptions.remove(action_descriptions[-2])
            if actionable_event.tracked_object not in pickable_objects:
                if len(pickable_objects) > 0:
                    square_obj = [obj for obj in pickable_objects if "3" in obj.name]
                    if len(square_obj) > 0:
                        to_pick_object = square_obj[0]
                        pickable_objects.remove(to_pick_object)
                    else:
                        to_pick_object = pickable_objects.pop()
                else:
                    action_descriptions.remove(action_descriptions[-1])
                    break
            else:
                to_pick_object = actionable_event.tracked_object
                pickable_objects.remove(actionable_event.tracked_object)
            loginfo(f"Object to Pick is {to_pick_object.name}")
            mapped_objects[actionable_event.tracked_object] = to_pick_object
            arm, grasp = get_arm_and_grasp_description_for_object(to_pick_object)
            end_effector = RobotDescription.current_robot_description.get_arm_chain(arm).end_effector
            pose = to_pick_object.get_grasp_pose(end_effector, grasp)
            object_picked_arm_and_grasp[to_pick_object] = (arm, grasp, pose)
            action_to_add = (actionable_event,
                             PickUpActionDescription(to_pick_object, arm=arm, grasp_description=grasp))
            if len(action_descriptions) >= 1:
                action_descriptions[-1] = action_to_add
            else:
                action_descriptions.append(action_to_add)

        elif isinstance(actionable_event, (PlacingEvent, InsertionEvent)):
            if actionable_event.tracked_object in mapped_objects:
                object_to_place = mapped_objects[actionable_event.tracked_object]
            else:
                action_descriptions.remove(action_descriptions[-1])
                continue
            place_pose = actionable_event.tracked_object.pose
            arm, grasp, pose = object_picked_arm_and_grasp[object_to_place]
            if isinstance(actionable_event, PlacingEvent):
                place_pose.orientation = object_to_place.orientation
            elif isinstance(actionable_event, InsertionEvent):
                if match_shapes:
                    scene_obj = actionable_event.through_hole.parent_entity
                    place_pose = scene_obj.links[obj_hole_map[object_to_place.name]].pose
                else:
                    loginfo("Not Matching Shapes")
                    loginfo(f"Placing in the {actionable_event.through_hole.name}")
                    place_pose = actionable_event.through_hole.pose
                    loginfo(f"Placing pose is {place_pose}")
                place_pose.position.z += 0.002
                place_pose.orientation = pose.orientation
            loginfo(f"Object to Place is {object_to_place.name}")
            action_descriptions[-1] = (actionable_event,
                                       PlaceActionDescription(object_to_place, target_location=place_pose,
                                                              arm=arm,
                                                              insert=isinstance(actionable_event, InsertionEvent),
                                                              pre_place_vertical_distance=0.1))
            objects_to_insert.append(object_to_place)
            action_descriptions.append((None, ParkArmsActionDescription(Arms.BOTH)))

    # episode_segmenter.reset()
    latest_event = sorted(episode_segmenter.logger.get_events(), key=lambda e: e.timestamp)[-1]

    validate = True
    failed_insertion_action: Optional[ActionDescription] = None
    failed_insertion_object_tracker: Optional[ObjectTracker] = None
    event_that_led_to_insertion_action: Optional[InsertionEvent] = None
    with real_robot:
        for event, partial_designator in action_descriptions:
            performable = partial_designator.current_node
            try:
                performable.perform()
            except Exception as e:
                logerr(str(e))
                continue
            if not validate:
                continue
            if issubclass(performable.action, PlaceAction):
                logdebug(f"Placing Pose is {performable.kwargs['target_location']}")
                obj = performable.kwargs["object_designator"]
                obj_tracker = ObjectTrackerFactory.get_tracker(obj)
                latest_pick_up = obj_tracker.get_first_event_of_type_after_event(PickUpEvent, latest_event)
                if latest_pick_up is not None:
                    time.sleep(6)
                    insertion_event = obj_tracker.get_first_event_of_type_after_event(InsertionEvent, latest_pick_up)
                    if insertion_event is None:
                        failed_insertion_action = performable
                        failed_insertion_object_tracker = obj_tracker
                        event_that_led_to_insertion_action = event
                        text_to_speech(f"Hmmm, Looks like the {obj_name_map[obj.name]} was not inserted,"
                                       f"Could you show me how or where to insert it?")
                        ParkArmsActionDescription(Arms.BOTH).current_node.perform()
                        break
                    else:
                        all_inserted_objects.add(obj)

    # episode_segmenter.reset()
    latest_event = sorted(episode_segmenter.logger.get_events(), key=lambda e: e.timestamp)[-1]

    if failed_insertion_action is None:
        continue

    while True:
        teacher_insertion_event: Optional[InsertionEvent] = failed_insertion_object_tracker.get_first_event_of_type_after_event(InsertionEvent, latest_event)
        if teacher_insertion_event is None:
            time.sleep(0.5)
        else:
            break
    all_inserted_objects.add(teacher_insertion_event.tracked_object)
    failed_insertion_hole: Link = event_that_led_to_insertion_action.through_hole
    teacher_insertion_hole: Link = teacher_insertion_event.through_hole
    failed_hole_shape = failed_insertion_hole.geometry[0]
    teacher_hole_shape = teacher_insertion_hole.geometry[0]
    obj_shape = teacher_insertion_event.tracked_object.root_link.geometry[0]

    if type(obj_shape) is type(teacher_hole_shape) and type(obj_shape) is not type(failed_hole_shape):
        match_shapes = True
        text_to_speech("Ok thank you, I think I understand now what you want,"
                       "you want me to match the object shape with the hole shape.")
    elif type(teacher_hole_shape) is type(failed_hole_shape) and type(obj_shape) is type(failed_hole_shape):
        match_shapes = True
        if teacher_insertion_hole.name != failed_insertion_hole.name:
            size_diff = np.linalg.norm(np.array(teacher_hole_shape.size) - np.array(failed_hole_shape.size))
            if size_diff >= 1e-3:
                teacher_difference = np.linalg.norm(np.array(teacher_hole_shape.size) - np.array(obj_shape.size))
                failed_difference = np.linalg.norm(np.array(failed_hole_shape.size) - np.array(obj_shape.size))
                if teacher_difference < failed_difference:
                    text_to_speech("Hmm ok so you chose a similar shaped hole like me, but you chose a different size,"
                                   "might be because it is closer to the size of the object shape, Ok I see!")
                else:
                    text_to_speech("Hmm ok so you chose a similar shaped hole like me,"
                                   " but I think the one I chose should be more fitting to that object,"
                                   " I didn't get the idea, I do not see the patter here, sorry!")
            else:
                text_to_speech("Hmm, I didn't get why you chose this specific hole, but I guess it could be a matter of "
                               "preference from your side")
        else:
            text_to_speech("Hmm ok so you placed the object in the same hole I chose,"
                           " maybe I didn't manipulate it correctly and that's why it was not inserted,"
                           "I will try to do better next time!")
    elif type(obj_shape) is not type(teacher_hole_shape):
        text_to_speech("Hmm, I didn't get the idea, I do not see the patter here, sorry!")

    if match_shapes:
        pickable_objects = set(select_transportable_objects(World.current_world.objects, not_contained=True))
        pickable_objects = {pobj for pobj in pickable_objects if pobj not in excluded_objects}
        pickable_objects = {obj for obj in pickable_objects if obj not in all_inserted_objects}
        if len(pickable_objects) > 1:
            obj_names_to_insert = ', the'.join([obj_name_map[obj.name] for obj in list(pickable_objects)[:-1]])
            obj_names_to_insert += (
                f', and the {obj_name_map[list(pickable_objects)[-1].name]} have not been inserted yet,'
                f' I can insert them if you let me.')
        elif len(pickable_objects) == 1:
            obj = pickable_objects.pop()
            obj_name = obj.name
            if obj_name in obj_name_map:
                obj_name = obj_name_map[obj_name]
            obj_names_to_insert = obj_name + 'has not been inserted yet, I can insert it if you let me.'
        else:
            continue
        text_to_speech(f"I see that the {obj_names_to_insert}")

    # episode_segmenter.reset()
    latest_event = sorted(episode_segmenter.logger.get_events(), key=lambda e: e.timestamp)[-1]

multiverse_player.stop()
multiverse_player.join()

episode_segmenter.stop()
thread.join()
loginfo("Joined Thread.")
