import time
from datetime import timedelta

import numpy as np
from pycram.testing import SemanticWorldTestCase, setup_world
from semantic_digital_twin.collision_checking.trimesh_collision_detector import TrimeshCollisionDetector
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.world_description.connections import FixedConnection
from semantic_digital_twin.world_description.world_entity import Body
from typing_extensions import List, Type

from segmind.datastructures.events import TranslationEvent, StopMotionEvent, StopTranslationEvent, \
    ContactEvent, Event, EventUnion
from segmind.datastructures.object_tracker import ObjectTrackerFactory
from segmind.detectors.atomic_event_detectors import TranslationDetector, AtomicEventDetector
from segmind.detectors.coarse_event_detectors import GeneralPickUpDetector
from segmind.detectors.spatial_relation_detector import InsertionDetector
from segmind.detectors.motion_detection_helpers import has_consistent_direction, is_displaced
from segmind.event_logger import EventLogger




class TestEventDetectors():

    def test_general_pick_up_start_condition_checker(self):
        self.world = setup_world()
        tcd = TrimeshCollisionDetector(self.world)
        milk = self.world.get_body_by_name("milk.stl")
        robot = self.world.get_body_by_name("base_link")
        event = ContactEvent(contact_points=tcd.check_collision_between_bodies(milk, robot), of_object=robot)
        GeneralPickUpDetector.start_condition_checker(event)

    def test_translation_detector(self):
        self.world = setup_world()
        self.milk = self.world.get_body_by_name("milk.stl")
        milk_tracker = ObjectTrackerFactory.get_tracker(self.milk)
        translation_detector = self.run_and_get_translation_detector(self.milk)

        try:
            translation_detector.update_with_latest_motion_data()
            self.fridge = self.world.get_body_by_name("cabinet1")
            root_T_milk_connection = HomogeneousTransformationMatrix.from_xyz_rpy(
                x = self.fridge.global_pose.x, y=self.fridge.global_pose.y, z=self.fridge.global_pose.z
            )
            with self.world.modify_world():
                milk_conn = FixedConnection(
                    parent=self.world.root,
                    child=self.milk,
                    parent_T_connection_expression=root_T_milk_connection
                )
                self.world.add_connection(milk_conn)

            # update twice to detect two displacements between three poses, since window size is 2
            time.sleep(5)
            translation_detector.update_with_latest_motion_data()
            time.sleep(5)
            translation_detector.update_with_latest_motion_data()
            time.sleep(5)
            # wait one timestep to detect that it is moving
            time.sleep(translation_detector.get_n_changes_wait_time(1))
            time.sleep(5)
            translation_event = milk_tracker.get_latest_event_of_type(TranslationEvent)
            assert(translation_event is not None)

            # update once to detect two consistent gradients of zero value between last three updates.
            translation_detector.update_with_latest_motion_data()

            # wait one timestep to detect that it is not moving
            time.sleep(translation_detector.get_n_changes_wait_time(1))

            assert(milk_tracker.get_first_event_of_type_after_event(StopTranslationEvent, translation_event)
                            is not None)
        except Exception as e:
            raise e
        finally:
            translation_detector.stop()
            translation_detector.join()

    def test_insertion_detector(self):
        milk_tracker = ObjectTrackerFactory.get_tracker(self.milk)
        time_between_frames = timedelta(seconds=0.01)
        translation_detector = self.run_and_get_translation_detector(self.milk, time_between_frames)

        sr_detector = InsertionDetector(wait_time=time_between_frames)
        sr_detector.start()

        try:
            self.assertFalse(self.kitchen.links["iai_fridge_main"].contains_body(self.milk))
            fridge_position = self.kitchen.links["iai_fridge_main"].position.to_list()
            
            # Get initial translation data
            translation_detector.update_with_latest_motion_data()
            
            self.milk.set_position(fridge_position)

            # update trhice, the first three updates will trigger the displacement threshold, while the last update will trigger the consistent zero gradient
            translation_detector.update_with_latest_motion_data()
            translation_detector.update_with_latest_motion_data()
            translation_detector.update_with_latest_motion_data()

            # because milk goes to moving state then to stop state thus we need to wait for 2 changes
            time.sleep(translation_detector.get_n_changes_wait_time(2))
            
            self.assertTrue(milk_tracker.get_latest_event_of_type(StopMotionEvent) is not None)
            self.assertTrue(self.kitchen.links["iai_fridge_main"].contains_body(self.milk))
        except Exception as e:
            raise e
        finally:
            translation_detector.stop()
            sr_detector.stop()
            translation_detector.join()
            sr_detector.join()

    @staticmethod
    def run_and_get_translation_detector(obj: Body, time_between_frames: timedelta = timedelta(seconds=0.01))\
            -> TranslationDetector:
        logger = EventLogger()
        translation_detector = TranslationDetector(logger, obj,
                                                   time_between_frames=time_between_frames,
                                                   window_size=2)
        translation_detector.start()
        # wait one timestep to detect the initial state
        time.sleep(translation_detector.get_n_changes_wait_time(1))
        return translation_detector

    def test_consistent_gradient_motion_detection_method(self):
        for i in range(3):
            a = np.zeros((3, 3))
            a[:, i] = 1
            self.assertTrue(has_consistent_direction(a.tolist()))
            a = np.zeros((3, 3))
            a[:, i] = -1
            self.assertTrue(has_consistent_direction(a.tolist()))
            a = np.zeros((3, 3))
            a[:, i] = -1
            a[1, i] = 1
            self.assertFalse(has_consistent_direction(a.tolist()))

    def test_displacement_motion_detection_method(self):
        for i in range(3):
            a = np.zeros((3, 3))
            a[:, i] = 1
            self.assertTrue(is_displaced(a.tolist(), 1.5))
            a = np.zeros((3, 3))
            a[:, i] = -1
            self.assertTrue(is_displaced(a.tolist(), 1.5))
            a = np.zeros((3, 3))
            a[:, i] = -1
            a[1, i] = 1
            self.assertFalse(is_displaced(a.tolist(), 1.5))
