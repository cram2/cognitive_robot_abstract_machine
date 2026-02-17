import time

import rclpy

from Segmind.test import setup_contact_world
from segmind import set_logger_level, LogLevel, logger
from segmind.datastructures.events import CloseContactEvent, ContactEvent, LossOfCloseContactEvent, LossOfContactEvent
from segmind.datastructures.object_tracker import ObjectTrackerFactory
from segmind.detectors.atomic_event_detectors import ContactDetector, LossOfContactDetector
from segmind.event_logger import EventLogger
from semantic_digital_twin.adapters.ros.visualization.viz_marker import VizMarkerPublisher
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import FixedConnection
from semantic_digital_twin.world_description.world_entity import Body

set_logger_level(LogLevel.DEBUG)

class TestContactEvent:
    world: World
    viz_marker_publisher: VizMarkerPublisher

    def test_contact_events(self):
        self.world = setup_contact_world()
        self.visualize(self.world)
        self.tracked_obj = self.world.get_body_by_name("cylinder_body")
        obj_tracker = ObjectTrackerFactory.get_tracker(self.tracked_obj)
        contact_detector = self.run_and_get_contact_detector(self.tracked_obj)
        loss_contact_detector = self.run_and_get_loss_contact_detector(self.tracked_obj)

        try:
            assert (len(contact_detector.latest_contact_bodies)) == 0
            assert (len(contact_detector.latest_close_bodies)) == 0
            assert (len(obj_tracker.get_event_history())) == 0

            with self.world.modify_world():
                root_C_cylinder = HomogeneousTransformationMatrix.from_xyz_rpy(
                    x=self.tracked_obj.global_pose.x ,y=-3.05, z=self.tracked_obj.global_pose.z
                )
                with self.world.modify_world():
                    cylinder_conn = FixedConnection(
                        parent=self.world.root,
                        child=self.tracked_obj,
                        parent_T_connection_expression=root_C_cylinder
                    )
                    self.world.add_connection(cylinder_conn)

            time.sleep(1)

            assert (len(contact_detector.latest_close_bodies)) == 2
            assert (len(contact_detector.latest_contact_bodies)) == 0

            assert (len(obj_tracker.get_event_history())) == 2
            assert type(obj_tracker.get_latest_event()) == CloseContactEvent

            with self.world.modify_world():
                root_C_cylinder = HomogeneousTransformationMatrix.from_xyz_rpy(
                    x=self.tracked_obj.global_pose.x, y=-3.2, z=self.tracked_obj.global_pose.z
                )
                with self.world.modify_world():
                    cylinder_conn = FixedConnection(
                        parent=self.world.root,
                        child=self.tracked_obj,
                        parent_T_connection_expression=root_C_cylinder
                    )
                    self.world.add_connection(cylinder_conn)

            time.sleep(1)

            assert (len(contact_detector.latest_contact_bodies)) == 2
            assert (len(contact_detector.latest_close_bodies)) == 2

            assert (len(obj_tracker.get_event_history())) == 4
            assert type(obj_tracker.get_latest_event()) == ContactEvent
            assert obj_tracker.get_latest_event_of_type(ContactEvent) is not None

            with self.world.modify_world():
                root_C_cylinder = HomogeneousTransformationMatrix.from_xyz_rpy(
                    x=self.tracked_obj.global_pose.x, y=-2.2, z=self.tracked_obj.global_pose.z
                )
                with self.world.modify_world():
                    cylinder_conn = FixedConnection(
                        parent=self.world.root,
                        child=self.tracked_obj,
                        parent_T_connection_expression=root_C_cylinder
                    )
                    self.world.add_connection(cylinder_conn)

            time.sleep(2)

            assert (len(contact_detector.latest_contact_bodies)) == 0
            assert (len(contact_detector.latest_close_bodies)) == 0

            assert (len(obj_tracker.get_event_history())) == 6
            assert type(obj_tracker.get_latest_event()) == LossOfContactEvent


        except Exception as e:
            raise e

        finally:
            contact_detector.stop()
            loss_contact_detector.stop()

    @staticmethod
    def run_and_get_contact_detector(obj: Body) -> ContactDetector:
        logger = EventLogger()
        contact_detector = ContactDetector(logger, obj)
        contact_detector.start()
        #time.sleep(2)
        return contact_detector

    @staticmethod
    def run_and_get_loss_contact_detector(obj: Body) -> LossOfContactDetector:
        logger = EventLogger()
        loss_contact_detector = LossOfContactDetector(logger, obj)
        loss_contact_detector.start()
        #time.sleep(2)
        return loss_contact_detector

    def visualize(self, world):
        logger.debug("Starting Visualization")
        rclpy.init()
        self.node = rclpy.create_node("test_node")
        self.world = world
        logger.debug("Node created")
        self.viz_marker_publisher = VizMarkerPublisher(world=self.world, node=self.node)
        self.viz_marker_publisher.with_tf_publisher()
