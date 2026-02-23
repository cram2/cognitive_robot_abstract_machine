import time

import rclpy

from Segmind.test import setup_contact_world
from segmind import set_logger_level, LogLevel, logger
from segmind.datastructures.events import (
    CloseContactEvent,
    ContactEvent,
    LossOfCloseContactEvent,
    LossOfContactEvent,
)
from segmind.datastructures.object_tracker import ObjectTrackerFactory
from segmind.detectors.atomic_event_detectors import (
    ContactDetector,
    LossOfContactDetector,
)
from segmind.event_logger import EventLogger
from semantic_digital_twin.adapters.ros.visualization.viz_marker import (
    VizMarkerPublisher,
)
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
        contact_detector = self.run_and_get_contact_detector(
            self.tracked_obj, max_distance=0.1
        )
        loss_contact_detector = self.run_and_get_loss_contact_detector(self.tracked_obj)

        try:
            assert (len(contact_detector.latest_contact_bodies)) == 0
            assert (len(contact_detector.latest_close_bodies)) == 0
            assert (len(obj_tracker.get_event_history())) == 0

            self.tracked_obj.parent_connection.origin = (
                HomogeneousTransformationMatrix.from_xyz_rpy(y=-0.05)
            )
            time.sleep(1)

            assert (len(contact_detector.latest_close_bodies)) == 2
            assert (len(contact_detector.latest_contact_bodies)) == 0

            assert (len(obj_tracker.get_event_history())) == 2
            assert type(obj_tracker.get_latest_event()) == CloseContactEvent

            self.tracked_obj.parent_connection.origin = (
                HomogeneousTransformationMatrix.from_xyz_rpy(y=-0.2)
            )
            time.sleep(1)

            assert (len(contact_detector.latest_contact_bodies)) == 2
            assert (len(contact_detector.latest_close_bodies)) == 2

            assert (len(obj_tracker.get_event_history())) == 4
            assert type(obj_tracker.get_latest_event()) == ContactEvent
            assert obj_tracker.get_latest_event_of_type(ContactEvent) is not None

            self.tracked_obj.parent_connection.origin = (
                HomogeneousTransformationMatrix.from_xyz_rpy(y=-1.2)
            )
            time.sleep(1)

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
    def run_and_get_contact_detector(obj: Body, max_distance: float) -> ContactDetector:
        logger = EventLogger()
        contact_detector = ContactDetector(
            logger, obj, max_closeness_distance=max_distance
        )
        contact_detector.start()
        return contact_detector

    @staticmethod
    def run_and_get_loss_contact_detector(obj: Body) -> LossOfContactDetector:
        logger = EventLogger()
        loss_contact_detector = LossOfContactDetector(logger, obj)
        loss_contact_detector.start()
        return loss_contact_detector

    def visualize(self, world):
        logger.debug("Starting Visualization")
        rclpy.init()
        self.node = rclpy.create_node("test_node")
        self.world = world
        logger.debug("Node created")
        self.viz_marker_publisher = VizMarkerPublisher(world=self.world, node=self.node)
        self.viz_marker_publisher.with_tf_publisher()
