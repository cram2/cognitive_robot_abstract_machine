import time
from collections import defaultdict
from typing import Dict, DefaultDict, Set

import rclpy

from Segmind.test import setup_contact_world, setup_support_world
from krrood.symbolic_math.symbolic_math import trinary_logic_or
from segmind.datastructures.events import (
    ContactEvent,
    LossOfContactEvent,
    SupportEvent,
    LossOfSupportEvent,
)
from segmind.detectors.atomic_event_detectors_nodes import (
    DetectorStateChart,
    ContactDetector,
    LossOfContactDetector,
)
from segmind.detectors.spatial_relation_detector import SupportDetectorNode
from segmind.event_logger import EventLogger

from giskardpy.executor import Executor
from giskardpy.motion_statechart.context import SegmindContext
from giskardpy.motion_statechart.goals.templates import Sequence
from semantic_digital_twin.adapters.ros.visualization.viz_marker import (
    VizMarkerPublisher,
)
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.world_description.world_entity import Body


class TestMotionStatechart:

    def test_contact_detector(self):
        world = setup_contact_world()
        self.visualize(world)
        sc = DetectorStateChart()
        logger = EventLogger()
        cylinder = world.get_body_by_name("cylinder_body")
        self.context = SegmindContext(
            world=world,
            logger=logger,
            latest_contact_bodies={},
        )
        kin_sim = Executor(self.context)

        contact_detector = ContactDetector(
            name="contact_detector", context=self.context, tracked_object=cylinder
        )
        loss_of_contact_detector = LossOfContactDetector(
            name="loss_of_contact_detector",
            context=self.context,
            tracked_object=cylinder,
        )

        sc.add_nodes([contact_detector, loss_of_contact_detector])

        kin_sim.compile(motion_statechart=sc)
        kin_sim.tick()
        # No Contact yet
        assert (
            len(
                [
                    i
                    for i in self.context.logger.get_events()
                    if isinstance(i, ContactEvent)
                ]
            )
            == 0
        )
        assert contact_detector.observation_state == 0.0
        assert loss_of_contact_detector.observation_state == 0.0

        cylinder.parent_connection.origin = (
            HomogeneousTransformationMatrix.from_xyz_rpy(y=-0.4)
        )
        kin_sim.tick()

        # Contact with 2 objects
        assert len([i for i in logger.get_events() if isinstance(i, ContactEvent)]) == 2
        assert contact_detector.observation_state == 1.0

        cylinder.parent_connection.origin = (
            HomogeneousTransformationMatrix.from_xyz_rpy(z=2)
        )

        kin_sim.tick()

        kin_sim.motion_statechart.draw(
            "/home/sorin/dev/workspace/Segmind/test/img/" + "sony.pdf"
        )
        assert (
            len([i for i in logger.get_events() if isinstance(i, LossOfContactEvent)])
            == 2
        )
        assert loss_of_contact_detector.observation_state == 1.0
        rclpy.shutdown()

    def test_support_detector(self):
        world = setup_support_world()
        self.visualize(world)
        sc = DetectorStateChart()
        logger = EventLogger()

        cylinder = world.get_body_by_name("cylinder_body")
        table = world.get_body_by_name("table_body")
        cabinet = world.get_body_by_name("cabinet")
        self.latest_contact_bodies: Dict[Body, Set[Body]] = defaultdict(set)
        self.latest_support: Dict[Body, Body] = {}
        self.context = SegmindContext(
            world=world,
            latest_support=self.latest_support,
            latest_contact_bodies=self.latest_contact_bodies,
        )
        kin_sim = Executor(self.context)

        contact_detector = ContactDetectorNode(
            name="contact_detector",
            logger=logger,
            tracked_object=cylinder,
        )
        contact_detector_v2 = ContactDetectorv2(
            name="contact_detector_v2",
            context=self.context,
        )

        loss_of_contact_detector = LossOfContactDetectorNode(
            name="loss_of_contact_detector", logger=logger, tracked_object=cylinder
        )

        support_detector = SupportDetectorNode(
            name="support_detector", logger=logger, tracked_object=cylinder
        )
        sc.add_nodes([support_detector, contact_detector, loss_of_contact_detector])
        support_detector.start_condition = trinary_logic_or(
            loss_of_contact_detector.observation_variable,
            contact_detector.observation_variable,
        )

        kin_sim.compile(motion_statechart=sc)
        contact_detector_v2.print_something()

        assert len([i for i in logger.get_events() if isinstance(i, ContactEvent)]) == 0
        assert (
            len([i for i in logger.get_events() if isinstance(i, LossOfContactEvent)])
            == 0
        )
        assert len([i for i in logger.get_events() if isinstance(i, SupportEvent)]) == 0
        assert (
            len([i for i in logger.get_events() if isinstance(i, LossOfSupportEvent)])
            == 0
        )
        assert support_detector.observation_state == 0.0 or 0.5
        assert loss_of_contact_detector.observation_state == 0.0 or 0.5
        assert contact_detector.observation_state == 0.0 or 0.5
        kin_sim.tick()

        cylinder.parent_connection.origin = (
            HomogeneousTransformationMatrix.from_xyz_rpy(
                x=table.global_pose.x,
                y=table.global_pose.y,
                z=table.global_pose.z + 0.2,
            )
        )
        kin_sim.tick()
        kin_sim.tick()
        assert support_detector.observation_state == 1
        assert len([i for i in logger.get_events() if isinstance(i, ContactEvent)]) == 1
        assert len([i for i in logger.get_events() if isinstance(i, SupportEvent)]) == 1

        cylinder.parent_connection.origin = (
            HomogeneousTransformationMatrix.from_xyz_rpy(
                x=cabinet.global_pose.x,
                y=cabinet.global_pose.y,
                z=cabinet.global_pose.z,
            )
        )
        kin_sim.tick()
        assert support_detector.observation_state == 1
        assert (
            len([i for i in logger.get_events() if isinstance(i, LossOfContactEvent)])
            == 1
        )
        assert (
            len([i for i in logger.get_events() if isinstance(i, LossOfSupportEvent)])
            == 1
        )

    def visualize(self, world):
        rclpy.init()
        node = rclpy.create_node("test_node")
        viz = VizMarkerPublisher(_world=world, node=node)
        viz.with_tf_publisher()
