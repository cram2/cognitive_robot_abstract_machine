import time
from collections import defaultdict
from typing import Dict, DefaultDict, Set

import rclpy


from krrood.symbolic_math.symbolic_math import trinary_logic_or
from segmind.datastructures.events import (
    ContactEvent,
    LossOfContactEvent,
    SupportEvent,
    LossOfSupportEvent, LossOfContainmentEvent, ContainmentEvent,
)
from segmind.detectors.atomic_event_detectors_nodes import (
    DetectorStateChart,
    ContactDetector,
    LossOfContactDetector,
    SegmindContext,
)

from segmind.detectors.spatial_relation_detector_nodes import (
    SupportDetector,
    LossOfSupportDetector, ContainmentDetector, LossOfContainmentDetector
)
from segmind.episode_segmenter import EpisodeSegmenterExecutor
from segmind.event_logger import EventLogger

from giskardpy.executor import Executor

from giskardpy.motion_statechart.goals.templates import Sequence
from semantic_digital_twin.adapters.ros.visualization.viz_marker import (
    VizMarkerPublisher,
)
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.world_description.world_entity import Body
from test import setup_contact_world, setup_support_world


class TestMotionStatechart:


    #ToDo: We need to add some constraints on ticking, how many time do we tick per update?
    def test_contact_detector(self):
        """
        We will test the following cases:
        1. No contact yet.
        2. Contact with 2 objects.
        3. Tick Again to see if there was no new contact-event added.
        4. Loss of contact with 2 objects.
        5. Tick Again to see if there was no new loss of contact-event added.
        6. Same Contact as in 2
        7. Loss of contact with 2 objects as in 4.
        """
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

        #ToDo: We need the EpisodeSegmenter here as an Executer and change the name of kin_sim
        self.segmind_executor = EpisodeSegmenterExecutor(context=self.context)

        contact_detector = ContactDetector(
            name="contact_detector", context=self.context, tracked_object=cylinder
        )
        loss_of_contact_detector = LossOfContactDetector(
            name="loss_of_contact_detector",
            context=self.context,
            tracked_object=cylinder,
        )

        sc.add_nodes([contact_detector, loss_of_contact_detector])

        self.segmind_executor.compile(sc)
        self.segmind_executor.tick()
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
        self.segmind_executor.tick()

        # Contact with 2 objects
        assert len([i for i in logger.get_events() if isinstance(i, ContactEvent)]) == 2
        assert contact_detector.observation_state == 1.0

        # 3. Tick Again to see if there was no new contact-event added, also the observation_state should be 0.0
        self.segmind_executor.tick()

        assert len([i for i in logger.get_events() if isinstance(i, ContactEvent)]) == 2
        assert contact_detector.observation_state == 0.0

        cylinder.parent_connection.origin = (
            HomogeneousTransformationMatrix.from_xyz_rpy(z=2)
        )

        # 4. Loss of contact with 2 objects
        self.segmind_executor.tick()
        assert (
            len([i for i in logger.get_events() if isinstance(i, LossOfContactEvent)])
            == 2
        )
        assert loss_of_contact_detector.observation_state == 1.0

        # 5. Tick Again to check if the obs state turned back to 0.0
        self.segmind_executor.tick()
        assert (
            len([i for i in logger.get_events() if isinstance(i, LossOfContactEvent)])
            == 2
        )
        assert loss_of_contact_detector.observation_state == 0.0

        # 6. Contact again
        cylinder.parent_connection.origin = (
            HomogeneousTransformationMatrix.from_xyz_rpy(y=-0.4)
        )
        self.segmind_executor.tick()

        # Contact with 2 objects
        assert len([i for i in logger.get_events() if isinstance(i, ContactEvent)]) == 4
        assert contact_detector.observation_state == 1.0

        # Loss Of Contact agin
        cylinder.parent_connection.origin = (
            HomogeneousTransformationMatrix.from_xyz_rpy(z=2)
        )

        self.segmind_executor.tick()
        assert (
            len([i for i in logger.get_events() if isinstance(i, LossOfContactEvent)])
            == 4
        )
        assert loss_of_contact_detector.observation_state == 1.0

        rclpy.shutdown()

    def test_new_support_detector(self):
        world = setup_support_world()
        self.visualize(world)
        sc = DetectorStateChart()
        logger = EventLogger()

        cylinder = world.get_body_by_name("cylinder_body")
        table = world.get_body_by_name("table_body")
        cabinet = world.get_body_by_name("cabinet")

        self.context = SegmindContext(
            world=world,
            latest_support={},
            latest_contact_bodies={},
            latest_containments={},
            logger=logger,
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

        support_detector = SupportDetector(
            name="support_detector",
            context=self.context,
            tracked_object=cylinder,
        )

        loss_of_support_detector = LossOfSupportDetector(
            name="los_detector",
            context=self.context,
            tracked_object=cylinder,
        )
        containment_detector = ContainmentDetector(
            name="containment_detector",
            context=self.context,
            tracked_object=cylinder,
        )
        loss_of_containment_detector = LossOfContainmentDetector(
            name="los_containment_detector",
            context=self.context,
            tracked_object=cylinder,
        )

        sc.add_nodes(
            [
                support_detector,
                contact_detector,
                loss_of_contact_detector,
                loss_of_support_detector,
                containment_detector,
                loss_of_containment_detector,
            ]
        )
        support_detector.start_condition = contact_detector.observation_variable
        loss_of_support_detector.start_condition = (
            loss_of_contact_detector.observation_variable
        )
        # Normally containment requires support, but for testing the object will float inside another obj.
        #containment_detector.start_condition = support_detector.observation_variable
        #loss_of_containment_detector.start_condition = (loss_of_support_detector.observation_variable)
        kin_sim.compile(motion_statechart=sc)

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

        # Second tick needed to actually set the obs state to 1.0
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
        kin_sim.tick()
        assert loss_of_support_detector.observation_state == 1
        assert (
            len([i for i in logger.get_events() if isinstance(i, LossOfContactEvent)])
            == 1
        )
        assert len([i for i in logger.get_events() if isinstance(i, ContainmentEvent)]) == 1


        # Doing it again as sanity check
        cylinder.parent_connection.origin = (
            HomogeneousTransformationMatrix.from_xyz_rpy(
                x=table.global_pose.x,
                y=table.global_pose.y,
                z=table.global_pose.z + 0.2,
            )
        )
        kin_sim.tick()

        assert support_detector.observation_state == 1
        assert len([i for i in logger.get_events() if isinstance(i, ContactEvent)]) == 2
        assert len([i for i in logger.get_events() if isinstance(i, SupportEvent)]) == 2
        assert len([i for i in logger.get_events() if isinstance(i, LossOfContainmentEvent)]) == 1

        cylinder.parent_connection.origin = (
            HomogeneousTransformationMatrix.from_xyz_rpy(
                x=cabinet.global_pose.x,
                y=cabinet.global_pose.y,
                z=cabinet.global_pose.z,
            )
        )
        kin_sim.tick()

        assert loss_of_support_detector.observation_state == 1
        assert (
            len([i for i in logger.get_events() if isinstance(i, LossOfContactEvent)])
            == 2
        )
        kin_sim.motion_statechart.draw("/home/sorin/dev/Segmind/test/img/" + "test.pdf")
        rclpy.shutdown()
    def visualize(self, world):
        rclpy.init()
        node = rclpy.create_node("test_node")
        viz = VizMarkerPublisher(_world=world, node=node)
        viz.with_tf_publisher()
