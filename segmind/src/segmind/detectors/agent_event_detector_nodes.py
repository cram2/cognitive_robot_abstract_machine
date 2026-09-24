"""
Detecting what an agent does to an object rather than what happens to it: taking hold of
one, and letting go of it again.

What is read here is a relation between a body and the hand of an agent, which is what
sets it apart from the relations any two bodies stand in, and from the events read off
other events.
"""

from __future__ import annotations

from dataclasses import dataclass

from giskardpy.motion_statechart.context import MotionStatechartContext
from typing_extensions import List

from segmind.datastructures.events import (
    DetectionEvent,
    GraspEvent,
    LossOfGraspEvent,
)
from segmind.detectors.base import AbstractDetector, IndexedBodyPairs, SegmindContext
from semantic_digital_twin.reasoning.predicates import contact
from semantic_digital_twin.robots.robot_part_mixins import HasTwoFingers
from semantic_digital_twin.robots.robot_parts import EndEffector
from semantic_digital_twin.world_description.world_entity import Body


@dataclass(eq=False, repr=False)
class GraspDetector(AbstractDetector):
    """
    Reports an object being taken hold of by an agent, and being let go of again.
    """

    @staticmethod
    def has_hold_of(end_effector: EndEffector, tracked_object: Body) -> bool:
        """
        Whether ``end_effector`` holds ``tracked_object``.

        A hand holds what lies between its fingers, so each of them has to touch it.
        Brushing one of them, as a gripper does passing whatever stands beside what it
        reaches for, is not taking hold of anything. A hand that is not made of two
        fingers is read as one part, since nothing here knows what holding means for it.

        :param end_effector: The hand asked about.
        :param tracked_object: The body it may hold.
        :return: True when every side of the hand touches the body.
        """
        sides = (
            [end_effector.thumb.bodies, end_effector.finger.bodies]
            if isinstance(end_effector, HasTwoFingers)
            else [end_effector.bodies]
        )
        return all(
            any(
                contact(tracked_object, body)
                for body in side
                if body is not tracked_object
                and body.collision
                and body.collision.shapes
            )
            for side in sides
        )

    def tool_frames_holding(
        self, context: MotionStatechartContext, tracked_objects: List[Body]
    ) -> IndexedBodyPairs:
        """
        Which tool frames have hold of each body.

        A tool frame is a place rather than a thing and has no geometry to touch, so
        what is asked is whether the hand around it holds the body.

        :param context: The current motion statechart context.
        :param tracked_objects: The bodies to check.
        :return: The tool frames holding each body, per body.
        """
        holding: IndexedBodyPairs = {}
        for end_effector in context.world.get_semantic_annotations_by_type(EndEffector):
            for tracked_object in tracked_objects:
                if self.has_hold_of(end_effector, tracked_object):
                    holding.setdefault(tracked_object, set()).add(
                        end_effector.tool_frame
                    )
        return holding

    def update_context_and_events(
        self,
        context: MotionStatechartContext,
        segmind_context: SegmindContext,
        tracked_objects: List[Body],
    ) -> List[DetectionEvent]:
        """
        Detects bodies newly taken hold of and bodies that are no longer held.

        :param context: The current motion statechart context.
        :param segmind_context: The shared SegmindContext holding what is already known.
        :param tracked_objects: The bodies to check.
        :return: One event per body newly held and one per body let go of, per tool
            frame.
        """
        holding = self.tool_frames_holding(context, tracked_objects)
        latest_grasps = segmind_context.latest_grasps
        taken_hold_of = self.remember_new_relations(latest_grasps, holding)
        let_go_of = self.forget_lost_relations(latest_grasps, holding, tracked_objects)
        return [
            GraspEvent(tracked_object=body, with_object=tool_frame)
            for body, tool_frames in taken_hold_of.items()
            for tool_frame in tool_frames
        ] + [
            LossOfGraspEvent(tracked_object=body, with_object=tool_frame)
            for body, tool_frames in let_go_of.items()
            for tool_frame in tool_frames
        ]
