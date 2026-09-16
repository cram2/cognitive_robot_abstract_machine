"""
Detecting that an agent has taken hold of an object, and that it has let go again.
"""

from __future__ import annotations

from abc import ABC
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
from semantic_digital_twin.robots.robot_parts import EndEffector
from semantic_digital_twin.world_description.world_entity import Body


@dataclass(eq=False, repr=False)
class AbstractGraspDetector(AbstractDetector, ABC):
    """
    Shared reading of which tool frames have hold of which bodies.
    """

    def tool_frames_holding(
        self, context: MotionStatechartContext, tracked_objects: List[Body]
    ) -> IndexedBodyPairs:
        """
        Which tool frames have hold of each body.

        A tool frame is a place rather than a thing and has no geometry to touch, so
        what is asked is whether the hand around it touches the body.

        :param context: The current motion statechart context.
        :param tracked_objects: The bodies to check.
        :return: The tool frames holding each body, per body.
        """
        holding: IndexedBodyPairs = {}
        for end_effector in context.world.get_semantic_annotations_by_type(EndEffector):
            hand = [
                body
                for body in end_effector.bodies
                if body.collision and body.collision.shapes
            ]
            for tracked_object in tracked_objects:
                if any(
                    contact(tracked_object, body)
                    for body in hand
                    if body is not tracked_object
                ):
                    holding.setdefault(tracked_object, set()).add(
                        end_effector.tool_frame
                    )
        return holding


@dataclass(eq=False, repr=False)
class GraspDetector(AbstractGraspDetector):
    """
    Reports an object being taken hold of by an agent.
    """

    def update_context_and_events(
        self,
        context: MotionStatechartContext,
        segmind_context: SegmindContext,
        tracked_objects: List[Body],
    ) -> List[DetectionEvent]:
        """
        Detects bodies newly taken hold of.

        :param context: The current motion statechart context.
        :param segmind_context: The shared SegmindContext holding what is already known.
        :param tracked_objects: The bodies to check.
        :return: One event per body newly held, per tool frame holding it.
        """
        events = []
        for body, tool_frames in self.tool_frames_holding(
            context, tracked_objects
        ).items():
            taken_hold_of = tool_frames - segmind_context.latest_grasps.get(body, set())
            if not taken_hold_of:
                continue
            segmind_context.latest_grasps.setdefault(body, set()).update(taken_hold_of)
            events.extend(
                GraspEvent(tracked_object=body, with_object=tool_frame)
                for tool_frame in taken_hold_of
            )
        return events


@dataclass(eq=False, repr=False)
class LossOfGraspDetector(AbstractGraspDetector):
    """
    Reports an agent letting go of an object it had hold of.
    """

    def update_context_and_events(
        self,
        context: MotionStatechartContext,
        segmind_context: SegmindContext,
        tracked_objects: List[Body],
    ) -> List[DetectionEvent]:
        """
        Detects bodies that are no longer held.

        :param context: The current motion statechart context.
        :param segmind_context: The shared SegmindContext holding what is already known.
        :param tracked_objects: The bodies to check.
        :return: One event per body let go of, per tool frame that let go.
        """
        let_go_of = self.forget_lost_relations(
            segmind_context.latest_grasps,
            self.tool_frames_holding(context, tracked_objects),
            tracked_objects,
        )
        return [
            LossOfGraspEvent(tracked_object=body, with_object=tool_frame)
            for body, tool_frames in let_go_of.items()
            for tool_frame in tool_frames
        ]
