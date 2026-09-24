from __future__ import annotations

from abc import abstractmethod, ABC
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, List, Dict, Set, Any
import numpy as np

from giskardpy.motion_statechart.context import MotionStatechartContext
from segmind.datastructures.events import (
    DetectionEvent,
    ContactEvent,
    LossOfContactEvent,
    TranslationEvent,
    RotationEvent,
    StopTranslationEvent,
    StopRotationEvent,
)
from segmind.detectors.base import SegmindContext, AbstractDetector
from semantic_digital_twin.reasoning.predicates import contact
from semantic_digital_twin.spatial_types.spatial_types import Pose
from semantic_digital_twin.world_description.world_entity import Body


@dataclass(eq=False, repr=False)
class ContactDetector(AbstractDetector):
    """
    Detects contacts between bodies being established and being lost.

    The detector reports a :class:`ContactEvent` when a body touches something new and a
    :class:`LossOfContactEvent` when it stops touching something it touched.
    """

    def update_context_and_events(
        self,
        context: MotionStatechartContext,
        segmind_context: SegmindContext,
        tracked_objects: List[Body],
    ) -> List[DetectionEvent]:
        """
        Detects newly formed and newly lost contacts and updates the stored contact
        state.

        :param context: The current motion statechart context.
        :param segmind_context: The shared SegmindContext containing the information
            required to track events.
        :param tracked_objects: List of bodies to check for contacts.
        :return: The contacts formed, then the contacts lost.
        """
        contacts_now = self.get_relation(context, tracked_objects, contact)
        latest_contacts = segmind_context.latest_contact_bodies
        new_contacts = self.remember_new_relations(latest_contacts, contacts_now)
        lost_contacts = self.forget_lost_relations(
            latest_contacts, contacts_now, tracked_objects
        )
        return [
            ContactEvent(tracked_object=body, with_object=other)
            for body, others in new_contacts.items()
            for other in others
        ] + [
            LossOfContactEvent(tracked_object=body, with_object=other)
            for body, others in lost_contacts.items()
            for other in others
        ]


@dataclass(eq=False, repr=False)
class MotionDetector(AbstractDetector):
    """
    Base class for motion-based detectors.

    Provides shared functionality for monitoring poses of bodies and generating events
    when movement is detected.
    """

    window_size: int = 4
    """
    The window size indicates how many poses to consider for movement.
    """

    distance_threshold: float = 0.005
    """
    Threshold for the distance between two poses to be considered movement.
    """

    rotation_threshold: float = 0.1
    """
    Threshold for the rotation error between two poses to be considered rotation.
    """

    _pose_history: Dict[Body, List[Pose]] = field(
        default_factory=dict, init=False, repr=False
    )
    """
    Pose window per tracked body.

    Owned by this detector rather than shared through the context, so that the window
    always spans ``window_size`` ticks regardless of how many other motion detectors are
    registered in the statechart.
    """

    def update_context_and_events(
        self,
        context: MotionStatechartContext,
        segmind_context: SegmindContext,
        tracked_objs: List[Body],
    ) -> List[DetectionEvent]:
        """
        Updates the pose history for each tracked object and checks for motion events.

        :param context: The current motion statechart context.
        :param segmind_context: The shared SegmindContext containing the information
            required to track events.
        :param tracked_objs: List of bodies to update and check.
        :return: A list of events triggered during this update.
        """
        events = []
        for obj in tracked_objs:
            poses = self._pose_history.setdefault(obj, [])
            poses.append(obj.global_pose)
            if len(poses) < self.window_size:
                continue

            event = self._check_and_trigger_event(segmind_context, obj, poses)
            if event:
                events.append(event)

            poses.pop(0)
        return events

    @abstractmethod
    def _check_and_trigger_event(
        self, context: SegmindContext, obj: Body, poses: List[Pose]
    ) -> Optional[DetectionEvent]:
        """
        Subclass-specific logic to trigger a Motion or StopMotion event.

        Called once per tick per tracked object, with the full pose window, so each
        detector evaluates only the condition it is responsible for.

        :param context: The shared SegmindContext containing the information required to
            track events.
        :param obj: The body to check.
        :param poses: The pose window of ``obj``, oldest first.
        :return: A MotionEvent, StopMotionEvent or None.
        """
        pass

    def _is_moving(self, poses: List[Pose]) -> bool:
        """
        Determines whether an object is moving by evaluating the distance between the
        first and the last recorded position of the window.

        :param poses: The pose window of the body, oldest first.
        :return: True if the object is moving, False otherwise.
        """
        return (
            poses[0].to_position().euclidean_distance(poses[-1].to_position())
            > self.distance_threshold
        )

    def _is_rotating(self, poses: List[Pose]) -> bool:
        """
        Determines whether an object is rotating by evaluating the rotation error
        between the first and the last recorded pose of the window.

        :param poses: The pose window of the body, oldest first.
        :return: True if the object is rotating, False otherwise.
        """
        return (
            float(
                poses[0]
                .to_rotation_matrix()
                .rotational_distance(poses[-1].to_rotation_matrix())
            )
            > self.rotation_threshold
        )


@dataclass(eq=False, repr=False)
class TranslationDetector(MotionDetector):
    """
    Detector for translations.

    Reports a :class:`TranslationEvent` when an object starts moving and a
    :class:`StopTranslationEvent` when an object that was moving stops.
    """

    def _check_and_trigger_event(
        self, context: SegmindContext, obj: Body, poses: List[Pose]
    ) -> Optional[DetectionEvent]:
        """
        Reports the object starting to move or, when it was moving, coming to a stop.

        No event is reported while the object stays as it was: stationary, or moving
        while a motion event for it is already active.

        :param context: The shared SegmindContext containing the information required to
            track events.
        :param obj: The object being monitored for movement.
        :param poses: The pose window of ``obj``, oldest first.
        :return: A TranslationEvent, a StopTranslationEvent or None.
        """
        latest_motion_event = context.latest_motion_events.get(obj)
        if not self._is_moving(poses):
            if latest_motion_event is None:
                return None
            context.latest_motion_events.pop(obj)
            return StopTranslationEvent(
                tracked_object=obj,
                world_T_start_pose=latest_motion_event.world_T_start_pose,
                world_T_current_pose=poses[-1],
            )

        if latest_motion_event is not None:
            return None

        new_event = TranslationEvent(
            tracked_object=obj,
            world_T_start_pose=poses[0],
            world_T_current_pose=poses[-1],
        )
        context.latest_motion_events[obj] = new_event
        return new_event


@dataclass(eq=False, repr=False)
class RotationDetector(MotionDetector):
    """
    Detector for rotations.

    Reports a :class:`RotationEvent` when an object starts rotating and a
    :class:`StopRotationEvent` when an object that was rotating stops.
    """

    def _check_and_trigger_event(
        self, context: SegmindContext, obj: Body, poses: List[Pose]
    ) -> Optional[DetectionEvent]:
        """
        Reports the object starting to rotate or, when it was rotating, coming to a
        stop.

        No event is reported while the object stays as it was: not rotating, or
        rotating while a rotation event for it is already active.

        :param context: The shared SegmindContext containing the information required to
            track events.
        :param obj: The object to check.
        :param poses: The pose window of ``obj``, oldest first.
        :return: A RotationEvent, a StopRotationEvent or None.
        """
        latest_rotation_event = context.latest_rotation_events.get(obj)
        if not self._is_rotating(poses):
            if latest_rotation_event is None:
                return None
            context.latest_rotation_events.pop(obj)
            return StopRotationEvent(
                tracked_object=obj,
                world_T_start_pose=latest_rotation_event.world_T_start_pose,
                world_T_current_pose=poses[-1],
            )

        if latest_rotation_event is not None:
            return None

        new_event = RotationEvent(
            tracked_object=obj,
            world_T_start_pose=poses[0],
            world_T_current_pose=poses[-1],
        )
        context.latest_rotation_events[obj] = new_event
        return new_event
