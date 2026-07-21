from __future__ import annotations

from abc import abstractmethod, ABC
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, List, Dict, Set, Any
import numpy as np

from giskardpy.motion_statechart.context import MotionStatechartContext
from krrood.symbolic_math.symbolic_math import Scalar
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
    Detector responsible for identifying newly established contacts
    between bodies.
    """

    def update_context_and_events(
        self,
        context: MotionStatechartContext,
        segmind_context: SegmindContext,
        tracked_objects: List[Body],
    ) -> List[DetectionEvent]:
        """
        Detects newly formed contacts and updates the stored contact state.

        Generates a ContactEvent whenever a new contact between two bodies
        is detected.

        :param context: The current motion statechart context.
        :param segmind_context: The shared SegmindContext containing the information required to track events.
        :param tracked_objects: List of bodies to check for new contacts.
        :return: List of ContactEvent instances generated during this update.
        """
        new_contact_pairs = self.get_relation(context, tracked_objects, contact)

        events = []
        for obj, contact_list in new_contact_pairs.items():
            new_contacts = (
                contact_list
                if obj not in segmind_context.latest_contact_bodies
                else contact_list - segmind_context.latest_contact_bodies[obj]
            )
            if new_contacts:
                segmind_context.latest_contact_bodies.setdefault(obj, set()).update(new_contacts)
                events.extend(
                    [ContactEvent(tracked_object=obj, with_object=c) for c in new_contacts]
                )

        return events


@dataclass(eq=False, repr=False)
class LossOfContactDetector(AbstractDetector):
    """
    Detector responsible for identifying when previously existing
    contacts between bodies are lost.
    """

    def update_context_and_events(
        self,
        context: MotionStatechartContext,
        segmind_context: SegmindContext,
        tracked_objects: List[Body],
    ) -> List[DetectionEvent]:
        """
        Detects when existing contacts are no longer present and updates
        the stored contact state accordingly.

        Generates a LossOfContactEvent whenever a previously detected
        contact no longer exists.

        :param context: The current motion statechart context.
        :param segmind_context: The shared SegmindContext containing the information required to track events.
        :param tracked_objects: List of bodies to check for lost contacts.
        :return: List of LossOfContactEvent instances generated during this update.
        """

        new_contact_pairs = self.get_relation(context, tracked_objects, contact)

        events = []
        for obj, contact_list in list(segmind_context.latest_contact_bodies.items()):
            loss_contacts = (
                contact_list.copy()
                if obj not in new_contact_pairs
                else contact_list - new_contact_pairs[obj]
            )
            if loss_contacts:

                segmind_context.latest_contact_bodies[obj] -= loss_contacts
                if not segmind_context.latest_contact_bodies[obj]:
                    segmind_context.latest_contact_bodies.pop(obj)

                events.extend(
                    [
                        LossOfContactEvent(tracked_object=obj, with_object=s)
                        for s in loss_contacts
                    ]
                )

        return events


@dataclass(eq=False, repr=False)
class MotionDetector(AbstractDetector):
    """
    Base class for motion-based detectors.

    Provides shared functionality for monitoring poses of
    bodies and generating events when movement is detected.
    """

    window_size: int = 4
    """
    The window size indicates how many poses to consider for movement.
    """

    distance_threshold: Scalar = 0.005
    """
    Threshold for the distance between two poses to be considered movement.
    """

    rotation_threshold: float = 0.1
    """
    Threshold for the rotation error between two poses to be considered rotation.
    """

    _pose_history: Dict[Body, List[Pose]] = field(default_factory=dict, init=False, repr=False)
    """
    Pose window per tracked body. Owned by this detector rather than shared through the
    context, so that the window always spans ``window_size`` ticks regardless of how many
    other motion detectors are registered in the statechart.
    """

    def update_context_and_events(self, context: MotionStatechartContext, segmind_context:SegmindContext, tracked_objs: List[Body]) -> List[DetectionEvent]:
        """
        Updates the pose history for each tracked object and checks for motion events.

        :param context: The current motion statechart context.
        :param segmind_context: The shared SegmindContext containing the information required to track events.
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
    def _check_and_trigger_event(self, context: SegmindContext, obj: Body, poses: List[Pose]) -> Optional[DetectionEvent]:
        """
        Subclass-specific logic to trigger a Motion or StopMotion event.

        Called once per tick per tracked object, with the full pose window, so each detector
        evaluates only the condition it is responsible for.

        :param context: The shared SegmindContext containing the information required to track events.
        :param obj: The body to check.
        :param poses: The pose window of ``obj``, oldest first.
        :return: A MotionEvent, StopMotionEvent or None.
        """
        pass

    def _is_moving(self, poses: List[Pose]) -> bool:
        """
        Determines whether an object is moving by evaluating the distance between the first
        and the last recorded position of the window.

        :param poses: The pose window of the body, oldest first.
        :return: True if the object is moving, False otherwise.
        """
        return poses[0].to_position().euclidean_distance(poses[-1].to_position()) > self.distance_threshold

    def _is_rotating(self, poses: List[Pose]) -> bool:
        """
        Determines whether an object is rotating by evaluating the rotation error between the
        first and the last recorded pose of the window.

        :param poses: The pose window of the body, oldest first.
        :return: True if the object is rotating, False otherwise.
        """
        rotational_error = float(
            poses[0].to_rotation_matrix().rotational_error(poses[-1].to_rotation_matrix())
        )
        return rotational_error > self.rotation_threshold


@dataclass(eq=False, repr=False)
class TranslationDetector(MotionDetector):
    """
    Detector for translation events.
    Triggers a TranslationEvent when an object starts moving.
    """

    def _check_and_trigger_event(self, context: SegmindContext, obj: Body, poses: List[Pose]) -> Optional[DetectionEvent]:
        """
        Triggers a TranslationEvent when an object starts moving.

        No event is triggered while the object is stationary, or while a motion event for it
        is already active.

        :param context: The shared SegmindContext containing the information required to track events.
        :param obj: The object being monitored for movement.
        :param poses: The pose window of ``obj``, oldest first.
        :return: A TranslationEvent if the object started moving, otherwise None.
        """
        if not self._is_moving(poses):
            return None

        if context.latest_motion_events.get(obj) is not None:
            return None

        new_event = TranslationEvent(
            tracked_object=obj,
            start_pose=poses[0],
            current_pose=poses[-1],
        )

        context.latest_motion_events[obj] = new_event
        return new_event


@dataclass(eq=False, repr=False)
class StopTranslationDetector(MotionDetector):
    """
    Detector for stop translation events.
    Triggers a StopTranslationEvent when an object that was moving stops.
    """

    def _check_and_trigger_event(self, context: SegmindContext, obj: Body, poses: List[Pose]) -> Optional[DetectionEvent]:
        """
        Triggers a StopTranslationEvent when an object that was moving comes to a stop.

        Requires an active motion event, which is created by the :class:`TranslationDetector`.

        :param context: The shared SegmindContext containing the information required to track events.
        :param obj: The object to check for movement.
        :param poses: The pose window of ``obj``, oldest first.
        :return: A StopTranslationEvent if the object stopped moving, otherwise None.
        """
        if self._is_moving(poses):
            return None

        latest_motion_event = context.latest_motion_events.get(obj)
        if latest_motion_event is None:
            return None

        stop_event = StopTranslationEvent(
            tracked_object=obj,
            start_pose=latest_motion_event.start_pose,
            current_pose=poses[-1],
        )

        context.latest_motion_events.pop(obj, None)

        return stop_event



@dataclass(eq=False, repr=False)
class RotationDetector(MotionDetector):
    """
    Detector for rotation events.
    Triggers a RotationEvent when an object starts rotating.
    """

    def _check_and_trigger_event(self, context: SegmindContext, obj: Body, poses: List[Pose]) -> Optional[DetectionEvent]:
        """
        Triggers a RotationEvent when an object starts rotating.

        No event is triggered while the object is not rotating, or while a rotation event for
        it is already active.

        :param context: The shared SegmindContext containing the information required to track events.
        :param obj: The object to check.
        :param poses: The pose window of ``obj``, oldest first.
        :return: A RotationEvent if the object started rotating, otherwise None.
        """
        if not self._is_rotating(poses):
            return None

        if context.latest_rotation_events.get(obj) is not None:
            return None

        new_event = RotationEvent(
            tracked_object=obj,
            start_pose=poses[0],
            current_pose=poses[-1],
        )

        context.latest_rotation_events[obj] = new_event
        return new_event


@dataclass(eq=False, repr=False)
class StopRotationDetector(MotionDetector):
    """
    Detector for stop rotation events.
    Triggers a StopRotationEvent when an object that was rotating stops.
    """

    def _check_and_trigger_event(self, context: SegmindContext, obj: Body, poses: List[Pose]) -> Optional[DetectionEvent]:
        """
        Triggers a StopRotationEvent when an object that was rotating comes to a stop.

        Requires an active rotation event, which is created by the :class:`RotationDetector`.

        :param context: The shared SegmindContext containing the information required to track events.
        :param obj: The object to check for movement.
        :param poses: The pose window of ``obj``, oldest first.
        :return: A StopRotationEvent if the object stopped rotating, otherwise None.
        """
        if self._is_rotating(poses):
            return None

        latest_rotation_event = context.latest_rotation_events.get(obj)
        if latest_rotation_event is None:
            return None

        stop_event = StopRotationEvent(
            tracked_object=obj,
            start_pose=latest_rotation_event.start_pose,
            current_pose=poses[-1],
        )

        context.latest_rotation_events.pop(obj, None)

        return stop_event


