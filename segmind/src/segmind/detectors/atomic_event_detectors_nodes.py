import math
from abc import abstractmethod, ABC
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Set, Any
import numpy as np

from segmind.datastructures.events import (
    Event,
    ContactEvent,
    LossOfContactEvent,
    TranslationEvent,
    RotationEvent,
    StopTranslationEvent,
    StopRotationEvent,
)
from segmind.detectors.base import (
    AbstractDetector,
)
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
        tracked_objects: List[Body],
    ) -> List[Event]:
        """
        Detects newly formed contacts and updates the stored contact state.

        Generates a ContactEvent whenever a new contact between two bodies
        is detected.

        :param tracked_objects: List of bodies to check for new contacts.
        :return: List of ContactEvent instances generated during this update.
        """

        new_contact_pairs = self.get_relation(tracked_objects, contact)
        latest_contact_bodies = self.context.latest_contact_bodies
        events = []
        for obj, contact_list in new_contact_pairs.items():
            new_contacts = (
                contact_list
                if obj not in latest_contact_bodies
                else contact_list - latest_contact_bodies[obj]
            )
            if new_contacts:
                latest_contact_bodies.setdefault(obj, set()).update(new_contacts)
                events.extend(
                    [ContactEvent(of_object=obj, with_object=c) for c in new_contacts]
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
        tracked_objects: List[Body],
    ) -> List[Event]:
        """
        Detects when existing contacts are no longer present and updates
        the stored contact state accordingly.

        Generates a LossOfContactEvent whenever a previously detected
        contact no longer exists.

        :param tracked_objects: List of bodies to check for lost contacts.
        :return: List of LossOfContactEvent instances generated during this update.
        """

        new_contact_pairs = self.get_relation(tracked_objects, contact)
        latest_contact_bodies = self.context.latest_contact_bodies
        events = []
        for obj, contact_list in list(latest_contact_bodies.items()):
            loss_contacts = (
                contact_list
                if obj not in new_contact_pairs
                else contact_list - new_contact_pairs[obj]
            )
            if loss_contacts:
                latest_contact_bodies.pop(obj)
                events.extend(
                    [
                        LossOfContactEvent(of_object=obj, with_object=s)
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

    distance_threshold: float = 0.005
    """
    Threshold for the distance between two poses to be considered movement.
    """

    def update_context_and_events(self, tracked_objs: List[Body]) -> List[Event]:
        """
        Updates the pose history for each tracked object and checks for motion events.

        :param tracked_objs: List of bodies to update and check.
        :return: A list of events triggered during this update.
        """
        events = []
        for obj in tracked_objs:
            latest_poses = self.context.latest_poses.setdefault(obj, [])
            latest_poses.append(obj.global_pose)
            if len(latest_poses) >= self.window_size:
                event = self.check_obj_movement(obj)
                if event:
                    events.append(event)
                latest_poses.pop(0)
        return events

    def check_obj_movement(self, obj: Body) -> Optional[Event]:
        """
        Determines if an object is moving based on its pose history and delegates event creation.

        :param obj: The body to check.
        :return: An Event if movement/stop is detected, otherwise None.
        """
        is_moving = self._calculate_is_moving(obj)
        self.context.object_moving_status[obj] = is_moving
        return self._check_movement_and_trigger_event(obj)

    @abstractmethod
    def _check_movement_and_trigger_event(self, obj: Body) -> Optional[Event]:
        """
        Subclass-specific logic to trigger Motion or StopMotion events.

        :param obj: The body to check.
        :return: A MotionEvent, StopMotionEvent or None.
        """
        pass

    def _calculate_is_moving(self, obj: Body) -> bool:
        """
        Determines whether an object is moving by evaluating the distance between its
        recorded positions within the window.

        :param obj: The body to check.
        :return: True if the object is moving, False otherwise.
        """

        latest_poses = self.context.latest_poses[obj]
        p_last = np.array(latest_poses[-1].to_position().to_list())
        for p in latest_poses[:-1]:
            p_prev = np.array(p.to_position().to_list())
            distance = np.linalg.norm(p_last - p_prev)
            if distance > self.distance_threshold:
                return True
        return False

    def _is_stationary(self, poses: List[Pose]) -> bool:
        """
        Determines if the object is stationary based on average movement.

        :param poses: List of recent poses.
        :return: True if stationary, False otherwise.
        """
        p_last = np.array(poses[-1].to_position().to_list())

        distances = [
            np.linalg.norm(p_last - np.array(p.to_position().to_list()))
            for p in poses[:-1]
        ]

        if not distances:
            return True

        avg_distance = np.mean(distances)

        return avg_distance < self.distance_threshold

@dataclass(eq=False, repr=False)
class TranslationDetector(MotionDetector):
    """
    Detector for translation events.
    Triggers a TranslationEvent when an object starts moving.
    """

    def _check_movement_and_trigger_event(self, obj: Body) -> Optional[Event]:
        """
        Checks the movement of an object and triggers a motion event if applicable.

        If an object is detected as moving and no active motion event exists, a new
        motion event is created and returned. If an object is moving and an existing
        motion event is found, the current pose is updated in the motion event. When
        the object is not moving, no event is triggered or updated.

        :param obj: The object being monitored for movement.
        :return: A TranslationEvent if the object is moving, otherwise None.
        """
        latest_motion_event = self.context.latest_motion_events.get(obj)
        latest_poses = self.context.latest_poses[obj]
        is_moving = self.context.object_moving_status.get(obj)

        if not is_moving or latest_motion_event is not None:
            return None

        new_event = TranslationEvent(
            tracked_object=obj,
            start_pose=latest_poses[0],
            current_pose=latest_poses[-1],
        )

        self.context.latest_motion_events[obj] = new_event
        return new_event


@dataclass(eq=False, repr=False)
class StopTranslationDetector(MotionDetector):
    """
    Detector for stop translation events.
    Triggers a StopTranslationEvent when an object that was moving stops.
    """

    def _check_movement_and_trigger_event(self, obj: Body) -> Optional[Event]:
        """
        Checks the movement of an object and triggers an event if necessary.

        This method examines the motion status of an object within the context. If
        the object is not moving and meets specific conditions, a stop event is
        triggered. The stop event indicates that the object has stopped translation
        based on a configured distance threshold.

        :param obj: The object to check for movement.
        :return: A StopTranslationEvent if the object stops moving, otherwise None.
        """
        latest_motion_event = self.context.latest_motion_events.get(obj)
        latest_poses = self.context.latest_poses[obj]
        is_moving = self.context.object_moving_status.get(obj)

        if is_moving:
            return None

        if latest_motion_event is None:
            return None

        if not self._is_stationary(latest_poses):
            return None

        stop_event = StopTranslationEvent(
            tracked_object=obj,
            start_pose=latest_motion_event.start_pose,
            current_pose=latest_poses[-1],
        )

        self.context.latest_motion_events.pop(obj, None)

        return stop_event



#Todo: Refactor the Rotationdetector
@dataclass(eq=False, repr=False)
class RotationDetector(MotionDetector):
    """
    Detector for rotation events.
    Triggers a RotationEvent when an object starts rotating.
    """

    def check_obj_movement(self, obj: Body) -> Optional[Event]:
        """
        Checks if an object is rotating based on its pose history.

        :param obj: The object to check.
        :return: A RotationEvent if the object is rotating, otherwise None.
        """
        latest_poses = self.context.latest_poses[obj]
        is_moving = False
        q_last = latest_poses[-1].to_quaternion().to_list()
        for p in latest_poses[:-1]:
            if not np.allclose(q_last, p.to_quaternion().to_list()):
                is_moving = True
                break
        self.context.object_moving_status[obj] = is_moving
        return self._check_movement_and_trigger_event(obj)

    def _check_movement_and_trigger_event(self, obj: Body) -> Optional[Event]:
        """
        Checks if an object is rotating and triggers a RotationEvent if necessary.

        :param obj: The object to check.
        :return: The RotationEvent if the object is rotating, otherwise None.
        """
        latest_motion_event = self.context.latest_motion_events.get(obj)
        latest_poses = self.context.latest_poses[obj]
        is_moving = self.context.object_moving_status.get(obj)

        if is_moving:
            if latest_motion_event is None:
                new_event = RotationEvent(
                    tracked_object=obj,
                    start_pose=latest_poses[0],
                    current_pose=latest_poses[-1],
                )
                self.context.latest_motion_events[obj] = new_event
                return new_event
            else:
                return None

        return None


@dataclass(eq=False, repr=False)
class StopRotationDetector(MotionDetector):
    """
    Detector for stop rotation events.
    Triggers a StopRotationEvent when an object that was rotating stops.
    """

    def check_obj_movement(self, obj: Body) -> Optional[Event]:
        """
        Check if an object is rotating based on its pose history.

        :param obj: The object to check.
        :return: A StopRotationEvent if the object is rotating, otherwise None.
        """
        latest_poses = self.context.latest_poses[obj]
        # Check orientation closeness using quaternions.
        is_moving = False
        q_last = latest_poses[-1].to_quaternion().to_list()
        for p in latest_poses[:-1]:
            if not np.allclose(q_last, p.to_quaternion().to_list()):
                is_moving = True
                break
        self.context.object_moving_status[obj] = is_moving
        return self._check_movement_and_trigger_event(obj)

    def _check_movement_and_trigger_event(self, obj: Body) -> Optional[Event]:
        """
        Checks if an object is rotating and triggers a StopRotationEvent if necessary.

        :param obj: The object to check for movement.
        :return: The StopRotationEvent if the object stops rotating, otherwise None.
        """
        latest_motion_event = self.context.latest_motion_events.get(obj)
        latest_poses = self.context.latest_poses[obj]
        is_moving = self.context.object_moving_status.get(obj)

        if is_moving:
            return None

        if latest_motion_event is None:
            return None

        all_poses_same = all(
            np.allclose(
                p.to_quaternion().to_list(),
                latest_poses[0].to_quaternion().to_list(),
            )
            for p in latest_poses
        )
        if all_poses_same:
            stop_event = StopRotationEvent(
                tracked_object=obj,
                start_pose=latest_motion_event.start_pose,
                current_pose=latest_poses[-1],
            )
            del self.context.latest_motion_events[obj]
            return stop_event

