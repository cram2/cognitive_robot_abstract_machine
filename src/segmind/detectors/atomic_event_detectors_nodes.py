from abc import abstractmethod, ABC
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Set, Any
import numpy as np

from giskardpy.motion_statechart.context import MotionStatechartContext
from giskardpy.motion_statechart.data_types import ObservationStateValues
from giskardpy.motion_statechart.graph_node import MotionStatechartNode
from giskardpy.motion_statechart.motion_statechart import MotionStatechart
from segmind.datastructures.events import Event, ContactEvent, LossOfContactEvent, MotionEvent, TranslationEvent, RotationEvent, StopTranslationEvent, StopRotationEvent
from semantic_digital_twin.reasoning.predicates import contact
from semantic_digital_twin.spatial_types.spatial_types import Pose
from semantic_digital_twin.world_description.connections import Connection6DoF
from semantic_digital_twin.world_description.world_entity import Body


@dataclass
class SegmindContext(MotionStatechartContext):
    """
    Context object shared across the motion statechart detectors.

    Stores the latest detected contact and support relationships
    between bodies in the simulation as well as the event logger.
    ToDo:  Why circular import for EventLogger? and move this away
    """


    IndexedBodyPairs = Dict[Body, Set[Body]]
    """
    Type hint for dictionaries mapping bodies to sets of bodies
    """
    
    object_moving_status: Dict[Body, bool] = None
    """
    
    """
    
    latest_contact_bodies: IndexedBodyPairs = None
    """
    :param latest_contact_bodies: Dictionary mapping each body to the set of
    bodies it is currently in contact with.
    """

    latest_support: IndexedBodyPairs = None
    """
    :param latest_support: Dictionary mapping each body to the set of bodies
    that currently support it.
    """

    latest_containments: IndexedBodyPairs = None
    """
    :param latest_support: Dictionary mapping each body to the set of bodies
    that currently support it.
    """
    
    latest_poses: Dict[Body, List[Pose]] = field(default_factory=dict)
    
    latest_motion_events: Dict[Body, MotionEvent] = field(default_factory=dict)
    
    logger: Any = None
    """
    :param latest_support: Dictionary mapping each body to the set of bodies
    that currently support it.
    """

#ToDo: See if we can create our own MotionStatechartNode or change its name (talk to simon)
@dataclass(repr=False, eq=False)
class DetectorStateChartNode(MotionStatechartNode):
    pass


@dataclass
class DetectorStateChart(MotionStatechart):
    """
    Statechart responsible for running the different motion detectors.

    Currently acts as a container for the detectors and inherits the
    functionality from MotionStatechart.
    """

    pass


#ToDo: there is a lot of duplication with SupportDetector, so we have to make it more robust
@dataclass(eq=False, repr=False)
class BaseContactDetector(DetectorStateChartNode, ABC):
    """
    Abstract base class for contact-based detectors.

    Provides shared functionality for detecting contacts between
    bodies and generating events when contact relationships change.
    """

    tracked_object: Optional[Body] = field(kw_only=True, default=None)
    """
    :param tracked_object: Optional body that should be monitored.
    If None, all trackable objects in the world are checked.
    """

    context: SegmindContext = field(kw_only=True)
    """
    :param context: Segmind context containing world information,
    contact history and logging utilities.
    """

    def on_tick(
        self, context: SegmindContext
    ) -> Optional[ObservationStateValues]:
        """
        Executes one update cycle of the detector.

        Determines the objects that should be checked for contacts,
        computes new contact relationships and triggers events if
        contact changes are detected.

        :param context: The statechart execution context.
        :return: ObservationStateValues.TRUE if events were triggered,
        otherwise ObservationStateValues.FALSE.
        """

        objects_to_check = (
            [self.tracked_object] if self.tracked_object else [
                body
                for body in self.context.world.bodies
                if type(body.parent_connection) is Connection6DoF
            ]
        )
        events = self.update_latest_contact_bodies_and_trigger_events(objects_to_check)
        for e in events:
            self.context.logger.log_event(e)
        return ObservationStateValues.TRUE if events else ObservationStateValues.FALSE

    def get_contact_bodies(self, tracked_objects: List[Body]) -> Dict[Body, Set[Body]]:
        """
        Computes the contact relationships for a set of tracked objects.

        Iterates over all bodies with collision enabled and checks whether
        they are in contact with the tracked objects.

        :param tracked_objects: List of bodies that should be checked for contacts.
        :return: Dictionary mapping each tracked body to the set of bodies
            it is currently in contact with.
        """

        contact_bodies: Dict[Body, Set[Body]] = {}
        bodies_with_collision = self.context.world.bodies_with_collision
        for obj in tracked_objects:
            for body in bodies_with_collision:
                if body is obj:
                    continue
                if contact(obj, body):
                    contact_bodies.setdefault(obj, set()).add(body)
        return contact_bodies

    @abstractmethod
    def update_latest_contact_bodies_and_trigger_events(
        self,
        tracked_objects: List[Body],
    ) -> List[Event]:
        """
        Updates the stored contact relationships and generates events
        when changes are detected.

        Implementations define how contact changes are interpreted
        (e.g. new contact or loss of contact).

        :param tracked_objects: List of bodies to check for contact changes.
        :return: List of events generated during the update.
        """

        pass


@dataclass(eq=False, repr=False)
class ContactDetector(BaseContactDetector):
    """
    Detector responsible for identifying newly established contacts
    between bodies.
    """

    def update_latest_contact_bodies_and_trigger_events(
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

        new_contact_pairs = self.get_contact_bodies(tracked_objects)
        latest_contact_bodies = self.context.latest_contact_bodies
        events = []
        for obj, contact_list in new_contact_pairs.items():
            new_contacts = contact_list if obj not in latest_contact_bodies else contact_list - latest_contact_bodies[obj]
            if new_contacts:
                latest_contact_bodies.setdefault(obj, set()).update(new_contacts)
                events.extend([ContactEvent(of_object=obj, with_object=c) for c in new_contacts])

        return events


@dataclass(eq=False, repr=False)
class LossOfContactDetector(BaseContactDetector):
    """
    Detector responsible for identifying when previously existing
    contacts between bodies are lost.
    """

    def update_latest_contact_bodies_and_trigger_events(
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

        new_contact_pairs = self.get_contact_bodies(tracked_objects)
        latest_contact_bodies = self.context.latest_contact_bodies
        events = []
        for obj, contact_list in list(latest_contact_bodies.items()):
            loss_contacts = contact_list if obj not in new_contact_pairs else contact_list - new_contact_pairs[obj]
            if loss_contacts:
                latest_contact_bodies.pop(obj)
                events.extend([LossOfContactEvent(of_object=obj, with_object=s) for s in loss_contacts])

        return events


@dataclass(eq=False, repr=False)
class MotionDetector(DetectorStateChartNode, ABC):
    """
    Abstract base class for motion-based detectors.

    Provides shared functionality for monitoring poses of
    bodies and generating events when movement is detected.
    """

    tracked_object: Optional[Body] = field(kw_only=True, default=None)
    """
    :param tracked_object: Optional body that should be monitored.
    If None, all trackable objects in the world are checked.
    """

    context: SegmindContext = field(kw_only=True)
    """
    :param context: Segmind context containing world information,
    contact history and logging utilities.
    """
    window_size: int = 4

    def on_tick(
            self, context: SegmindContext
    ) -> Optional[ObservationStateValues]:
        """
        Executes one update cycle of the detector.

        Determines the objects that should be monitored for movement,
        computes new movement status and triggers events if
        motion is detected.

        :param context: The statechart execution context.
        :return: ObservationStateValues.TRUE if events were triggered,
        otherwise ObservationStateValues.FALSE.
        """

        objects_to_check = (
            [self.tracked_object] if self.tracked_object else [
                body
                for body in self.context.world.bodies
                if type(body.parent_connection) is Connection6DoF
            ]
        )
        events = self.update_latest_pose_and_trigger_events(objects_to_check)
        for e in events:
            self.context.logger.log_event(e)
        return ObservationStateValues.TRUE if events else ObservationStateValues.FALSE
    
    
    def update_latest_pose_and_trigger_events(self, tracked_objs: List[Body]) -> List[Event]:
        events = []
        for obj in tracked_objs:
            latest_poses = self.context.latest_poses.setdefault(obj, [])
            latest_poses.append(obj.global_pose.to_pose())
            if len(latest_poses) > self.window_size:
                event = self.check_obj_movement(obj)
                if event:
                    events.append(event)
                latest_poses.pop(0)
        return events

    @abstractmethod
    def check_obj_movement(self, obj: Body) -> Optional[Event]:
        """
        Determines if an object is moving based on its pose history.

        :param obj: The body to check.
        :return: An Event if movement is detected, otherwise None.
        """
        pass

    def _check_movement_and_trigger_event(
        self,
        obj: Body,
        is_moving: bool,
        motion_event_type: type,
        stop_motion_event_type: type
    ) -> Optional[Event]:
        latest_motion_event = self.context.latest_motion_events.get(obj)
        latest_poses = self.context.latest_poses[obj]

        if is_moving:
            # If the object is moving:
            # 1. If it wasn't moving before, create a new MotionEvent.
            # 2. If it was already moving, update the current_pose but don't return a new event.
            if latest_motion_event is None:
                new_event = motion_event_type(
                    tracked_object=obj,
                    start_pose=latest_poses[0],
                    current_pose=latest_poses[-1]
                )
                self.context.latest_motion_events[obj] = new_event
                return new_event
            else:
                # Update current pose of the existing event
                latest_motion_event.current_pose = latest_poses[-1]
                return None
        else:
            # If the object is not moving:
            # 1. If it was moving before, check if it truly stopped (all poses in window are same).
            # 2. If it stopped, create a StopMotionEvent and clear the active event.
            if latest_motion_event is None:
                return None
            
            # Check if ALL poses in the window are the same.
            all_poses_same = all(np.allclose(p.to_list(), latest_poses[0].to_list()) for p in latest_poses)
            if all_poses_same:
                stop_event = stop_motion_event_type(
                    tracked_object=obj,
                    start_pose=latest_motion_event.start_pose,
                    current_pose=latest_poses[-1]
                )
                del self.context.latest_motion_events[obj]
                return stop_event
        
        return None


@dataclass(eq=False, repr=False)
class TranslationDetector(MotionDetector):
    def check_obj_movement(self, obj: Body) -> Optional[Event]:
        latest_poses = self.context.latest_poses[obj]
        # Simple movement check: is the last pose different from the previous one?
        # Or should it be: is the last pose different from the first one in the window?
        # The requirement says "when all 4 Poses are not showing any difference" for StopMotionEvent.
        is_moving = not np.allclose(latest_poses[-1].to_position().to_list(), latest_poses[-2].to_position().to_list())
        return self._check_movement_and_trigger_event(
            obj, is_moving, TranslationEvent, StopTranslationEvent
        )


@dataclass(eq=False, repr=False)
class StopTranslationDetector(MotionDetector):
    def check_obj_movement(self, obj: Body) -> Optional[Event]:
        latest_poses = self.context.latest_poses[obj]
        
@dataclass(eq=False, repr=False)
class RotationDetector(MotionDetector):
    def check_obj_movement(self, obj: Body) -> Optional[Event]:
        latest_poses = self.context.latest_poses[obj]
        # We need to define how to check orientation closeness.
        is_moving = not np.allclose(latest_poses[-1].to_quaternion().to_list(), latest_poses[-2].to_quaternion().to_list())
        return self._check_movement_and_trigger_event(
            obj, is_moving, RotationEvent, StopRotationEvent
        )