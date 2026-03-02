from dataclasses import dataclass, field
from typing import Optional, List, Dict, Set, Any

from giskardpy.motion_statechart.context import MotionStatechartContext
from giskardpy.motion_statechart.data_types import ObservationStateValues
from giskardpy.motion_statechart.graph_node import MotionStatechartNode
from giskardpy.motion_statechart.motion_statechart import MotionStatechart
from segmind.datastructures.events import Event, ContactEvent, LossOfContactEvent
from semantic_digital_twin.reasoning.predicates import contact
from semantic_digital_twin.world_description.connections import Connection6DoF
from semantic_digital_twin.world_description.world_entity import Body


@dataclass
class SegmindContext(MotionStatechartContext):
    latest_contact_bodies: Dict[Body, Set[Body]] = None
    latest_support: Dict[Body, Set[Body]] = None
    # ToDo:  Why circular import for EventLogger?
    logger: Optional[Any] = None


class DetectorStateChart(MotionStatechart):
    pass


@dataclass(eq=False, repr=False)
class ContactDetector(MotionStatechartNode):
    tracked_object: Optional[Body] = field(kw_only=True, default=None)
    context: SegmindContext = field(kw_only=True)

    def on_tick(
        self, context: MotionStatechartContext
    ) -> Optional[ObservationStateValues]:
        trackable_objects = [
            body
            for body in self.context.world.bodies
            if type(body.parent_connection) is Connection6DoF
        ]
        objects_to_check = (
            [self.tracked_object] if self.tracked_object else trackable_objects
        )

        events = self.update_latest_contact_bodies_and_trigger_events(objects_to_check)
        for e in events:
            self.context.logger.log_event(e)

        if events:
            for e in events:
                self.context.logger.log_event(e)

            return ObservationStateValues.TRUE

        return ObservationStateValues.FALSE

    def print_something(self):
        trackable_objects = [
            body
            for body in self.context.world.bodies
            if type(body.parent_connection) is Connection6DoF
        ]
        print(trackable_objects)

    def get_contact_bodies(self, tracked_objects: List[Body]) -> Dict[Body, Set[Body]]:
        contact_bodies = {}
        for obj in tracked_objects:
            for body in self.context.world.bodies_with_collision:
                if body is obj:
                    continue
                if contact(obj, body):
                    contact_bodies[obj] = contact_bodies.get(obj, set()) | {body}
        return contact_bodies

    def update_latest_contact_bodies_and_trigger_events(
        self,
        tracked_objects: List[Body],
    ) -> List[Event]:
        new_contact_pairs = self.get_contact_bodies(tracked_objects)
        latest_contact_bodies = self.context.latest_contact_bodies
        events = []
        for obj, contact_list in new_contact_pairs.items():
            if obj not in latest_contact_bodies:
                latest_contact_bodies[obj] = contact_list
                for body in contact_list:
                    events.append(
                        ContactEvent(
                            close_bodies=contact_list,
                            latest_close_bodies=self.context.latest_contact_bodies,
                            of_object=obj,
                            with_object=body,
                        )
                    )

            elif obj in latest_contact_bodies:
                if latest_contact_bodies[obj] == contact_list:
                    continue
                latest_contact_bodies[obj] |= contact_list
                for body in contact_list:
                    if body in latest_contact_bodies[obj]:
                        continue
                    events.append(
                        ContactEvent(
                            close_bodies=contact_list,
                            latest_close_bodies=self.context.latest_contact_bodies,
                            of_object=obj,
                            with_object=body,
                        )
                    )
        self.context.latest_contact_bodies = latest_contact_bodies
        return events


@dataclass(eq=False, repr=False)
class LossOfContactDetector(MotionStatechartNode):
    tracked_object: Optional[Body] = field(kw_only=True, default=None)
    context: SegmindContext = field(kw_only=True)

    def on_tick(
        self, context: MotionStatechartContext
    ) -> Optional[ObservationStateValues]:
        trackable_objects = [
            body
            for body in self.context.world.bodies
            if type(body.parent_connection) is Connection6DoF
        ]
        objects_to_check = (
            [self.tracked_object] if self.tracked_object else trackable_objects
        )
        # self.context.latest_contact_bodies = self.update_latest_contact_bodies(objects_to_check)
        # Now we need to trigger the event
        events = self.update_latest_contact_bodies_and_trigger_events(objects_to_check)
        for e in events:
            self.context.logger.log_event(e)
        if events:
            for e in events:
                self.context.logger.log_event(e)
            return ObservationStateValues.TRUE
        return ObservationStateValues.FALSE

    def print_something(self):
        trackable_objects = [
            body
            for body in self.context.world.bodies
            if type(body.parent_connection) is Connection6DoF
        ]
        print(trackable_objects)

    def get_contact_bodies(self, tracked_objects: List[Body]) -> Dict[Body, Set[Body]]:
        contact_bodies = {}
        for obj in tracked_objects:
            for body in self.context.world.bodies_with_collision:
                if body is obj:
                    continue
                if contact(obj, body):
                    contact_bodies[obj] = contact_bodies.get(obj, set()) | {body}
        return contact_bodies

    def update_latest_contact_bodies_and_trigger_events(
        self,
        tracked_objects: List[Body],
    ) -> List[Event]:
        new_contact_pairs = self.get_contact_bodies(tracked_objects)
        latest_contact_bodies = self.context.latest_contact_bodies
        events = []
        for obj, contact_list in list(latest_contact_bodies.items()):
            if obj not in new_contact_pairs:
                latest_contact_bodies.pop(obj)
                for body in contact_list:
                    events.append(
                        LossOfContactEvent(
                            close_bodies=contact_list,
                            latest_close_bodies=self.context.latest_contact_bodies,
                            of_object=obj,
                            with_object=body,
                        )
                    )

            elif obj in new_contact_pairs:
                new_contacts = new_contact_pairs[obj]
                lost_contact = contact_list - new_contacts
                for body in lost_contact:
                    events.append(
                        LossOfContactEvent(
                            close_bodies=contact_list,
                            latest_close_bodies=self.context.latest_contact_bodies,
                            of_object=obj,
                            with_object=body,
                        )
                    )
                latest_contact_bodies[obj] = new_contacts
        self.context.latest_contact_bodies = latest_contact_bodies
        return events
