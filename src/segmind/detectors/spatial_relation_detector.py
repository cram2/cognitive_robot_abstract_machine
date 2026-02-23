import time
from abc import abstractmethod
from os.path import dirname
from queue import Queue, Empty

from ripple_down_rules.rdr_decorators import RDRDecorator
from segmind import logger, set_logger_level, LogLevel
from semantic_digital_twin.reasoning.predicates import (
    InsideOf,
    contact,
    is_supported_by,
)
from semantic_digital_twin.world_description.world_entity import Body
from typing_extensions import Any, Dict, List, Optional, Type, Callable, Union


from segmind.datastructures.events import PlacingEvent, CloseContactEvent
from segmind.episode_player import EpisodePlayer
from .atomic_event_detectors import AtomicEventDetector
from ..datastructures.events import (
    MotionEvent,
    EventUnion,
    StopMotionEvent,
    NewObjectEvent,
    InsertionEvent,
    Event,
    ContainmentEvent,
    ContactEvent,
    LossOfContactEvent,
    StopTranslationEvent,
    LossOfSupportEvent,
    SupportEvent,
)
from ..datastructures.object_tracker import ObjectTrackerFactory
from ..utils import get_support

set_logger_level(LogLevel.DEBUG)
EventCondition = Callable[[EventUnion], bool]


class SpatialRelationDetector(AtomicEventDetector):
    """
    A class that detects spatial relations between objects.
    """

    check_on_events = {NewObjectEvent: None, StopMotionEvent: None}

    def __init__(
        self,
        check_on_events: Optional[Dict[Type[Event], EventCondition]] = None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.event_queue: Queue[EventUnion] = Queue()
        self.queues.append(self.event_queue)
        self.check_on_events = (
            check_on_events if check_on_events is not None else self.check_on_events
        )
        self.bodies_states: Dict[Body, Any] = {}
        self.update_initial_state()
        for event, cond in self.check_on_events.items():
            self.logger.add_callback(event, self.on_event, cond)

    def reset(self):
        self.bodies_states = {}
        self.update_initial_state()
        self.event_queue = Queue()

    def update_initial_state(self):
        for body in self.world.bodies:
            self.update_body_state(body)

    @abstractmethod
    def update_body_state(self, body: Body):
        """
        Update the state of a body.
        """
        pass

    def on_event(self, event: MotionEvent):
        """
        A callback that is called when a MotionEvent occurs, and adds the event to the event queue.

        :param event: The MotionEvent that occurred.
        """
        self.event_queue.put(event)

    def detect_events(self) -> None:
        """
        Detect spatial relations between objects.
        """
        try:
            checked_bodies: List[Body] = []
            while True:
                event = self.event_queue.get_nowait()
                self.event_queue.task_done()
                logger.debug(f"Checking event {event}")
                involved_bodies = event.involved_bodies
                bodies_to_check = list(
                    filter(lambda body: body not in checked_bodies, involved_bodies)
                )
                checked_bodies.extend(
                    self.world.update_containment_for(bodies_to_check)
                )
                logger.debug(
                    f"Checked bodies: {[body.name for body in checked_bodies]}"
                )
        except Empty:
            pass

    def __str__(self):
        return self.__class__.__name__

    def _join(self, timeout=None):
        pass


class ContainmentDetector(SpatialRelationDetector):

    # This one is messy, its just for the sake of the test, since the contained object is actually not supported by anything and just floats around.
    # check_on_events = {ContactEvent: None, CloseContactEvent: None, LossOfContactEvent: None, PlacingEvent: None, StopTranslationEvent: None, NewObjectEvent: None, StopMotionEvent: None}
    check_on_events = {
        ContactEvent: None,
        CloseContactEvent: None,
        LossOfContactEvent: None,
        PlacingEvent: None,
        StopTranslationEvent: None,
    }

    def update_body_state(self, body: Body):
        """
        Update the state of a body.
        """
        for b in self.world.bodies:
            if b == body:
                continue
            if (
                InsideOf(body, b).compute_containment_ratio() > 0.6
                and b not in self.bodies_states
            ):
                logger.debug(
                    f"We are getting in here for {body.name} and {b.name} and "
                    f"the ratio is {InsideOf(body, b).compute_containment_ratio()}"
                )

                self.bodies_states[body] = [b]
        # if isinstance(body, Body):
        #     for link in body.links.values():
        #         self.bodies_states[link] = link.update_containment(intersection_ratio=0.6)
        # self.bodies_states[body] = body.update_containment(intersection_ratio=0.6)

    def detect_events(self) -> None:
        """
        Detect Containment relations between objects.
        """
        try:
            while True:
                event = self.event_queue.get_nowait()
                self.event_queue.task_done()
                if event.tracked_object in self.bodies_states:
                    known_containments = self.bodies_states[event.tracked_object]
                else:
                    known_containments = []
                self.update_body_state(event.tracked_object)
                logger.debug(
                    f"Known containments for {event.tracked_object.name}: {known_containments}"
                )
                new_containments = set(self.bodies_states[event.tracked_object]) - set(
                    known_containments
                )
                logger.debug(
                    f"New containments for {event.tracked_object.name}: {new_containments}"
                )
                for new_containment in new_containments:
                    self.logger.log_event(
                        ContainmentEvent(
                            tracked_object=event.tracked_object,
                            with_object=new_containment,
                            timestamp=event.timestamp,
                        )
                    )
                time.sleep(self.wait_time.total_seconds())
        except Empty:
            pass

    @classmethod
    def event_types(cls) -> List[Type[ContainmentEvent]]:
        return [ContainmentEvent]


class InsertionDetector(SpatialRelationDetector):

    def event_condition(self, event: ContainmentEvent) -> bool:
        return self.get_latest_interference_with_hole(event) is not None

    check_on_events = {ContainmentEvent: event_condition}

    @staticmethod
    def get_latest_interference_with_hole(
        event: ContainmentEvent,
    ) -> Optional[ContactEvent]:
        """
        Get the latest interference event with a hole before the given event.
        """

        def conditions(event_to_check: EventUnion) -> bool:
            return (
                isinstance(event_to_check, ContactEvent)
                and event_to_check.timestamp < event.timestamp
                and "hole" in event_to_check.with_object.name
            )

        return event.object_tracker.get_nearest_event_to_event_with_conditions(
            event, conditions
        )

    def update_body_state(self, body: Body): ...

    def detect_events(self) -> None:
        """
        Detect Containment relations between objects.
        """
        try:
            checked_bodies: List[Body] = []
            while True:
                event: ContainmentEvent = self.event_queue.get_nowait()
                self.event_queue.task_done()
                if event.tracked_object in checked_bodies:
                    logger.debug(
                        f"tracked object {event.tracked_object.name} is already checked"
                    )
                    continue
                while True:
                    latest_interference_with_hole = (
                        self.get_latest_interference_with_hole(event)
                    )
                    hole = latest_interference_with_hole.with_object
                    if not self.hole_insertion_verifier(hole, event):
                        if event.tracked_object.is_moving:
                            time.sleep(self.wait_time.total_seconds())
                            continue
                        else:
                            break
                    agent = event.agent if hasattr(event, "agent") else None
                    end_timestamp = (
                        event.end_timestamp if hasattr(event, "end_timestamp") else None
                    )
                    insertion_event = InsertionEvent(
                        inserted_object=event.tracked_object,
                        inserted_into_objects=[event.with_object],
                        through_hole=hole,
                        agent=agent,
                        timestamp=event.timestamp,
                        end_timestamp=end_timestamp,
                    )
                    self.logger.log_event(insertion_event)
                    break
                time.sleep(self.wait_time.total_seconds())
        except Empty:
            pass

    @staticmethod
    def ask_now(case_dict):
        self_ = case_dict["self_"]
        hole = case_dict["hole"]
        event = case_dict["event"]
        return "object_3" in event.tracked_object.name

    hole_insertion_verifier_rdr = RDRDecorator(
        f"{dirname(__file__)}/models",
        (bool,),
        True,
        fit=False,
        fitting_decorator=EpisodePlayer.pause_resume,
        package_name="segmind",
        ask_now=ask_now,
    )

    @hole_insertion_verifier_rdr.decorator
    def hole_insertion_verifier(self, hole: Body, event: ContainmentEvent) -> bool:
        pass

    @classmethod
    def event_types(cls) -> List[Type[InsertionEvent]]:
        return [InsertionEvent]


class SupportDetector(SpatialRelationDetector):

    pause_condition = (
        lambda event: isinstance(event, StopTranslationEvent)
        and event.tracked_object in event.world.bodies_with_enabled_collision
    )

    @staticmethod
    # @EpisodePlayer.pause_resume_with_condition(pause_condition)
    def event_condition(event: StopTranslationEvent) -> bool:
        for obj in event.tracked_object._world.bodies_with_enabled_collision:
            is_supported = is_supported_by(event.tracked_object, obj)
            if not is_supported:
                continue
            return True

        return False

    check_on_events = {
        StopTranslationEvent: None,
        LossOfContactEvent: None,
        ContactEvent: None,
        CloseContactEvent: None,
    }

    def update_body_state(self, body: Body, with_bodies: Optional[List[Body]] = None):
        """
        Update the state of a body.
        """
        support = get_support(body)
        if body in self.bodies_states:
            if support is None:
                if self.bodies_states[body] is None:
                    return
                else:
                    self.logger.log_event(
                        LossOfSupportEvent(
                            tracked_object=body, with_object=self.bodies_states[body]
                        )
                    )
            else:
                if self.bodies_states[body] is None:
                    self.logger.log_event(
                        SupportEvent(tracked_object=body, with_object=support)
                    )
                elif support != self.bodies_states[body]:
                    self.logger.log_event(
                        LossOfSupportEvent(
                            tracked_object=body, with_object=self.bodies_states[body]
                        )
                    )
                    self.logger.log_event(
                        SupportEvent(tracked_object=body, with_object=support)
                    )
                else:
                    return
        self.bodies_states[body] = support
        ObjectTrackerFactory.get_tracker(body).support = support

    def detect_events(self) -> None:
        """
        Detect Containment relations between objects.
        """
        try:
            while True:
                event = self.event_queue.get_nowait()
                self.event_queue.task_done()
                if event.tracked_object in self.world.bodies_with_enabled_collision:
                    self.update_body_state(event.tracked_object)
                time.sleep(self.wait_time.total_seconds())
        except Empty:
            pass

    @classmethod
    def event_types(cls) -> List[Type[Union[SupportEvent, LossOfSupportEvent]]]:
        return [SupportEvent, LossOfSupportEvent]
