from __future__ import annotations

import os.path

from pycram.designators.location_designator import Location
from pycram.plan import Plan

from ripple_down_rules.rdr_decorators import RDRDecorator
from semantic_digital_twin import world
from semantic_digital_twin.semantic_annotations.semantic_annotations import Floor

try:
    from matplotlib import pyplot as plt
except ImportError:
    plt = None

from typing_extensions import Optional, List, Union, Dict


from .atomic_event_detectors import *
from ..datastructures.events import *
from ..utils import (
    get_angle_between_vectors,
    get_support,
    is_object_supported_by_container_body,
)
from ..episode_player import EpisodePlayer


class DetectorWithStarterEvent(AtomicEventDetector, ABC):
    """
    A type of event detector that requires an event to occur as a start condition.
    """

    def __init__(
        self,
        logger: EventLogger,
        starter_event: EventUnion,
        wait_time: Optional[timedelta] = None,
        *args,
        **kwargs,
    ):
        """
        :param logger: An instance of the EventLogger class that is used to log the events.
        :param starter_event: An instance of the Event class that represents the event to start the event detector.
        :param wait_time: An optional timedelta value that introduces a delay between calls to the event detector.
        """
        super().__init__(logger, wait_time, *args, **kwargs)
        self.starter_event: EventUnion = starter_event
        self._start_timestamp = self.starter_event.timestamp

    @classmethod
    @abstractmethod
    def action_type(cls):
        """
        The action type that this detector detects.
        """
        pass

    @classmethod
    @abstractmethod
    def event_types(cls) -> List[Type[Event]]:
        """
        The event type that this detector invokes.
        """
        pass

    @classmethod
    @abstractmethod
    def start_condition_checker(cls, event: Event) -> bool:
        """
        Check if the event is a starter event.

        :param event: The Event instance that represents the event.
        """
        pass

    @property
    def start_timestamp(self) -> float:
        return self._start_timestamp

    @start_timestamp.setter
    def start_timestamp(self, timestamp: float):
        self._start_timestamp = timestamp

    def _no_event_found_log(self, event_type: Type[Event]):
        print(
            f"{self} with starter event: {self.starter_event} found no event of type: {event_type}"
        )


class DetectorWithTrackedObjectAndStarterEvent(
    DetectorWithStarterEvent, HasPrimaryTrackedObject, ABC
):
    """
    A type of event detector that requires an event to occur as a start condition and has one tracked object.
    """

    currently_tracked_objects: Optional[
        Dict[Body, DetectorWithTrackedObjectAndStarterEvent]
    ] = None
    """
    All the objects that are currently tracked by a detector with a starter event.
    """

    def __init__(
        self,
        logger: EventLogger,
        starter_event: EventUnion,
        wait_time: Optional[timedelta] = None,
        *args,
        **kwargs,
    ):
        """
        :param logger: An instance of the EventLogger class that is used to log the events.
        :param starter_event: An instance of the Event class that represents the event to start the event detector.
        :param wait_time: An optional timedelta value that introduces a delay between calls to the event detector.
        """
        DetectorWithStarterEvent.__init__(
            self, logger, starter_event, wait_time, *args, **kwargs
        )
        object_to_track = self.get_object_to_track_from_starter_event(starter_event)
        HasPrimaryTrackedObject.__init__(self, tracked_object=object_to_track)
        if self.currently_tracked_objects is None:
            self.currently_tracked_objects = {}
        self.currently_tracked_objects[self.tracked_object] = self

    def __hash__(self):
        return DetectorWithStarterEvent.__hash__(self)

    def reset(self):
        self.currently_tracked_objects = {}

    def check_for_event_pre_starter_event(
        self, event_type: Type[Event], time_tolerance: timedelta
    ) -> Optional[EventUnion]:
        """
        Check if the tracked_object was involved in an event before the starter event.

        :param event_type: The event type to check for.
        :param time_tolerance: The time tolerance to consider the event as before the starter event.
        """
        event = self.object_tracker.get_first_event_of_type_before_event(
            event_type, self.starter_event
        )
        if (
            event is not None
            and self.start_timestamp - event.timestamp <= time_tolerance.total_seconds()
        ):
            return event
        else:
            self._no_event_found_log(event_type)

    def check_for_event_post_starter_event(
        self, event_type: Type[Event]
    ) -> Optional[EventUnion]:
        """
        Check if the tracked_object was involved in an event after the starter event.

        :param event_type: The event type to check for.
        :return: The event if the tracked_object was involved in an event, else None.
        """
        event = self.object_tracker.get_first_event_of_type_after_event(
            event_type, self.starter_event
        )
        if event is None:
            self._no_event_found_log(event_type)
        return event

    def check_for_event_near_starter_event(
        self, event_type: Type[Event], time_tolerance: timedelta
    ) -> Optional[EventUnion]:
        """
        Check if the tracked_object was involved in an event near the starter event (i.e. could be before or after).

        :param event_type: The event type to check for.
        :param time_tolerance: The time tolerance to consider the event as near the starter event.
        :return: The event if the tracked_object was involved in an event, else None.
        """
        event = self.object_tracker.get_nearest_event_of_type_to_event(
            self.starter_event, tolerance=time_tolerance, event_type=event_type
        )
        if event is None:
            self._no_event_found_log(event_type)
        return event

    @classmethod
    @abstractmethod
    def get_object_to_track_from_starter_event(cls, starter_event: EventUnion) -> Body:
        """
        Get the object to track from the starter event.

        :param starter_event: The starter event that can be used to get the object to track.
        """
        pass

    def __str__(self):
        return f"{self.thread_id} - {self.tracked_object.name if self.tracked_object is not None else None}"


class AbstractInteractionDetector(DetectorWithTrackedObjectAndStarterEvent, ABC):
    """
    An abstract detector that detects an interaction between the agent and an object.
    """

    models_path: str = os.path.join(os.path.dirname(__file__), "models")
    """
    The path to the directory where the Ripple Down Rules models are stored.
    """

    def __init__(self, logger: EventLogger, starter_event: EventUnion, *args, **kwargs):
        """
        :param logger: An instance of the EventLogger class that is used to log the events.
        :param starter_event: An instance of a type of Event that represents the event to
         start the event detector.
        """
        DetectorWithTrackedObjectAndStarterEvent.__init__(
            self, logger, starter_event, *args, **kwargs
        )
        self.run_once: bool = True

    def detect_events(self) -> List[EventUnion]:
        """
        Detect if the tracked_object was interacted with by the agent.

        :return: An instance of the interaction event if the tracked_object was interacted with, else None.
        """
        event = None
        while not self.kill_event.is_set():

            event = self.get_interaction_event()
            if not event:
                time.sleep(0.01)
                continue
            self.currently_tracked_objects.pop(self.tracked_object, None)
            break

        if event:
            print(
                f"{self.__class__.__name__} detected an interaction with: {self.tracked_object.name}"
            )
            return [event]

        return []

    @abstractmethod
    def get_interaction_event(self) -> Optional[EventUnion]:
        """
        Perform checks to determine if the object was interacted with, and return the interaction event.

        :return: A boolean value that represents if all the checks passed and the object was interacted with.
        """
        pass

    @classmethod
    @abstractmethod
    def get_object_to_track_from_starter_event(cls, starter_event: EventUnion) -> Body:
        """
        Get the object to track for interaction from the possible starter event.

        :param starter_event: The possible starter event that can be used to get the object to track.
        """
        pass

    def __str__(self):
        return f"{self.thread_id} - {self.tracked_object.name}"

    def _join(self, timeout=None):
        pass


class AbstractPickUpDetector(AbstractInteractionDetector, ABC):
    """
    An abstract detector that detects if the tracked_object was picked up.
    """

    currently_tracked_objects: Dict[Body, AbstractPickUpDetector] = {}

    @classmethod
    def action_type(cls):
        return PickUpAction

    @classmethod
    def event_types(cls) -> List[Type[PickUpEvent]]:
        return [PickUpEvent]


class GeneralPickUpDetector(AbstractPickUpDetector):
    """
    A detector that detects pick-up events based on incremental learning using Ripple Down Rules.
    """

    models_path: str = AbstractInteractionDetector.models_path
    interaction_checks_rdr: RDRDecorator = RDRDecorator(
        models_path,
        (PickUpEvent, type(None)),
        True,
        package_name="segmind",
        fit=False,
        update_existing_rules=False,
        use_generated_classifier=True,
        fitting_decorator=EpisodePlayer.pause_resume,
    )
    """
    A decorator that uses a Ripple Down Rules model to check if the tracked_object was picked up and returns the PickUp Event.
    """

    object_to_track_rdr: RDRDecorator = RDRDecorator(
        models_path,
        (Body, type(None)),
        True,
        package_name="segmind",
        fit=False,
        use_generated_classifier=False,
        fitting_decorator=EpisodePlayer.pause_resume,
    )
    """
    A decorator that uses a Ripple Down Rules model to get the object to track from the starter event.
    """

    @staticmethod
    def ask_now(case_dict):
        cls_ = case_dict["cls_"]
        event = case_dict["event"]
        return (
            isinstance(event, LossOfSupportEvent)
            and event.tracked_object.name == "montessori_object_5"
        )

    start_condition_rdr: RDRDecorator = RDRDecorator(
        models_path,
        (bool,),
        True,
        package_name="segmind",
        fit=False,
        use_generated_classifier=False,
        fitting_decorator=EpisodePlayer.pause_resume,
        ask_now=ask_now,
        generate_dot_file=True,
    )
    """
    A decorator that uses a Ripple Down Rules model to check for starting conditions for the pick up event.
    """

    @interaction_checks_rdr.decorator
    def get_interaction_event(self) -> Optional[PickUpEvent]:
        pass

    @classmethod
    @object_to_track_rdr.decorator
    def get_object_to_track_from_starter_event(cls, starter_event: EventUnion) -> Body:
        pass

    @classmethod
    @start_condition_rdr.decorator
    def start_condition_checker(cls, event: Event) -> bool:
        pass

    def __str__(self):
        if (
            hasattr(self.starter_event, "agent")
            and self.starter_event.agent is not None
        ):
            return f"{super().__str__()} - Agent: {self.agent.name}"
        else:
            return super().__str__()


class PlacingDetector(AbstractInteractionDetector):
    # """
    # An abstract detector that detects if the tracked_object was placed by the agent.
    # """
    #
    # thread_prefix = "placing_"
    #
    # models_path: str = AbstractInteractionDetector.models_path
    # interaction_checks_rdr: RDRDecorator = RDRDecorator(models_path, (PlacingEvent, type(None)), True, package_name="segmind",
    #  fit=False, update_existing_rules=True, use_generated_classifier=False, fitting_decorator=EpisodePlayer.pause_resume)
    # """
    # A decorator that uses a Ripple Down Rules model to check if the tracked_object was picked up and returns the PickUp Event.
    # """
    #
    # object_to_track_rdr: RDRDecorator = RDRDecorator(models_path, (Body, type(None)), True, package_name="segmind",
    #  fit=False, use_generated_classifier=False, fitting_decorator=EpisodePlayer.pause_resume)
    # """
    # A decorator that uses a Ripple Down Rules model to get the object to track from the starter event.
    # """
    # @staticmethod
    # def ask_now(case_dict):
    #     cls_ = case_dict["cls_"]
    #     event = case_dict["event"]
    #     return isinstance(event, SupportEvent)
    # start_condition_rdr: RDRDecorator = RDRDecorator(models_path, (bool,), True, package_name="segmind",
    #  fit=False, use_generated_classifier=False, fitting_decorator=EpisodePlayer.pause_resume, ask_now=ask_now)
    # """
    # A decorator that uses a Ripple Down Rules model to check for starting conditions for the pick up event.
    # """
    # print("a")

    @classmethod
    def start_condition_checker(cls, event: SupportEvent) -> bool:
        if isinstance(event, SupportEvent):
            return True

        return False

    @classmethod
    def get_object_to_track_from_starter_event(cls, starter_event: EventUnion) -> Body:
        return starter_event.tracked_object

    @classmethod
    def action_type(cls):
        return PlaceAction

    @classmethod
    def event_types(cls) -> List[Type[PlacingEvent]]:
        return [PlacingEvent]

    def get_interaction_event(self) -> Optional[PlacingEvent]:
        if isinstance(self.starter_event, SupportEvent):
            return PlacingEvent(
                tracked_object=self.tracked_object,
                timestamp=self.starter_event.timestamp - self.wait_time.total_seconds(),
                end_timestamp=self.starter_event.timestamp,
            )

    # @interaction_checks_rdr.decorator
    # def get_interaction_event(self) -> Optional[PlacingEvent]:
    #     pass
    #
    # @classmethod
    # @object_to_track_rdr.decorator
    # def get_object_to_track_from_starter_event(cls, starter_event: CloseContactEvent) -> Body:
    #     pass
    #
    # @classmethod
    # @start_condition_rdr.decorator
    # def start_condition_checker(cls, event: EventWithOneTrackedObject) -> bool:
    #     """
    #     Check if an agent is in contact with the tracked_object.
    #
    #     :param event: The ContactEvent instance that represents the contact event.
    #     """
    #     pass


def check_for_supporting_surface(
    tracked_object: Body, possible_surfaces: Optional[List[Body]] = None
) -> Optional[Body]:
    """
    Check if any of the possible surfaces are supporting the tracked_object.

    :param tracked_object: An instance of the Object class that represents the tracked_object to check.
    :param possible_surfaces: A list of Object instances that represent the possible surfaces.
    :return: An instance of the Object class that represents the supporting surface if found, else None.
    """
    with world.World():
        dt = 0.07
        World.current_world.simulate(dt)
        prospection_obj = World.current_world.get_prospection_object_for_object(
            tracked_object
        )
        contact_points = prospection_obj.contact_points
        contacted_bodies = contact_points.get_all_bodies()
        contacted_bodies = [
            body for body in contacted_bodies if body.name != tracked_object.name
        ]
        contacted_body_names = [body.name for body in contacted_bodies]
        contacted_bodies = dict(zip(contacted_body_names, contacted_bodies))
        if possible_surfaces is None:
            possible_surface_names = contacted_body_names
        else:
            possible_surface_names = [obj.name for obj in possible_surfaces]
            possible_surface_names = list(
                set(contacted_body_names).intersection(possible_surface_names)
            )
    supporting_surface = None
    opposite_gravity = [0, 0, 1]
    smallest_angle = np.pi / 8
    for obj_name in possible_surface_names:
        obj = World.current_world.get_object_by_name(obj_name)
        normals = [
            n.to_list()
            for n in contact_points.get_normals_of_object(contacted_bodies[obj_name])
        ]
        normal = np.mean(np.array(normals), axis=0)
        angle = get_angle_between_vectors(normal, opposite_gravity)
        if 0 <= angle <= smallest_angle:
            smallest_angle = angle
            supporting_surface = obj
    if supporting_surface is not None:
        print(f"found surface {supporting_surface.name}")
    return supporting_surface


def select_transportable_objects_from_contact_event(
    event: Union[CloseContactEvent, AgentContactEvent],
) -> List[Body]:
    """
    Select the objects that can be transported from the contact event.

    :param event: The contact event
    """
    contacted_objects = event.contact_points.get_objects_that_have_points()
    return select_transportable_objects(contacted_objects + [event.tracked_object])


def select_transportable_objects_from_loss_of_contact_event(
    event: LossOfCloseContactEvent,
) -> List[Body]:
    """
    Select the objects that can be transported from the loss of contact event.
    """
    return select_transportable_objects([event.tracked_object])


def select_transportable_objects(
    objects: List[Body], not_contained: bool = False
) -> List[Body]:
    """
    Select the objects that can be transported

    :param objects: A list of Object instances.
    """
    transportable_objects = [obj for obj in objects]
    # Here check if this: not issubclass(obj.ontology_concept, (Agent, Location, Supporter, Floor))]
    if not_contained:
        transportable_objects = [
            obj
            for obj in transportable_objects
            if not is_object_supported_by_container_body(obj)
        ]

    return transportable_objects


EventDetectorUnion = Union[
    NewObjectDetector,
    MotionDetector,
    TranslationDetector,
    RotationDetector,
    PlacingDetector,
]
TypeEventDetectorUnion = Union[
    Type[AtomicEventDetector],
    Type[MotionDetector],
    Type[TranslationDetector],
    Type[RotationDetector],
    Type[NewObjectDetector],
    Type[PlacingDetector],
]
