import time
from abc import abstractmethod, ABC
from dataclasses import dataclass, field


from pycram.datastructures.partial_designator import PartialDesignator
from typing_extensions import Optional, List, Union, Type

from pycram.datastructures.pose import PoseStamped
from pycram.robot_plans import ActionDescription, PickUpActionDescription, PickUpAction, PlaceActionDescription, \
    PlaceAction
from segmind.datastructures.mixins import HasPrimaryTrackedObject, HasPrimaryAndSecondaryTrackedObjects
from segmind.datastructures.object_tracker import ObjectTrackerFactory
from semantic_digital_twin.orm.ormatic_interface import BodyDAO
from semantic_digital_twin.spatial_types.spatial_types import Pose
from semantic_digital_twin.world_description.geometry import BoundingBox, Color
from semantic_digital_twin.world_description.world_entity import Body, Agent


@dataclass
class Event(ABC):
    timestamp: float = field(default_factory=time.time)
    """
    The time at which the event occurred, defaults to current time.
    """
    detector_thread_id: Optional[str] = None
    """
    The id of the detector that detected the event.
    """

    @abstractmethod
    def __eq__(self, other):
        pass

    @abstractmethod
    def __hash__(self):
        pass

    def annotate(self, color: Optional[Color] = None) -> None:
        """
        Annotates the object with the given color.

        :param color: The color of the annotation text and/or object.
        :return: The TextAnnotation object that references the annotation text.
        """
        color = color or self.color
        self.set_color(color)

    @abstractmethod
    def set_color(self, color: Color):
        pass

    @property
    @abstractmethod
    def color(self) -> Color:
        pass

    @abstractmethod
    def __str__(self):
        pass

    def __repr__(self):
        return self.__str__()


@dataclass
class EventWithTrackedObjects(Event, ABC):
    """
    An abstract class that represents an event that involves one or more tracked objects.
    """

    @property
    @abstractmethod
    def tracked_objects(self) -> List[Body]:
        """
        The tracked objects involved in the event.
        """
        pass

    @property
    @abstractmethod
    def involved_bodies(self) -> List[Body]:
        """
        The bodies involved in the event.
        """
        pass

    @abstractmethod
    def update_object_trackers_with_event(self) -> None:
        """
        Update the object trackers of the involved objects with the event.
        """
        pass


@dataclass(kw_only=True)
class EventWithOneTrackedObject(EventWithTrackedObjects, HasPrimaryTrackedObject, ABC):
    """
    An abstract class that represents an event that involves one tracked object.
    """

    @property
    def tracked_objects(self) -> List[Body]:
        return [self.tracked_object]

    def update_object_trackers_with_event(self) -> None:
        ObjectTrackerFactory.get_tracker(self.tracked_object).add_event(self)

    def __str__(self):
        return f"{self.__class__.__name__}: {self.tracked_object.name} - {self.timestamp}"

    def __eq__(self, other):
        return (other.__class__ == self.__class__
                and self.tracked_object == other
                and round(self.timestamp, 1) == round(other.timestamp, 1))

    def __hash__(self):
        return hash((self.__class__, self.tracked_object, round(self.timestamp, 1)))


@dataclass(kw_only=True)
class EventWithTwoTrackedObjects(EventWithTrackedObjects, HasPrimaryAndSecondaryTrackedObjects, ABC):
    """
    An abstract class that represents an event that involves two tracked objects.
    """

    @property
    def tracked_objects(self) -> List[Body]:
        return [self.tracked_object, self.with_object] if self.with_object is not None else [self.tracked_object]

    def update_object_trackers_with_event(self) -> None:
        ObjectTrackerFactory.get_tracker(self.tracked_object).add_event(self)
        if self.with_object is not None:
            ObjectTrackerFactory.get_tracker(self.with_object).add_event(self)

    def __str__(self):
        with_object_name = f" - {self.with_object.name}" if self.with_object is not None else ""
        return f"{self.__class__.__name__}: {self.tracked_object.name}{with_object_name} - {self.timestamp}"

    def __eq__(self, other):
        return (other.__class__ == self.__class__
                and self.tracked_object == other.tracked_object
                and self.with_object == other.with_object
                and round(self.timestamp, 1) == round(other.timestamp, 1))

    def __hash__(self):
        hash_tuple = (self.__class__, self.tracked_object, round(self.timestamp, 1))
        if self.with_object is not None:
            hash_tuple += (self.with_object,)
        return hash(hash_tuple)


@dataclass(unsafe_hash=True)
class DefaultEventWithTwoTrackedObjects(EventWithTwoTrackedObjects):
    """
    A default implementation of EventWithTwoTrackedObjects that does not require a with_object.
    This is useful for events that only involve one tracked object.
    """

    @property
    def involved_bodies(self) -> List[Body]:
        return self.tracked_objects

    def set_color(self, color: Color):
        ...

    @property
    def color(self) -> Color:
        return self.tracked_object.color


@dataclass(unsafe_hash=True)
class NewObjectEvent(EventWithOneTrackedObject):
    """
    The NewObjectEvent class is used to represent an event that involves the addition of a new object to the world.
    """

    @property
    def involved_bodies(self) -> List[Body]:
        return self.tracked_objects

    def set_color(self, color: Color):
        ...

    @property
    def color(self) -> Color:
        return self.tracked_object.color


@dataclass(unsafe_hash=True)
class SupportEvent(DefaultEventWithTwoTrackedObjects):
    """
    The SupportEvent class is used to represent an event that involves an object that is supported by another object.
    """
    ...


@dataclass(unsafe_hash=True)
class LossOfSupportEvent(DefaultEventWithTwoTrackedObjects):
    """
    The LossOfSupportEvent class is used to represent an event that involves an object that was supported by another
    object and then lost support.
    """
    ...


@dataclass(init=False, unsafe_hash=True)
class MotionEvent(EventWithOneTrackedObject, ABC):
    """
    The MotionEvent class is used to represent an event that involves an object that was stationary and then moved or
    vice versa.
    """
    start_pose: Pose = field(init=False)
    current_pose: Pose = field(init=False)

    def __init__(self, tracked_object: Body, start_pose: Pose, current_pose: Pose,
                 timestamp: Optional[float] = None):
        EventWithOneTrackedObject.__init__(self, tracked_object=tracked_object,
                                           timestamp=timestamp if timestamp is not None else time.time())
        self.start_pose: Pose = start_pose
        self.current_pose: Pose = current_pose

    @property
    def involved_bodies(self) -> List[Body]:
        return self.tracked_objects

    def set_color(self, color: Color):
        self.tracked_object.set_color(color)


@dataclass(init=False, unsafe_hash=True)
class TranslationEvent(MotionEvent):
    @property
    def color(self) -> Color:
        return Color(0, 1, 1, 1)


@dataclass(init=False, unsafe_hash=True)
class RotationEvent(MotionEvent):
    @property
    def color(self) -> Color:
        return Color(0, 1, 1, 1)


@dataclass(init=False, unsafe_hash=True)
class StopMotionEvent(MotionEvent):
    @property
    def color(self) -> Color:
        return Color(1, 0, 0, 1)


@dataclass(init=False, unsafe_hash=True)
class StopTranslationEvent(StopMotionEvent):
    ...


@dataclass(init=False, unsafe_hash=True)
class StopRotationEvent(StopMotionEvent):
    ...


@dataclass(init=False, unsafe_hash=True)
class AbstractContactEvent(EventWithTwoTrackedObjects, ABC):
    close_bodies: list[Body] = field(init=False, default_factory=list[Body])
    latest_close_bodies: list[Body] = field(init=False, default_factory=list[Body])
    bounding_box: BoundingBox = field(init=False)
    pose: PoseStamped = field(init=False)
    with_object_bounding_box: Optional[BoundingBox] = field(init=False, default=None)
    with_object_pose: Optional[PoseStamped] = field(init=False, default=None)

    def __init__(self,
                 of_object: Body,
                 close_bodies: Optional[list[Body]] = None,
                 latest_close_bodies: Optional[list[Body]] = None,
                 with_object: Optional[Body] = None,
                 timestamp: Optional[float] = None):
        EventWithTwoTrackedObjects.__init__(self,
                                            tracked_object=of_object,
                                            with_object=with_object,
                                            timestamp=timestamp if timestamp is not None else time.time())
        self.close_bodies: list[Body] = close_bodies
        self.latest_close_bodies: list[Body] = latest_close_bodies
        self.bounding_box: BoundingBox = BoundingBox.from_mesh(
            of_object.collision.combined_mesh,
            origin=of_object.global_pose
        )
        self.pose: Pose = of_object.global_pose.to_pose()
        self.with_object_bounding_box: Optional[BoundingBox] = (
            BoundingBox.from_mesh(
                with_object.collision.combined_mesh,
                origin=with_object.global_pose.from_xyz_rpy(),
            )
            if with_object is not None
            else None
        )
        self.with_object_pose: Optional[Pose] = with_object.global_pose.to_pose() if with_object is not None else None

    @property
    def involved_bodies(self) -> List[Body]:
        return list(set(self.links))

    def set_color(self, color: Color):
        self.main_link.color = color
        [link.set_color(color) for link in self.links]

    @property
    def object_names(self) -> List[str]:
        return [obj.name for obj in self.objects]

    @property
    def link_names(self) -> List[str]:
        return [link.name for link in self.links]

    @property
    @abstractmethod
    def main_link(self) -> Body:
        pass

    @property
    @abstractmethod
    def links(self) -> List[Body]:
        pass

    @property
    @abstractmethod
    def objects(self) -> List[Body]:
        pass


@dataclass(init=False, unsafe_hash=True)
class CloseContactEvent(AbstractContactEvent):

    @property
    def color(self) -> Color:
        return Color(0, 0, 1, 1)

    @property
    def objects(self):
        return self.close_bodies

    @property
    def main_link(self):
        if len(self.close_bodies) > 0:
            return self.close_bodies[0]
        else:
            print(f"No contact points found for {self.tracked_object.name} in {self.__class__.__name__}")

    @property
    def links(self):
        return self.close_bodies


@dataclass(init=False, unsafe_hash=True)
class ContactEvent(CloseContactEvent):
    ...


@dataclass(init=False, unsafe_hash=True)
class LossOfCloseContactEvent(AbstractContactEvent):

    @property
    def latest_objects_that_got_removed(self):
        return self.get_objects_that_got_removed(self.latest_close_bodies)

    def get_objects_that_got_removed(self, close_bodies: list[Body]):
        return self.close_bodies.get_objects_that_got_removed(close_bodies)

    @property
    def color(self) -> Color:
        return Color(1, 0, 0, 1)

    @property
    def main_link(self) -> Body:
        return self.latest_close_bodies[0]

    @property
    def links(self) -> List[Body]:
        return self.close_bodies.get_bodies_that_got_removed(self.latest_close_bodies)

    @property
    def objects(self):
        return self.close_bodies.get_objects_that_got_removed(self.close_bodies)


@dataclass(init=False, unsafe_hash=True)
class LossOfContactEvent(LossOfCloseContactEvent):
    ...


@dataclass(init=False, unsafe_hash=True)
class AbstractAgentContact(AbstractContactEvent, ABC):
    @property
    def agent(self) -> Body:
        return self.tracked_object

    @property
    def agent_link(self) -> Body:
        return self.main_link

    def with_object_contact_link(self) -> Optional[Body]:
        if self.with_object is not None:
            return [link for link in self.links if link.parent_connection.parent == self.with_object][0]
        else:
            return None

    @property
    @abstractmethod
    def object_link(self) -> Body:
        pass


@dataclass(init=False, unsafe_hash=True)
class AgentContactEvent(CloseContactEvent, AbstractAgentContact):

    @property
    def object_link(self) -> Body:
        if self.with_object is not None:
            return self.with_object_contact_link()
        else:
            return self.contact_bodies[0].parent_connection.child


@dataclass(init=False, unsafe_hash=True)
class AgentInterferenceEvent(ContactEvent, AgentContactEvent):
    ...


@dataclass(init=False, unsafe_hash=True)
class AgentLossOfContactEvent(LossOfCloseContactEvent, AbstractAgentContact):

    @property
    def object_link(self) -> Body:
        if self.with_object is not None:
            return self.with_object_contact_link()
        else:
            return self.latest_close_bodies[0].link_b


@dataclass(init=False, unsafe_hash=True)
class AgentLossOfInterferenceEvent(LossOfContactEvent, AgentLossOfContactEvent):
    ...


@dataclass(kw_only=True)
class AbstractAgentObjectInteractionEvent(EventWithTwoTrackedObjects, ABC):
    agent: Optional[Agent] = None
    timestamp: Optional[float] = None
    end_timestamp: Optional[float] = None
    agent_frozen_cp: Optional[BodyDAO] = field(init=False, default=None, repr=False, hash=False)

    def __post_init__(self):
        EventWithTwoTrackedObjects.__post_init__(self)
        if self.agent is not None:
            self.agent_frozen_cp = self.agent.frozen_copy()

    @property
    def involved_bodies(self) -> List[Body]:
        return self.tracked_objects

    @property
    def agent_state(self) -> Optional[Agent]:
        if self.agent is None:
            return None
        return self.agent

    def __eq__(self, other):
        if self.end_timestamp is None:
            return super().__eq__(other)
        return (super().__eq__(other)
                and round(self.end_timestamp, 1) == round(other.end_timestamp, 1))

    @property
    def hash_tuple(self):
        hash_tuple = (self.__class__, self.agent, self.tracked_object, round(self.timestamp, 1))
        if self.end_timestamp is not None:
            hash_tuple += (round(self.end_timestamp, 1),)
        return hash_tuple

    def __hash__(self):
        return hash(self.hash_tuple)

    def duration(self):
        if self.end_timestamp is None:
            return None
        return self.end_timestamp - self.timestamp

    def set_color(self, color: Color):
        if self.agent is not None:
            self.agent.set_color(color)
        self.tracked_object.set_color(color)

    @property
    @abstractmethod
    def action_description(self) -> PartialDesignator[ActionDescription]:
        pass

    @property
    @abstractmethod
    def action_type(self) -> Type[ActionDescription]:
        pass


@dataclass(unsafe_hash=True)
class PickUpEvent(AbstractAgentObjectInteractionEvent):

    @property
    def color(self) -> Color:
        return Color(0, 1, 0, 1)

    def action_description(self) -> PickUpActionDescription:
        return PickUpActionDescription(self.tracked_object)

    @property
    def action_type(self) -> Type[PickUpAction]:
        return PickUpAction


@dataclass(unsafe_hash=True)
class PlacingEvent(AbstractAgentObjectInteractionEvent):
    placement_pose: Optional[PoseStamped] = None

    @property
    def color(self) -> Color:
        return Color(1, 0, 1, 1)

    def action_description(self, pose: Optional[PoseStamped] = None) -> PlaceActionDescription:
        if pose is None:
            pose = self.tracked_object.pose
        return PlaceActionDescription(self.tracked_object, pose)

    @property
    def action_type(self) -> Type[PlaceAction]:
        return PlaceAction


@dataclass(init=False, unsafe_hash=True)
class InsertionEvent(AbstractAgentObjectInteractionEvent):
    inserted_into_objects: List[Body] = field(init=False, default_factory=list, repr=False, hash=False)
    inserted_into_objects_frozen_cp: List[BodyDAO] = field(init=False, default_factory=list, repr=False, hash=False)

    def __init__(self, inserted_object: Body,
                 inserted_into_objects: List[Body],
                 through_hole: Body,
                 agent: Optional[Body] = None,
                 timestamp: Optional[float] = None,
                 end_timestamp: Optional[float] = None):
        super().__init__(tracked_object=inserted_object, agent=agent,
                         timestamp=timestamp, end_timestamp=end_timestamp, with_object=through_hole)
        self.inserted_into_objects: List[Body] = inserted_into_objects
        self.inserted_into_objects_frozen_cp: List[BodyDAO] = [obj for obj in inserted_into_objects]

    @property
    def through_hole(self) -> Body:
        return self.with_object

    @property
    def action_type(self) -> Type[PlaceAction]:
        return PlaceAction

    def action_description(self) -> PlaceActionDescription:
        return PlaceActionDescription(self.tracked_object, self.through_hole.pose, insert=True)

    def hash_tuple(self):
        hash_tuple = (*super().hash_tuple, *(obj.name for obj in self.inserted_into_objects))
        return hash_tuple

    def __str__(self):
        with_object_name = " - " + f" - ".join([obj.name.name for obj in self.inserted_into_objects])
        return f"{self.__class__.__name__}: {self.tracked_object.name.name}{with_object_name} - {self.timestamp}"

    @property
    def color(self) -> Color:
        return Color(1, 0, 1, 1)


@dataclass(unsafe_hash=True)
class ContainmentEvent(DefaultEventWithTwoTrackedObjects):
    ...

@dataclass(unsafe_hash=True)
class LossOfContainmentEvent(DefaultEventWithTwoTrackedObjects):
    ...

# Create a type that is the union of all event types
EventUnion = Union[NewObjectEvent,
MotionEvent,
StopMotionEvent,
CloseContactEvent,
LossOfCloseContactEvent,
AgentContactEvent,
AgentLossOfContactEvent,
PickUpEvent,
PlacingEvent,
ContainmentEvent,
InsertionEvent]