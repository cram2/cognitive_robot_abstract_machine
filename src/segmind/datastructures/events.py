import time
from abc import abstractmethod, ABC
from dataclasses import dataclass, field

from pycram.datastructures.dataclasses import Color
from pycram.datastructures.enums import Arms
from pycram.datastructures.partial_designator import PartialDesignator
from pycram.datastructures.pose import PoseStamped
from pycram.robot_plans import ActionDescription, PlaceAction, PickUpAction, PlaceActionDescription, \
    PickUpActionDescription
from semantic_digital_twin.collision_checking.collision_detector import Collision
from semantic_digital_twin.collision_checking.trimesh_collision_detector import TrimeshCollisionDetector
from semantic_digital_twin.spatial_types.spatial_types import Pose
from semantic_digital_twin.world_description.geometry import BoundingBox
from semantic_digital_twin.world_description.world_entity import Body, Agent
from typing_extensions import Optional, List, Union, Type

from segmind.datastructures.mixins import HasPrimaryTrackedObject, HasPrimaryAndSecondaryTrackedObjects
from segmind.datastructures.object_tracker import ObjectTrackerFactory


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
    start_pose: PoseStamped = field(init=False)
    current_pose: PoseStamped = field(init=False)

    def __init__(self, tracked_object: Body, start_pose: PoseStamped, current_pose: PoseStamped,
                 timestamp: Optional[float] = None):
        EventWithOneTrackedObject.__init__(self, tracked_object=tracked_object,
                                           timestamp=timestamp if timestamp is not None else time.time())
        self.start_pose: PoseStamped = start_pose
        self.current_pose: PoseStamped = current_pose

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
    contact_points: Collision = field(init=False, default_factory=Collision)
    latest_contact_points: Collision = field(init=False, default_factory=Collision)
    bounding_box: BoundingBox = field(init=False)
    pose: PoseStamped = field(init=False)
    with_object_bounding_box: Optional[BoundingBox] = field(init=False, default=None)
    with_object_pose: Optional[PoseStamped] = field(init=False, default=None)

    def __init__(self,
                 contact_points: Collision,
                 of_object: Body,
                 latest_contact_points: Optional[Collision] = None,
                 with_object: Optional[Body] = None,
                 timestamp: Optional[float] = None):
        EventWithTwoTrackedObjects.__init__(self,
                                            tracked_object=of_object,
                                            with_object=with_object,
                                            timestamp=timestamp if timestamp is not None else time.time())
        self.contact_points: TrimeshCollisionDetector = contact_points
        self.latest_contact_points: TrimeshCollisionDetector = latest_contact_points
        self.bounding_box: BoundingBox = BoundingBox.from_mesh(
            of_object.collision.combined_mesh,
            origin=of_object.global_pose
        )
        self.pose: Pose = of_object.global_pose.to_pose()
        self.with_object_bounding_box: Optional[BoundingBox] = with_object.get_axis_aligned_bounding_box() \
            if with_object is not None else None
        self.with_object_pose: Optional[PoseStamped] = with_object.pose if with_object is not None else None

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
class ContactEvent(AbstractContactEvent):

    @property
    def color(self) -> Color:
        return Color(0, 0, 1, 1)

    @property
    def objects(self):
        return self.contact_points.get_objects_that_have_points()

    @property
    def main_link(self):
        if len(self.contact_points) > 0:
            return self.contact_points[0].link_a
        else:
            print(f"No contact points found for {self.tracked_object.name} in {self.__class__.__name__}")

    @property
    def links(self):
        return self.contact_points.get_all_bodies()


@dataclass(init=False, unsafe_hash=True)
class InterferenceEvent(ContactEvent):
    ...


@dataclass(init=False, unsafe_hash=True)
class LossOfContactEvent(AbstractContactEvent):

    @property
    def latest_objects_that_got_removed(self):
        return self.get_objects_that_got_removed(self.latest_contact_points)

    def get_objects_that_got_removed(self, contact_points: TrimeshCollisionDetector):
        return self.contact_points.get_objects_that_got_removed(contact_points)

    @property
    def color(self) -> Color:
        return Color(1, 0, 0, 1)

    @property
    def main_link(self) -> Body:
        return self.latest_contact_points[0].link_a

    @property
    def links(self) -> List[Body]:
        return self.contact_points.get_bodies_that_got_removed(self.latest_contact_points)

    @property
    def objects(self):
        return self.contact_points.get_objects_that_got_removed(self.latest_contact_points)


@dataclass(init=False, unsafe_hash=True)
class LossOfInterferenceEvent(LossOfContactEvent):
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
            return [link for link in self.links if link.parent_entity == self.with_object][0]
        else:
            return None

    @property
    @abstractmethod
    def object_link(self) -> Body:
        pass


@dataclass(init=False, unsafe_hash=True)
class AgentContactEvent(ContactEvent, AbstractAgentContact):

    @property
    def object_link(self) -> Body:
        if self.with_object is not None:
            return self.with_object_contact_link()
        else:
            return self.contact_points[0].link_b


@dataclass(init=False, unsafe_hash=True)
class AgentInterferenceEvent(InterferenceEvent, AgentContactEvent):
    ...


@dataclass(init=False, unsafe_hash=True)
class AgentLossOfContactEvent(LossOfContactEvent, AbstractAgentContact):

    @property
    def object_link(self) -> Body:
        if self.with_object is not None:
            return self.with_object_contact_link()
        else:
            return self.latest_contact_points[0].link_b


@dataclass(init=False, unsafe_hash=True)
class AgentLossOfInterferenceEvent(LossOfInterferenceEvent, AgentLossOfContactEvent):
    ...


@dataclass(kw_only=True)
class AbstractAgentObjectInteractionEvent(EventWithTwoTrackedObjects, ABC):
    agent: Optional[Body] = None
    timestamp: Optional[float] = None
    end_timestamp: Optional[float] = None
    agent_frozen_cp: Optional[Body] = field(init=False, default=None, repr=False, hash=False)

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
        return self.agent.state

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

    def action_description(self) -> Type[PickUpActionDescription]:
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

    def action_description(self, pose: Optional[PoseStamped] = None, arm: Optional[Arms] = None) -> Type[PlaceActionDescription]:
        if pose is None:
            pose = self.tracked_object.pose
        return PlaceActionDescription(self.tracked_object, pose, arm)

    @property
    def action_type(self) -> Type[PlaceAction]:
        return PlaceAction


@dataclass(init=False, unsafe_hash=True)
class InsertionEvent(AbstractAgentObjectInteractionEvent):
    inserted_into_objects: List[Body] = field(init=False, default_factory=list, repr=False, hash=False)
    inserted_into_objects_frozen_cp: List[Body] = field(init=False, default_factory=list, repr=False, hash=False)

    def __init__(self, inserted_object: Body,
                 inserted_into_objects: List[Body],
                 through_hole: Body,
                 agent: Optional[Body] = None,
                 timestamp: Optional[float] = None,
                 end_timestamp: Optional[float] = None):
        super().__init__(tracked_object=inserted_object, agent=agent,
                         timestamp=timestamp, end_timestamp=end_timestamp, with_object=through_hole)
        self.inserted_into_objects: List[Body] = inserted_into_objects
        self.inserted_into_objects_frozen_cp: List[Body] = [obj.frozen_copy() for obj in inserted_into_objects]

    @property
    def through_hole(self) -> Body:
        return self.with_object

    @property
    def action_type(self) -> Type[PlaceAction]:
        return PlaceAction

    def action_description(self) -> Type[PlaceActionDescription]:
        return PlaceActionDescription(self.tracked_object, self.through_hole.global_pose)

    def hash_tuple(self):
        hash_tuple = (*super().hash_tuple, *(obj.name for obj in self.inserted_into_objects))
        return hash_tuple

    def __str__(self):
        with_object_name = " - " + f" - ".join([obj.name for obj in self.inserted_into_objects])
        return f"{self.__class__.__name__}: {self.tracked_object.name}{with_object_name} - {self.timestamp}"

    @property
    def color(self) -> Color:
        return Color(1, 0, 1, 1)


@dataclass(unsafe_hash=True)
class ContainmentEvent(DefaultEventWithTwoTrackedObjects):
    ...


# Create a type that is the union of all event types
EventUnion = Union[NewObjectEvent,
MotionEvent,
StopMotionEvent,
ContactEvent,
LossOfContactEvent,
AgentContactEvent,
AgentLossOfContactEvent,
PickUpEvent,
PlacingEvent,
ContainmentEvent,
InsertionEvent]
