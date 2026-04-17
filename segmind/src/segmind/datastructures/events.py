import time
from abc import abstractmethod, ABC
from dataclasses import dataclass, field

from geometry_msgs.msg import PoseStamped

from typing_extensions import Optional, List, Union, Type

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
    contact_bodies: list[Body] = field(init=False, default_factory=list)
    latest_contact_bodies: list[Body] = field(init=False, default_factory=list)
    bounding_box: BoundingBox = field(init=False)
    pose: Pose = field(init=False)
    with_object_bounding_box: Optional[BoundingBox] = field(init=False, default=None)
    with_object_pose: Optional[PoseStamped] = field(init=False, default=None)

    def __init__(self,
                 of_object: Body,
                 contact_bodies: Optional[list[Body]] = None,
                 latest_contact_bodies: Optional[list[Body]] = None,
                 with_object: Optional[Body] = None,
                 timestamp: Optional[float] = None):
        EventWithTwoTrackedObjects.__init__(self,
                                            tracked_object=of_object,
                                            with_object=with_object,
                                            timestamp=timestamp if timestamp is not None else time.time())
        self.contact_bodies: list[Body] = contact_bodies
        self.latest_contact_bodies: list[Body] = latest_contact_bodies
        self.bounding_box: BoundingBox = BoundingBox.from_mesh(
            of_object.collision.combined_mesh,
            origin=of_object.global_pose.to_homogeneous_matrix()
        )
        self.pose: Pose = of_object.global_pose
        self.with_object_bounding_box: Optional[BoundingBox] = (
            BoundingBox.from_mesh(
                with_object.collision.combined_mesh,
                origin=with_object.global_pose.to_homogeneous_matrix()
            )
            if with_object is not None
            else None
        )
        self.with_object_pose: Optional[Pose] = with_object.global_pose if with_object is not None else None



@dataclass(init=False, unsafe_hash=True)
class ContactEvent(AbstractContactEvent):
    ...




@dataclass(init=False, unsafe_hash=True)
class LossOfContactEvent(AbstractContactEvent):
    ...


@dataclass(unsafe_hash=True)
class PickUpEvent(EventWithOneTrackedObject):
    ...


@dataclass(unsafe_hash=True)
class PlacingEvent(EventWithTwoTrackedObjects):
    ...



@dataclass(init=False, unsafe_hash=True)
class InsertionEvent(EventWithTwoTrackedObjects):
    inserted_into_objects: List[Body] = field(init=False, default_factory=list, repr=False, hash=False)
    inserted_into_objects_frozen_cp: List[BodyDAO] = field(init=False, default_factory=list, repr=False, hash=False)

    def __init__(self, inserted_object: Body,
                 inserted_into_objects: List[Body],
                 through_hole: Body,
                 timestamp: Optional[float] = None,
                 ):
        super().__init__(tracked_object=inserted_object,
                         timestamp=timestamp, with_object=through_hole)
        self.inserted_into_objects: List[Body] = inserted_into_objects
        self.inserted_into_objects_frozen_cp: List[BodyDAO] = [obj for obj in inserted_into_objects]

    @property
    def through_hole(self) -> Body:
        return self.with_object

    def __str__(self):
        with_object_name = " - " + f" - ".join([obj.name.name for obj in self.inserted_into_objects])
        return f"{self.__class__.__name__}: {self.tracked_object.name.name}{with_object_name} - {self.timestamp}"

@dataclass(unsafe_hash=True)
class ContainmentEvent(DefaultEventWithTwoTrackedObjects):
    ...

@dataclass(unsafe_hash=True)
class LossOfContainmentEvent(DefaultEventWithTwoTrackedObjects):
    ...

# Create a type that is the union of all event types
EventUnion = Union[
MotionEvent,
StopMotionEvent,
PickUpEvent,
PlacingEvent,
ContainmentEvent,
InsertionEvent]