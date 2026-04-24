import time
from abc import abstractmethod, ABC
from dataclasses import dataclass, field
from datetime import datetime

from geometry_msgs.msg import PoseStamped

from typing_extensions import Optional, List, Union, Type

from segmind.datastructures.mixins import HasPrimaryTrackedObject, HasPrimaryAndSecondaryTrackedObjects
from segmind.datastructures.object_tracker import ObjectTrackerFactory
from semantic_digital_twin.orm.ormatic_interface import BodyDAO
from semantic_digital_twin.spatial_types.spatial_types import Pose
from semantic_digital_twin.world_description.geometry import BoundingBox, Color
from semantic_digital_twin.world_description.world_entity import Body, Agent


@dataclass
class DetectionEvent(ABC):
    timestamp: datetime = field(default_factory=datetime.now().timestamp)
    """
    The time at which the event occurred, defaults to current time.
    """
    detector_thread_id: Optional[int] = field(default=None)
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
class EventWithTrackedObjects(DetectionEvent, ABC):
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
    Used to represent an event that involves an object that was stationary and then moved or
    vice versa.
    """
    start_pose: Pose = field(default_factory=Pose)
    current_pose: Pose = field(default_factory=Pose)

    def __init__(self, tracked_object: Body, start_pose: Pose, current_pose: Pose,
                 timestamp: Optional[float] = None):
        EventWithOneTrackedObject.__init__(self, tracked_object=tracked_object,
                                           timestamp=timestamp if timestamp is not None else time.time())
        self.start_pose: Pose = start_pose
        self.current_pose: Pose = current_pose


@dataclass(init=False, unsafe_hash=True)
class TranslationEvent(MotionEvent):
    """
    Represents an event where an object moves from one location to another.
    """
    ...


@dataclass(init=False, unsafe_hash=True)
class RotationEvent(MotionEvent):
    """
    Represents an event where an object rotates around a center point.
    """
    ...


@dataclass(init=False, unsafe_hash=True)
class StopTranslationEvent(MotionEvent):
    """
    Represents an event where an object stops moving.
    """
    ...


@dataclass(init=False, unsafe_hash=True)
class StopRotationEvent(MotionEvent):
    """
    Represents an event where an object stops rotating.
    """
    ...


@dataclass(unsafe_hash=True)
class AbstractContactEvent(EventWithTwoTrackedObjects, ABC):
    """
    Represents an event where two objects are in contact with each other.
    """

    contact_bodies: list[Body] = field(init=False, default_factory=list)
    """
    The bodies that are in contact with each other.
    """

    latest_contact_bodies: list[Body] = field(init=False, default_factory=list)
    """
    The bodies that were in contact with each other in the previous time step.
    """

    bounding_box: BoundingBox = field(init=False)
    """
    Bounding box of the object.
    """

    pose: Pose = field(init=False)
    """
    Pose of the object.
    """

    with_object_bounding_box: Optional[BoundingBox] = field(init=False, default=None)
    """
    Bounding box of the second object in contact.
    """

    with_object_pose: Optional[PoseStamped] = field(init=False, default=None)
    """
    Pose of the second object in contact.
    """

    def __post_init__(self):
        super().__post_init__()

        self.bounding_box = BoundingBox.from_mesh(
            self.tracked_object.collision.combined_mesh,
            origin=self.tracked_object.global_pose.to_homogeneous_matrix()
        )
        self.pose = self.tracked_object.global_pose

        if self.with_object is not None:
            self.with_object_bounding_box = BoundingBox.from_mesh(
                self.with_object.collision.combined_mesh,
                origin=self.with_object.global_pose.to_homogeneous_matrix()
            )
            self.with_object_pose = self.with_object.global_pose


@dataclass(init=False, unsafe_hash=True)
class ContactEvent(AbstractContactEvent):
    """
    Represents an event where two objects are in contact with each other.
    """
    ...




@dataclass(init=False, unsafe_hash=True)
class LossOfContactEvent(AbstractContactEvent):
    """
    Represents an event where two objects are no longer in contact with each other.
    """
    ...


@dataclass(unsafe_hash=True)
class PickUpEvent(EventWithOneTrackedObject):
    """
    Represents an event where an object is picked up by another object.
    """
    ...


@dataclass(unsafe_hash=True)
class PlacingEvent(EventWithTwoTrackedObjects):
    """
    Represents an event where an object is placed on another object.
    """
    ...



@dataclass(init=False, unsafe_hash=True)
class InsertionEvent(EventWithTwoTrackedObjects):
    """
    Represents an event where an object is inserted into another object.
    """

    inserted_into_objects: List[Body] = field(init=False, default_factory=list, repr=False, hash=False)
    """
    List of objects into which the object was inserted.
    """

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
    """
    Represents an event where an object is contained in another object.
    """
    ...

@dataclass(unsafe_hash=True)
class LossOfContainmentEvent(DefaultEventWithTwoTrackedObjects):
    """
    Represents an event where an object is no longer contained in another object.
    """
    ...

