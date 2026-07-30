from __future__ import annotations

from abc import abstractmethod, ABC
from dataclasses import dataclass, field
from datetime import datetime
from functools import cached_property

from geometry_msgs.msg import PoseStamped
from typing_extensions import Optional, List

from segmind.datastructures.object_tracker import ObjectEventTracker, ObjectTrackerFactory
from semantic_digital_twin.semantic_annotations.semantic_annotations import Aperture
from semantic_digital_twin.spatial_types.spatial_types import Pose
from semantic_digital_twin.world_description.geometry import BoundingBox
from semantic_digital_twin.world_description.world_entity import Body


@dataclass
class DetectionEvent(ABC):
    timestamp: datetime = field(default_factory=datetime.now)
    """
    The time at which the event occurred, defaults to current time.
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


@dataclass(kw_only=True)
class EventWithTrackedObjects(DetectionEvent, ABC):
    """
    An abstract event involving one or more tracked objects.

    Provides the primary :attr:`tracked_object` and an optional :attr:`with_object`,
    along with ORM frozen copies and per-object tracker access.
    """

    tracked_object: Body
    """The primary object involved in this event."""

    with_object: Optional[Body] = None
    """The secondary object involved in this event, if any."""

    @property
    def tracked_objects(self) -> List[Body]:
        """
        :return: the primary object, plus the secondary object when present.
        """
        return [self.tracked_object] if self.with_object is None else [self.tracked_object, self.with_object]

    @cached_property
    def object_tracker(self) -> ObjectEventTracker:
        """
        :return: the event tracker for :attr:`tracked_object`.
        """
        return ObjectTrackerFactory.get_tracker(self.tracked_object)

    @cached_property
    def with_object_tracker(self) -> Optional[ObjectEventTracker]:
        """
        :return: the event tracker for :attr:`with_object`, or ``None`` if absent.
        """
        return ObjectTrackerFactory.get_tracker(self.with_object) if self.with_object is not None else None

    def update_object_trackers_with_event(self, factory: ObjectTrackerFactory) -> None:
        """
        Register this event with the tracker of every involved object.

        :param factory: factory used to look up per-object trackers.
        """
        for obj in self.tracked_objects:
            factory.get_tracker(obj).add_event(self)

    def __str__(self) -> str:
        names = " - ".join(str(obj.name) for obj in self.tracked_objects)
        return f"{self.__class__.__name__}: {names} - {self.timestamp}"

    def __eq__(self, other) -> bool:
        return (other.__class__ == self.__class__
                and self.tracked_objects == other.tracked_objects
                and self.timestamp == other.timestamp)

    def __hash__(self) -> int:
        return hash((self.__class__, tuple(self.tracked_objects), self.timestamp))


@dataclass(unsafe_hash=True)
class SupportEvent(EventWithTrackedObjects):
    """
    The SupportEvent class is used to represent an event that involves an object that is supported by another object.
    """


@dataclass(unsafe_hash=True)
class LossOfSupportEvent(EventWithTrackedObjects):
    """
    The LossOfSupportEvent class is used to represent an event that involves an object that was supported by another
    object and then lost support.
    """


@dataclass(unsafe_hash=True)
class MotionEvent(EventWithTrackedObjects, ABC):
    """
    Used to represent an event that involves an object that was stationary and then moved or
    vice versa.
    """
    start_pose: Pose = field(default_factory=Pose)
    """
    The pose of the object at the start of the event.
    """
    current_pose: Pose = field(default_factory=Pose)
    """
    The pose of the object at the end of the event.
    """




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
class AbstractContactEvent(EventWithTrackedObjects, ABC):
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
class PickUpEvent(EventWithTrackedObjects):
    """
    Represents an event where an object is picked up by another object.
    """
    ...


@dataclass(unsafe_hash=True)
class PlacingEvent(EventWithTrackedObjects):
    """
    Represents an event where an object is placed on another object.
    """
    ...



@dataclass(unsafe_hash=True)
class InsertionEvent(EventWithTrackedObjects):
    """
    Represents an event where an object is inserted into another object.
    """

    inserted_into_objects: List[Body] = field(default_factory=list)
    """
    List of objects into which the object was inserted.
    """

    @property
    def through_hole(self) -> Aperture:
        return self.with_object.get_semantic_annotations_by_type(type_=Aperture)[0]


    def __str__(self) -> str:
        with_object_name = " - " + " - ".join([str(obj.name) for obj in self.inserted_into_objects])
        return f"{self.__class__.__name__}: {self.tracked_object.name}{with_object_name} - {self.timestamp}"

@dataclass(unsafe_hash=True)
class ContainmentEvent(EventWithTrackedObjects):
    """
    Represents an event where an object is contained in another object.
    """
    ...

@dataclass(unsafe_hash=True)
class LossOfContainmentEvent(EventWithTrackedObjects):
    """
    Represents an event where an object is no longer contained in another object.
    """
    ...

