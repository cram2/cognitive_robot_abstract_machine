from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property

from krrood.ormatic.data_access_objects.dao import DataAccessObject
from krrood.ormatic.data_access_objects.helper import to_dao
from segmind.datastructures.object_tracker import ObjectEventTracker, ObjectTrackerFactory
from semantic_digital_twin.orm.ormatic_interface import BodyDAO
from semantic_digital_twin.world_description.world_entity import Body
from typing_extensions import List, Optional

from test.krrood_test.dataset.ormatic_interface import WorldDAO


# Needs some refactoring, maybe merging with the semDT mixins at some point.

@dataclass
class HasTrackedObjects:
    """
    A mixin class that provides the tracked object for the event.
    """

    _involved_objects: List[Body]
    """
    List of objects involved in the event.
    """

    @property
    def involved_objects(self) -> List[Body]:
        """
        The objects involved in the event.
        """
        return self._involved_objects


@dataclass(kw_only=True, unsafe_hash=True)
class HasPrimaryTrackedObject:
    """
    A mixin class that provides the tracked object for the event.
    """
    tracked_object: Body
    """
    The tracked object.
    """

    tracked_object_frozen_copy: Optional[BodyDAO] = field(init=False, default=None, repr=False, hash=False)
    """
    The tracked object as a Data Access Object, to be used by ORMatic and the NEEMInterface.
    """

    world_frozen_copy: Optional[WorldDAO] = field(init=False, default=None, repr=False, hash=False)
    """
    The world as a Data Access Object, to be used by ORMatic and the NEEMInterface.
    """

    def __post_init__(self):
        self.world_frozen_cp = self.tracked_object._world.__deepcopy__(
            memo={id(self.tracked_object._world): self.tracked_object._world}
        )

    @cached_property
    def object_tracker(self) -> ObjectEventTracker:
        return ObjectTrackerFactory.get_tracker(self.tracked_object)


@dataclass(kw_only=True, unsafe_hash=True)
class HasSecondaryTrackedObject:
    """
    A mixin class that provides the tracked objects for the event.
    """

    with_object: Optional[Body] = None
    """
    The secondary tracked object.
    """

    with_object_frozen_cp: Optional[DataAccessObject[Body]] = field(init=False, default=None, repr=False, hash=False)
    """
    The secondary tracked object as a Data Access Object.
    """


    def __post_init__(self):
        if self.with_object is not None:
            self.with_object_frozen_cp = to_dao(self.with_object)


    @cached_property
    def with_object_tracker(self) -> Optional[ObjectEventTracker]:
        return ObjectTrackerFactory.get_tracker(self.with_object) if self.with_object is not None else None


@dataclass(kw_only=True, unsafe_hash=True)
class HasPrimaryAndSecondaryTrackedObjects(HasPrimaryTrackedObject, HasSecondaryTrackedObject):
    """
    A mixin class that provides the tracked objects for the event.
    """

    def __post_init__(self):
        HasPrimaryTrackedObject.__post_init__(self)
        HasSecondaryTrackedObject.__post_init__(self)
