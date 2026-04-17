from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property

import semantic_digital_twin
from krrood.ormatic.data_access_objects.dao import DataAccessObject
from krrood.ormatic.data_access_objects.helper import to_dao
from semantic_digital_twin.orm.ormatic_interface import BodyDAO
from semantic_digital_twin.world_description.world_entity import Body
from typing_extensions import List, Optional, TYPE_CHECKING

from .object_tracker import ObjectTrackerFactory, ObjectTracker


@dataclass
class HasTrackedObjects:
    """
    A mixin class that provides the tracked object for the event.
    """
    _involved_objects: List[Body]

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
    tracked_object_frozen_cp: Optional[BodyDAO] = field(init=False, default=None, repr=False, hash=False)
    def __init__(self, tracked_object: Body):
        self.tracked_object = tracked_object

    def __post_init__(self):
        self.world_frozen_cp = self.tracked_object._world.__deepcopy__(
            memo={id(self.tracked_object._world): self.tracked_object._world}
        )



    @cached_property
    def object_tracker(self) -> ObjectTracker:
        return ObjectTrackerFactory.get_tracker(self.tracked_object)


@dataclass(kw_only=True, unsafe_hash=True)
class HasSecondaryTrackedObject:
    """
    A mixin class that provides the tracked objects for the event.
    """
    with_object: Optional[Body] = None
    with_object_frozen_cp: Optional[DataAccessObject[Body]] = field(init=False, default=None, repr=False, hash=False)

    def __post_init__(self):
        if self.with_object is not None:
            self.with_object_frozen_cp = to_dao(self.with_object)


    @cached_property
    def with_object_tracker(self) -> Optional[ObjectTracker]:
        return ObjectTrackerFactory.get_tracker(self.with_object) if self.with_object is not None else None


@dataclass(kw_only=True, unsafe_hash=True)
class HasPrimaryAndSecondaryTrackedObjects(HasPrimaryTrackedObject, HasSecondaryTrackedObject):
    """
    A mixin class that provides the tracked objects for the event.
    """

    def __post_init__(self):
        HasPrimaryTrackedObject.__post_init__(self)
        HasSecondaryTrackedObject.__post_init__(self)
