"""
The work items and states the conversions between DAOs and domain objects share.
"""

from __future__ import annotations

import abc
from collections import deque
from dataclasses import dataclass, field
from typing_extensions import (
    Dict,
    TYPE_CHECKING,
    Generic,
    Any,
    TypeAlias,
    TypeVar,
)

if TYPE_CHECKING:
    from krrood.ormatic.data_access_objects.dao import (
        DataAccessObject,
    )

DomainObject: TypeAlias = Any
"""
An object of a domain class that a DAO maps.
"""

InstanceDict = Dict[int, Any]  # Dictionary that maps object ids to objects
WorkItemT = TypeVar("WorkItemT", bound="DataAccessObjectWorkItem")


@dataclass(kw_only=True)
class DataAccessObjectWorkItem(abc.ABC):
    """
    Abstract base class for conversion work items.
    """

    dao_instance: DataAccessObject


@dataclass(kw_only=True)
class DataAccessObjectState(Generic[WorkItemT], abc.ABC):
    """
    Abstract base class for conversion states.
    """

    memo: InstanceDict = field(default_factory=dict)
    """
    Cache for converted objects to prevent duplicates and handle circular references.
    """

    work_items: deque[WorkItemT] = field(default_factory=deque)
    """
    Deque of work items to be processed.
    """

    @abc.abstractmethod
    def push_work_item(self, *args: Any, **kwargs: Any) -> None:
        """
        Add a new work item to the processing queue.

        :param args: Positional arguments for the work item.
        :param kwargs: Keyword arguments for the work item.
        """
        pass

    def has(self, *, source: Any) -> bool:
        """
        Check if the given source object has already been converted.

        :param source: The object to check.
        :return: True if already converted.
        """
        return id(source) in self.memo

    def get(self, *, source: Any) -> Any | None:
        """
        Get the converted object for the given source object.

        :param source: The source object.
        :return: The converted object if it exists.
        """
        return self.memo.get(id(source))

    def register(self, *, source: Any, target: Any) -> None:
        """
        Register a conversion result in the memoization store.

        :param source: The source object.
        :param target: The conversion result.
        """
        self.memo[id(source)] = target

    def pop(self, *, source: Any) -> Any | None:
        """
        Remove and return the conversion result for the given source object.

        :param source: The source object.
        :return: The conversion result if it existed.
        """
        return self.memo.pop(id(source), None)
