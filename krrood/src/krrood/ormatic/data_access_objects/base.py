from __future__ import annotations

import abc
from collections import deque
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Generic, Any, Optional, Type, _GenericAlias, TypeVar
from typing_extensions import get_origin, get_args, Dict, TYPE_CHECKING

from krrood.ormatic.exceptions import NoGenericError

if TYPE_CHECKING:
    from krrood.ormatic.data_access_objects.dao import (
        DataAccessObject,
    )

InstanceDict = Dict[int, Any]  # Dictionary that maps object ids to objects
WorkItemT = TypeVar("WorkItemT", bound="DataAccessObjectWorkItem")


@dataclass
class DataAccessObjectWorkItem(abc.ABC):
    """
    Abstract base class for conversion work items.
    """

    dao_instance: DataAccessObject


@dataclass
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

    def has(self, source: Any) -> bool:
        """
        Check if the given source object has already been converted.

        :param source: The object to check.
        :return: True if already converted.
        """
        return id(source) in self.memo

    def get(self, source: Any) -> Optional[Any]:
        """
        Get the converted object for the given source object.

        :param source: The source object.
        :return: The converted object if it exists.
        """
        return self.memo.get(id(source))

    def register(self, source: Any, target: Any) -> None:
        """
        Register a conversion result in the memoization store.

        :param source: The source object.
        :param target: The conversion result.
        """
        self.memo[id(source)] = target

    def pop(self, source: Any) -> Optional[Any]:
        """
        Remove and return the conversion result for the given source object.

        :param source: The source object.
        :return: The conversion result if it existed.
        """
        return self.memo.pop(id(source), None)


T = TypeVar("T")


class HasGeneric(Generic[T]):
    """
    Base class for classes that carry a generic type argument.
    """

    @classmethod
    @lru_cache(maxsize=None)
    def original_class(cls) -> T:
        """
        Get the concrete generic argument.

        :return: The generic type argument.
        :raises NoGenericError: If no generic argument is found.
        """
        tp = cls._dao_like_argument()
        if tp is None:
            raise NoGenericError(cls)
        return tp

    @classmethod
    @lru_cache(maxsize=None)
    def constructable_original_class(cls) -> T:
        """
        Return the constructable original class. Use this for object allocation in from_dao cycles, as Generic Aliases
        cannot be constructed directly.
        """
        original_class = cls.original_class()
        if type(original_class) is _GenericAlias:
            return get_origin(original_class)
        else:
            return original_class

    @classmethod
    def _dao_like_argument(cls) -> Optional[Type]:
        """
        Extract the generic argument from the class hierarchy.

        :return: The generic type or None.
        """
        # filter for instances of generic aliases in the superclasses
        for base in filter(
            lambda x: isinstance(x, _GenericAlias),
            cls.__orig_bases__,
        ):
            return get_args(base)[0]

        # No acceptable base found
        return None
