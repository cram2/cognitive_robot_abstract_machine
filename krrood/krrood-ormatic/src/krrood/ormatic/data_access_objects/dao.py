"""
The data access object that maps a class to its database rows and back.
"""

from __future__ import annotations

import threading
from functools import lru_cache

import sqlalchemy.inspection
import sqlalchemy.orm
from typing_extensions import Any, TypeVar

from krrood.ormatic.data_access_objects.alternative_mappings import AlternativeMapping
from krrood.ormatic.data_access_objects.from_dao_conversion import (
    FromDataAccessObjectConversion,
)
from krrood.ormatic.data_access_objects.helper import (
    clear_dao_lookup_caches,
    get_data_access_object_class,
)
from krrood.ormatic.data_access_objects.stored_object_row import StoredObjectRowHolder
from krrood.ormatic.data_access_objects.to_dao_conversion import (
    ToDataAccessObjectConversion,
)
from krrood.ormatic.utils import is_data_column
from krrood.patterns.classproperty import classproperty
from krrood.patterns.exceptions import NoGenericError
from krrood.patterns.has_generic import HasGeneric

_repr_thread_local = threading.local()

T = TypeVar("T")


# %% data access objects


class DataAccessObject(
    ToDataAccessObjectConversion,
    FromDataAccessObjectConversion,
    StoredObjectRowHolder,
    HasGeneric[T],
):
    """
    Base class for Data Access Objects (DAOs), which convert domain object graphs, with
    inheritance, circular references and :class:`AlternativeMapping` objects, to their
    database rows and back; see :class:`ToDataAccessObjectConversion` and
    :class:`FromDataAccessObjectConversion`.
    """

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """
        Forget the failed lookups of DAO classes, which the new class may answer.
        """
        super().__init_subclass__(**kwargs)
        # New DAO classes invalidate previously failed (None) lookups.
        clear_dao_lookup_caches()

    @classmethod
    def _find_alternative_mapping_base(cls) -> type[DataAccessObject] | None:
        """
        Find the first base class using an alternative mapping.

        :return: The base class or None.
        """
        for base_clazz in cls.__mro__[1:]:
            try:
                if issubclass(base_clazz, DataAccessObject) and issubclass(
                    base_clazz.original_class(), AlternativeMapping
                ):
                    return base_clazz
            except (AttributeError, TypeError, NoGenericError):
                continue
        return None

    @classmethod
    @lru_cache(maxsize=None)
    def _uses_alternative_mapping(cls) -> bool:
        """
        :return: Whether the original class of this DAO class is an
            :class:`AlternativeMapping`.
        """
        try:
            return issubclass(cls.original_class(), AlternativeMapping)
        except (AttributeError, TypeError, NoGenericError):
            return False

    def __repr__(self) -> str:
        """
        Return a string representation including columns and relationships.

        :return: The string representation.
        """
        if not hasattr(_repr_thread_local, "seen"):
            _repr_thread_local.seen = set()

        if id(self) in _repr_thread_local.seen:
            return f"{self.__class__.__name__}(...)"

        _repr_thread_local.seen.add(id(self))
        try:
            mapper: sqlalchemy.orm.Mapper = sqlalchemy.inspection.inspect(type(self))
            representations = []

            for column in mapper.columns:
                if is_data_column(column):
                    value = getattr(self, column.name)
                    representations.append(f"{column.name}={repr(value)}")

            for relationship in mapper.relationships:
                value = getattr(self, relationship.key)
                if value is not None:
                    representations.append(f"{relationship.key}={repr(value)}")

            return f"{self.__class__.__name__}({', '.join(representations)})"
        finally:
            _repr_thread_local.seen.remove(id(self))


class PartDataAccessObject(DataAccessObject[T]):
    """
    The row of an object in the table of a base class outside the object's primary
    chain, mapped to that base class's table and held by the object's own DAO.

    It is never the DAO an object is converted to or loaded as.
    """

    @classproperty
    def is_part(cls) -> bool:
        """
        :return: True, as overriding :meth:`StoredObjectRowHolder.is_part`.
        """
        return True

    @classmethod
    @lru_cache(maxsize=None)
    def type_column_key(cls) -> str | None:
        """
        :return: The attribute of the type column of the base class's hierarchy, if the
            base class's table is the root of that hierarchy and so holds it.
        """
        base_dao = get_data_access_object_class(cls.original_class())
        type_column = sqlalchemy.inspection.inspect(base_dao).polymorphic_on
        if type_column is None or type_column.table is not cls.__table__:
            return None
        return type_column.key
