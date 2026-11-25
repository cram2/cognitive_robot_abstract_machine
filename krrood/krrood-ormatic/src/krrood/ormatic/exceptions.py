"""
Errors raised while generating ORM interfaces and while converting between domain
objects and their data access objects.
"""

from __future__ import annotations

import zoneinfo
from dataclasses import dataclass
from typing_extensions import Any, TYPE_CHECKING

from sqlalchemy.orm import RelationshipProperty

from krrood.exceptions import DataclassException

if TYPE_CHECKING:
    from krrood.ormatic.data_access_objects.alternative_mappings import FunctionMapping
    from krrood.ormatic.data_access_objects.conversion_order import (
        ConversionOrderConstraint,
    )
    from krrood.ormatic.result_loading import ResultLoading


@dataclass(kw_only=True)
class NoDAOFoundError(DataclassException, TypeError):
    """
    Represents an error raised when no DAO (Data Access Object) class is found for a
    given class.

    This exception is typically used when an attempt to convert a class into a
    corresponding DAO fails. It provides information about the class and the DAO
    involved.
    """

    obj: Any
    """
    The class that no dao was found for.
    """

    def error_message(self) -> str:
        return f"Class {type(self.obj)} does not have a DAO."

    def suggest_correction(self) -> str:
        return (
            "Check that the correct ormatic interface is imported, that the ormatic interfaces are "
            "up to date by running the script that generates the ormatic interface, and that the target class is "
            "actually mapped."
        )


@dataclass(kw_only=True)
class QueryCannotBePersisted(DataclassException, TypeError):
    """
    Raised when an object that is to be stored holds an entity query language query in
    place of a value.

    A query describes the objects that would satisfy it rather than one of them, so
    there is nothing to store until it is answered.
    """

    query: Any
    """
    The query that was to be stored.
    """

    def error_message(self) -> str:
        return f"The query {self.query} cannot be stored."

    def suggest_correction(self) -> str:
        return "Store the objects the query was answered with instead of the query."


@dataclass(kw_only=True)
class NoDAOFoundForTypeError(NoDAOFoundError):
    """
    Raised when no DAO class is found for a domain *type* rather than for a concrete
    instance.

    Type-driven lookups (such as EQL translation, which resolves DAOs from variable
    types) store the offending type itself in :attr:`obj`, so the message reports it
    directly instead of its metaclass.
    """

    def error_message(self) -> str:
        return f"No DAO found for type {self.obj}."


@dataclass(kw_only=True)
class NoDAOFoundForSelectionError(NoDAOFoundError):
    """
    Raised when none of the selected expressions of a query resolve to a DAO-bearing
    type.
    """

    def error_message(self) -> str:
        return f"No DAO could be derived from the selected expressions: {self.obj}."


@dataclass(kw_only=True)
class NoDAOFoundDuringParsingError(NoDAOFoundError):
    dao: type
    """
    The DAO class that tried to convert the cls to a DAO if any.
    """

    relationship: RelationshipProperty
    """
    The relationship that tried to create the DAO.
    """

    def error_message(self) -> str:
        return (
            f"Class {type(self.obj)} does not have a DAO. This happened when trying "
            f"to create a dao for {self.dao}) on the relationship {self.relationship} with the "
            f"relationship value {self.obj}. "
            f"Expected a relationship value of type {self.relationship.target if self.relationship else 'Unknown'}."
        )


@dataclass(kw_only=True)
class UnsupportedRelationshipError(DataclassException, ValueError):
    """
    Raised when a relationship direction is not supported by the ORM mapping.

    This error indicates that the relationship configuration could not be interpreted
    into a domain mapping.
    """

    relationship: RelationshipProperty

    def error_message(self) -> str:
        return f"Unsupported relationship direction for {self.relationship}."

    def suggest_correction(self) -> str:
        return ""


@dataclass(kw_only=True)
class UncallableFunction(NotImplementedError):
    """
    Exception raised when anonymous functions are reconstructed and then called.
    """

    function_mapping: FunctionMapping
    """
    The mapping that was used to reconstruct the function.
    """

    def __post_init__(self):
        super().__init__(
            f"The reconstructed function was a lambda function and hence cannot be called again. "
            f"The function tried to be reconstructed from {self.function_mapping}"
        )


@dataclass(kw_only=True)
class UnsupportedColumnType(DataclassException, TypeError):
    """
    Exception raised when a column type is neither a type_mapping nor a builtin
    sqlalchemy type.
    """

    column_type: type

    def error_message(self) -> str:
        return f"Column type: {self.column_type} is neither a builtin sqlalchemy type nor does it exist in the dict of type_mappings."

    def suggest_correction(self) -> str:
        return ""


@dataclass(kw_only=True)
class ZoneInfoWithoutKey(DataclassException, ValueError):
    """
    Raised when a timezone that has no IANA key, such as one read from a file, is
    written to the database, where timezones are stored by their key.
    """

    zone: zoneinfo.ZoneInfo
    """
    The timezone that has no key.
    """

    def error_message(self) -> str:
        return f"The timezone {self.zone!r} has no IANA key to store it by."

    def suggest_correction(self) -> str:
        return (
            "create the timezone from its key, e.g. zoneinfo.ZoneInfo('Europe/Berlin')."
        )


@dataclass(kw_only=True)
class ConversionOrderCycle(DataclassException, ValueError):
    """
    Raised when the alternative mappings of a conversion cannot be put in any order,
    because the orders they ask for form a cycle.
    """

    cycle: list[ConversionOrderConstraint]
    """
    The constraints that close the cycle, each one putting its earlier mapping type
    before its later one.
    """

    def error_message(self) -> str:
        orders = "; ".join(constraint.description() for constraint in self.cycle)
        return f"No order converts these alternative mappings: {orders}."

    def suggest_correction(self) -> str:
        corrections = "; ".join(
            constraint.suggest_correction() for constraint in self.cycle
        )
        return f"break the cycle by one of: {corrections}."


@dataclass(kw_only=True)
class StoredObjectTableNotUniqueError(DataclassException, LookupError):
    """
    Raised when not exactly one class maps the table every stored object has a row in.
    """

    table_classes: list[type]
    """
    The classes mapping that table in the process.
    """

    def error_message(self) -> str:
        if not self.table_classes:
            return "No class maps the table of every stored object."
        return (
            f"Several classes map the table of every stored object: "
            f"{self.table_classes}."
        )

    def suggest_correction(self) -> str:
        return (
            "Import krrood.ormatic.stored_object, which every generated ORM interface "
            "imports, and map the table with no other class."
        )


@dataclass(kw_only=True)
class PartOfDetachedObjectNotLoadedError(DataclassException, LookupError):
    """
    Raised when a DAO outside any session is asked for a part it never loaded, so the
    fields of that secondary base cannot be read.
    """

    data_access_object: Any
    """
    The DAO asked for the part.
    """

    part_class: type
    """
    The class of the part that was never loaded.
    """

    def error_message(self) -> str:
        return (
            f"{type(self.data_access_object).__name__} with id "
            f"{self.data_access_object.database_id} is not in a session and never "
            f"loaded its {self.part_class.__name__}."
        )

    def suggest_correction(self) -> str:
        return (
            "Convert or read the DAO while its session is open, or add it to a session "
            "again before asking for its parts."
        )


@dataclass(kw_only=True)
class UnsupportedResultLoadingError(DataclassException, ValueError):
    """
    Raised when a ``set_of`` query is asked to load its rows otherwise than as it
    selects them.
    """

    result_loading: ResultLoading
    """
    The result loading the query was asked to evaluate with.
    """

    def error_message(self) -> str:
        return (
            f"A set_of query loads the rows it selects as they are, so it cannot be "
            f"loaded with {self.result_loading}."
        )

    def suggest_correction(self) -> str:
        return (
            "Keep the subclass loading and the result content of the set_of query at "
            "their defaults, or select one variable with entity()."
        )
