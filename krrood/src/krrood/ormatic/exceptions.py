from __future__ import annotations

from dataclasses import dataclass
from typing_extensions import Type, Any

from sqlalchemy.orm import RelationshipProperty

from ..utils import DataclassException


@dataclass
class NoGenericError(DataclassException, TypeError):
    """
    Exception raised when the original class for a DataAccessObject subclass cannot
    be determined.

    This exception is typically raised when a DataAccessObject subclass has not
    been parameterized properly, which prevents identifying the original class
    associated with it.
    """

    clazz: Type

    def __post_init__(self):
        self.message = (
            f"Cannot determine original class for {self.clazz}. "
            "Did you forget to parameterise the DataAccessObject subclass?"
        )


@dataclass
class NoDAOFoundError(DataclassException, TypeError):
    """
    Represents an error raised when no DAO (Data Access Object) class is found for a given class.

    This exception is typically used when an attempt to convert a class into a corresponding DAO fails.
    It provides information about the class and the DAO involved.
    """

    obj: Any
    """
    The class that no dao was found for
    """

    def __post_init__(self):
        self.message = (
            f"Class {type(self.obj)} does not have a DAO. Did you forget to import your ORM Interface? "
            f"Otherwise the class may not be in the ORM Interface"
        )


@dataclass
class NoDAOFoundDuringParsingError(NoDAOFoundError):

    dao: Type
    """
    The DAO class that tried to convert the cls to a DAO if any.
    """

    relationship: RelationshipProperty
    """
    The relationship that tried to create the DAO.
    """

    def __init__(self, obj: Any, dao: Type, relationship: RelationshipProperty = None):
        self.message = (
            f"Class {type(obj)} does not have a DAO. This happened when trying "
            f"to create a dao for {dao}) on the relationship {relationship} with the "
            f"relationship value {obj}. "
            f"Expected a relationship value of type {relationship.target if relationship else "Unknown"}."
        )


@dataclass
class UnsupportedRelationshipError(DataclassException, ValueError):
    """
    Raised when a relationship direction is not supported by the ORM mapping.

    This error indicates that the relationship configuration could not be
    interpreted into a domain mapping.
    """

    relationship: RelationshipProperty

    def __post_init__(self):
        self.message = f"Unsupported relationship direction for {self.relationship}."
