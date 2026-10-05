"""
Decides how ORMatic stores each field of a mapped class.

Every way of storing a field is a :class:`FieldStorage` member, and one
:class:`FieldStorageRule` per member states the complete condition under which it
applies. The rules form a :class:`~krrood.patterns.specificity_ranking.SpecificityRule`
family: their conditions exclude each other, so no rule relies on the order in which the
rules are checked, and a field claimed by two rules raises
:class:`~krrood.patterns.exceptions.AmbiguousRuleError`.
"""

from __future__ import annotations

import enum
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from inspect import isclass
from types import NoneType

from typing_extensions import TYPE_CHECKING, ClassVar, Type

from krrood.adapters.json_serializer import JSONSerializableTypeRegistry
from krrood.class_diagrams.wrapped_field import WrappedField
from krrood.patterns.specificity_ranking import SpecificityRule

if TYPE_CHECKING:
    from krrood.ormatic.ormatic import ORMatic

logger = logging.getLogger(__name__)


class FieldStorage(enum.Enum):
    """
    The ways ORMatic can store a field.
    """

    NOT_STORED = enum.auto()
    """
    The field gets no column.
    """

    TYPE = enum.auto()
    """
    A column holding a class, for fields annotated with ``type`` or ``Type[...]``.
    """

    BUILTIN = enum.auto()
    """
    A column whose type SQLAlchemy derives from the annotation by itself.
    """

    MANY_TO_ONE = enum.auto()
    """
    A foreign key to the table of a mapped class.
    """

    MANY_TO_MANY = enum.auto()
    """
    An association table to the table of a mapped class.
    """

    CUSTOM_TYPE = enum.auto()
    """
    A column of the type the type mappings give for the field's type.
    """

    JSON = enum.auto()
    """
    A JSON column written by krrood's JSON serializer.
    """


# %% the field to decide on


@dataclass
class FieldToStore:
    """
    A field of a mapped class, together with the ORMatic instance whose mapped classes and
    type mappings decide how it is stored.
    """

    wrapped_field: WrappedField
    """
    The field to decide on.
    """

    ormatic: ORMatic = field(repr=False)
    """
    The ORMatic instance that stores the field.
    """

    sqlalchemy_builtins: ClassVar[tuple[Type, ...]] = (
        int,
        float,
        str,
        bool,
        bytes,
        NoneType,
    )
    """
    The builtins SQLAlchemy maps to a column type by itself.
    """

    def __repr__(self) -> str:
        return (
            f"{self.wrapped_field.clazz.clazz.__name__}.{self.wrapped_field.field.name}"
        )

    @property
    def type_endpoint(self) -> Type:
        """
        :return: The type the field ultimately points to.
        """
        return self.wrapped_field.type_endpoint

    @property
    def holds_a_sqlalchemy_builtin(self) -> bool:
        """
        :return: True if the field's type is a builtin SQLAlchemy maps by itself.
        """
        return self.type_endpoint in self.sqlalchemy_builtins

    def is_mapped(self, clazz: Type) -> bool:
        """
        :return: True if a table maps the class.
        """
        return clazz in self.ormatic.mapped_classes

    def has_type_mapping(self, clazz: Type) -> bool:
        """
        :return: True if the type mappings give a column type for the class.
        """
        return clazz in self.ormatic.type_mappings

    def has_json_serializer(self, clazz: Type) -> bool:
        """
        :return: True if a JSON serializer is registered for the class hierarchy.
        """
        return JSONSerializableTypeRegistry().has_type_specific_serializer(clazz)

    def is_stored_as_a_value(self, clazz: Type) -> bool:
        """
        Whether values of the class are stored whole in their owner's row, so a free type
        parameter of the class leaves nothing undecided about how to store them.

        :return: True if no table maps the class and a type mapping or a JSON serializer
            stores it.
        """
        return not self.is_mapped(clazz) and (
            self.has_type_mapping(clazz) or self.has_json_serializer(clazz)
        )

    @property
    def cannot_be_stored(self) -> bool:
        """
        Whether the field's type leaves ORMatic no way to store it: a dictionary, or a
        generic class with free type parameters that nothing in the class diagram could
        fill.
        """
        type_endpoint = self.type_endpoint
        if isclass(type_endpoint) and issubclass(type_endpoint, dict):
            return True
        return (
            self.wrapped_field.is_underspecified_generic
            and isclass(type_endpoint)
            and not self.is_mapped(type_endpoint)
            and not self.is_stored_as_a_value(type_endpoint)
            and not any(
                issubclass(type_endpoint, alternative_mapping.original_class())
                for alternative_mapping in self.ormatic.alternative_mappings
            )
        )

    @property
    def holds_a_storable_value(self) -> bool:
        """
        :return: True if the field holds values rather than classes, and ORMatic can
            store them.
        """
        return not self.wrapped_field.is_type_type and not self.cannot_be_stored


# %% rules


class FieldStorageRule(SpecificityRule, ABC):
    """
    The condition under which a field is stored in one particular way.
    """

    storage: ClassVar[FieldStorage]
    """
    The way of storing a field that this rule decides on.
    """

    @classmethod
    @abstractmethod
    def applies(cls, field_to_store: FieldToStore) -> bool:
        """
        :param field_to_store: The field to decide on.
        :return: True if the field is stored the way this rule stands for.
        """

    @classmethod
    @abstractmethod
    def describe(cls, field_to_store: FieldToStore) -> str:
        """
        :param field_to_store: A field this rule applies to.
        :return: A sentence for the log saying how the field is stored.
        """


class NotStoredRule(FieldStorageRule):
    """
    Fields whose type ORMatic has no way to store.
    """

    storage = FieldStorage.NOT_STORED

    @classmethod
    def applies(cls, field_to_store: FieldToStore) -> bool:
        return field_to_store.cannot_be_stored

    @classmethod
    def describe(cls, field_to_store: FieldToStore) -> str:
        if issubclass(field_to_store.type_endpoint, dict):
            return "Not storing, since dictionaries cannot be stored."
        return "Not storing the underspecified generic field."


class TypeRule(FieldStorageRule):
    """
    Fields that hold classes.
    """

    storage = FieldStorage.TYPE

    @classmethod
    def applies(cls, field_to_store: FieldToStore) -> bool:
        return (
            field_to_store.wrapped_field.is_type_type
            and not field_to_store.cannot_be_stored
        )

    @classmethod
    def describe(cls, field_to_store: FieldToStore) -> str:
        return "Storing as type."


class BuiltinRule(FieldStorageRule):
    """
    Single values of a builtin that SQLAlchemy maps by itself.
    """

    storage = FieldStorage.BUILTIN

    @classmethod
    def applies(cls, field_to_store: FieldToStore) -> bool:
        return (
            field_to_store.holds_a_storable_value
            and not field_to_store.wrapped_field.is_container
            and field_to_store.holds_a_sqlalchemy_builtin
        )

    @classmethod
    def describe(cls, field_to_store: FieldToStore) -> str:
        return "Storing as builtin type."


class ManyToOneRule(FieldStorageRule):
    """
    Single references to an instance of a mapped class.
    """

    storage = FieldStorage.MANY_TO_ONE

    @classmethod
    def applies(cls, field_to_store: FieldToStore) -> bool:
        return (
            field_to_store.holds_a_storable_value
            and not field_to_store.wrapped_field.is_container
            and field_to_store.is_mapped(field_to_store.type_endpoint)
        )

    @classmethod
    def describe(cls, field_to_store: FieldToStore) -> str:
        return "Storing as many to one relationship."


class ManyToManyRule(FieldStorageRule):
    """
    Collections of instances of a mapped class.
    """

    storage = FieldStorage.MANY_TO_MANY

    @classmethod
    def applies(cls, field_to_store: FieldToStore) -> bool:
        return (
            field_to_store.holds_a_storable_value
            and field_to_store.wrapped_field.is_container
            and not field_to_store.wrapped_field.is_optional
            and field_to_store.is_mapped(field_to_store.type_endpoint)
        )

    @classmethod
    def describe(cls, field_to_store: FieldToStore) -> str:
        return "Storing as many to many relationship."


class CustomTypeRule(FieldStorageRule):
    """
    Single values of a type that the type mappings give a column type for.
    """

    storage = FieldStorage.CUSTOM_TYPE

    @classmethod
    def applies(cls, field_to_store: FieldToStore) -> bool:
        type_endpoint = field_to_store.type_endpoint
        return (
            field_to_store.holds_a_storable_value
            and not field_to_store.wrapped_field.is_container
            and not field_to_store.holds_a_sqlalchemy_builtin
            and not field_to_store.is_mapped(type_endpoint)
            and field_to_store.has_type_mapping(type_endpoint)
        )

    @classmethod
    def describe(cls, field_to_store: FieldToStore) -> str:
        column_type = field_to_store.ormatic.type_mappings[field_to_store.type_endpoint]
        return f"Storing as custom type {column_type.__name__}."


class JSONRule(FieldStorageRule):
    """
    Values that krrood's JSON serializer writes: collections of builtins or of mapped
    types, and values or collections of a type with a JSON serializer of its own.
    """

    storage = FieldStorage.JSON

    @classmethod
    def applies(cls, field_to_store: FieldToStore) -> bool:
        type_endpoint = field_to_store.type_endpoint
        if not field_to_store.holds_a_storable_value or field_to_store.is_mapped(
            type_endpoint
        ):
            return False
        if field_to_store.wrapped_field.is_container:
            return (
                field_to_store.wrapped_field.is_collection_of_builtins
                or field_to_store.has_type_mapping(type_endpoint)
                or field_to_store.has_json_serializer(type_endpoint)
            )
        return (
            not field_to_store.holds_a_sqlalchemy_builtin
            and not field_to_store.has_type_mapping(type_endpoint)
            and field_to_store.has_json_serializer(type_endpoint)
        )

    @classmethod
    def describe(cls, field_to_store: FieldToStore) -> str:
        return "Storing as JSON."


# %% classification


@dataclass
class FieldClassifier:
    """
    Decides how each field is stored by asking the :class:`FieldStorageRule` family.
    """

    ormatic: ORMatic
    """
    The ORMatic instance whose mapped classes and type mappings the rules consult.
    """

    def classify(self, wrapped_field: WrappedField) -> FieldStorage:
        """
        :param wrapped_field: The field to decide on.
        :return: How the field is stored, or :attr:`FieldStorage.NOT_STORED` if no rule
            applies.
        :raises AmbiguousRuleError: If more than one rule applies.
        """
        field_to_store = FieldToStore(wrapped_field, self.ormatic)
        rule = FieldStorageRule.most_applicable(field_to_store)
        if rule is None:
            logger.info("Not storing, since no rule handles the field's type.")
            return FieldStorage.NOT_STORED
        logger.info(rule.describe(field_to_store))
        return rule.storage
