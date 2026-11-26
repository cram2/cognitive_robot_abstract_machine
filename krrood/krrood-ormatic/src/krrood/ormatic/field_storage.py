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

from typing_extensions import TYPE_CHECKING, ClassVar, Type, get_origin

from krrood.adapters.json_serializer import (
    DataclassJSONSerializer,
    JSONSerializableTypeRegistry,
)
from krrood.class_diagrams.utils import behaves_like_a_built_in_type
from krrood.class_diagrams.wrapped_field import WrappedField
from krrood.ormatic.collection_shape import CollectionShape
from krrood.patterns.classproperty import classproperty
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
    An association table with a row per value of a collection that holds a mapped
    class, as its values or as its keys.
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


@dataclass(kw_only=True)
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

    sqlalchemy_builtins: ClassVar[tuple[type, ...]] = (
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
    def type_endpoint(self) -> type:
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

    def is_mapped(self, *, clazz: type) -> bool:
        """
        :return: True if a table maps the class.
        """
        return clazz in self.ormatic.mapped_classes

    def has_type_mapping(self, *, clazz: type) -> bool:
        """
        :return: True if the type mappings give a column type for the class.
        """
        return clazz in self.ormatic.type_mappings

    def has_json_serializer(self, *, clazz: type) -> bool:
        """
        :return: True if a JSON serializer is registered for the class hierarchy, or the
            class is a dataclass, which :class:`DataclassJSONSerializer` writes.
        """
        return JSONSerializableTypeRegistry().has_type_specific_serializer(
            clazz=clazz
        ) or DataclassJSONSerializer.matches_generic_type(clazz=clazz)

    def is_stored_as_a_value(self, *, clazz: type) -> bool:
        """
        Whether values of the class are stored whole in their owner's row, so a free type
        parameter of the class leaves nothing undecided about how to store them.

        :return: True if no table maps the class and a type mapping or a JSON serializer
            stores it.
        """
        return not self.is_mapped(clazz=clazz) and (
            self.has_type_mapping(clazz=clazz) or self.has_json_serializer(clazz=clazz)
        )

    @property
    def collection_shape(self) -> CollectionShape | None:
        """
        :return: The shape of the collection the field holds, or None if it holds none.
        """
        return CollectionShape.from_annotation(
            annotation=self.wrapped_field.resolved_type
        )

    def value_storage(self, *, value_type: type) -> ValueStorage:
        """
        :param value_type: The type of a key or value of the collection the field holds.
        :return: How keys or values of the type are stored, as the
            :class:`ValueStorageRule` family decides.
        """
        rule = ValueStorageRule.most_applicable(
            value_to_store=ValueToStore(value_type=value_type, field_to_store=self)
        )
        if rule is None:
            return ValueStorage.NOT_STORED
        return rule.storage

    @property
    def cannot_be_stored(self) -> bool:
        """
        Whether the field's type leaves ORMatic no way to store it: a dictionary without
        key and value types, or a generic class with free type parameters that nothing
        in the class diagram could fill.
        """
        type_endpoint = self.type_endpoint
        if isclass(type_endpoint) and issubclass(type_endpoint, dict):
            return True
        return (
            self.wrapped_field.is_underspecified_generic
            and isclass(type_endpoint)
            and not self.is_mapped(clazz=type_endpoint)
            and not self.is_stored_as_a_value(clazz=type_endpoint)
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


# %% the keys and values of a collection


class ValueStorage(enum.Enum):
    """
    The ways ORMatic can store a key or a value of a collection.
    """

    REFERENCE = enum.auto()
    """
    A foreign key to the row of a mapped class.
    """

    COLUMN = enum.auto()
    """
    A column of the value's own type.
    """

    JSON = enum.auto()
    """
    A JSON column written by krrood's JSON serializer.
    """

    NOT_STORED = enum.auto()
    """
    No way to store the value.
    """


@dataclass(kw_only=True)
class ValueToStore:
    """
    The type of a key or value of a collection a field holds.
    """

    value_type: type
    """
    The type.
    """

    field_to_store: FieldToStore
    """
    The field holding the collection.
    """

    @property
    def is_mapped(self) -> bool:
        """
        :return: True if a table maps the type.
        """
        return self.field_to_store.is_mapped(clazz=self.value_type)

    @property
    def has_column_type(self) -> bool:
        """
        :return: True if a column type exists for the type: a builtin SQLAlchemy maps by
            itself, an enum, or a type with a type mapping.
        """
        return (
            self.value_type in FieldToStore.sqlalchemy_builtins
            or (isclass(self.value_type) and issubclass(self.value_type, enum.Enum))
            or self.field_to_store.has_type_mapping(clazz=self.value_type)
        )


class ValueStorageRule(SpecificityRule, ABC):
    """
    The condition under which a key or value of a collection is stored in one
    particular way.
    """

    @classproperty
    @abstractmethod
    def storage(cls) -> ValueStorage:
        """
        :return: The way of storing a value that this rule decides on.
        """

    @classmethod
    @abstractmethod
    def applies(cls, *, value_to_store: ValueToStore) -> bool:
        """
        :param value_to_store: The type of the key or value to decide on.
        :return: True if the value is stored the way this rule stands for.
        """


class ReferenceRule(ValueStorageRule):
    """
    Instances of a mapped class.
    """

    @classproperty
    def storage(cls) -> ValueStorage:
        return ValueStorage.REFERENCE

    @classmethod
    def applies(cls, *, value_to_store: ValueToStore) -> bool:
        return value_to_store.is_mapped


class ColumnRule(ValueStorageRule):
    """
    Values of a type no table maps, but a column type exists for.
    """

    @classproperty
    def storage(cls) -> ValueStorage:
        return ValueStorage.COLUMN

    @classmethod
    def applies(cls, *, value_to_store: ValueToStore) -> bool:
        return not value_to_store.is_mapped and value_to_store.has_column_type


class JSONValueRule(ValueStorageRule):
    """
    Values of a type that krrood's JSON serializer writes and no column type exists
    for: other builtins, classes, and types with a JSON serializer.
    """

    @classproperty
    def storage(cls) -> ValueStorage:
        return ValueStorage.JSON

    @classmethod
    def applies(cls, *, value_to_store: ValueToStore) -> bool:
        value_type = value_to_store.value_type
        return (
            not value_to_store.is_mapped
            and not value_to_store.has_column_type
            and (
                behaves_like_a_built_in_type(value_type)
                or value_type in (type, Type)
                or get_origin(value_type) is type
                or value_to_store.field_to_store.has_json_serializer(clazz=value_type)
            )
        )


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
    def applies(cls, *, field_to_store: FieldToStore) -> bool:
        """
        :param field_to_store: The field to decide on.
        :return: True if the field is stored the way this rule stands for.
        """

    @classmethod
    @abstractmethod
    def describe(cls, *, field_to_store: FieldToStore) -> str:
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
    def applies(cls, *, field_to_store: FieldToStore) -> bool:
        return field_to_store.cannot_be_stored

    @classmethod
    def describe(cls, *, field_to_store: FieldToStore) -> str:
        if issubclass(field_to_store.type_endpoint, dict):
            return "Not storing, since the dictionary declares no key and value types."
        return "Not storing the underspecified generic field."


class TypeRule(FieldStorageRule):
    """
    Fields that hold classes.
    """

    storage = FieldStorage.TYPE

    @classmethod
    def applies(cls, *, field_to_store: FieldToStore) -> bool:
        return (
            field_to_store.wrapped_field.is_type_type
            and not field_to_store.cannot_be_stored
        )

    @classmethod
    def describe(cls, *, field_to_store: FieldToStore) -> str:
        return "Storing as type."


class BuiltinRule(FieldStorageRule):
    """
    Single values of a builtin that SQLAlchemy maps by itself.
    """

    storage = FieldStorage.BUILTIN

    @classmethod
    def applies(cls, *, field_to_store: FieldToStore) -> bool:
        return (
            field_to_store.holds_a_storable_value
            and not field_to_store.wrapped_field.is_container
            and field_to_store.holds_a_sqlalchemy_builtin
        )

    @classmethod
    def describe(cls, *, field_to_store: FieldToStore) -> str:
        return "Storing as builtin type."


class ManyToOneRule(FieldStorageRule):
    """
    Single references to an instance of a mapped class.
    """

    storage = FieldStorage.MANY_TO_ONE

    @classmethod
    def applies(cls, *, field_to_store: FieldToStore) -> bool:
        return (
            field_to_store.holds_a_storable_value
            and not field_to_store.wrapped_field.is_container
            and field_to_store.is_mapped(clazz=field_to_store.type_endpoint)
        )

    @classmethod
    def describe(cls, *, field_to_store: FieldToStore) -> str:
        return "Storing as many to one relationship."


class ManyToManyRule(FieldStorageRule):
    """
    Collections, nested ones and dictionaries included, that hold instances of a mapped
    class as their values or keys.
    """

    storage = FieldStorage.MANY_TO_MANY

    @classmethod
    def applies(cls, *, field_to_store: FieldToStore) -> bool:
        shape = field_to_store.collection_shape
        if (
            not field_to_store.holds_a_storable_value
            or shape is None
            or len(shape.leaf_types) != 1
        ):
            return False
        key_storages = [
            field_to_store.value_storage(value_type=key_type)
            for key_type in shape.key_types
        ]
        leaf_storage = field_to_store.value_storage(value_type=shape.leaf_type)
        return (
            ValueStorage.REFERENCE in (*key_storages, leaf_storage)
            and all(
                storage in (ValueStorage.REFERENCE, ValueStorage.COLUMN)
                for storage in key_storages
            )
            and leaf_storage is not ValueStorage.NOT_STORED
        )

    @classmethod
    def describe(cls, *, field_to_store: FieldToStore) -> str:
        return "Storing as many to many relationship."


class CustomTypeRule(FieldStorageRule):
    """
    Single values of a type that the type mappings give a column type for.
    """

    storage = FieldStorage.CUSTOM_TYPE

    @classmethod
    def applies(cls, *, field_to_store: FieldToStore) -> bool:
        type_endpoint = field_to_store.type_endpoint
        return (
            field_to_store.holds_a_storable_value
            and not field_to_store.wrapped_field.is_container
            and not field_to_store.holds_a_sqlalchemy_builtin
            and not field_to_store.is_mapped(clazz=type_endpoint)
            and field_to_store.has_type_mapping(clazz=type_endpoint)
        )

    @classmethod
    def describe(cls, *, field_to_store: FieldToStore) -> str:
        column_type = field_to_store.ormatic.type_mappings[field_to_store.type_endpoint]
        return f"Storing as custom type {column_type.__name__}."


class JSONRule(FieldStorageRule):
    """
    Values that krrood's JSON serializer writes: collections, nested ones and
    dictionaries included, whose keys and values hold no mapped class, and values of a
    type with a JSON serializer of its own.
    """

    storage = FieldStorage.JSON

    @classmethod
    def applies(cls, *, field_to_store: FieldToStore) -> bool:
        type_endpoint = field_to_store.type_endpoint
        if not field_to_store.holds_a_storable_value or field_to_store.is_mapped(
            clazz=type_endpoint
        ):
            return False
        shape = field_to_store.collection_shape
        if shape is not None:
            return all(
                field_to_store.value_storage(value_type=value_type)
                in (ValueStorage.COLUMN, ValueStorage.JSON)
                for value_type in (*shape.key_types, *shape.leaf_types)
            )
        return (
            not field_to_store.holds_a_sqlalchemy_builtin
            and not field_to_store.has_type_mapping(clazz=type_endpoint)
            and field_to_store.has_json_serializer(clazz=type_endpoint)
        )

    @classmethod
    def describe(cls, *, field_to_store: FieldToStore) -> str:
        return "Storing as JSON."


# %% classification


@dataclass(kw_only=True)
class FieldClassifier:
    """
    Decides how each field is stored by asking the :class:`FieldStorageRule` family.
    """

    ormatic: ORMatic
    """
    The ORMatic instance whose mapped classes and type mappings the rules consult.
    """

    def classify(self, *, wrapped_field: WrappedField) -> FieldStorage:
        """
        :param wrapped_field: The field to decide on.
        :return: How the field is stored, or :attr:`FieldStorage.NOT_STORED` if no rule
            applies.
        :raises AmbiguousRuleError: If more than one rule applies.
        """
        field_to_store = FieldToStore(wrapped_field=wrapped_field, ormatic=self.ormatic)
        rule = FieldStorageRule.most_applicable(field_to_store=field_to_store)
        if rule is None:
            logger.info("Not storing, since no rule handles the field's type.")
            return FieldStorage.NOT_STORED
        logger.info(rule.describe(field_to_store=field_to_store))
        return rule.storage
