"""
Conversion of dataclass instances to eval-able Python constructor source and back.

:class:`ObjectSourceSerializer` is the pluggable strategy; the default,
:class:`DataclassConstructorSerializer`, handles flat and nested dataclasses whose
fields are scalars, enum members or further dataclasses.
"""

from __future__ import annotations

import dataclasses
import enum
from abc import ABC, abstractmethod
from dataclasses import dataclass

from typing_extensions import Any

from krrood.class_diagrams.utils import get_type_hints_of_object
from krrood.code_generation.exceptions import ObjectNotSerializableError
from krrood.code_generation.object_to_source import value_to_source
from krrood.ormatic.utils import classproperty

# %% ObjectSource


@dataclass
class ObjectSource:
    """
    The eval-able Python constructor source for an object, plus its type dependencies.
    """

    source: str
    """
    Eval-able Python constructor expression, e.g. ``Animal(name='Rex')``.
    """

    referenced_types: set[type]
    """
    Every type referenced in :attr:`source` that must be imported for it to evaluate.
    """


# %% ObjectSourceSerializer


@dataclass
class ObjectSourceSerializer(ABC):
    """
    Converts an object to Python constructor source and reconstructs it from stored
    data.
    """

    @abstractmethod
    def to_source(self, instance: Any) -> ObjectSource:
        """
        Return the constructor source and referenced types for ``instance``.

        :param instance: The object to serialize.
        :return: The eval-able constructor expression and the types it needs imported.
        """

    @abstractmethod
    def from_data(self, data: Any, instance_type: type) -> Any:
        """
        Reconstruct an instance of ``instance_type`` from ``data``.

        :param data: The data representation of the object as stored by the subclass.
        :param instance_type: The expected type of the reconstructed instance.
        :return: An instance equal to the original.
        """


# %% DataclassConstructorSerializer


@dataclass
class DataclassConstructorSerializer(ObjectSourceSerializer):
    """
    Emits ``DataclassType(field=value, ...)`` source for dataclass instances whose
    fields are ``None``, a :attr:`supported_scalar_types` value or a nested dataclass,
    and raises :class:`ObjectNotSerializableError` for any other field value.

    .. warning::
        Instances must be ``dataclasses.asdict``-safe: no circular references, and every
        field settable through the constructor.
    """

    @classproperty
    def supported_scalar_types(cls) -> tuple[type, ...]:
        """
        The scalar types emitted directly as field values.
        """
        return bool, int, float, str, enum.Enum

    def to_source(self, instance: Any) -> ObjectSource:
        if not dataclasses.is_dataclass(instance) or isinstance(instance, type):
            raise ObjectNotSerializableError(instance, self.supported_scalar_types)
        referenced_types: set[type] = {type(instance)}
        field_parts = []
        for instance_field in dataclasses.fields(instance):
            value = getattr(instance, instance_field.name)
            value_source = self._emit_value(value)
            referenced_types.update(value_source.referenced_types)
            field_parts.append(f"{instance_field.name}={value_source.source}")
        source = f"{type(instance).__name__}({', '.join(field_parts)})"
        return ObjectSource(source, referenced_types)

    def _emit_value(self, value: Any) -> ObjectSource:
        """
        Emit a single field value as source, recursing into nested dataclasses.

        :param value: The value of one field of an instance.
        :return: The value's constructor source and the types it references.
        :raises ObjectNotSerializableError: When ``value`` is neither ``None``, a
            supported scalar, nor a dataclass instance.
        """
        if dataclasses.is_dataclass(value) and not isinstance(value, type):
            return self.to_source(value)
        if value is not None and not isinstance(value, self.supported_scalar_types):
            raise ObjectNotSerializableError(value, self.supported_scalar_types)
        referenced_types: set[type] = set()
        if isinstance(value, enum.Enum):
            referenced_types.add(type(value))
        return ObjectSource(value_to_source(value), referenced_types)

    def from_data(self, data: Any, instance_type: type) -> Any:
        """
        Reconstructs from a ``dataclasses.asdict``-style mapping, recursing into fields
        whose type hint is a dataclass; a non-dataclass ``instance_type`` returns
        ``data`` unchanged.
        """
        if not dataclasses.is_dataclass(instance_type):
            return data
        type_hints = get_type_hints_of_object(instance_type)
        constructor_arguments = {}
        for instance_field in dataclasses.fields(instance_type):
            value = data[instance_field.name]
            field_type = type_hints.get(instance_field.name, type(value))
            if isinstance(value, dict) and dataclasses.is_dataclass(field_type):
                constructor_arguments[instance_field.name] = self.from_data(
                    value, field_type
                )
            else:
                constructor_arguments[instance_field.name] = value
        return instance_type(**constructor_arguments)
