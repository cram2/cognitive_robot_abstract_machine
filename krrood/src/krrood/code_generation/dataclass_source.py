"""
Conversion of dataclass instances to eval-able Python constructor source.

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
    Converts an object to eval-able Python constructor source.
    """

    @abstractmethod
    def to_source(self, instance: Any) -> ObjectSource:
        """
        Return the constructor source and referenced types for ``instance``.

        :param instance: The object to serialize.
        :return: The eval-able constructor expression and the types it needs imported.
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
