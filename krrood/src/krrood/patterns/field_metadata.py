from __future__ import annotations

from dataclasses import dataclass, fields, is_dataclass

from typing_extensions import Dict, Optional, Self, TYPE_CHECKING, Type

if TYPE_CHECKING:
    from krrood.class_diagrams.wrapped_field import WrappedField


@dataclass
class FieldMetadata:
    """
    Krrood-specific metadata carried inside a dataclass field's ``metadata`` mapping.

    A field carries at most one instance of a given :class:`FieldMetadata` subclass,
    stored under that subclass itself as the key (attach it with :meth:`as_dict`, read
    it back with :meth:`of_field`).
    """

    def as_dict(self) -> Dict[type, Self]:
        """
        :return: a dataclass-field ``metadata`` mapping carrying this metadata under its own
            type, ready to pass to ``field(metadata=...)``.
        """
        return {type(self): self}

    @classmethod
    def of_field(cls, clazz: Type, field_name: str) -> Optional[Self]:
        """
        :return: the instance of *cls* attached to *field_name* of *clazz*, or ``None`` when
            *clazz* is not a dataclass, has no such field, or the field carries no metadata of
            type *cls*.
        """
        if not is_dataclass(clazz):
            return None
        field_ = next((f for f in fields(clazz) if f.name == field_name), None)
        if field_ is None:
            return None
        return field_.metadata.get(cls)

    @classmethod
    def of_wrapped_field(cls, wrapped_field: WrappedField) -> Optional[Self]:
        """
        :return: the instance of *cls* attached to the dataclass field behind
            *wrapped_field*, or ``None`` when that field carries no metadata of type *cls*.

        ..note:: The lookup goes through the dataclass field name, which may differ from
            the public name under which the field is addressed.
        """
        return cls.of_field(wrapped_field.clazz.clazz, wrapped_field.field.name)


@dataclass
class JSONMetadata(FieldMetadata):
    serialize: bool = True
    """
    Whether the field should be serialized to JSON.
    """
