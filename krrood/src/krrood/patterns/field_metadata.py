from __future__ import annotations

from dataclasses import dataclass, fields, is_dataclass

from typing_extensions import Dict, Optional, Self, Type


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


@dataclass
class JSONMetadata(FieldMetadata):
    serialize: bool = True
    """
    Whether the field should be serialized to JSON.
    """
