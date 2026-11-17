"""
A module that manages fields by hand as well as through the field descriptor base.
"""

from __future__ import annotations

from dataclasses import dataclass

from typing_extensions import Any

from krrood.patterns.dataclass_field_descriptor import DataclassFieldDescriptor


class HandWrittenDataDescriptor:
    """
    A data descriptor that does not build on the field descriptor base.
    """

    def __get__(self, instance: Any, owner: type) -> Any:
        """
        :return: Nothing.
        """
        return None

    def __set__(self, instance: Any, value: Any) -> None:
        """
        Discards *value*.
        """


@dataclass
class DataclassWithPropertySetter:
    """
    A dataclass whose field is managed by a property, next to a computed property with a
    setter.
    """

    _value: int = 0
    """
    The value the property manages.
    """

    @property
    def value(self) -> int:
        """
        The stored value.
        """
        return self._value

    @value.setter
    def value(self, value: int) -> None:
        self._value = value

    @property
    def doubled(self) -> int:
        """
        Twice the stored value, which no field holds.
        """
        return 2 * self._value

    @doubled.setter
    def doubled(self, doubled: int) -> None:
        self._value = doubled // 2


class PlainClassWithPropertySetter:
    """
    A class that is no dataclass and may have a property setter.
    """

    stored: int = 0
    """
    The value the property manages.
    """

    @property
    def value(self) -> int:
        """
        The stored value.
        """
        return self.stored

    @value.setter
    def value(self, value: int) -> None:
        self.stored = value


@dataclass(eq=False)
class DescriptorOnTheBase(DataclassFieldDescriptor):
    """
    A field descriptor built on the base, which may implement reading and writing.
    """

    def read(self, instance: Any) -> Any:
        return None

    def write(self, instance: Any, value: Any) -> None:
        pass
