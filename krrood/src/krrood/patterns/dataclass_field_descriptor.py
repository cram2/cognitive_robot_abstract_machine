"""
Dataclass fields whose value is kept somewhere other than the instance's own attribute.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from copy import copy
from dataclasses import MISSING, Field, dataclass
from itertools import repeat

from typing_extensions import Any


# %% dataclass field descriptor


@dataclass(eq=False)
class DataclassFieldDescriptor(ABC):
    """
    A data descriptor that manages a field of a dataclass and keeps the field's value
    where :meth:`read` and :meth:`write` say.

    The field stays an ordinary dataclass field, declared with any default and
    :func:`dataclasses.field` arguments. Assign the descriptor to the field's name on
    the class once the class is created, e.g. ``Owner.value = KeptInStorage()``.

    .. warning:: A subclass that declares the field again with a default of its own
        replaces the descriptor with that default, so the value is no longer managed.
    """

    @abstractmethod
    def read(self, instance: Any) -> Any:
        """
        :param instance: The instance that owns the field.
        :return: The value of the field.
        """

    @abstractmethod
    def write(self, instance: Any, value: Any) -> None:
        """
        Keep the value of the field.

        :param instance: The instance that owns the field.
        :param value: The new value of the field.
        """

    def __get__(self, instance: Any | None, owner: type | None = None) -> Any:
        """
        :return: The value of the field, or the descriptor itself when read on the class.
        """
        if instance is None:
            return self
        return self.read(instance)

    def __set__(self, instance: Any, value: Any) -> None:
        """
        Write the value of the field, unless the value is a descriptor, which a subclass
        that declares the field again without a default takes as the field's default.
        """
        if isinstance(value, DataclassFieldDescriptor):
            return
        self.write(instance, value)

    @staticmethod
    def without_class_level_default(field_: Field) -> Field:
        """
        :param field_: A field of a subclass that a descriptor manages on a base class.
        :return: A copy of the field with its default given by a factory, so that
            :func:`dataclasses.dataclass` does not set the default on the subclass over
            the inherited descriptor.
        """
        if field_.default is MISSING:
            return field_
        result = copy(field_)
        result.default_factory = repeat(field_.default).__next__
        result.default = MISSING
        return result
