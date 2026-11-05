"""
Dataclass fields whose value is kept somewhere other than the instance's own attribute.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from typing_extensions import Any


# %% dataclass field descriptor


@dataclass
class DataclassFieldDescriptor(ABC):
    """
    A data descriptor that is the default of a dataclass field and keeps the field's
    value where :meth:`read` and :meth:`write` say. The field defaults to ``None``, and
    writing the value it already has does nothing.

    Declare the field with the descriptor as its plain default, behind
    :data:`dataclasses.KW_ONLY` if it has to be keyword only, never through
    :func:`dataclasses.field`.
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
        :return: The value of the field, or ``None`` when read on the class, which
            dataclasses takes as the field's default.
        """
        if instance is None:
            return None
        return self.read(instance)

    def __set__(self, instance: Any, value: Any) -> None:
        """
        Write the value of the field, unless the field already has it.
        """
        if value is self.read(instance):
            return
        self.write(instance, value)
