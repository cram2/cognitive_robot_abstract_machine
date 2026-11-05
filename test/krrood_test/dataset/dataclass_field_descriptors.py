"""
A dataclass field kept in another object through a
:class:`~krrood.patterns.dataclass_field_descriptor.DataclassFieldDescriptor`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from krrood.patterns.dataclass_field_descriptor import DataclassFieldDescriptor


# %% storage


@dataclass
class Storage:
    """
    The place a descriptor backed field keeps its value, counting how often it is written.
    """

    value: int | None = None
    """
    The stored value.
    """

    number_of_writes: int = 0
    """
    How often the value was written.
    """


# %% descriptor backed field


@dataclass
class ValueKeptInStorage(DataclassFieldDescriptor):
    """
    A field whose value lives in the :attr:`HolderOfValueKeptInStorage.storage` of its
    owner.
    """

    def read(self, instance: HolderOfValueKeptInStorage) -> int | None:
        return instance.storage.value

    def write(self, instance: HolderOfValueKeptInStorage, value: int | None) -> None:
        instance.storage.value = value
        instance.storage.number_of_writes += 1


@dataclass
class HolderOfValueKeptInStorage:
    """
    A dataclass with a field kept in its storage.
    """

    storage: Storage = field(default_factory=Storage)
    """
    Where :attr:`value` is kept.
    """

    value: int | None = ValueKeptInStorage()
    """
    The value kept in :attr:`storage`.
    """
