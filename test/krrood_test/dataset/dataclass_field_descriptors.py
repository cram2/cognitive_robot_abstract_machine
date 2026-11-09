"""
A dataclass field kept in another object through a
:class:`~krrood.patterns.dataclass_field_descriptor.DataclassFieldDescriptor`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from typing_extensions import Generic, TypeVar

from krrood.patterns.dataclass_field_descriptor import (
    DataclassFieldDescriptor,
    DataclassFieldDescriptorOwner,
)
from krrood.patterns.field_metadata import JSONMetadata
from krrood.patterns.subclass_safe_generic import SubClassSafeGeneric


# %% storage


@dataclass(kw_only=True)
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


@dataclass(kw_only=True, eq=False)
class ValueKeptInStorage(DataclassFieldDescriptor):
    """
    A field whose value lives in the storage of its owner.
    """

    def read(
        self, *, instance: HolderOfValueKeptInStorage | GenericHolderOfValueKeptInStorage
    ) -> int | None:
        return instance.storage.value

    def write(
        self,
        *, instance: HolderOfValueKeptInStorage | GenericHolderOfValueKeptInStorage,
        value: int | None,
    ) -> None:
        instance.storage.value = value
        instance.storage.number_of_writes += 1


@dataclass(kw_only=True)
class HolderOfValueKeptInStorage(DataclassFieldDescriptorOwner):
    """
    A dataclass with a field kept in its storage.
    """

    storage: Storage = field(default_factory=Storage)
    """
    Where :attr:`value` is kept.
    """

    value: int | None = field(
        default=None, kw_only=True, metadata=JSONMetadata(serialize=False).as_dict()
    )
    """
    The value kept in :attr:`storage`.
    """


HolderOfValueKeptInStorage.value = ValueKeptInStorage()


# %% subclasses of the holder


@dataclass(kw_only=True)
class SubclassOfHolder(HolderOfValueKeptInStorage):
    """
    A subclass that inherits the descriptor backed field unchanged.
    """

    other_value: int = 0
    """
    A field of the subclass's own.
    """


@dataclass(kw_only=True)
class SubclassDeclaringTheFieldAgain(HolderOfValueKeptInStorage):
    """
    A subclass that declares the descriptor backed field again without a default.
    """

    value: int | None = field(kw_only=True)
    """
    The value kept in :attr:`storage`.
    """


@dataclass(kw_only=True)
class SubclassDeclaringTheFieldAgainWithDefault(HolderOfValueKeptInStorage):
    """
    A subclass that declares the descriptor backed field again with a default of its own.
    """

    value: int | None = 7
    """
    The value kept in :attr:`storage`.
    """


@dataclass(kw_only=True)
class SubclassDeclaringTheFieldAgainWithField(HolderOfValueKeptInStorage):
    """
    A subclass that declares the descriptor backed field again through
    :func:`dataclasses.field`.
    """

    value: int | None = field(default=7, kw_only=True)
    """
    The value kept in :attr:`storage`.
    """


ValueType = TypeVar("ValueType")
"""
The type of the value of :class:`GenericHolderOfValueKeptInStorage`.
"""


@dataclass(kw_only=True)
class GenericHolderOfValueKeptInStorage(
    Generic[ValueType], SubClassSafeGeneric, DataclassFieldDescriptorOwner
):
    """
    A generic dataclass with a field kept in its storage whose type is the parameter.
    """

    storage: Storage = field(default_factory=Storage)
    """
    Where :attr:`value` is kept.
    """

    value: ValueType | None = field(default=None, kw_only=True)
    """
    The value kept in :attr:`storage`.
    """


GenericHolderOfValueKeptInStorage.value = ValueKeptInStorage()


@dataclass(kw_only=True)
class IntegerHolderOfValueKeptInStorage(GenericHolderOfValueKeptInStorage[int]):
    """
    A subclass that narrows the type of the descriptor backed field.
    """
