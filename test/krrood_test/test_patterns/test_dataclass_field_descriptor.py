"""
Tests for :class:`~krrood.patterns.dataclass_field_descriptor.DataclassFieldDescriptor`.
"""

from __future__ import annotations

from dataclasses import fields

import pytest

from krrood.patterns.field_metadata import JSONMetadata

from ..dataset.dataclass_field_descriptors import (
    GenericHolderOfValueKeptInStorage,
    HolderOfValueKeptInStorage,
    IntegerHolderOfValueKeptInStorage,
    SubclassDeclaringTheFieldAgain,
    SubclassOfHolder,
    ValueKeptInStorage,
)


# %% field declaration


def test_field_keeps_its_declared_default():
    [value_field] = [
        field_
        for field_ in fields(HolderOfValueKeptInStorage)
        if field_.name == "value"
    ]
    assert value_field.default is None


def test_field_keeps_its_metadata():
    assert JSONMetadata.of_field(HolderOfValueKeptInStorage, "value") == JSONMetadata(
        serialize=False
    )


def test_descriptor_is_read_on_the_class():
    assert isinstance(HolderOfValueKeptInStorage.value, ValueKeptInStorage)


# %% reading and writing


def test_value_given_at_construction_is_written():
    value = 3
    assert HolderOfValueKeptInStorage(value=value).storage.value == value


def test_default_is_written_at_construction():
    holder = HolderOfValueKeptInStorage()
    assert holder.storage.number_of_writes == 1
    assert holder.storage.value is None


def test_reading_returns_the_stored_value():
    holder = HolderOfValueKeptInStorage()
    holder.storage.value = 5
    assert holder.value == holder.storage.value


def test_value_is_not_kept_on_the_instance():
    assert "value" not in vars(HolderOfValueKeptInStorage(value=3))


# %% subclasses


@pytest.mark.parametrize(
    "holder_type",
    [
        SubclassOfHolder,
        GenericHolderOfValueKeptInStorage,
        IntegerHolderOfValueKeptInStorage,
    ],
)
def test_subclass_keeps_the_value_in_storage(holder_type: type):
    value = 3
    holder = holder_type(value=value)
    assert holder.storage.value == value
    assert "value" not in vars(holder)


def test_narrowed_field_keeps_its_default():
    assert IntegerHolderOfValueKeptInStorage().value is None


def test_narrowed_field_has_the_bound_type():
    [value_field] = [
        field_
        for field_ in fields(IntegerHolderOfValueKeptInStorage)
        if field_.name == "value"
    ]
    assert value_field.type == int | None


def test_field_declared_again_without_default_keeps_the_value_in_storage():
    value = 3
    holder = SubclassDeclaringTheFieldAgain(value=value)
    assert holder.storage.value == value
    assert "value" not in vars(holder)
