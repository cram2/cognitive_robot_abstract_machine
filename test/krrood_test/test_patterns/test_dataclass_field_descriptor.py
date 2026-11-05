"""
Tests for :class:`~krrood.patterns.dataclass_field_descriptor.DataclassFieldDescriptor`.
"""

from __future__ import annotations

from dataclasses import fields

from ..dataset.dataclass_field_descriptors import HolderOfValueKeptInStorage


# %% field default


def test_field_default_is_none():
    [value_field] = [
        field_
        for field_ in fields(HolderOfValueKeptInStorage)
        if field_.name == "value"
    ]
    assert value_field.default is None


def test_construction_without_value_does_not_write():
    assert HolderOfValueKeptInStorage().storage.number_of_writes == 0


# %% reading and writing


def test_value_given_at_construction_is_written():
    value = 3
    assert HolderOfValueKeptInStorage(value=value).storage.value == value


def test_reading_returns_the_stored_value():
    holder = HolderOfValueKeptInStorage()
    holder.storage.value = 5
    assert holder.value == holder.storage.value


def test_writing_the_stored_value_again_is_skipped():
    holder = HolderOfValueKeptInStorage(value=3)
    holder.value = holder.storage.value
    assert holder.storage.number_of_writes == 1
