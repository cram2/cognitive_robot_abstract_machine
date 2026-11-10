"""
Tests for turning dataclass instances into constructor source and back
(:mod:`krrood.code_generation.dataclass_source`).
"""

from __future__ import annotations

import dataclasses

import pytest

from krrood.code_generation.dataclass_source import DataclassConstructorSerializer
from krrood.code_generation.exceptions import ObjectNotSerializableError
from test.krrood_test.dataset.nested_dataclass_records import (
    Animal,
    Owner,
    Species,
    UnserializableRecord,
    rex,
)

# %% DataclassConstructorSerializer.to_source


def test_to_source_emits_eval_able_constructor_source_for_a_nested_dataclass():
    serializer = DataclassConstructorSerializer()

    object_source = serializer.to_source(rex())

    rebuilt = eval(
        object_source.source,
        {record_type.__name__: record_type for record_type in (Animal, Species, Owner)},
    )
    assert rebuilt == rex()
    assert object_source.referenced_types == {Animal, Species, Owner}


def test_to_source_raises_for_a_non_dataclass_value():
    serializer = DataclassConstructorSerializer()

    with pytest.raises(ObjectNotSerializableError) as error:
        serializer.to_source({"not": "a dataclass"})
    assert error.value.value == {"not": "a dataclass"}
    assert (
        error.value.supported_types
        == DataclassConstructorSerializer.supported_scalar_types
    )


def test_to_source_raises_for_an_unsupported_field_type():
    serializer = DataclassConstructorSerializer()

    with pytest.raises(ObjectNotSerializableError) as error:
        serializer.to_source(UnserializableRecord(tags=["a", "b"]))
    assert error.value.value == ["a", "b"]


# %% DataclassConstructorSerializer.from_data


def test_from_data_reconstructs_a_nested_dataclass_from_an_asdict_style_mapping():
    serializer = DataclassConstructorSerializer()
    data = dataclasses.asdict(rex())

    rebuilt = serializer.from_data(data, Animal)

    assert rebuilt == rex()
