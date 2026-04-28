import pytest
from _pytest import raises

from krrood.ormatic.exceptions import UnsupportedColumnType
from krrood.utils import inheritance_path_length
from probabilistic_model.probabilistic_circuit.relational.learn_rspn import (
    get_python_type_from_sqlalchemy_column,
)
from semantic_digital_twin.spatial_types import Point3
from semantic_digital_twin.world_description.world_entity import Body
from ..dataset.example_classes import *


class A: ...


class B(A): ...


class C(A): ...


class D(B, C): ...


class E(D, B): ...


class WackyEnum(E, Enum): ...


def test_distance_between_classes():
    assert inheritance_path_length(KRROODPosition5D, KRROODPosition) == 2
    assert inheritance_path_length(KRROODPosition, KRROODPosition5D) is None
    assert inheritance_path_length(Atom, Symbol) == 1
    assert inheritance_path_length(MultipleInheritance, PrimaryBase) == 1
    assert inheritance_path_length(D, A) == 2
    assert inheritance_path_length(E, A) == 2
    assert inheritance_path_length(WackyEnum, Enum) == 1


def test_sqlalchemy_column_type_extraction():
    from sqlalchemy import Column, Integer, String, Boolean
    from krrood.ormatic.custom_types import PolymorphicEnumType

    assert get_python_type_from_sqlalchemy_column(Column(Integer)) == int
    assert get_python_type_from_sqlalchemy_column(Column(String)) == str
    assert get_python_type_from_sqlalchemy_column(Column(Boolean)) == bool
    assert get_python_type_from_sqlalchemy_column(Column(PolymorphicEnumType)) == Enum

    class TestUnsupportedCustomType(TypeDecorator):
        impl = types.Unicode(50)
        cache_ok = True

        def process_bind_param(self, value, dialect):
            return value

        def process_result_value(self, value, dialect):
            return value

    with pytest.raises(UnsupportedColumnType):
        get_python_type_from_sqlalchemy_column(Column(TestUnsupportedCustomType))
