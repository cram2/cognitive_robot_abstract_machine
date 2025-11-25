from __future__ import annotations

from dataclasses import fields, Field

from typing_extensions import Type

from krrood.class_diagrams.class_diagram import WrappedClass
from krrood.class_diagrams.wrapped_field import WrappedField
from ..dataset.example_classes import (
    Position,
    Orientation,
    Pose,
    Positions,
    PositionTypeWrapper,
)


def get_field_by_name(cls: Type, name: str) -> Field:
    for f in fields(cls):
        if f.name == name:
            return f
    raise ValueError


def test_builtin_not_optional():
    wrapped_class = WrappedClass(clazz=Position)
    wrapped_field = WrappedField(wrapped_class, get_field_by_name(Position, "x"))
    assert wrapped_field.resolved_type is float
    assert not wrapped_field.is_container
    assert wrapped_field.is_builtin_type
    assert not wrapped_field.is_optional
    assert not wrapped_field.is_type_type


def test_builtin_optional():
    wrapped_class = WrappedClass(clazz=Orientation)
    wrapped_field = WrappedField(wrapped_class, get_field_by_name(Orientation, "w"))

    assert wrapped_field.contained_type is float
    assert wrapped_field.is_optional
    assert not wrapped_field.is_container
    assert wrapped_field.is_builtin_type


def test_one_to_one_relationship():
    wrapped_class = WrappedClass(clazz=Pose)
    wrapped_field = WrappedField(wrapped_class, get_field_by_name(Pose, "position"))

    assert not wrapped_field.is_optional
    assert wrapped_field.container_type is None
    assert wrapped_field.resolved_type is Position
    assert not wrapped_field.is_builtin_type


def test_one_to_many_relationship():
    wrapped_class = WrappedClass(clazz=Positions)
    wrapped_field = WrappedField(
        wrapped_class, get_field_by_name(Positions, "positions")
    )

    assert not wrapped_field.is_optional
    assert wrapped_field.container_type is list
    assert wrapped_field.contained_type is Position
    assert not wrapped_field.is_builtin_type


def test_is_type_type():
    wrapped_class = WrappedClass(clazz=PositionTypeWrapper)
    wrapped_field = WrappedField(
        wrapped_class, get_field_by_name(PositionTypeWrapper, "position_type")
    )
    assert wrapped_field.is_type_type
