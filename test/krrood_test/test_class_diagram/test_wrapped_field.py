from __future__ import annotations

from dataclasses import dataclass, field, fields, Field
from types import NoneType

from typing_extensions import List, Type, TypeVar

from krrood.class_diagrams.class_diagram import WrappedClass
from krrood.class_diagrams.utils import common_base_class
from krrood.class_diagrams.wrapped_field import WrappedField
from ..dataset.example_classes import (
    KRROODPosition,
    KRROODOrientation,
    KRROODPose,
    KRROODPositions,
    KRROODPositionTypeWrapper,
    KRROODBarePositionTypeWrapper,
    GenericClassAssociation,
    KRROODCup,
    KRROODBowl,
    KRROODPhysicalObject,
    KRROODPosition4D,
    KRROODPosition5D,
)


def get_field_by_name(cls: Type, name: str) -> Field:
    for f in fields(cls):
        if f.name == name:
            return f
    raise ValueError


def test_builtin_not_optional():
    wrapped_class = WrappedClass(clazz=KRROODPosition)
    wrapped_field = WrappedField(wrapped_class, get_field_by_name(KRROODPosition, "x"))
    assert wrapped_field.resolved_type is float
    assert not wrapped_field.is_container
    assert wrapped_field.is_builtin_type
    assert not wrapped_field.is_optional
    assert not wrapped_field.is_type_type


def test_builtin_optional():
    wrapped_class = WrappedClass(clazz=KRROODOrientation)
    wrapped_field = WrappedField(
        wrapped_class, get_field_by_name(KRROODOrientation, "w")
    )

    assert wrapped_field.contained_type is float
    assert wrapped_field.is_optional
    assert not wrapped_field.is_container
    assert wrapped_field.is_builtin_type
    assert not wrapped_field.is_instantiation_of_generic_class


def test_one_to_one_relationship():
    wrapped_class = WrappedClass(clazz=KRROODPose)
    wrapped_field = WrappedField(
        wrapped_class, get_field_by_name(KRROODPose, "position")
    )

    assert not wrapped_field.is_optional
    assert wrapped_field.container_type is None
    assert wrapped_field.resolved_type is KRROODPosition
    assert not wrapped_field.is_builtin_type


def test_one_to_many_relationship():
    wrapped_class = WrappedClass(clazz=KRROODPositions)
    wrapped_field = WrappedField(
        wrapped_class, get_field_by_name(KRROODPositions, "positions")
    )

    assert not wrapped_field.is_optional
    assert wrapped_field.container_type is list
    assert wrapped_field.contained_type is KRROODPosition
    assert not wrapped_field.is_builtin_type
    assert not wrapped_field.is_instantiation_of_generic_class


def test_is_type_type():
    wrapped_class = WrappedClass(clazz=KRROODPositionTypeWrapper)
    wrapped_field = WrappedField(
        wrapped_class, get_field_by_name(KRROODPositionTypeWrapper, "position_type")
    )
    assert wrapped_field.is_type_type


def test_is_type_type_for_bare_type_annotation():
    """
    A field annotated with the bare builtin ``type`` (not the parametrized ``Type[X]``)
    is also a type-type field — ``get_origin(type)`` is ``None``, unlike
    ``get_origin(Type[X])``, so this is a distinct case from :func:`test_is_type_type`.
    """
    wrapped_class = WrappedClass(clazz=KRROODBarePositionTypeWrapper)
    wrapped_field = WrappedField(
        wrapped_class,
        get_field_by_name(KRROODBarePositionTypeWrapper, "position_type"),
    )
    assert wrapped_field.is_type_type


def test_is_specialized_generic():
    wrapped_class = WrappedClass(clazz=GenericClassAssociation)
    associated_value = WrappedField(
        wrapped_class, get_field_by_name(GenericClassAssociation, "associated_value")
    )
    assert associated_value.is_instantiation_of_generic_class
    associated_value_not_parametrized = WrappedField(
        wrapped_class,
        get_field_by_name(GenericClassAssociation, "associated_value_not_parametrized"),
    )
    assert not associated_value_not_parametrized.is_instantiation_of_generic_class

    associated_value_list = WrappedField(
        wrapped_class,
        get_field_by_name(GenericClassAssociation, "associated_value_list"),
    )
    assert associated_value_list.is_container
    assert associated_value_list.is_instantiation_of_generic_class

    associated_value_not_parametrized_list = WrappedField(
        wrapped_class,
        get_field_by_name(
            GenericClassAssociation, "associated_value_not_parametrized_list"
        ),
    )
    assert associated_value_not_parametrized_list.is_container
    assert not associated_value_not_parametrized_list.is_instantiation_of_generic_class


class TypeVariableBound:
    """
    Bound used by the bounded-type-variable mimic below.
    """


_unbounded_type_variable = TypeVar("_unbounded_type_variable")
_bounded_type_variable = TypeVar("_bounded_type_variable", bound=TypeVariableBound)


@dataclass
class FieldTypedAsUnboundedTypeVariable:
    items: List[_unbounded_type_variable] = field(default_factory=list)


@dataclass
class FieldTypedAsBoundedTypeVariable:
    items: List[_bounded_type_variable] = field(default_factory=list)


def test_unbounded_type_variable_field_is_underspecified():
    wrapped_class = WrappedClass(clazz=FieldTypedAsUnboundedTypeVariable)
    wrapped_field = WrappedField(
        wrapped_class, get_field_by_name(FieldTypedAsUnboundedTypeVariable, "items")
    )
    assert wrapped_field.is_underspecified_generic


def test_bounded_type_variable_field_is_specified_to_its_bound():
    wrapped_class = WrappedClass(clazz=FieldTypedAsBoundedTypeVariable)
    wrapped_field = WrappedField(
        wrapped_class, get_field_by_name(FieldTypedAsBoundedTypeVariable, "items")
    )
    assert not wrapped_field.is_underspecified_generic
    assert wrapped_field.type_endpoint is TypeVariableBound


def test_common_base_class():
    assert common_base_class([KRROODCup, KRROODBowl]) is KRROODPhysicalObject
    assert common_base_class([KRROODPosition4D, KRROODPosition5D]) is KRROODPosition4D
    assert common_base_class([KRROODPosition, KRROODPhysicalObject]) is None
    assert common_base_class([KRROODPosition]) is KRROODPosition

    assert common_base_class([]) is None
    assert (
        common_base_class(["UnresolvedRef", KRROODCup, KRROODBowl])
        is KRROODPhysicalObject
    )

    assert common_base_class([KRROODCup, KRROODBowl, NoneType]) is None
