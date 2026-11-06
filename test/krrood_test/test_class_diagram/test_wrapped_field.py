from __future__ import annotations

from dataclasses import dataclass, field, fields, Field

import pytest
from types import NoneType

from sortedcontainers import SortedSet
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
    KRROODPipeOptionalOrientation,
)


def get_field_by_name(cls: Type, name: str) -> Field:
    for f in fields(cls):
        if f.name == name:
            return f
    raise ValueError


def test_builtin_not_optional():
    wrapped_class = WrappedClass(clazz=KRROODPosition)
    wrapped_field = WrappedField(
        clazz=wrapped_class, field=get_field_by_name(KRROODPosition, "x")
    )
    assert wrapped_field.resolved_type is float
    assert not wrapped_field.is_container
    assert wrapped_field.is_builtin_type
    assert not wrapped_field.is_optional
    assert not wrapped_field.is_type_type


def test_builtin_optional():
    wrapped_class = WrappedClass(clazz=KRROODOrientation)
    wrapped_field = WrappedField(
        clazz=wrapped_class, field=get_field_by_name(KRROODOrientation, "w")
    )

    assert wrapped_field.contained_type is float
    assert wrapped_field.is_optional
    assert not wrapped_field.is_container
    assert wrapped_field.is_builtin_type
    assert not wrapped_field.is_instantiation_of_generic_class


def test_builtin_optional_written_with_a_pipe():
    # `X | None` and `Optional[X]` denote the same type, so they must be introspected
    # identically; otherwise a field spelled with a pipe silently loses its mapping.
    wrapped_class = WrappedClass(clazz=KRROODPipeOptionalOrientation)
    wrapped_field = WrappedField(
        clazz=wrapped_class, field=get_field_by_name(KRROODPipeOptionalOrientation, "w")
    )

    assert wrapped_field.contained_type is float
    assert wrapped_field.is_optional
    assert not wrapped_field.is_container
    assert wrapped_field.is_builtin_type
    assert not wrapped_field.is_instantiation_of_generic_class


def test_optional_relationship_written_with_a_pipe():
    wrapped_class = WrappedClass(clazz=KRROODPipeOptionalOrientation)
    wrapped_field = WrappedField(
        clazz=wrapped_class,
        field=get_field_by_name(KRROODPipeOptionalOrientation, "position"),
    )

    assert wrapped_field.is_optional
    assert wrapped_field.contained_type is KRROODPosition
    assert not wrapped_field.is_builtin_type


def test_one_to_one_relationship():
    wrapped_class = WrappedClass(clazz=KRROODPose)
    wrapped_field = WrappedField(
        clazz=wrapped_class, field=get_field_by_name(KRROODPose, "position")
    )

    assert not wrapped_field.is_optional
    assert wrapped_field.container_type is None
    assert wrapped_field.resolved_type is KRROODPosition
    assert not wrapped_field.is_builtin_type


def test_one_to_many_relationship():
    wrapped_class = WrappedClass(clazz=KRROODPositions)
    wrapped_field = WrappedField(
        clazz=wrapped_class, field=get_field_by_name(KRROODPositions, "positions")
    )

    assert not wrapped_field.is_optional
    assert wrapped_field.container_type is list
    assert wrapped_field.contained_type is KRROODPosition
    assert not wrapped_field.is_builtin_type
    assert not wrapped_field.is_instantiation_of_generic_class


def test_is_type_type():
    wrapped_class = WrappedClass(clazz=KRROODPositionTypeWrapper)
    wrapped_field = WrappedField(
        clazz=wrapped_class,
        field=get_field_by_name(KRROODPositionTypeWrapper, "position_type"),
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
        clazz=wrapped_class,
        field=get_field_by_name(KRROODBarePositionTypeWrapper, "position_type"),
    )
    assert wrapped_field.is_type_type


def test_is_specialized_generic():
    wrapped_class = WrappedClass(clazz=GenericClassAssociation)
    associated_value = WrappedField(
        clazz=wrapped_class,
        field=get_field_by_name(GenericClassAssociation, "associated_value"),
    )
    assert associated_value.is_instantiation_of_generic_class
    associated_value_not_parametrized = WrappedField(
        clazz=wrapped_class,
        field=get_field_by_name(
            GenericClassAssociation, "associated_value_not_parametrized"
        ),
    )
    assert not associated_value_not_parametrized.is_instantiation_of_generic_class

    associated_value_list = WrappedField(
        clazz=wrapped_class,
        field=get_field_by_name(GenericClassAssociation, "associated_value_list"),
    )
    assert associated_value_list.is_container
    assert associated_value_list.is_instantiation_of_generic_class

    associated_value_not_parametrized_list = WrappedField(
        clazz=wrapped_class,
        field=get_field_by_name(
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


@dataclass(kw_only=True)
class FieldTypedAsUnboundedTypeVariable:
    items: List[_unbounded_type_variable] = field(default_factory=list)


@dataclass(kw_only=True)
class FieldTypedAsBoundedTypeVariable:
    items: List[_bounded_type_variable] = field(default_factory=list)


def test_unbounded_type_variable_field_is_underspecified():
    wrapped_class = WrappedClass(clazz=FieldTypedAsUnboundedTypeVariable)
    wrapped_field = WrappedField(
        clazz=wrapped_class,
        field=get_field_by_name(FieldTypedAsUnboundedTypeVariable, "items"),
    )
    assert wrapped_field.is_underspecified_generic


def test_bounded_type_variable_field_is_specified_to_its_bound():
    wrapped_class = WrappedClass(clazz=FieldTypedAsBoundedTypeVariable)
    wrapped_field = WrappedField(
        clazz=wrapped_class,
        field=get_field_by_name(FieldTypedAsBoundedTypeVariable, "items"),
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


# %% WrappedField.holds_collection


@dataclass(kw_only=True)
class FieldsOfEveryValueShape:
    """
    Fields typed as single values and as collections, each optionally wrapped in a
    union.
    """

    single_value: float
    """
    A single value.
    """

    optional_single_value: float | None
    """
    A single value that may be missing.
    """

    class_reference: Type[KRROODPosition]
    """
    A single class, which is not a collection although its annotation has an origin.
    """

    collection: List[str]
    """
    A collection of values.
    """

    optional_collection: list[str] | None
    """
    A collection of values that may be missing.
    """

    mapping: dict[str, int]
    """
    A mapping, which is a collection of key-value pairs.
    """

    frozen_collection: frozenset[str]
    """
    An immutable collection of values.
    """

    sorted_collection: SortedSet[str]
    """
    A sorted collection of values, whose class is not a subclass of :class:`set`.
    """


@pytest.mark.parametrize(
    "field_name, holds_collection",
    [
        ("single_value", False),
        ("optional_single_value", False),
        ("class_reference", False),
        ("collection", True),
        ("optional_collection", True),
        ("mapping", True),
        ("frozen_collection", True),
        ("sorted_collection", True),
    ],
)
def test_holds_collection(field_name: str, holds_collection: bool):
    wrapped_class = WrappedClass(clazz=FieldsOfEveryValueShape)
    wrapped_field = WrappedField(
        clazz=wrapped_class,
        field=get_field_by_name(FieldsOfEveryValueShape, field_name),
    )
    assert wrapped_field.holds_collection is holds_collection
