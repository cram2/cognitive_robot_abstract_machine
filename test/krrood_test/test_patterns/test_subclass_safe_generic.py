import sys
from dataclasses import fields

from typing_extensions import (
    get_type_hints,
    get_args,
    get_origin,
)

from krrood.entity_query_language.factories import variable_from
from krrood.patterns.subclass_safe_generic import SubClassSafeGeneric
from krrood.utils import get_generic_type_param
from ..dataset.classes_with_generic import (
    FirstGeneric,
    SubClassGenericThatUpdatesGenericTypeToBuiltInType,
    SubClassGenericThatUpdatesGenericTypeToTypeDefinedInSameModule,
    SubClassGenericThatUpdatesGenericTypeToAnotherTypeVar,
    SubClassGenericThatUpdatesGenericTypeToTypeDefinedInImportedModuleOfThisLibrary,
    SubClassGenericThatRecreatesAField,
    SubClassGenericThatRecreatesAFieldWithAnotherVar,
    SubClassGenericThatRecreatesAFieldWithNonBuiltInType,
    TwoGenericContainerBoundToBuiltIns,
)


def test_resolve_generic_type_same_class():
    _assert_generic_type_is_resolved(FirstGeneric)


def test_resolve_generic_type_subclass_with_built_in_type_as_generic_type():
    cls = SubClassGenericThatUpdatesGenericTypeToBuiltInType
    _assert_generic_type_is_resolved(cls)


def test_resolving_generic_type_preserves_field_kwargs():
    cls = SubClassGenericThatUpdatesGenericTypeToBuiltInType
    field_name = variable_from(cls).generic_attribute_using_generic._attribute_name_
    field_ = next(f for f in fields(cls) if f.name == field_name)
    assert field_.default_factory is list
    assert field_.kw_only


def test_resolving_generic_type_preserves_parent_field_kwargs():
    cls = FirstGeneric
    assert_field_kwargs_are_preserved_when_resolving_generic_type(cls, kw_only=True)


def test_recreated_field_with_built_in_type_is_preserved_when_resolving_generic_type():
    cls = SubClassGenericThatRecreatesAField
    assert_field_kwargs_are_preserved_when_resolving_generic_type(cls)


def test_recreated_field_with_non_builtin_type_is_preserved_when_resolving_generic_type():
    cls = SubClassGenericThatRecreatesAFieldWithNonBuiltInType
    assert_field_kwargs_are_preserved_when_resolving_generic_type(cls)


def test_recreated_field_with_var_type_is_preserved_when_resolving_generic_type():
    cls = SubClassGenericThatRecreatesAFieldWithAnotherVar
    assert_field_kwargs_are_preserved_when_resolving_generic_type(cls)


def assert_field_kwargs_are_preserved_when_resolving_generic_type(cls, kw_only=False):
    field_name = variable_from(cls).generic_attribute_using_generic._attribute_name_
    field_ = next(f for f in fields(cls) if f.name == field_name)
    assert field_.default_factory is list
    assert field_.kw_only == kw_only
    evaluated_type = eval(field_.type, sys.modules[cls.__module__].__dict__)
    assert get_origin(evaluated_type) is list
    assert (
        get_args(evaluated_type)[0]
        is get_generic_type_param(cls, SubClassSafeGeneric)[0]
    )


def test_resolve_generic_type_subclass_with_type_defined_in_same_module_as_generic_type():
    cls = SubClassGenericThatUpdatesGenericTypeToTypeDefinedInSameModule
    _assert_generic_type_is_resolved(cls)


def test_resolve_generic_type_subclass_with_type_defined_in_imported_module_of_this_library():
    cls = (
        SubClassGenericThatUpdatesGenericTypeToTypeDefinedInImportedModuleOfThisLibrary
    )
    _assert_generic_type_is_resolved(cls)


def test_resolve_generic_type_subclass_with_new_type_var_as_generic_type():
    cls = SubClassGenericThatUpdatesGenericTypeToAnotherTypeVar
    _assert_generic_type_is_resolved(cls)


def test_resolve_two_generic_types_subclass_with_built_in_types():
    cls = TwoGenericContainerBoundToBuiltIns
    resolved_hints = get_type_hints(cls, include_extras=True)
    assert resolved_hints[variable_from(cls).first_attribute._attribute_name_] is int
    assert resolved_hints[variable_from(cls).second_attribute._attribute_name_] is str
    list_of_first = resolved_hints[variable_from(cls).list_of_first._attribute_name_]
    list_of_second = resolved_hints[variable_from(cls).list_of_second._attribute_name_]
    assert get_origin(list_of_first) is list and get_args(list_of_first)[0] is int
    assert get_origin(list_of_second) is list and get_args(list_of_second)[0] is str


def _assert_generic_type_is_resolved(cls):
    resolved_hints = get_type_hints(cls, include_extras=True)
    generic_type = get_generic_type_param(cls, SubClassSafeGeneric)[0]
    assert (
        resolved_hints[variable_from(cls).attribute_using_generic._attribute_name_]
        is generic_type
    )
    nested_generic_type = resolved_hints[
        variable_from(cls).generic_attribute_using_generic._attribute_name_
    ]
    assert (
        get_origin(nested_generic_type) is list
        and get_args(nested_generic_type)[0] is generic_type
    )
