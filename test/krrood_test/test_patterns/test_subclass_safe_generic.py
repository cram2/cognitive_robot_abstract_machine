import sys
from dataclasses import fields, field as dataclass_field

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
)


from dataclasses import dataclass

from typing_extensions import TypeVar

from krrood.class_diagrams.class_diagram import ClassDiagram
from krrood.patterns.subclass_safe_generic import SubClassSafeGeneric


def test_multi_generic_through_inheritance():

    T = TypeVar("T")
    U = TypeVar("U")

    @dataclass
    class A(SubClassSafeGeneric[T]):
        a: T

    @dataclass
    class B(A[int]): ...

    @dataclass
    class C(B, SubClassSafeGeneric[U]):
        b: U

    @dataclass
    class D(C[str]): ...

    class_diagram = ClassDiagram([A, B, C, D])
    D_wrapped = class_diagram.get_wrapped_class(D)
    for f in D_wrapped.fields:
        if f.name == "a":
            assert f.type_endpoint is int
        if f.name == "b":
            assert f.type_endpoint is str


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


# ---------------------------------------------------------------------------
# Regression tests for field-pollution bugs in SubClassSafeGeneric
# ---------------------------------------------------------------------------


def test_update_field_kwargs_finds_field_via_full_mro_not_only_nearest_ancestor():
    """
    Regression – _update_field_kwargs field lookup must search the full MRO.

    Before the fix, the else-branch used fields(cls) which reads only the FIRST
    ancestor's __dataclass_fields__ via MRO lookup.  For a class like:

        class A:              # has field 'items' with default_factory=list
        class B(A): pass      # B.__dataclass_fields__ not in B.__dict__ yet
        class C(OtherBase, B[T]): pass   # OtherBase.__dataclass_fields__ is found first

    fields(C) before @dataclass runs returns OtherBase's fields, missing 'items'.
    _update_field_kwargs then hit the else/else path and:
      1. (old) created bare field() → MISSING default → required argument; or
      2. (with only `if non_type_kwargs:`) skipped the setattr but still wrote the
         annotation, so @dataclass created a required field anyway.

    The fix searches cls.__mro__[1:] __dataclass_fields__ directly.
    """
    from dataclasses import MISSING, Field as DField

    T_local = TypeVar("T_local")
    U_local = TypeVar("U_local")

    @dataclass
    class _Inner(SubClassSafeGeneric[T_local]):
        inner_items: list = dataclass_field(default_factory=list, kw_only=True)

    @dataclass
    class _Unrelated:
        unrelated: int = 0

    # _Combo inherits from _Unrelated first, then _Inner[int].
    # Before @dataclass runs, fields(_Combo) returns _Unrelated.__dataclass_fields__
    # which doesn't contain 'inner_items'.  SubClassSafeGeneric must still find
    # it via the full MRO search.
    @dataclass
    class _Combo(_Unrelated, _Inner[int]):
        pass

    f = next(f for f in fields(_Combo) if f.name == "inner_items")
    assert f.default_factory is list, (
        "inner_items.default_factory was lost: _update_field_kwargs did not find "
        "the field via full MRO search and created a bare required field instead"
    )
    instance = _Combo()
    assert instance.inner_items == []


def test_subclass_safe_generic_type_resolution_failure_does_not_kill_class_definition():
    """
    Regression – SubClassSafeGeneric.__init_subclass__ robustness.

    If get_and_resolve_generic_type_hints_of_object_using_substitutions raises
    (e.g. due to a circular TYPE_CHECKING import), the class definition must still
    succeed.  Before the fix the exception propagated out of __init_subclass__ and
    killed the class entirely, causing an ImportError cascade.
    """
    from unittest.mock import patch

    T_local = TypeVar("T_local")

    @dataclass
    class _Storage(SubClassSafeGeneric[T_local]):
        items: list = dataclass_field(default_factory=list, kw_only=True)
        required: T_local = dataclass_field(kw_only=True)

    target = (
        "krrood.patterns.subclass_safe_generic"
        ".get_and_resolve_generic_type_hints_of_object_using_substitutions"
    )
    with patch(target, side_effect=RuntimeError("simulated circular-import failure")):

        @dataclass
        class _IntStorage(_Storage[int]):
            pass

    # Class must have been defined despite the resolution failure.
    assert _IntStorage is not None
    # Fields from the parent must still be intact with their original defaults.
    items_field = next(f for f in fields(_IntStorage) if f.name == "items")
    assert (
        items_field.default_factory is list
    ), "items.default_factory was lost after type-resolution failure"
    instance = _IntStorage(required=1)
    assert instance.items == []


def test_subclass_safe_generic_inherited_default_factory_survives_type_update():
    """
    Regression – SubClassSafeGeneric type update must not strip default_factory.

    When the type of an inherited field that has default_factory is updated via
    _update_field_kwargs (else-branch, field_ is a copy), the copy must retain
    default_factory so the field remains optional in the child class __init__.
    """
    T_local = TypeVar("T_local")

    @dataclass
    class _Container(SubClassSafeGeneric[T_local]):
        objects: list = dataclass_field(default_factory=list, kw_only=True)
        key: T_local = dataclass_field(kw_only=True)

    @dataclass
    class _IntContainer(_Container[int]):
        pass

    objects_field = next(f for f in fields(_IntContainer) if f.name == "objects")
    assert objects_field.default_factory is list, (
        "default_factory was dropped from 'objects' when SubClassSafeGeneric "
        "updated its type — field became required"
    )
    # Must be constructible without passing objects
    instance = _IntContainer(key=42)
    assert instance.objects == []
