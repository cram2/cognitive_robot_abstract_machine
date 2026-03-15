from typing_extensions import (
    get_type_hints,
    get_args,
    get_origin,
    TypeVar,
    List,
    Tuple,
    Optional,
    Union,
)

from dataset.classes_with_generic import (
    SubClassGenericThatUpdatesGenericTypeToBuiltInType,
    SubClassGenericThatUpdatesGenericTypeToTypeDefinedInSameModule,
    SubClassGenericThatUpdatesGenericTypeToAnotherTypeVar,
    SubClassGenericThatUpdatesGenericTypeToTypeDefinedInImportedModuleOfThisLibrary,
)
from krrood.class_diagrams.utils import resolve_type
from krrood.entity_query_language.factories import variable_from
from krrood.patterns.subclass_safe_generic import SubClassSafeGeneric
from krrood.utils import get_generic_type_param
from ..dataset.classes_with_generic import FirstGeneric


def test_generic_type_resolution_simple():
    T = TypeVar("T")
    U = TypeVar("U")
    resolution_result = resolve_type(T, {T: U})
    assert resolution_result.resolved
    assert resolution_result.resolved_type is U


def test_generic_type_resolution_nested_single():
    T = TypeVar("T")
    U = TypeVar("U")
    resolution_result = resolve_type(List[T], {T: U})
    assert resolution_result.resolved
    assert get_origin(resolution_result.resolved_type) is list
    assert get_args(resolution_result.resolved_type)[0] is U


def test_generic_type_resolution_nested_double():
    T = TypeVar("T")
    U = TypeVar("U")
    resolution_result = resolve_type(List[List[T]], {T: U})
    assert resolution_result.resolved
    assert get_origin(resolution_result.resolved_type) is list
    assert get_args(resolution_result.resolved_type)[0] == List[U]


def test_multiple_generic_type_resolution_nested():
    T = TypeVar("T")
    U = TypeVar("U")
    resolution_result = resolve_type(Tuple[T, U], {T: U, U: T})
    assert resolution_result.resolved
    assert get_origin(resolution_result.resolved_type) is tuple
    assert get_args(resolution_result.resolved_type) == (U, T)


def test_generic_type_resolution_with_optional_generic_type():
    T = TypeVar("T")
    U = TypeVar("U")
    resolution_result = resolve_type(Optional[T], {T: U})
    assert resolution_result.resolved
    assert get_origin(resolution_result.resolved_type) == Union
    assert get_args(resolution_result.resolved_type) == (U, type(None))


def test_generic_type_resolution_with_union_of_generic_type():
    T = TypeVar("T")
    U = TypeVar("U")
    resolution_result = resolve_type(Union[T, int], {T: U})
    assert resolution_result.resolved
    assert get_origin(resolution_result.resolved_type) == Union
    assert get_args(resolution_result.resolved_type) == (U, int)


def test_generic_type_resolution_with_union_of_generic_type_and_optional_generic_type():
    T = TypeVar("T")
    U = TypeVar("U")
    resolution_result = resolve_type(Union[T, int, Optional[T]], {T: U})
    assert resolution_result.resolved
    assert get_origin(resolution_result.resolved_type) == Union
    assert get_args(resolution_result.resolved_type) == (U, int, type(None))


def test_generic_type_resolution_with_tuple_of_multiple_generic_types_including_optional_generic_type():
    T = TypeVar("T")
    U = TypeVar("U")
    resolution_result = resolve_type(Tuple[T, int, Optional[T]], {T: U})
    assert resolution_result.resolved
    assert get_origin(resolution_result.resolved_type) is tuple
    assert get_args(resolution_result.resolved_type) == (U, int, Union[U, type(None)])


def test_resolve_generic_type_same_class():
    _assert_generic_type_is_resolved(FirstGeneric)


def test_resolve_generic_type_subclass_with_built_in_type_as_generic_type():
    cls = SubClassGenericThatUpdatesGenericTypeToBuiltInType
    _assert_generic_type_is_resolved(cls)


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
