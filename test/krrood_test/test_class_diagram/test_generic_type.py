from typing_extensions import (
    TypeVar,
    List,
    get_origin,
    get_args,
    Optional,
    Union,
    Tuple,
)

from krrood.class_diagrams.utils import resolve_type


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
