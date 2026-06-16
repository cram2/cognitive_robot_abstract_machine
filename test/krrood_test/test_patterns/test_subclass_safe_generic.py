import sys
import pytest
from dataclasses import fields, dataclass
from typing import (
    Union,
    TypeVar,
    Generic,
    List,
    Optional,
    Callable,
    Annotated,
    Dict,
    Tuple,
)

from typing_extensions import (
    get_type_hints,
    get_args,
    get_origin,
    TypeVarTuple,
    Unpack,
)

from krrood.entity_query_language.factories import variable_from
from krrood.patterns.subclass_safe_generic import (
    SubClassSafeGeneric,
    AbstractSubClassSafeGeneric,
)
from krrood.utils import get_generic_type_params
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
    CombinedClass,
    ComplexCombinedThreeGenericSubClassSafe,
    ExampleClass,
    ComplexCombinedThreeGenericSubClassSafeWithThirdTypes,
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
        is get_generic_type_params(cls, SubClassSafeGeneric)[0]
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
    generic_type = get_generic_type_params(cls, SubClassSafeGeneric)[0]
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


def test_combined_class_with_generic_subclass_safe_generic_inheritance_second_doesnt_die_from_missing_type_and_is_still_updated():
    cls = CombinedClass
    resolved_hints = get_type_hints(cls, include_extras=True)
    field_name = "generic_list"
    assert resolved_hints[field_name] == list[str]

    field_ = next(f for f in fields(cls) if f.name == field_name)
    assert field_.type == list[str]


def test_ComplexCombinedThreeGenericSubClassSafe():
    cls = ComplexCombinedThreeGenericSubClassSafe
    resolved_hints = get_type_hints(cls, include_extras=True)
    assert resolved_hints["combined_three_generic_first_argument"] == ExampleClass
    assert resolved_hints["combined_three_generic_second_argument"] == CombinedClass
    assert (
        resolved_hints["one_generic_first_argument"]
        == Union[ExampleClass, CombinedClass]
    )


def test_ComplexCombinedThreeGenericSubClassSafeWithThirdTypes():
    cls = ComplexCombinedThreeGenericSubClassSafeWithThirdTypes
    resolved_hints = get_type_hints(cls, include_extras=True)
    assert resolved_hints["combined_three_generic_first_argument"] == ExampleClass
    assert resolved_hints["combined_three_generic_second_argument"] == CombinedClass
    assert resolved_hints["one_generic_first_argument"] == int


T = TypeVar("T")
T2 = TypeVar("T2")
U = TypeVar("U")
V = TypeVar("V")


def test_multiple_generic_parameters_specialization():
    """
    Test that when a base has multiple generic parameters and some are specialized
    with concrete types while others stay generic, all are correctly handled.
    The current implementation using u_roots[0] and zip(unique_bases, specialized_types)
    is expected to fail here because it only maps the first parameter of the first base.
    """

    @dataclass
    class MultiBase(Generic[T, T2], AbstractSubClassSafeGeneric):
        attr1: T
        attr2: T2

    @dataclass
    class Intermediate(Generic[U], MultiBase[U, int]):
        pass

    @dataclass
    class Final(Intermediate[str]):
        pass

    field_dict = {f.name: f for f in fields(Final)}
    assert field_dict["attr1"].type == str
    assert field_dict["attr2"].type == int


def test_deep_inheritance_substitution():
    """
    Test that substitutions are correctly propagated through deep inheritance.
    """

    @dataclass
    class Base(Generic[T], AbstractSubClassSafeGeneric):
        attr: T

    @dataclass
    class Intermediate(Generic[U], Base[U]):
        pass

    @dataclass
    class Final(Intermediate[int]):
        pass

    field_dict = {f.name: f for f in fields(Final)}
    assert field_dict["attr"].type == int


def test_transitive_resolution_complex():
    """
    Test that transitive resolution works for complex types like List[T].
    If T -> U and U -> int, then List[T] should become List[int].
    """

    @dataclass
    class Base(Generic[T], AbstractSubClassSafeGeneric):
        attr: List[T]

    @dataclass
    class Intermediate(Generic[U], Base[U]):
        pass

    @dataclass
    class Final(Intermediate[int]):
        pass

    fields_by_name = {field.name: field for field in fields(Final)}
    assert fields_by_name["attr"].type == List[int]


W = TypeVar("W")


def test_multiple_generic_bases_map_failure():
    """
    Test that multiple generic bases are correctly reconstructed in the substitution map.
    The current zip-based implementation is expected to fail.
    """
    from krrood.utils import ensure_hashable

    @dataclass
    class Base1(Generic[T, T2], AbstractSubClassSafeGeneric):
        pass

    @dataclass
    class Base2(Generic[V], AbstractSubClassSafeGeneric):
        pass

    @dataclass
    class Combined(Generic[U, W], Base1[U, int], Base2[W]):
        pass

    @dataclass
    class Final(Combined[str, bool]):
        pass

    substitutions = Final._get_generic_type_substitutions()

    # T2 should be mapped to int (via Combined)
    assert substitutions.get(ensure_hashable(T2)) == int
    # V should be mapped to W and then to bool, so resolved V should be bool
    # We use substitutions.get because it might be missing or wrongly mapped

    # In a perfect world, we should be able to resolve V to bool through the chain V -> W -> bool
    from krrood.class_diagrams.utils import resolve_type

    result = resolve_type(V, substitutions)
    assert result.resolved_type == bool


def test_transitive_map_failure():
    """
    Test that the substitution map itself is not transitively resolved.
    If T -> U and U -> int, resolving List[T] should give List[int].
    """

    @dataclass
    class Base(Generic[T], AbstractSubClassSafeGeneric):
        pass

    @dataclass
    class Intermediate(Generic[U], Base[U]):
        pass

    @dataclass
    class Final(Intermediate[int]):
        pass

    substitutions = Final._get_generic_type_substitutions()
    from krrood.class_diagrams.utils import resolve_type

    result = resolve_type(List[T], substitutions)
    assert result.resolved_type == List[int]


@pytest.mark.parametrize(
    "type_hint, expected_resolved",
    [
        (Union[T, U], Union[int, str]),
        ("float | T", float | int),
        (Optional[U], Optional[str]),
        (Callable[[T], U], Callable[[int], str]),
        (Annotated[V, "metadata"], Annotated[float, "metadata"]),
        (List[Union[T, Dict[str, U]]], List[Union[int, Dict[str, str]]]),
    ],
)
def test_complex_type_resolution(type_hint, expected_resolved):
    """
    Test that complex types are correctly resolved when bound in a subclass.
    """

    @dataclass
    class Base(Generic[T, U, V], AbstractSubClassSafeGeneric):
        attribute: type_hint

    @dataclass
    class Final(Base[int, str, float]):
        pass

    fields_by_name = {field.name: field for field in fields(Final)}
    assert fields_by_name["attribute"].type == expected_resolved


def test_circular_reference_resolution():
    """
    Test that circular references in type substitutions are handled without infinite recursion.
    """
    T_local = TypeVar("T_local")
    U_local = TypeVar("U_local")

    # Direct cycle: T -> U, U -> T
    substitutions = {T_local: U_local, U_local: T_local}
    resolved = AbstractSubClassSafeGeneric._resolve_substitutions_transitively(
        substitutions
    )
    assert resolved[T_local] in {T_local, U_local}
    assert resolved[U_local] in {T_local, U_local}

    # Indirect cycle: T -> List[U], U -> T
    substitutions = {T_local: List[U_local], U_local: T_local}
    resolved = AbstractSubClassSafeGeneric._resolve_substitutions_transitively(
        substitutions
    )
    assert resolved[T_local] is not None
    assert resolved[U_local] is not None


Ts = TypeVarTuple("Ts")


def test_typevartuple_basic_substitution():
    """
    Test that TypeVarTuple is correctly substituted when bound in a subclass.
    """

    @dataclass
    class Base(Generic[Unpack[Ts]], AbstractSubClassSafeGeneric):
        attribute: Tuple[Unpack[Ts]]

    @dataclass
    class Final(Base[int, str]):
        pass

    fields_by_name = {field.name: field for field in fields(Final)}
    assert fields_by_name["attribute"].type == Tuple[int, str]


def test_typevartuple_mixed_with_typevar():
    """
    Test that TypeVarTuple mixed with normal TypeVar is correctly substituted.
    """

    @dataclass
    class Base(Generic[T, Unpack[Ts]], AbstractSubClassSafeGeneric):
        attribute: Tuple[T, Unpack[Ts]]

    @dataclass
    class Final(Base[float, int, str]):
        pass

    fields_by_name = {field.name: field for field in fields(Final)}
    assert fields_by_name["attribute"].type == Tuple[float, int, str]


def test_typevartuple_multiple_subclasses():
    """
    Test that TypeVarTuple resolution works through multiple levels of inheritance.
    """

    @dataclass
    class Base(Generic[Unpack[Ts]], AbstractSubClassSafeGeneric):
        attribute: Tuple[Unpack[Ts]]

    @dataclass
    class Intermediate(Generic[T, Unpack[Ts]], Base[Unpack[Ts]]):
        other: T

    @dataclass
    class Final(Intermediate[bool, int, str]):
        pass

    fields_by_name = {field.name: field for field in fields(Final)}
    assert fields_by_name["attribute"].type == Tuple[int, str]
    assert fields_by_name["other"].type == bool


def test_typevartuple_unpack_in_different_positions():
    """
    Test Unpack[Ts] in different positions within a Tuple or other generic.
    """

    @dataclass
    class Base(Generic[Unpack[Ts]], AbstractSubClassSafeGeneric):
        attribute: Tuple[int, Unpack[Ts], float]

    @dataclass
    class Final(Base[str, bool]):
        pass

    fields_by_name = {field.name: field for field in fields(Final)}
    # Expected: Tuple[int, str, bool, float]
    assert fields_by_name["attribute"].type == Tuple[int, str, bool, float]


def test_typevartuple_at_start():
    """
    Test TypeVarTuple at the start of Generic declaration.
    """

    @dataclass
    class Base(Generic[Unpack[Ts], T], AbstractSubClassSafeGeneric):
        first: Tuple[Unpack[Ts]]
        last: T

    @dataclass
    class Final(Base[int, str, float]):
        pass

    fields_by_name = {field.name: field for field in fields(Final)}
    assert fields_by_name["first"].type == Tuple[int, str]
    assert fields_by_name["last"].type == float


def test_typevartuple_in_middle():
    """
    Test TypeVarTuple in the middle of Generic declaration.
    """

    @dataclass
    class Base(Generic[T, Unpack[Ts], U], AbstractSubClassSafeGeneric):
        first: T
        middle: Tuple[Unpack[Ts]]
        last: U

    @dataclass
    class Final(Base[int, str, float, bool]):
        pass

    fields_by_name = {field.name: field for field in fields(Final)}
    assert fields_by_name["first"].type == int
    assert fields_by_name["middle"].type == Tuple[str, float]
    assert fields_by_name["last"].type == bool


def test_typevartuple_empty_middle():
    """
    Test TypeVarTuple in the middle with empty arguments for it.
    """

    @dataclass
    class Base(Generic[T, Unpack[Ts], U], AbstractSubClassSafeGeneric):
        first: T
        middle: Tuple[Unpack[Ts]]
        last: U

    @dataclass
    class Final(Base[int, bool]):
        pass

    fields_by_name = {field.name: field for field in fields(Final)}
    assert fields_by_name["first"].type == int
    assert fields_by_name["middle"].type == Tuple[()]
    assert fields_by_name["last"].type == bool


def test_typevartuple_multiple_usage_in_fields():
    """
    Test that TypeVarTuple can be used in multiple fields.
    """

    @dataclass
    class Base(Generic[Unpack[Ts]], AbstractSubClassSafeGeneric):
        first: Tuple[Unpack[Ts]]
        second: List[Tuple[Unpack[Ts]]]

    @dataclass
    class Final(Base[int, str]):
        pass

    fields_by_name = {field.name: field for field in fields(Final)}
    assert fields_by_name["first"].type == Tuple[int, str]
    assert fields_by_name["second"].type == List[Tuple[int, str]]
