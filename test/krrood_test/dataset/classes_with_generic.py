from __future__ import annotations

from abc import ABC
from dataclasses import dataclass, field
from typing import Union

from typing_extensions import List, TypeVar, Generic

from krrood.entity_query_language.core.mapped_variable import MappedVariable
from krrood.entity_query_language.factories import variable
from krrood.patterns.subclass_safe_generic import SubClassSafeGeneric
from krrood.utils import T

U = TypeVar("U")
V = TypeVar("V")


@dataclass(kw_only=True)
class FirstGeneric(Generic[T], SubClassSafeGeneric):
    attribute_using_generic: T
    generic_attribute_using_generic: List[T] = field(default_factory=list, kw_only=True)


@dataclass(kw_only=True)
class SubClassGenericThatUpdatesGenericTypeToBuiltInType(FirstGeneric[int]): ...


@dataclass(kw_only=True, eq=False)
class BareGenericFieldHolder:
    """
    Holds a field typed to a bare, unparametrized generic class.

    ``FirstGeneric`` never binds its own type parameter, but its concrete
    parametrizations, such as :class:`SubClassGenericThatUpdatesGenericTypeToBuiltInType`,
    are mapped, so ``item`` must round trip polymorphically through that base rather than
    being dropped.
    """

    item: FirstGeneric


@dataclass(kw_only=True)
class SubClassGenericThatRecreatesAField(FirstGeneric[int]):
    generic_attribute_using_generic: List[int] = field(default_factory=list)


@dataclass(kw_only=True)
class SubClassGenericThatRecreatesAFieldWithNonBuiltInType(FirstGeneric[FirstGeneric]):
    generic_attribute_using_generic: List[FirstGeneric] = field(default_factory=list)


@dataclass(kw_only=True)
class SubClassGenericThatUpdatesGenericTypeToTypeDefinedInSameModule(
    FirstGeneric[FirstGeneric]
): ...


@dataclass(kw_only=True)
class SubClassGenericThatUpdatesGenericTypeToTypeDefinedInImportedModuleOfThisLibrary(
    FirstGeneric[MappedVariable]
): ...


NewTypeVar = TypeVar("NewTypeVar", bound=FirstGeneric)


@dataclass(kw_only=True)
class SubClassGenericThatUpdatesGenericTypeToAnotherTypeVar(
    FirstGeneric[NewTypeVar]
): ...


@dataclass(kw_only=True)
class SubClassGenericThatRecreatesAFieldWithAnotherVar(FirstGeneric[NewTypeVar]):
    generic_attribute_using_generic: List[NewTypeVar] = field(default_factory=list)


T2 = TypeVar("T2")


@dataclass(kw_only=True)
class TwoGenericSubClassSafe(Generic[T, T2], SubClassSafeGeneric, ABC): ...


@dataclass(kw_only=True)
class TwoGenericContainer(TwoGenericSubClassSafe[U, V]):
    first_attribute: U
    second_attribute: V
    list_of_first: List[U] = field(default_factory=list, kw_only=True)
    list_of_second: List[V] = field(default_factory=list, kw_only=True)


@dataclass(kw_only=True)
class TwoGenericContainerBoundToBuiltIns(TwoGenericContainer[int, str]): ...


@dataclass(kw_only=True, eq=False)
class GenericListClass(Generic[T], SubClassSafeGeneric, ABC):
    generic_variable: T = field(default=None)
    generic_list: list[T] = field(default_factory=list)


@dataclass(kw_only=True, eq=False)
class ExampleClass: ...


@dataclass(kw_only=True, eq=False)
class CombinedClass(ExampleClass, GenericListClass[str]): ...


@dataclass(kw_only=True, eq=False)
class OneGenericSubClassSafe(Generic[T], SubClassSafeGeneric, ABC):
    one_generic_first_argument: T


@dataclass(kw_only=True, eq=False)
class CombinedThreeGenericSubClassSafe(
    Generic[U, V], OneGenericSubClassSafe[Union[U, V]]
):
    combined_three_generic_first_argument: U
    combined_three_generic_second_argument: V


@dataclass(kw_only=True, eq=False)
class ComplexCombinedThreeGenericSubClassSafe(
    CombinedThreeGenericSubClassSafe[ExampleClass, CombinedClass]
): ...


@dataclass(kw_only=True, eq=False)
class CombinedThreeGenericSubClassSafeWithThirdType(
    Generic[U, V], OneGenericSubClassSafe[int]
):
    combined_three_generic_first_argument: U
    combined_three_generic_second_argument: V


@dataclass(kw_only=True, eq=False)
class ComplexCombinedThreeGenericSubClassSafeWithThirdTypes(
    CombinedThreeGenericSubClassSafeWithThirdType[ExampleClass, CombinedClass]
): ...
