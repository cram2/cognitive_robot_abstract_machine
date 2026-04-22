from __future__ import annotations

from dataclasses import dataclass, field

from typing_extensions import List, TypeVar

from krrood.entity_query_language.core.mapped_variable import MappedVariable
from krrood.entity_query_language.factories import variable
from krrood.patterns.subclass_safe_generic import SubClassSafeGeneric
from krrood.utils import T


@dataclass
class FirstGeneric(SubClassSafeGeneric[T]):
    attribute_using_generic: T
    generic_attribute_using_generic: List[T] = field(default_factory=list, kw_only=True)


@dataclass
class SubClassGenericThatUpdatesGenericTypeToBuiltInType(FirstGeneric[int]): ...


@dataclass
class SubClassGenericThatRecreatesAField(FirstGeneric[int]):
    generic_attribute_using_generic: List[int] = field(default_factory=list)


@dataclass
class SubClassGenericThatRecreatesAFieldWithNonBuiltInType(FirstGeneric[FirstGeneric]):
    generic_attribute_using_generic: List[FirstGeneric] = field(default_factory=list)


@dataclass
class SubClassGenericThatUpdatesGenericTypeToTypeDefinedInSameModule(
    FirstGeneric[FirstGeneric]
): ...


@dataclass
class SubClassGenericThatUpdatesGenericTypeToTypeDefinedInImportedModuleOfThisLibrary(
    FirstGeneric[MappedVariable]
): ...


NewTypeVar = TypeVar("NewTypeVar", bound=FirstGeneric)


@dataclass
class SubClassGenericThatUpdatesGenericTypeToAnotherTypeVar(
    FirstGeneric[NewTypeVar]
): ...


@dataclass
class SubClassGenericThatRecreatesAFieldWithAnotherVar(FirstGeneric[NewTypeVar]):
    generic_attribute_using_generic: List[NewTypeVar] = field(default_factory=list)
