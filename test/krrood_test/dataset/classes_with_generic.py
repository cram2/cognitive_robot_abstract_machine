from __future__ import annotations

from dataclasses import dataclass

from typing_extensions import List, TypeVar

from krrood.entity_query_language.core.mapped_variable import MappedVariable
from krrood.entity_query_language.factories import variable
from krrood.patterns.subclass_safe_generic import SubClassSafeGeneric
from krrood.utils import T


@dataclass
class FirstGeneric(SubClassSafeGeneric[T]):
    attribute_using_generic: T
    generic_attribute_using_generic: List[T]

    @classmethod
    def get_attributes_using_generic_type(cls) -> List[MappedVariable]:
        return [
            variable(cls, None).attribute_using_generic,
            variable(cls, None).generic_attribute_using_generic,
        ]


@dataclass
class SubClassGenericThatUpdatesGenericTypeToBuiltInType(FirstGeneric[int]): ...


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
