from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from typing_extensions import Set, List

from krrood.entity_query_language.predicate import Symbol

if TYPE_CHECKING:
    from .university_ontology_like_classes_without_descriptors import (
        Company,
        Country,
        Course,
        RecognizedGroup,
    )

@dataclass(eq=False)
class Person(Symbol):
    name: str
    works_for: RecognizedGroup = field(kw_only=True, default=None)
    member_of: List[RecognizedGroup] = field(kw_only=True, default_factory=list)
    head_of: RecognizedGroup = field(init=False)
    representative_of: RecognizedGroup = field(init=False)
    delegate_of: RecognizedGroup = field(init=False)
    teacher_of: List[Course] = field(init=False)

@dataclass
class RoleForPerson(Person):
    person: Person
    name: str = field(init=False)
    works_for: RecognizedGroup = field(init=False)
    member_of: List[RecognizedGroup] = field(init=False)

@dataclass(eq=False)
class CEOAsFirstRole(RoleForPerson):
    # Original Owner of the head_of field
    head_of: RecognizedGroup = field(default=None, kw_only=True)

@dataclass(eq=False)
class ProfessorAsFirstRole(RoleForPerson):
    # Original Owner of the teacher_of field
    teacher_of: List[Course] = field(default_factory=list, kw_only=True)

@dataclass
class RoleForCEOAsFirstRole(CEOAsFirstRole):
    ceo: CEOAsFirstRole
    person: Person = field(init=False)
    head_of: RecognizedGroup = field(init=False)

@dataclass(eq=False)
class RepresentativeAsSecondRole(RoleForCEOAsFirstRole):
    # Original Owner of the representative_of field
    representative_of: RecognizedGroup = field(default=None, kw_only=True)

@dataclass
class RoleForRepresentativeAsSecondRole(RepresentativeAsSecondRole):
    representative: RepresentativeAsSecondRole
    ceo: CEOAsFirstRole = field(init=False)
    representative_of: RecognizedGroup = field(init=False)

@dataclass(eq=False)
class DelegateAsThirdRole(RoleForRepresentativeAsSecondRole):
    # Original Owner of the delegate_of field
    delegate_of: RecognizedGroup = field(default=None, kw_only=True)
