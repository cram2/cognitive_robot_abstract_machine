from __future__ import annotations
from dataclasses import dataclass, field
from .role_taker_for_university_ontology import PersonAsRoleTakerInAnotherModule
from krrood.entity_query_language.predicate import HasType, HasTypes, Predicate, Symbol, length
from typing_extensions import List, Set


@dataclass
class RoleForPersonAsRoleTakerInAnotherModule(PersonAsRoleTakerInAnotherModule):
    person: PersonAsRoleTakerInAnotherModule
    name: str = field(init=False)


@dataclass(eq=False)
class Student(RoleForPersonAsRoleTakerInAnotherModule):
    # Original Owner of the takes_course field
    takes_course: List[Course] = field(default_factory=list, kw_only=True)


@dataclass(eq=False)
class RecognizedGroup(Symbol):
    name: str
    members: Set[Person] = field(default_factory=set)
    sub_organization_of: List[RecognizedGroup] = field(default_factory=list)


@dataclass(eq=False)
class Person(Symbol):
    name: str
    works_for: RecognizedGroup = field(default=None, kw_only=True)
    member_of: List[RecognizedGroup] = field(default_factory=list, kw_only=True)
    head_of: RecognizedGroup = field(init=False)
    delegate_of: RecognizedGroup = field(init=False)
    teacher_of: List[Course] = field(init=False)
    representative_of: RecognizedGroup = field(init=False)


@dataclass
class RoleForPerson(Person):
    person: Person
    name: str = field(init=False)
    works_for: RecognizedGroup = field(init=False)
    member_of: List[RecognizedGroup] = field(init=False)


@dataclass(eq=False)
class ProfessorAsFirstRole(RoleForPerson):
    # Original Owner of the teacher_of field
    teacher_of: List[Course] = field(default_factory=list, kw_only=True)


@dataclass(unsafe_hash=True)
class Course(Symbol):
    name: str


@dataclass(eq=False)
class Country(RecognizedGroup):
    ...


@dataclass(eq=False)
class Company(RecognizedGroup):
    ...


@dataclass(eq=False)
class CEOAsFirstRole(RoleForPerson):
    # Original Owner of the head_of field
    head_of: RecognizedGroup = field(default=None, kw_only=True)


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


@dataclass(eq=False)
class AssociateProfessorAsSubClassOfARoleInSameModule(ProfessorAsFirstRole):
    ...


