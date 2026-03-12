from __future__ import annotations

from dataclasses import dataclass, field, Field, fields

from typing_extensions import Set, List

from .role_taker_for_university_ontology import PersonAsRoleTakerInAnotherModule
from krrood.patterns.role import Role
from krrood.entity_query_language.predicate import (
    Symbol,
    Predicate,  # type: ignore
    HasType,  # type: ignore
    HasTypes,  # type: ignore
    length,  # type: ignore
)


@dataclass(eq=False)
class HasName:
    name: str

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        return self.name == other.name


@dataclass(eq=False)
class RecognizedGroup(HasName, Symbol):
    members: Set[Person] = field(default_factory=set)
    sub_organization_of: List[RecognizedGroup] = field(default_factory=list)


@dataclass(eq=False)
class Company(RecognizedGroup): ...


@dataclass(eq=False)
class Country(RecognizedGroup): ...


@dataclass(unsafe_hash=True)
class Course(HasName, Symbol): ...


@dataclass(eq=False)
class Person(HasName, Symbol):
    works_for: RecognizedGroup = None
    member_of: List[RecognizedGroup] = field(default_factory=list)


@dataclass(eq=False)
class CEOAsFirstRole(Role[Person], Symbol):
    person: Person
    head_of: RecognizedGroup = None

    @classmethod
    def role_taker_field(cls) -> Field:
        return [f for f in fields(cls) if f.name == "person"][0]


@dataclass(eq=False)
class DirectDiamondShapedInheritanceWhereOneIsRole(Role[Person], HasName):
    person: Person

    @classmethod
    def role_taker_field(cls) -> Field:
        return [f for f in fields(cls) if f.name == "person"][0]


@dataclass(eq=False)
class InDirectDiamondShapedInheritanceWhereOneIsRole(RecognizedGroup, Role[Person]):
    person: Person = field(kw_only=True)

    @classmethod
    def role_taker_field(cls) -> Field:
        return [f for f in fields(cls) if f.name == "person"][0]


@dataclass(eq=False)
class ProfessorAsFirstRole(Role[Person], Symbol):
    person: Person
    teacher_of: List[Course] = field(default_factory=list, kw_only=True)

    @classmethod
    def role_taker_field(cls) -> Field:
        return [f for f in fields(cls) if f.name == "person"][0]


@dataclass(eq=False)
class AssociateProfessorAsSubClassOfARoleInSameModule(ProfessorAsFirstRole): ...


@dataclass(eq=False)
class RepresentativeAsSecondRole(Role[CEOAsFirstRole], Symbol):
    ceo: CEOAsFirstRole
    representative_of: RecognizedGroup = field(default=None, kw_only=True)

    @classmethod
    def role_taker_field(cls) -> Field:
        return [f for f in fields(cls) if f.name == "ceo"][0]


@dataclass(eq=False)
class DelegateAsThirdRole(Role[RepresentativeAsSecondRole], Symbol):
    representative: RepresentativeAsSecondRole

    delegate_of: RecognizedGroup = field(kw_only=True, default=None)

    @classmethod
    def role_taker_field(cls) -> Field:
        return [f for f in fields(cls) if f.name == "representative"][0]


@dataclass(eq=False)
class Student(Role[PersonAsRoleTakerInAnotherModule], Symbol):
    person: PersonAsRoleTakerInAnotherModule
    takes_course: List[Course] = field(default_factory=list, kw_only=True)

    @classmethod
    def role_taker_field(cls) -> Field:
        return next(f for f in fields(cls) if f.name == "person")
