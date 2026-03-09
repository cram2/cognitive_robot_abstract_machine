from __future__ import annotations

from dataclasses import dataclass, field, Field, fields

from typing_extensions import Set, List

from krrood.patterns.role import Role
from krrood.entity_query_language.predicate import Symbol


@dataclass(eq=False)
class RecognizedGroup(Symbol):
    name: str

    members: Set[Person] = field(default_factory=set)
    sub_organization_of: List[RecognizedGroup] = field(default_factory=list)

    def __hash__(self):
        return hash(self.name)


@dataclass(eq=False)
class Company(RecognizedGroup): ...


@dataclass(eq=False)
class Country(RecognizedGroup): ...


@dataclass(eq=False)
class Person(Symbol):
    name: str
    works_for: RecognizedGroup = None
    member_of: List[RecognizedGroup] = field(default_factory=list)

    def __hash__(self):
        return hash(self.name)


@dataclass(eq=False)
class CEOAsFirstRole(Role[Person], Symbol):
    person: Person
    head_of: RecognizedGroup = None

    @classmethod
    def role_taker_field(cls) -> Field:
        return [f for f in fields(cls) if f.name == "person"][0]


@dataclass(eq=False)
class RepresentativeAsSecondRole(Role[CEOAsFirstRole], Symbol):
    ceo: CEOAsFirstRole
    representative_of: RecognizedGroup = None

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
