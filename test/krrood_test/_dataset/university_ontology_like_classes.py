from __future__ import annotations

from dataclasses import dataclass, field

from typing_extensions import Set, List, Type

from krrood.class_diagrams.utils import Role
from krrood.entity_query_language.predicate import Symbol
from krrood.ontomatic.property_descriptor.mixins import (
    HasInverseProperty,
    TransitiveProperty,
)
from krrood.ontomatic.property_descriptor.property_descriptor import (
    PropertyDescriptor,
)


@dataclass
class Company(Symbol):
    name: str
    members: Set[Person] = field(default_factory=set)
    sub_organization_of: List[Company] = field(default_factory=list)

    def __hash__(self):
        return hash(self.name)


@dataclass
class Person(Symbol):
    name: str
    works_for: Company = None
    member_of: List[Company] = field(default_factory=list)

    def __hash__(self):
        return hash(self.name)


@dataclass
class CEO(Role[Person], Symbol):
    person: Person
    head_of: Company = None

    def __hash__(self):
        return hash(self.person)


@dataclass
class Member(PropertyDescriptor, HasInverseProperty):

    @classmethod
    def get_inverse(cls) -> Type[MemberOf]:
        return MemberOf


@dataclass
class MemberOf(PropertyDescriptor, HasInverseProperty):
    @classmethod
    def get_inverse(cls) -> Type[Member]:
        return Member


@dataclass
class WorksFor(MemberOf):
    pass


@dataclass
class HeadOf(WorksFor):
    pass


@dataclass
class SubOrganizationOf(PropertyDescriptor, TransitiveProperty): ...


# Person fields' descriptors
Person.works_for = WorksFor(Person, "works_for")
Person.member_of = MemberOf(Person, "member_of")

# CEO fields' descriptors
CEO.head_of = HeadOf(CEO, "head_of")

# Company fields' descriptors
Company.members = Member(Company, "members")
Company.sub_organization_of = SubOrganizationOf(Company, "sub_organization_of")
