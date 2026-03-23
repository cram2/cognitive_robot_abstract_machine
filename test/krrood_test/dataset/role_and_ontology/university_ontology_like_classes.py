from __future__ import annotations

from dataclasses import dataclass, field, Field, fields

from typing_extensions import Set, List, Type

from krrood.entity_query_language.core.mapped_variable import Attribute
from krrood.entity_query_language.factories import variable_from
from krrood.patterns.role.role import Role
from krrood.entity_query_language.predicate import Symbol
from krrood.ontomatic.property_descriptor.mixins import (
    HasInverseProperty,
    TransitiveProperty,
)
from krrood.ontomatic.property_descriptor.property_descriptor import (
    PropertyDescriptor,
)


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
    def role_taker_attribute(cls) -> Person:
        return variable_from(cls).person


@dataclass(eq=False)
class RepresentativeAsSecondRole(Role[CEOAsFirstRole], Symbol):
    ceo: CEOAsFirstRole
    representative_of: RecognizedGroup = None

    @classmethod
    def role_taker_attribute(cls) -> CEOAsFirstRole:
        return variable_from(cls).ceo


@dataclass(eq=False)
class DelegateAsThirdRole(Role[RepresentativeAsSecondRole], Symbol):
    representative: RepresentativeAsSecondRole

    delegate_of: RecognizedGroup = field(kw_only=True, default=None)

    @classmethod
    def role_taker_attribute(cls) -> RepresentativeAsSecondRole:
        return variable_from(cls).representative


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
class RepresentativeOf(HeadOf):
    pass


@dataclass
class DelegateOf(RepresentativeOf):
    pass


@dataclass
class SubOrganizationOf(PropertyDescriptor, TransitiveProperty): ...


# Person fields' descriptors
Person.works_for = WorksFor(Person, "works_for")
Person.member_of = MemberOf(Person, "member_of")

# CEO fields' descriptors
CEOAsFirstRole.head_of = HeadOf(CEOAsFirstRole, "head_of")

# Representative fields' descriptors
RepresentativeAsSecondRole.representative_of = RepresentativeOf(
    RepresentativeAsSecondRole, "representative_of"
)

# Delegate fields' descriptors
DelegateAsThirdRole.delegate_of = DelegateOf(DelegateAsThirdRole, "delegate_of")

# RecognizedGroup fields' descriptors
RecognizedGroup.members = Member(RecognizedGroup, "members")
RecognizedGroup.sub_organization_of = SubOrganizationOf(
    RecognizedGroup, "sub_organization_of"
)
