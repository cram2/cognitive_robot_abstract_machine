from __future__ import annotations

from dataclasses import dataclass, field

from typing_extensions import Set, List, Type

from krrood.patterns.role import Role, role_taker_field
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
    person: Person = role_taker_field()
    head_of: RecognizedGroup = None


@dataclass(eq=False)
class RepresentativeAsSecondRole(Role[CEOAsFirstRole], Symbol):
    ceo: CEOAsFirstRole = role_taker_field()
    representative_of: RecognizedGroup = None


@dataclass(eq=False)
class DelegateAsThirdRole(Role[RepresentativeAsSecondRole], Symbol):
    representative: RepresentativeAsSecondRole = role_taker_field()

    delegate_of: RecognizedGroup = field(kw_only=True, default=None)


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
